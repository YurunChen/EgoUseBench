"""Export the registered human-quality pages as a browser-only offline package."""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from flask import Flask

from .frame_inputs import observation_images
from .human_reference import SAMPLE
from .sample_review import DECISIONS, PROTOCOL, SampleReviewWorkspace
from .sample_review_web import register_sample_review
from .workspace import resolve_within


class _PortablePage(HTMLParser):
    """Keep the existing material view; replace only transport and form storage."""

    def __init__(self, links: dict[str, str]) -> None:
        super().__init__(convert_charrefs=False)
        self.links = links
        self.parts: list[str] = []
        self.active = False

    def handle_starttag(self, tag, attrs):
        if tag == "main":
            self.active = True
        if not self.active or (tag == "input" and dict(attrs).get("name") == "token"):
            return
        result = []
        for key, value in attrs:
            if key in {"src", "href", "data-src"} and value:
                value = self.links.get(value, value)
                match = re.fullmatch(r"/review/sample/(\d+)", value)
                if match:
                    value = f"#task={match[1]}"
                elif value == "/review/sample":
                    value = "#"
                if value.startswith("/"):
                    raise ValueError(f"离线页面仍引用服务器：{value}")
            if tag == "form" and key == "method":
                continue
            result.append(
                key if value is None else f'{key}="{escape(value, quote=True)}"'
            )
        self.parts.append(f"<{tag}{' ' if result else ''}{' '.join(result)}>")

    def handle_endtag(self, tag):
        if self.active:
            self.parts.append(f"</{tag}>")
        if tag == "main":
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_entityref(self, name):
        if self.active:
            self.parts.append(f"&{name};")

    def handle_charref(self, name):
        if self.active:
            self.parts.append(f"&#{name};")


def export_review_package(root: Path, destination: Path) -> Path:
    """Build a fresh package without existing reviewer judgments or runtime secrets."""
    root = root.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    assets = destination / "assets"
    materials = destination / "materials"
    assets.mkdir()
    materials.mkdir()
    bundle_id = f"human-quality-{uuid4().hex}"
    review = SampleReviewWorkspace(root, "package-builder")
    # Rendering reads no participant records, including any local preview.
    review.output = destination / ".unused-records"
    app = Flask("egocua.review_web")
    app.add_url_rule("/review", "review_index", lambda: "")
    app.add_url_rule("/review/update-count", "update_count_index", lambda: "")
    register_sample_review(app, review)
    client = app.test_client()
    copied: dict[Path, str] = {}
    for item in range(1, len(review.materials.rows) + 1):
        case = review.materials.case(item)
        links = {}
        for index, relative in enumerate(case.assets()):
            path = resolve_within(case.asset_root, relative, context="离线候选图片")
            if path not in copied:
                suffix = path.suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"}:
                    raise ValueError(f"不支持的图片类型：{suffix}")
                copied[path] = f"assets/a{len(copied) + 1:05d}{suffix}"
                shutil.copyfile(path, destination / copied[path])
            links[f"/review/sample/{item}/asset/{index}"] = copied[path]
        for batch, observation in enumerate(
            case.session.initial_payload()["observations"]
        ):
            for index, image in enumerate(
                observation_images(observation, case.asset_root)
            ):
                relative = f"assets/w{item:02d}-{batch:03d}-{index}.jpg"
                if image.media_type != "image/jpeg":
                    raise ValueError(f"不支持的回放图片类型：{image.media_type}")
                (destination / relative).write_bytes(image.encoded_bytes)
                links[f"/review/sample/{item}/frame/{batch}/{index}"] = relative
        response = client.get(f"/review/sample/{item}")
        if response.status_code != 200:
            raise ValueError(f"第{item}题导出失败：{response.status_code}")
        page = _PortablePage(links)
        page.feed(response.get_data(as_text=True))
        payload = {"item": item, "html": "".join(page.parts)}
        (materials / f"{item:02d}.js").write_text(
            f"window.showReviewTask({json.dumps(payload, ensure_ascii=False)});",
            encoding="utf-8",
        )
    metadata = {
        "package_id": bundle_id,
        "protocol": PROTOCOL,
        "sample": SAMPLE,
        "created_at": datetime.now(UTC).isoformat(),
        "decisions": DECISIONS,
        "tasks": review.materials.rows,
    }
    (destination / "package.js").write_text(
        f"window.reviewPackage = {json.dumps(metadata, ensure_ascii=False)};",
        encoding="utf-8",
    )
    static = Path(__file__).parent / "static"
    for name in (
        "review.css",
        "review.js",
        "portable_review.js",
        "portable_review.css",
    ):
        shutil.copyfile(static / name, destination / name)
    shutil.copyfile(
        Path(__file__).parent / "templates/review/portable.html",
        destination / "打开审核.html",
    )
    archive = destination.with_suffix(".zip")
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                output.write(
                    path, Path(destination.name) / path.relative_to(destination)
                )
    return archive
