from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from egocua.environment import load_site_data
from egocua.run_ledger import canonical_sha256
from egocua.schema import load_tasks
from egocua.workspace import resolve_within, write_json_atomic
from tools.render_site_audit import referenced_entity_ids, render_site_audit


def _exact_duplicate_groups(
    products: list[dict[str, object]], key: str,
) -> dict[str, list[str]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for product in products:
        if key == "image_sha256":
            provenance = product.get("image_provenance", {})
            value = str(
                provenance.get("sha256", "")
                if isinstance(provenance, dict)
                else ""
            ).strip()
        else:
            value = str(product.get(key, "")).strip()
        if key == "name":
            value = " ".join(value.casefold().split())
        grouped[value].append(str(product["item_id"]))
    return {value: ids for value, ids in grouped.items() if value and len(ids) > 1}


def render_all(
    tasks_path: str,
    output_path: str,
    *,
    root: str = ".",
) -> dict[str, object]:
    workspace = Path(root).resolve()
    task_root = resolve_within(
        workspace, tasks_path, context="tasks", must_exist=False
    )
    if not task_root.is_dir():
        raise ValueError(f"tasks directory does not exist: {task_root}")
    output_root = resolve_within(
        workspace, output_path, context="site audits", must_exist=False
    )
    tasks = load_tasks(task_root)
    site_data = {
        site: load_site_data(workspace, site)
        for site in sorted({task["site"] for task in tasks})
    }
    egomart_entities = site_data.get("egomart", {}).get("entities", [])
    duplicate_images = _exact_duplicate_groups(egomart_entities, "image_sha256")
    duplicate_names = _exact_duplicate_groups(egomart_entities, "name")
    image_group_by_id = {
        item_id: ids for ids in duplicate_images.values() for item_id in ids
    }
    name_group_by_id = {
        item_id: ids for ids in duplicate_names.values() for item_id in ids
    }
    artifacts: dict[str, dict[str, object]] = {}
    tasks_with_duplicates = 0
    expected_images = {f"{task['id']}.jpg" for task in tasks}
    for task in tasks:
        task_file = task_root / task["track"] / f"{task['id']}.json"
        destination = output_root / f"{task['id']}.jpg"
        render_site_audit(
            task_file.relative_to(workspace),
            destination.relative_to(workspace),
            root=workspace,
        )
        referenced_ids = referenced_entity_ids(task)
        exact_duplicates = [
            {
                "item_id": item_id,
                "same_image_item_ids": image_group_by_id.get(item_id, []),
                "same_name_item_ids": name_group_by_id.get(item_id, []),
            }
            for item_id in referenced_ids
            if item_id in image_group_by_id or item_id in name_group_by_id
        ]
        tasks_with_duplicates += bool(exact_duplicates)
        artifacts[task["id"]] = {
            "audit_path": destination.relative_to(workspace).as_posix(),
            "audit_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "task_sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
            "site": task["site"],
            "site_data_sha256": canonical_sha256(site_data[task["site"]]),
            "referenced_entity_ids": referenced_ids,
            "exact_duplicate_references": exact_duplicates,
        }
    for stale_image in output_root.glob("*.jpg"):
        if stale_image.name not in expected_images:
            stale_image.unlink()
    index_path = output_root / "index.json"
    write_json_atomic(index_path, {
        "schema_version": 2,
        "sites": {
            site: {
                "entities": len(data["entities"]),
                "site_data_sha256": canonical_sha256(data),
            }
            for site, data in site_data.items()
        },
        "egomart_duplicate_audit": {
            "distinct_images": len({
                str(entity.get("image_path"))
                for entity in egomart_entities if entity.get("image_path")
            }),
            "exact_duplicate_image_groups": len(duplicate_images),
            "listings_in_duplicate_image_groups": sum(map(len, duplicate_images.values())),
            "exact_duplicate_name_groups": len(duplicate_names),
            "listings_in_duplicate_name_groups": sum(map(len, duplicate_names.values())),
            "tasks_with_exact_duplicate_references": tasks_with_duplicates,
        },
        "tasks": artifacts,
    })
    return {
        "rendered": len(tasks),
        "task_ids": [task["id"] for task in tasks],
        "output": output_root.as_posix(),
        "index": index_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render observation/site-product audits for every task."
    )
    parser.add_argument("--tasks", default="data/tasks")
    parser.add_argument("--output", default="artifacts/reports/site-audits")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(
        render_all(args.tasks, args.output, root=args.root),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
