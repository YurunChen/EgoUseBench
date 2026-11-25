"""Exercise the shareable review package using file URLs with networking disabled."""

import json
import re
from zipfile import ZipFile

import pytest

from egocua.portable_review import _PortablePage, export_review_package
from egocua.sample_review import SampleReviewWorkspace
from tests.test_answer_validation import workspace  # noqa: F401


@pytest.fixture
def package(workspace, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(
        SampleReviewWorkspace, "materials", property(lambda self: workspace.materials)
    )
    # Existing author opinions must not travel to the recipient.
    author = SampleReviewWorkspace(tmp_path, "reviewer-a")
    author.submit(1, "problem", "AUTHOR_PRIVATE_OPINION")
    (tmp_path / ".env").write_text("DO_NOT_PACKAGE_THIS")
    path = tmp_path / "deliverable"
    archive = export_review_package(tmp_path, path)
    # Recipient opens only the extracted archive, not the builder's directory.
    extracted = tmp_path / "recipient"
    with ZipFile(archive) as bundle:
        bundle.extractall(extracted)
    return extracted / path.name


def test_archive_contains_only_materials_and_browser_runtime(package):
    names = [
        p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()
    ]
    assert all(
        n.startswith(("assets/", "materials/"))
        or n
        in {
            "打开审核.html",
            "review.css",
            "review.js",
            "package.js",
            "portable_review.js",
            "portable_review.css",
        }
        for n in names
    )
    text = "\n".join(p.read_text() for p in package.rglob("*.js"))
    assert "AUTHOR_PRIVATE_OPINION" not in text
    assert "DO_NOT_PACKAGE_THIS" not in text
    assert str(package.parent.parent) not in text
    assert len(list((package / "materials").glob("*.js"))) == 2
    page = (package / "materials/01.js").read_text()
    assert "The target variant." in page and "预设答案" in page
    assert "/review/sample/" not in page
    assert 'name=\\"token\\"' not in page
    for asset in re.findall(r"assets/[A-Za-z0-9._-]+", page):
        assert (package / asset).is_file()


def test_export_refuses_to_overwrite_a_package(package):
    with pytest.raises(FileExistsError):
        export_review_package(package, package)


def test_renderer_rejects_unmapped_server_assets():
    page = _PortablePage({})
    with pytest.raises(ValueError, match="服务器"):
        page.feed('<main><img src="/private/file"></main>')


@pytest.fixture
def browser():
    sync = pytest.importorskip("playwright.sync_api")
    with sync.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


def open_offline(browser, package):
    context = browser.new_context(
        accept_downloads=True, viewport={"width": 1100, "height": 900}
    )
    context.set_offline(True)
    page = context.new_page()
    errors, external = [], []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on(
        "request",
        lambda r: (
            external.append(r.url)
            if not r.url.startswith(("file:", "data:", "blob:"))
            else None
        ),
    )
    page.goto((package / "打开审核.html").as_uri())
    return context, page, errors, external


def test_file_only_review_save_resume_export_import_and_first_judgment(
    browser, package, tmp_path
):
    from playwright.sync_api import expect

    context, page, errors, external = open_offline(browser, package)
    page.locator("#reviewer-id").fill("P01")
    page.get_by_role("button", name="开始核查").click()
    expect(page.locator(".task-row")).to_have_count(2)
    assert page.evaluate("getComputedStyle(document.body).overflowWrap") == "anywhere"
    page.locator(".task-row").first.click()
    expect(page.locator("#note")).to_be_visible()
    expect(page.get_by_text("The target variant.", exact=False)).to_be_visible()
    page.locator(".frame-thumb").first.click()
    expect(page.locator(".lightbox")).to_be_visible()
    page.get_by_role("button", name="关闭", exact=True).click()
    page.locator("[value=problem]").check()
    page.get_by_role("button", name="保存并进入下一题").click()
    expect(page.locator("#storage-error")).to_contain_text("原因")
    assert not page.evaluate(
        'Object.values(JSON.parse(localStorage.getItem(Object.keys(localStorage).find(k=>k.endsWith(":P01")))) .records)[0].submitted_at'
    )
    page.locator("#note").fill("Need a clearer frame")
    page.reload()
    expect(page.locator("#note")).to_have_value("Need a clearer frame")
    expect(page.locator("[value=problem]")).to_be_checked()
    page.get_by_role("button", name="保存并进入下一题").click()
    expect(page).to_have_url(re.compile(r"#task=2$"))
    page.locator("#note").fill("Second task draft")
    with page.expect_download() as download:
        page.locator("#export").click()
    record_path = tmp_path / "results.json"
    download.value.save_as(record_path)
    records = json.loads(record_path.read_text())
    assert records["reviewer"] == "P01"
    assert records["records"][0]["decision"] == "problem"
    assert records["records"][0]["submitted_at"]
    assert records["records"][1]["submitted_at"] is None
    page.evaluate('location.hash="task=1"')
    expect(page.get_by_text("已保存：有问题", exact=True)).to_be_visible()
    expect(page.locator("#note")).to_have_count(0)
    assert not errors and not external
    context.close()

    restored, page, errors, external = open_offline(browser, package)
    page.locator("#import").set_input_files(record_path)
    expect(page.locator("#saved-status")).to_have_text("记录已导入并保存")
    expect(page.locator(".task-row")).to_have_count(2)
    page.locator(".task-row").nth(1).click()
    expect(page.locator("#note")).to_have_value("Second task draft")
    page.reload()
    expect(page.locator("#note")).to_have_value("Second task draft")
    # Complete the last task after restoring into a fresh browser.
    page.locator("[value=pass]").check()
    page.get_by_role("button", name="保存并进入下一题").click()
    expect(page.locator(".task-row")).to_have_count(2)
    expect(page.locator(".progress-number strong")).to_have_text("2/2")
    with page.expect_download() as complete:
        page.locator("#export").click()
    completed_path = tmp_path / "completed.json"
    complete.value.save_as(completed_path)
    completed = json.loads(completed_path.read_text())
    assert len(completed["records"]) == 2
    assert all(row["submitted_at"] for row in completed["records"])
    # A returned file cannot replace the first submitted judgment.
    completed["records"][0]["note"] = "Conflicting imported opinion"
    conflict = tmp_path / "conflict.json"
    conflict.write_text(json.dumps(completed))
    page.locator("#import").set_input_files(conflict)
    expect(page.locator("#storage-error")).to_contain_text("未覆盖任何记录")
    page.locator(".task-row").first.click()
    expect(page.get_by_text("Need a clearer frame", exact=True)).to_be_visible()
    assert not errors and not external
    restored.close()


def test_wrong_package_import_leaves_current_judgments_unchanged(
    browser, package, tmp_path
):
    context, page, errors, external = open_offline(browser, package)
    page.locator("#reviewer-id").fill("P02")
    page.get_by_role("button", name="开始核查").click()
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"package_id": "other", "reviewer": "P02", "records": []})
    )
    page.locator("#import").set_input_files(bad)
    from playwright.sync_api import expect

    expect(page.locator("#storage-error")).to_contain_text("不属于当前审核包")
    expect(page.locator(".task-row")).to_have_count(2)
    assert not errors and not external
    context.close()


def test_storage_failure_is_visible_and_never_claimed_saved(browser, package, tmp_path):
    from playwright.sync_api import expect

    context, page, errors, external = open_offline(browser, package)
    page.locator("#reviewer-id").fill("P03")
    page.get_by_role("button", name="开始核查").click()
    page.locator(".task-row").first.click()
    page.locator("#note").wait_for()
    page.evaluate(
        "() => { Storage.prototype.setItem=function(){throw new Error('QuotaExceededError')}; }"
    )
    page.locator("[value=pass]").check()
    expect(page.locator("#saved-status")).to_have_text("未保存")
    expect(page.locator("#storage-error")).to_contain_text("立即点击“导出记录”")
    page.get_by_role("button", name="保存并进入下一题").click()
    expect(page).to_have_url(re.compile(r"#task=1$"))
    with page.expect_download() as result:
        page.locator("#export").click()
    result.value.save_as(tmp_path / "emergency.json")
    assert json.loads((tmp_path / "emergency.json").read_text())["records"][0][
        "submitted_at"
    ]
    assert not errors and not external
    context.close()
