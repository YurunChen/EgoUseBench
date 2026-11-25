from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

import pytest

from tests.test_human_page import human_run


def test_participant_can_work_offline_entirely_in_the_web_page(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    with human_run(tmp_path, track="clarify") as (url, token, site, result):
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1440, "height": 1000}, accept_downloads=True
            )
            external_requests = []

            def local_only(route):
                if urlsplit(route.request.url).hostname not in {
                    None,
                    "127.0.0.1",
                    "localhost",
                }:
                    external_requests.append(route.request.url)
                    route.abort()
                else:
                    route.continue_()

            context.route("**/*", local_only)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url)
            page.get_by_text("Find this product", exact=False).wait_for()
            assert "无步数限制" in page.locator("#step").inner_text()
            assert (
                page.locator("#targets").count() == 0
                and page.locator("#action-dialog").count() == 0
            )
            page.locator("#evidence img").first.click()
            page.get_by_role("button", name="关闭大图").click()
            page.get_by_label("需要确认指代时，向佩戴者提问").fill("Which variant?")
            page.get_by_role("button", name="提问", exact=True).click()
            page.get_by_text("The target variant.", exact=False).wait_for()
            website = page.frame_locator("#website")
            website.locator("input[name=q]").fill("Product")
            website.locator("input[name=q]").press("Enter")
            website.locator("a.card-title").filter(
                has_text="clarify Product 1"
            ).first.click()
            website.get_by_role("button", name="Add to Cart", exact=True).click()
            # Several native actions are permitted without control-panel steps.
            website.get_by_text("Cart changes are saved.", exact=True).wait_for()
            page.get_by_role("button", name="完成任务", exact=True).click()
            page.get_by_role("button", name="导出操作记录", exact=True).wait_for()
            record = result.result(timeout=10)
            assert record["meta"]["max_steps"] is None
            assert page.locator("#finish").is_disabled()
            with page.expect_download() as download:
                page.get_by_role("button", name="导出操作记录", exact=True).click()
            exported = json.loads(download.value.path().read_text())
            assert exported["run_id"] == record["meta"]["run_id"]
            assert "task_snapshot" not in exported
            assert [row["command"] for row in exported["trace"]["commands"]] == [
                "ask Which variant?",
                "finish",
            ]
            assert not external_requests and not errors
            browser.close()


def test_opening_raw_template_explains_how_to_start_without_fetching():
    from pathlib import Path
    from egocua.adapters import human

    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        requests = []
        errors = []
        page.on("request", lambda request: requests.append(request.url))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(Path(human.__file__).with_name("human.html").as_uri())
        sync_api.expect(page.locator("#step")).to_have_text("请从启动入口打开")
        sync_api.expect(page.locator("#website-cover")).to_have_text("尚未启动任务服务")
        assert "启动人工执行.command" in page.locator("#notice").inner_text()
        assert page.locator("#finish").is_disabled()
        assert not any("/state" in url or "%7B" in url for url in requests)
        assert not errors
        browser.close()


def test_startup_failure_is_not_shown_as_task_finished():
    from pathlib import Path
    from egocua.adapters import human

    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        html = Path(human.__file__).with_name("human.html").read_text()
        page.route(
            "http://127.0.0.1:19999/",
            lambda route: route.fulfill(body=html, content_type="text/html"),
        )
        page.route("http://127.0.0.1:19999/state", lambda route: route.abort())
        page.goto("http://127.0.0.1:19999/")
        sync_api.expect(page.locator("#step")).to_have_text("连接未建立")
        sync_api.expect(page.locator("#website-cover")).to_have_text(
            re.compile(r"^无法连接任务服务")
        )
        assert page.locator("#finish").is_disabled()
        assert not page.locator("#website").get_attribute("src")
        browser.close()
