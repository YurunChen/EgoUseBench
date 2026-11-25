"""Open the existing review UI as an independent, offline answer-choice check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import webbrowser

from werkzeug.serving import make_server

from egocua.answer_validation import AnswerValidationWorkspace
from egocua.human_reference import SAMPLE
from egocua.review_web import create_answer_validation_app
from egocua.workspace import read_json_strict
from sites.runtime import QuietRequestHandler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="人工答案核验：看材料、选结果，无需搜索"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--item", type=int)
    parser.add_argument("--evaluator")
    parser.add_argument(
        "--preview", action="store_true", help="组织者预览，记录不计入正式结果"
    )
    parser.add_argument(
        "--summary", action="store_true", help="只读取已保存结果，不打开页面"
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    sample = read_json_strict(root / SAMPLE)
    if args.preview and (args.evaluator or args.summary):
        parser.error("预览不使用参与者编号，也不汇总正式结果")
    if not args.evaluator and not args.preview:
        if args.item is not None or args.summary:
            parser.error("正式核验需要 --evaluator 填写独立参与者编号")
        print(
            f"人工答案核验：{len(sample['tasks'])}题。这里只显示序号，不展示题目或答案。"
        )
        for index in range(1, len(sample["tasks"]) + 1):
            print(f"任务 {index:02d}")
        return
    if args.item is not None and not 1 <= args.item <= len(sample["tasks"]):
        parser.error("序号不在当前人工参考清单中")
    participant = args.evaluator or "preview-not-participant"
    output = (
        root / ".scratch/人工答案核验" / f"preview-{secrets.token_hex(6)}"
        if args.preview
        else root / "artifacts/human-answer-validation" / participant
    )
    workspace = AnswerValidationWorkspace(root, participant, output)
    if args.summary:
        print(json.dumps(workspace.summary(), ensure_ascii=False, indent=2))
        return
    app = create_answer_validation_app(workspace, preview=args.preview)
    server = make_server(
        "127.0.0.1", args.port, app, threaded=True, request_handler=QuietRequestHandler
    )
    url = f"http://127.0.0.1:{server.server_port}/"
    if args.item is not None:
        workspace.case(args.item)
        url += f"task/{args.item}"
    print(f"人工答案核验：{url}", flush=True)
    print(f"本地记录：{output}；保持此窗口开启。", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
