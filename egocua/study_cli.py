"""Command line for preparing, executing and summarizing research studies."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .controlled_experiments import (
    CONTROLLED_VARIABLES,
    DEFAULT_EVALUATION_CONCURRENCY,
    execution_schedule,
    export_study,
    load_composite_validation_study,
    load_composite_formal_study,
    load_update_count_study,
    model_variant,
    load_formal_study,
    load_frozen_study,
    output_labels,
    read_study,
    repeated_controlled_study,
    sampled_task_ids,
    study_matrix,
)
from .execution import run_isolated
from .experiment import FORMAL_PLAN
from .experiment_output import experiment_folder_name
from .run_ledger import load_run_record, read_run_identity
from .study_execution import EvaluationSettings, run_exported_evaluation
from .study_summary import summarize_records
from .synthesis_workflow import TaskStatus
from .update_count import UPDATE_COUNT


def main() -> None:

    parser = argparse.ArgumentParser(
        description="检查或导出正式评测与受控变量实验；不会调用模型。"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--formal-plan",
        help="登记的正式协议，默认 configs/benchmark.json",
    )
    source.add_argument(
        "--study-from", type=Path, help="读取已准备的实验输入，不重新读取当前成员或审核"
    )
    source.add_argument(
        "--repeat-from", nargs="+", type=Path,
        help="合并已冻结的受控变量并扩展重复；复用原首轮和共享基准",
    )
    parser.add_argument("--repetitions", type=int)
    parser.add_argument(
        "--replicate-id", type=int, action="append",
        help="从已冻结计划选择重复序号后导出；可重复，不增加新轮次",
    )
    source.add_argument(
        "--composite", action="store_true", help="准备已通过语义审核的组合题开发试跑"
    )
    source.add_argument(
        "--composite-formal", action="store_true",
        help="准备全部已合格组合题的 30 步正式评测",
    )
    parser.add_argument(
        "--with-probes",
        action="store_true",
        help="计划登记的探测条件与 online 在同一批正式任务上各跑一次",
    )
    parser.add_argument(
        "--interaction-settings", action="store_true",
        help="为每个正式任务准备 Computer Use、Offline Ego Use 和 Online Ego Use",
    )
    parser.add_argument(
        "--variable",
        choices=(*CONTROLLED_VARIABLES, UPDATE_COUNT),
        help="从正式 online 条件派生一个受控变量；须用 --task 登记样本",
    )
    parser.add_argument(
        "--task", action="append", metavar="TASK_ID",
        help="受控变量或组合题的样本任务，可重复",
    )
    parser.add_argument(
        "--model",
        help="冻结时选定被测模型；组合题必须明确指定",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export", type=Path)
    parser.add_argument(
        "--reuse-from",
        type=Path,
        help="导出新清单时引用旧冻结目录中实际输入相同的有效运行",
    )
    parser.add_argument(
        "--retain-site-for", action="append", default=[], metavar="TASK_ID",
        help="为指定组合题保留旧评测的网站快照，要求复用其有效运行",
    )
    mode.add_argument("--records", nargs="+", type=Path, help="已保存的运行档案目录")
    mode.add_argument(
        "--run-all",
        action="store_true",
        help="执行导出的唯一清单；服务异常或余额不足时停止",
    )
    parser.add_argument("--canary-index", type=int, default=0)
    parser.add_argument(
        "--defer-execution-index",
        type=int,
        action="append",
        default=[],
        help="暂缓已在其他进程执行的序号，仍计入未完成；可重复指定",
    )
    parser.add_argument(
        "--run-purpose",
        choices=("task_validation",),
        help="显式登记本轮用于检查和修复任务",
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_EVALUATION_CONCURRENCY)
    args = parser.parse_args()
    if args.replicate_id is not None and not args.study_from:
        parser.error("--replicate-id requires --study-from")
    if args.repetitions is not None and not args.repeat_from:
        parser.error("--repetitions requires --repeat-from")
    if args.repeat_from and (
        args.export is None or args.model or args.reuse_from or args.variable
        or args.interaction_settings or args.with_probes
    ):
        parser.error("--repeat-from requires --export and retains each source model and runs")
    if args.reuse_from is not None and args.export is None:
        parser.error("--reuse-from requires --export")
    if args.retain_site_for and (not args.composite_formal or args.reuse_from is None):
        parser.error("--retain-site-for requires --composite-formal and --reuse-from")
    composite = args.composite or args.composite_formal
    if not args.study_from and not composite and args.variable != UPDATE_COUNT and (args.variable is None) != (args.task is None):
        parser.error("--variable and --task are used together")
    if composite and (args.with_probes or args.interaction_settings or args.variable):
        parser.error("--composite is a separate development study")
    if composite and not args.model:
        parser.error("--composite requires --model")
    if args.variable == UPDATE_COUNT and not args.study_from and not args.model:
        parser.error("--variable update-count requires --model")
    if args.study_from and (
        args.with_probes or args.interaction_settings or args.task
        or (args.model and args.export is None)
    ):
        parser.error("--study-from reads a frozen study as exported")
    if args.study_from and (args.variable or args.replicate_id is not None) and args.records:
        parser.error("export the selected frozen study before --records")
    if args.with_probes and args.variable:
        parser.error("--with-probes and --variable are separate studies")
    if args.interaction_settings and (args.with_probes or args.variable):
        parser.error("--interaction-settings cannot be combined with probes or variables")
    if args.run_all:
        if args.study_from is None:
            parser.error("--run-all requires --study-from")
        if args.variable or args.replicate_id is not None:
            parser.error("export the selected frozen study before --run-all")
        summary = run_isolated(
            run_exported_evaluation(
                args.study_from.resolve(),
                EvaluationSettings(
                    args.canary_index,
                    args.concurrency,
                    args.run_purpose,
                    tuple(args.defer_execution_index),
                ),
                args.root.resolve(),
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if (
            summary["execution_stop"] is not None
            or summary["failed_request_executions"]
        ):
            raise SystemExit(1)
        return
    if args.repeat_from:
        arms, evidence, gate = repeated_controlled_study(
            args.repeat_from, 3 if args.repetitions is None else args.repetitions, args.root,
        )
    elif args.study_from:
        arms, evidence, gate = load_frozen_study(args.study_from)
        if args.variable:
            arms = [arm for arm in arms if arm.variable == args.variable]
            if not arms:
                parser.error(f"frozen study has no variable {args.variable}")
        if args.replicate_id is not None:
            selected_replicates = tuple(args.replicate_id)
            if len(set(selected_replicates)) != len(selected_replicates):
                parser.error("--replicate-id values must be distinct")
            if any(not set(selected_replicates) <= set(arm.plan.replicate_ids) for arm in arms):
                parser.error("--replicate-id must already exist in every selected arm")
            arms = [replace(arm, plan=replace(arm.plan, replicate_ids=selected_replicates)) for arm in arms]
        if args.model:
            arms = model_variant(arms, args.model)
    elif args.composite_formal:
        arms, evidence, gate = load_composite_formal_study(
            args.root, args.task, model=args.model
        )
    elif args.composite:
        arms, evidence, gate = load_composite_validation_study(
            args.root, args.task, model=args.model
        )
    elif args.variable == UPDATE_COUNT:
        arms, evidence, gate = load_update_count_study(
            args.root, args.task, model=args.model,
        )
    else:
        arms, evidence, gate = load_formal_study(
            args.root, args.formal_plan or FORMAL_PLAN,
            with_probes=args.with_probes, interaction_settings=args.interaction_settings,
            variable=args.variable,
            sample_ids=args.task, model=args.model,
        )
    if args.export:
        args.export = args.export / experiment_folder_name(output_labels(arms))
        export_study(
            arms,
            args.export,
            root=args.root,
            evidence=evidence,
            gate=gate,
            reuse_from=tuple(args.repeat_from) if args.repeat_from else args.reuse_from,
            retain_site_for=tuple(args.retain_site_for),
        )
    if args.records:
        records = []
        for path in args.records:
            record = load_run_record(path)
            experiment_id = read_run_identity(path)["experiment_id"]
            records.append((experiment_id, record))
        output = summarize_records(
            arms,
            records,
            root=args.root,
            evidence=evidence,
            execution_results=read_study(args.study_from).get("execution_results", [])
            if args.study_from
            else None,
        )
    else:
        matrix = study_matrix(arms, root=args.root, evidence=evidence)
        executions = execution_schedule(arms)
        output = {
            "planned_runs": len(matrix),
            "unique_executions": len(executions),
            "reused_conditions": len(matrix) - len(executions),
            "execution_schedule": executions,
            "blocked_by_confirmed_task_defect": sum(
                (gate or {}).get("formal_task_statuses", {}).get(row["run"]["task_id"])
                == TaskStatus.NEEDS_REPAIR
                for row in executions
            ),
            "paid_runs_started": 0,
            "matrix": matrix,
            "variables": {
                variable: {
                    "planned_runs": sum(row["variable"] == variable for row in matrix),
                    "tasks": len(
                        {
                            row["task_id"]
                            for row in matrix
                            if row["variable"] == variable
                        }
                    ),
                }
                for variable in dict.fromkeys(arm.variable for arm in arms)
            },
            "task_sample": sampled_task_ids(arms),
            "review_gate": gate,
        }
    if args.export is not None:
        output["experiment_directory"] = str(args.export.resolve())
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
