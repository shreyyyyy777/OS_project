"""Command line entry points for reproducible BGOpt experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .controller import Controller, load_config
from .dataset import create_label_template, prepare_training_frame
from .evaluation import run_evaluation
from .experiments import BASELINES, outcome_template, summarise_outcomes, write_plan
from .laps import write_inventory
from .laps_dataset import write_dataset_profile
from .model import evaluate, train
from .reporting import benchmark_log, write_report
from .telemetry import TelemetryCollector


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def command_collect(args: argparse.Namespace) -> int:
    collector = TelemetryCollector(current_user_only=not args.all_users, exclude_process_names=tuple(args.exclude))
    total = 0
    for index in range(args.samples):
        rows = collector.collect({"experiment_id": args.experiment_id, "scenario": args.scenario} if args.experiment_id or args.scenario else None)
        _write_jsonl(args.output, rows)
        total += len(rows)
        if index + 1 < args.samples:
            time.sleep(max(args.interval, 0.1))
    print(json.dumps({"samples": args.samples, "rows_written": total, "output": str(args.output)}))
    return 0


def command_inspect_laps(args: argparse.Namespace) -> int:
    report = write_inventory(args.artifact_root, args.output)
    print(json.dumps({"output": str(args.output), "laps_feature_count": report["laps_feature_count"], "inventory": report["inventory"]}))
    return 0


def command_profile_laps_dataset(args: argparse.Namespace) -> int:
    if not args.trust_pickle:
        raise SystemExit("This reads pickle files. Re-run only for a trusted artifact with --trust-pickle.")
    report = write_dataset_profile(args.dataset_dir, args.feature_map, args.output)
    print(json.dumps({"output": str(args.output), "rows": report["feature_shape"][0], "labels": report["label_count"]}))
    return 0


def command_label_template(args: argparse.Namespace) -> int:
    rows = create_label_template(args.telemetry, args.output)
    print(json.dumps({"rows": rows, "output": str(args.output)}))
    return 0


def command_preprocess(args: argparse.Namespace) -> int:
    report = prepare_training_frame(args.input, args.output, args.labels)
    print(json.dumps({**report, "output": str(args.output)}))
    return 0


def command_train(args: argparse.Namespace) -> int:
    report = train(args.dataset, args.model, args.report, model_kind=args.model_kind, group_column=args.group_column)
    print(json.dumps(report))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    report = evaluate(args.dataset, args.model)
    write_report(report, args.report)
    print(json.dumps({"rows": report["rows"], "output": str(args.report)}))
    return 0


def command_control(args: argparse.Namespace) -> int:
    if args.apply and args.iterations != 1:
        raise SystemExit("Real actions are limited to one iteration per invocation; remove --apply for dry-run loops.")
    controller = Controller(load_config(args.config), args.model)
    total = controller.run(
        args.log,
        interval=args.interval,
        iterations=args.iterations,
        dry_run=not args.apply,
        allow_pid=args.allow_pid,
        allow_cgroup=args.allow_cgroup,
        cgroup_path=args.cgroup_path,
        allow_restore=args.allow_restore,
    )
    print(json.dumps({"events_logged": total, "mode": "apply" if args.apply else "dry_run", "log": str(args.log)}))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    report = benchmark_log(args.controller_log)
    write_report(report, args.report)
    print(json.dumps({"events": report["events"], "output": str(args.report)}))
    return 0


def command_plan_experiment(args: argparse.Namespace) -> int:
    report = write_plan(args.config, args.output)
    print(json.dumps({"runs_planned": len(report["runs"]), "output": str(args.output), "status": "plan_only"}))
    return 0


def command_outcome_template(args: argparse.Namespace) -> int:
    rows = outcome_template(args.plan, args.output)
    print(json.dumps({"rows": rows, "output": str(args.output)}))
    return 0


def command_summarise_outcomes(args: argparse.Namespace) -> int:
    report = summarise_outcomes(args.outcomes)
    write_report(report, args.report)
    print(json.dumps({"runs": report["runs"], "output": str(args.report)}))
    return 0


def command_run_evaluation(args: argparse.Namespace) -> int:
    result = run_evaluation(
        mode=args.mode,
        repetition=args.repetition,
        workload=args.workload,
        output=args.output,
        probe_interval_ms=args.probe_interval_ms,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe, user-space ML-assisted background-process optimisation")
    subcommands = parser.add_subparsers(dest="command", required=True)

    collect = subcommands.add_parser("collect", help="collect portable telemetry to JSONL")
    collect.add_argument("--output", required=True)
    collect.add_argument("--interval", type=float, default=5.0)
    collect.add_argument("--samples", type=int, default=1)
    collect.add_argument("--all-users", action="store_true", help="include other users; off by default")
    collect.add_argument("--exclude", action="append", default=[], help="exact process name to skip; repeatable")
    collect.add_argument("--experiment-id", help="opaque experiment/run identifier written to raw telemetry")
    collect.add_argument("--scenario", help="human-defined scenario name written to raw telemetry")
    collect.set_defaults(handler=command_collect)

    inspect_laps = subcommands.add_parser("inspect-laps", help="read-only inventory of an extracted LAPS artifact")
    inspect_laps.add_argument("--artifact-root", required=True)
    inspect_laps.add_argument("--output", required=True)
    inspect_laps.set_defaults(handler=command_inspect_laps)

    profile_laps = subcommands.add_parser("profile-laps-dataset", help="profile the trusted released LAPS pickle dataset")
    profile_laps.add_argument("--dataset-dir", required=True)
    profile_laps.add_argument("--feature-map", required=True)
    profile_laps.add_argument("--output", required=True)
    profile_laps.add_argument("--trust-pickle", action="store_true", help="required: pickle loading executes deserialization code")
    profile_laps.set_defaults(handler=command_profile_laps_dataset)

    labels = subcommands.add_parser("label-template", help="make a manual workload-label CSV from telemetry")
    labels.add_argument("--telemetry", required=True)
    labels.add_argument("--output", required=True)
    labels.set_defaults(handler=command_label_template)

    preprocess = subcommands.add_parser("preprocess", help="join labels and produce a training CSV")
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--labels")
    preprocess.add_argument("--output", required=True)
    preprocess.set_defaults(handler=command_preprocess)

    training = subcommands.add_parser("train", help="train a lightweight workload classifier")
    training.add_argument("--dataset", required=True)
    training.add_argument("--model", required=True)
    training.add_argument("--report")
    training.add_argument("--model-kind", choices=("hist_gradient_boosting", "random_forest", "decision_tree"), default="hist_gradient_boosting")
    training.add_argument("--group-column", help="experiment/scenario/group column; inferred when omitted")
    training.set_defaults(handler=command_train)

    evaluation = subcommands.add_parser("evaluate", help="evaluate a saved model against labelled data")
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--report", required=True)
    evaluation.set_defaults(handler=command_evaluate)

    control = subcommands.add_parser("control", help="observe/classify/decide; dry-run by default")
    control.add_argument("--config", required=True)
    control.add_argument("--model", help="optional trained BGOpt model; fallback classifier is logged otherwise")
    control.add_argument("--log", required=True)
    control.add_argument("--interval", type=float, default=5.0)
    control.add_argument("--iterations", type=int, default=1)
    control.add_argument("--apply", action="store_true", help="enable narrowly-guarded action application")
    control.add_argument("--allow-pid", type=int, help="exact target PID required for a real action")
    control.add_argument("--allow-cgroup", action="store_true", help="required in addition to --apply for cgroup actions")
    control.add_argument("--cgroup-path", help="existing dedicated cgroup-v2 path; never created by BGOpt")
    control.add_argument("--allow-restore", action="store_true", help="allow a verified same-PID restoration only after --apply")
    control.set_defaults(handler=command_control)

    benchmark = subcommands.add_parser("benchmark", help="summarise a controller audit log")
    benchmark.add_argument("--controller-log", required=True)
    benchmark.add_argument("--report", required=True)
    benchmark.set_defaults(handler=command_benchmark)

    plan_experiment = subcommands.add_parser("plan-experiment", help="create a reviewable baseline matrix; does not run commands")
    plan_experiment.add_argument("--config", required=True)
    plan_experiment.add_argument("--output", required=True)
    plan_experiment.set_defaults(handler=command_plan_experiment)

    outcomes = subcommands.add_parser("outcome-template", help="create an empty measured-outcomes CSV from a plan")
    outcomes.add_argument("--plan", required=True)
    outcomes.add_argument("--output", required=True)
    outcomes.set_defaults(handler=command_outcome_template)

    summary = subcommands.add_parser("summarise-outcomes", help="summarise completed outcome measurements by baseline")
    summary.add_argument("--outcomes", required=True)
    summary.add_argument("--report", required=True)
    summary.set_defaults(handler=command_summarise_outcomes)
    evaluation = subcommands.add_parser(
        "run-evaluation",
        help="run one safe Experiment 001 measurement (Linux default only during validation)",
    )
    evaluation.add_argument("--mode", choices=BASELINES, required=True)
    evaluation.add_argument("--repetition", type=int, required=True)
    evaluation.add_argument("--workload", default="work/compile_workload/workload")
    evaluation.add_argument("--output", default="data/processed/experiment_001_evaluation.jsonl")
    evaluation.add_argument("--probe-interval-ms", type=float, default=50.0)
    evaluation.set_defaults(handler=command_run_evaluation)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "samples", 1) < 1 or getattr(args, "iterations", 1) < 1:
        parser.error("samples and iterations must be at least 1")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
