"""Reproducible experiment planning and outcome summaries.

Planning never launches a workload. Execution is intentionally a separate,
future Linux-host responsibility; a plan makes baseline matrices reviewable
before any resource-consuming command is run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


BASELINES = ("linux_default", "static_nice_plus_5", "bgopt_heuristic", "bgopt_ml_assisted")
OUTCOME_COLUMNS = (
    "experiment_id", "scenario", "baseline", "run_id", "foreground_latency_ms", "foreground_performance", "background_completion_s", "background_progress", "starvation_seconds", "controller_cpu_percent", "post_action_result",
)


def _toml(path: str | Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def plan(config_path: str | Path) -> dict[str, Any]:
    config = _toml(config_path)
    experiment = config.get("experiment", {})
    experiment_id = str(experiment.get("id", ""))
    scenario = str(experiment.get("scenario", ""))
    repetitions = int(experiment.get("repetitions", 0))
    if not experiment_id or not scenario or repetitions < 2:
        raise ValueError("Experiment config needs experiment.id, experiment.scenario, and repetitions >= 2")
    requested = tuple(config.get("baselines", {}).get("names", BASELINES))
    invalid = sorted(set(requested).difference(BASELINES))
    if invalid:
        raise ValueError(f"Unknown baselines: {invalid}")
    runs = [{"experiment_id": experiment_id, "scenario": scenario, "baseline": baseline, "run_id": f"{baseline}-{rep:02d}", "status": "planned", "dry_run_required": baseline.startswith("bgopt")} for baseline in requested for rep in range(1, repetitions + 1)]
    return {"experiment": experiment, "foreground": config.get("foreground", {}), "background": config.get("background", {}), "runs": runs, "safety_note": "This is a plan only. Commands are recorded for review and are not executed by BGOpt."}


def write_plan(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    report = plan(config_path)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def outcome_template(plan_path: str | Path, output: str | Path) -> int:
    import pandas as pd

    planned = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    frame = pd.DataFrame([{column: run.get(column, "") for column in OUTCOME_COLUMNS} for run in planned["runs"]], columns=OUTCOME_COLUMNS)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return len(frame)


def summarise_outcomes(path: str | Path) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"experiment_id", "scenario", "baseline", "run_id"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Outcome file missing columns: {sorted(missing)}")
    metrics = [column for column in OUTCOME_COLUMNS if column in frame.columns and column not in required | {"post_action_result"}]
    report: dict[str, Any] = {"runs": len(frame), "metrics": {}, "warning": "Summary statistics describe supplied measurements only; they are not evidence of improvement until matched completed repetitions exist."}
    for metric in metrics:
        numeric = pd.to_numeric(frame[metric], errors="coerce")
        subset = frame.assign(_metric=numeric).dropna(subset=["_metric"])
        if subset.empty:
            continue
        grouped: dict[str, Any] = {}
        for baseline, group in subset.groupby("baseline"):
            values = group["_metric"]
            n = len(values)
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            ci = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
            grouped[str(baseline)] = {"n": n, "mean": float(values.mean()), "median": float(values.median()), "stddev": std, "ci95_normal_approx": ci}
        report["metrics"][metric] = grouped
    return report

