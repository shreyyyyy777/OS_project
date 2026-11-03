"""Read-only adapters for the BGOpt Streamlit dashboard."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_CSV = PROJECT_ROOT / "data/processed/experiment_001_final_runs.csv"
FINAL_SUMMARY = PROJECT_ROOT / "reports/experiment_001_final_summary.json"
MODEL_PATH = PROJECT_ROOT / "models/bgopt_native_workload_classifier_v4.joblib"


def load_experiment_artifacts(root: Path = PROJECT_ROOT) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load final Experiment 001 artifacts without modifying them."""
    with (root / "data/processed/experiment_001_final_runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (root / "reports/experiment_001_final_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    return rows, summary


def mode_metrics(summary: dict[str, Any], mode: str) -> dict[str, Any]:
    return summary["per_mode"][mode]["metrics"]


def _decision_name(decision: Any) -> str:
    data = decision.as_dict() if hasattr(decision, "as_dict") else {}
    return str(data.get("action") or data.get("decision") or data.get("action_type") or "NO_ACTION")


def live_snapshot(mode: str = "Heuristic") -> dict[str, Any]:
    """Collect one telemetry snapshot and compute suggestions only; never applies them."""
    from bgopt.controller import annotate_protected_descendants, load_config
    from bgopt.features import heuristic_classify
    from bgopt.model import load, predict
    from bgopt.policy import PolicyConfig, decide
    from bgopt.pressure import estimate_pressure
    from bgopt.telemetry import TelemetryCollector

    config = load_config(PROJECT_ROOT / "config/example.toml")
    collector = TelemetryCollector(
        current_user_only=True,
        exclude_process_names=("bgopt", "streamlit"),
    )
    records = collector.collect({"source": "dashboard", "dry_run": True})
    annotate_protected_descendants(records, config.policy.protected_process_names)
    model_payload = load(MODEL_PATH) if mode == "Frozen ML v4" else None
    rows: list[dict[str, Any]] = []
    pressure = estimate_pressure(records[0]) if records else None
    policy_config = PolicyConfig()
    for record in records:
        if int(record.get("pid", -1)) == os.getpid():
            continue
        if model_payload is None:
            label, confidence, source = heuristic_classify(record)
        else:
            label, confidence, source = predict(model_payload, record)
        decision = decide(record, label, pressure, policy_config)
        protected = bool(record.get("is_protected") or record.get("is_protected_descendant"))
        rows.append({
            "Process": record.get("process_name", "unknown"),
            "PID": record.get("pid"),
            "CPU %": round(float(record.get("proc_cpu_percent", 0.0)), 1),
            "Memory MB": round(float(record.get("proc_rss_mb", 0.0)), 1),
            "Classification": label,
            "Confidence": round(float(confidence), 2),
            "Source": source,
            "Role": "foreground" if record.get("is_interactive_hint") or label == "interactive" else "background",
            "Terminal foreground": bool(record.get("is_terminal_foreground")),
            "Protected": protected,
            "Suggested decision": "NO_ACTION (protected)" if protected else _decision_name(decision),
        })
    rows.sort(key=lambda row: float(row["CPU %"]), reverse=True)
    host = records[0] if records else {}
    return {
        "rows": rows,
        "mode": mode,
        "pressure": pressure.as_dict() if pressure else {},
        "host": {
            "cpu_percent": host.get("host_cpu_percent", 0.0),
            "load_1_norm": host.get("host_load_1_norm", 0.0),
            "memory_percent": host.get("host_memory_percent", 0.0),
            "swap_percent": host.get("host_swap_percent", 0.0),
        },
        "classification_counts": dict(Counter(row["Classification"] for row in rows)),
        "protected_count": sum(1 for row in rows if row["Protected"]),
    }


def run_safe_evaluation(repetition: int = 1) -> dict[str, Any]:
    """Run the existing Linux-default evaluation only and return its fresh JSON output."""
    import subprocess
    import sys
    import tempfile

    target = Path(tempfile.gettempdir()) / f"bgopt_dashboard_evaluation_{repetition}.json"
    command = [
        sys.executable, "-m", "bgopt.cli", "run-evaluation",
        "--mode", "linux_default", "--repetition", str(repetition),
        "--workload", str(PROJECT_ROOT / "work/compile_workload/workload"),
        "--output", str(target),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "evaluation failed")
    return json.loads(target.read_text(encoding="utf-8"))
