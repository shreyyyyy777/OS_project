"""Offline reports for model and controller audit logs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def benchmark_log(controller_log: str | Path) -> dict[str, Any]:
    source = Path(controller_log)
    if not source.is_file():
        raise FileNotFoundError(f"Controller log not found: {source}")
    events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    actions = Counter(str(event.get("decision", {}).get("kind", "unknown")) for event in events)
    results = Counter(str(event.get("result", {}).get("status", "unknown")) for event in events)
    labels = Counter(str(event.get("classification", {}).get("workload_label", "unknown")) for event in events)
    pressure = Counter(str(event.get("pressure", {}).get("level", "unknown")) for event in events)
    proposals_under_pressure = sum(
        1 for event in events
        if event.get("pressure", {}).get("level") in {"medium", "high"}
        and event.get("decision", {}).get("kind") != "no_op"
    )
    return {
        "events": len(events),
        "action_counts": dict(actions),
        "result_counts": dict(results),
        "workload_label_counts": dict(labels),
        "pressure_level_counts": dict(pressure),
        "non_noop_proposals_under_medium_or_high_pressure": proposals_under_pressure,
        "interpretation": (
            "This is an audit/coverage report, not a performance claim. Compare matched baseline and "
            "treatment trials using your project-owned completion and foreground-latency outcomes."
        ),
    }


def write_report(report: dict[str, Any], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

