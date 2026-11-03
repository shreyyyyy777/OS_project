from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from utils import load_experiment_artifacts, mode_metrics


def test_load_experiment_artifacts_reads_structured_outputs(tmp_path: Path) -> None:
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    with (tmp_path / "data/processed/experiment_001_final_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "repetition"])
        writer.writeheader(); writer.writerow({"mode": "linux_default", "repetition": "1"})
    summary = {"per_mode": {"linux_default": {"metrics": {"scheduler_lag_p95_ms": {"mean": 1.0}}}}}
    (tmp_path / "reports/experiment_001_final_summary.json").write_text(json.dumps(summary))
    rows, loaded = load_experiment_artifacts(tmp_path)
    assert rows == [{"mode": "linux_default", "repetition": "1"}]
    assert mode_metrics(loaded, "linux_default")["scheduler_lag_p95_ms"]["mean"] == 1.0
