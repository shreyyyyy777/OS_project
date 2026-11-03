"""Preprocess raw telemetry and create a human-reviewable labelling sheet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .features import MODEL_FEATURES, WORKLOAD_CLASSES


def _pandas() -> Any:
    import pandas as pd

    return pd


def read_telemetry(path: str | Path) -> Any:
    pd = _pandas()
    source = Path(path)
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(source, lines=True)
    return pd.read_csv(source)


def create_label_template(telemetry_path: str | Path, output_path: str | Path) -> int:
    """Create one row per observed executable/PID for manual workload labels."""
    telemetry = read_telemetry(telemetry_path)
    columns = [column for column in ("experiment_id", "scenario", "pid", "process_name", "username") if column in telemetry.columns]
    if "pid" not in columns:
        raise ValueError("Telemetry must include pid")
    labels = telemetry.sort_values("timestamp" if "timestamp" in telemetry.columns else "pid")
    labels = labels.drop_duplicates(subset=["pid"], keep="last")[columns].copy()
    labels["workload_label"] = ""
    labels["process_role"] = ""  # foreground or background; supplied by the experiment owner
    labels["resource_label"] = ""  # cpu_intensive, io_intensive, network_intensive, mixed, other
    labels["allowed_labels"] = "|".join(WORKLOAD_CLASSES)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output, index=False)
    return len(labels)


def prepare_training_frame(raw_path: str | Path, output_path: str | Path, labels_path: str | Path | None = None) -> dict[str, int]:
    """Join project-owned labels to telemetry and write numeric model features.

    Label files use ``pid,workload_label``. A PID is intentionally a short-term
    labelling convenience; for a rigorous experiment, capture executable/hash
    and trial metadata as well so labels do not survive a PID reuse.
    """
    pd = _pandas()
    frame = read_telemetry(raw_path)
    if labels_path:
        labels = pd.read_csv(labels_path)
        if not {"pid", "workload_label"}.issubset(labels.columns):
            raise ValueError("Label file must have pid and workload_label columns")
        keep = [column for column in ("experiment_id", "scenario", "pid", "workload_label", "process_role", "resource_label") if column in labels.columns]
        labels = labels[keep].dropna(subset=["pid", "workload_label"]).drop_duplicates([column for column in ("experiment_id", "scenario", "pid") if column in keep] or ["pid"], keep="last")
        labels["workload_label"] = labels["workload_label"].astype(str).str.strip()
        join_keys = [column for column in ("experiment_id", "scenario", "pid") if column in labels.columns and column in frame.columns]
        frame = frame.drop(columns=["workload_label", "process_role", "resource_label"], errors="ignore").merge(labels, on=join_keys or ["pid"], how="left")
    if "workload_label" not in frame.columns:
        raise ValueError("No workload_label found. Use label-template, fill labels, then pass --labels.")
    for feature in MODEL_FEATURES:
        if feature not in frame.columns:
            frame[feature] = 0.0
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
    frame["workload_label"] = frame["workload_label"].fillna("").astype(str).str.strip()
    valid = frame[frame["workload_label"].isin(WORKLOAD_CLASSES)].copy()
    if valid.empty:
        raise ValueError(f"No valid labels. Allowed labels: {', '.join(WORKLOAD_CLASSES)}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wanted = [column for column in ("timestamp", "experiment_id", "scenario", "pid", "process_name", "process_role", "resource_label") if column in valid.columns] + list(MODEL_FEATURES) + ["workload_label"]
    valid[wanted].to_csv(output, index=False)
    return {"input_rows": len(frame), "labelled_rows": len(valid), "class_count": int(valid["workload_label"].nunique())}
