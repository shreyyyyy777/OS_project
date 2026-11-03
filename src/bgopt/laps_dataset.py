"""Read-only profiling of the released LAPS training dataset.

The artifact uses pickle files. Loading pickle is intentionally opt-in because
pickle is executable deserialization; use it only for a trusted local artifact.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from .laps import read_feature_map


WORKLOAD_SUFFIXES = (
    "archive_extraction",
    "blender_render",
    "kernel_compile",
    "llm_generate",
    "mixed_io",
    "network_intensive",
    "video_render",
)

RESOURCE_HINTS = {
    "archive_extraction": "cpu_and_io",
    "blender_render": "compute_intensive",
    "kernel_compile": "compute_intensive",
    "llm_generate": "compute_intensive",
    "mixed_io": "io_intensive",
    "network_intensive": "network_intensive",
    "video_render": "compute_intensive",
}


def _feature_family(name: str) -> str:
    low = name.lower().lstrip("@")
    if low.startswith("cpu"):
        return "cpu_and_load"
    if low.startswith("mem"):
        return "memory"
    if low.startswith(("process", "focus_process", "cmd_gpu", "gpu_process", "input_event")):
        return "process_and_interaction"
    if any(token in low for token in ("queue", "switch", "wakeup", "lock", "thread_block", "migration")):
        return "scheduling_and_latency"
    if any(token in low for token in ("iops", "read_", "write_", "disk_")):
        return "disk_io"
    return "network_and_interrupts"


def scene_reference(label: str) -> dict[str, str | bool]:
    for suffix in WORKLOAD_SUFFIXES:
        ending = f"_{suffix}"
        if label.endswith(ending):
            return {
                "laps_scene_label": label,
                "foreground_context": label[: -len(ending)],
                "paired_workload": suffix,
                "provisional_resource_hint": RESOURCE_HINTS[suffix],
                "human_review_required": True,
            }
    return {
        "laps_scene_label": label,
        "foreground_context": "unknown",
        "paired_workload": "unknown",
        "provisional_resource_hint": "unknown",
        "human_review_required": True,
    }


def profile_dataset(dataset_dir: str | Path, feature_map_path: str | Path) -> dict[str, Any]:
    dataset_root = Path(dataset_dir).expanduser().resolve()
    feature_names_path = dataset_root / "feature_names.json"
    features_path = dataset_root / "features.pkl"
    metadata_path = dataset_root / "metadata.pkl"
    missing = [str(path) for path in (feature_names_path, features_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LAPS dataset is incomplete: {missing}")
    names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    with features_path.open("rb") as handle:
        features = pickle.load(handle)
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    if not {"label", "vm_identifier", "group"}.issubset(metadata.columns):
        raise ValueError("LAPS metadata is missing label, vm_identifier, or group")
    map_names = read_feature_map(Path(feature_map_path))
    labels = sorted(str(value) for value in metadata["label"].unique())
    families = Counter(_feature_family(str(name)) for name in names)
    return {
        "dataset_dir": str(dataset_root),
        "feature_shape": list(features.shape),
        "metadata_shape": list(metadata.shape),
        "feature_names_count": len(names),
        "metadata_columns": list(metadata.columns),
        "feature_order_matches_feature_names_json": list(features.columns) == list(names),
        "feature_map_count": len(map_names),
        "feature_map_set_matches_dataset": set(map_names) == set(names),
        "feature_family_counts": dict(sorted(families.items())),
        "label_count": len(labels),
        "label_counts": {label: int((metadata["label"] == label).sum()) for label in labels},
        "vm_counts": {str(key): int(value) for key, value in metadata["vm_identifier"].value_counts().sort_index().items()},
        "scene_reference": [scene_reference(label) for label in labels],
        "reuse_boundary": (
            "The 28 LAPS scene labels and 150 BPF-derived inputs may inform feature and workload-taxonomy design, "
            "but cannot train BGOpt's per-process action controller directly. BGOpt must collect its own psutil-compatible "
            "features, process-behaviour labels, and action outcomes."
        ),
    }


def write_dataset_profile(dataset_dir: str | Path, feature_map_path: str | Path, output: str | Path) -> dict[str, Any]:
    report = profile_dataset(dataset_dir, feature_map_path)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

