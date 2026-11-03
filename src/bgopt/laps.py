"""Read-only LAPS artifact inventory and semantic feature crosswalk.

The adapter does not feed an arbitrary LAPS model: that would be invalid unless
the exact 150 inputs, order, scaling, and class vocabulary were reproduced.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .features import MODEL_FEATURES


FEATURE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "host_cpu_percent": ("cpu", "util", "usage"),
    "host_load_1_norm": ("load",),
    "host_memory_percent": ("memory", "mem", "ram"),
    "host_memory_available_mb": ("memory", "available", "free"),
    "host_swap_percent": ("swap",),
    "host_disk_busy_mb_s": ("disk", "io", "read", "write"),
    "proc_cpu_percent": ("process", "task", "pid", "cpu"),
    "proc_memory_percent": ("process", "task", "pid", "memory", "mem"),
    "proc_rss_mb": ("rss", "resident"),
    "proc_threads": ("thread",),
    "proc_open_files": ("fd", "file"),
    "proc_read_mb_s": ("read", "io"),
    "proc_write_mb_s": ("write", "io"),
    "proc_ctx_switches_s": ("context", "switch"),
    "proc_age_s": ("age", "runtime"),
    "is_terminal_bound": ("terminal", "tty"),
    "parent_is_shell": ("parent", "shell"),
    "is_interactive_hint": ("interactive", "foreground"),
}


def read_feature_map(path: Path) -> list[str]:
    """Return non-empty feature-map entries without assuming one file syntax."""
    names: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # LAPS feature maps commonly carry a name plus a position/delimiter.
        names.append(stripped.split()[0].split(":")[0].strip(","))
    return names


def feature_crosswalk(laps_features: list[str]) -> dict[str, list[str]]:
    """Suggest semantic overlap, never a claim of numeric/model compatibility."""
    result: dict[str, list[str]] = {feature: [] for feature in MODEL_FEATURES}
    for laps_feature in laps_features:
        low = laps_feature.lower()
        for bgopt_feature, keywords in FEATURE_KEYWORDS.items():
            if any(keyword in low for keyword in keywords):
                result[bgopt_feature].append(laps_feature)
    return result


def inspect_artifact(root: str | Path) -> dict[str, Any]:
    """Build a small reproducible inventory; only reads under ``root``."""
    artifact_root = Path(root).expanduser().resolve()
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"Artifact root does not exist or is not a directory: {artifact_root}")
    candidates: dict[str, list[str]] = {
        "readmes": [], "papers": [], "feature_maps": [], "models": [], "datasets": [],
    }
    ignored_directories = {".git", ".venv", ".pytest_cache", "__pycache__", "node_modules"}
    for directory, subdirectories, filenames in os.walk(artifact_root):
        subdirectories[:] = [name for name in subdirectories if name not in ignored_directories]
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            relative = str(path.relative_to(artifact_root))
            lower = path.name.lower()
            if lower.startswith("readme") or lower.endswith(".md"):
                candidates["readmes"].append(relative)
            if lower.endswith(".pdf") or "paper" in lower:
                candidates["papers"].append(relative)
            if "feature" in lower and (lower.endswith(".txt") or lower.endswith(".json") or lower.endswith(".csv")):
                candidates["feature_maps"].append(relative)
            if lower.endswith((".model", ".joblib", ".pkl", ".onnx", ".json")) and ("model" in lower or "classifier" in lower):
                candidates["models"].append(relative)
            if lower.endswith((".csv", ".parquet", ".pkl", ".npz", ".npy")) and "feature" not in lower:
                candidates["datasets"].append(relative)
    for values in candidates.values():
        values.sort()
    feature_names: list[str] = []
    parsed_from: str | None = None
    # Parse only an explicit text feature map, if one is present.
    for relative in candidates["feature_maps"]:
        candidate = artifact_root / relative
        if candidate.suffix.lower() == ".txt":
            feature_names, parsed_from = read_feature_map(candidate), relative
            break
    return {
        "artifact_root": str(artifact_root),
        "inventory": candidates,
        "parsed_feature_map": parsed_from,
        "laps_feature_count": len(feature_names),
        "laps_feature_names": feature_names,
        "bgopt_feature_crosswalk": feature_crosswalk(feature_names),
        "compatibility_note": (
            "This is a semantic crosswalk only. Do not load a LAPS model into BGOpt unless its exact "
            "feature order, feature computation, preprocessing, and workload-class vocabulary are reproduced."
        ),
        "action_layer_note": (
            "LAPS workload classes map through its scheduler Policy Mapping Table. BGOpt separately maps "
            "its workload classification plus pressure and safety rules to user-space actions."
        ),
    }


def write_inventory(root: str | Path, output: str | Path) -> dict[str, Any]:
    report = inspect_artifact(root)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
