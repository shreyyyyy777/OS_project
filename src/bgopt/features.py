"""Shared feature schema and conservative, explainable fallback classification."""

from __future__ import annotations

from typing import Any


# These are intentionally portable signals.  They overlap conceptually with a
# subset of LAPS' CPU/load/memory/I/O/context-switch observations but do not
# claim to be LAPS' ordered 150-feature vector.
MODEL_FEATURES: tuple[str, ...] = (
    "host_cpu_percent",
    "host_load_1_norm",
    "host_memory_percent",
    "host_memory_available_mb",
    "host_swap_percent",
    "host_disk_busy_mb_s",
    "proc_cpu_percent",
    "proc_memory_percent",
    "proc_rss_mb",
    "proc_threads",
    "proc_open_files",
    "proc_read_mb_s",
    "proc_write_mb_s",
    "proc_ctx_switches_s",
    "proc_age_s",
    "is_terminal_bound",
    "parent_is_shell",
    "is_interactive_hint",
)

WORKLOAD_CLASSES = (
    "interactive",
    "background_cpu",
    "background_io",
    "background_network",
    "background_mixed",
    "memory_heavy",
    "idle",
)


def number(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Return a finite-ish numeric value without making collection brittle."""
    try:
        value = float(record.get(key, default))
    except (TypeError, ValueError):
        return default
    return value


def interactive_hint(record: dict[str, Any]) -> bool:
    """Return True only for processes in a terminal's foreground process group."""
    if "is_terminal_foreground" in record:
        return bool(record.get("is_terminal_foreground"))
    # Backward-compatible fallback for older telemetry records.
    return bool(record.get("is_terminal_bound") or record.get("parent_is_shell"))


def heuristic_classify(record: dict[str, Any]) -> tuple[str, float, str]:
    """Transparent bootstrap classification for an untrained installation.

    This is deliberately not persisted as ground truth.  It exists only to
    exercise the controller pipeline before project-specific labels are ready.
    """
    if interactive_hint(record):
        return "interactive", 0.95, "heuristic"
    if number(record, "proc_cpu_percent") >= 20.0:
        return "background_cpu", 0.70, "heuristic"
    if number(record, "proc_read_mb_s") + number(record, "proc_write_mb_s") >= 5.0:
        return "background_io", 0.70, "heuristic"
    if number(record, "proc_memory_percent") >= 10.0 or number(record, "proc_rss_mb") >= 1024:
        return "memory_heavy", 0.65, "heuristic"
    return "idle", 0.75, "heuristic"


def resource_behavior(workload_label: str) -> str:
    return {
        "background_cpu": "cpu_bound",
        "background_io": "io_bound",
        "background_network": "network_bound",
        "background_mixed": "mixed",
        "memory_heavy": "memory_heavy",
        "interactive": "latency_sensitive",
        "idle": "idle",
    }.get(workload_label, "unknown")
