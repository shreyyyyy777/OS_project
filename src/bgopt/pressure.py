"""Explainable host resource-pressure estimate; it is separate from ML labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .features import number


@dataclass(frozen=True)
class Pressure:
    score: float
    level: str
    cpu_component: float
    memory_component: float
    swap_component: float
    disk_component: float
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def estimate_pressure(record: dict[str, Any]) -> Pressure:
    """Calculate a bounded score from host—not classifier—telemetry."""
    cpu = min(1.0, max(number(record, "host_cpu_percent") / 100, number(record, "host_load_1_norm")))
    memory = min(1.0, number(record, "host_memory_percent") / 100)
    swap = min(1.0, number(record, "host_swap_percent") / 100)
    # 200 MiB/s is deliberately only a rough saturation proxy and is logged.
    disk = min(1.0, number(record, "host_disk_busy_mb_s") / 200)
    score = min(1.0, 0.50 * cpu + 0.30 * memory + 0.12 * swap + 0.08 * disk)
    reasons: list[str] = []
    if cpu >= 0.80:
        reasons.append("high_cpu_or_load")
    if memory >= 0.85:
        reasons.append("high_memory")
    if swap >= 0.20:
        reasons.append("active_swap")
    if disk >= 0.75:
        reasons.append("high_disk_io")
    if not reasons:
        reasons.append("normal_resource_pressure")
    level = "high" if score >= 0.75 else "medium" if score >= 0.50 else "low"
    return Pressure(score, level, cpu, memory, swap, disk, tuple(reasons))

