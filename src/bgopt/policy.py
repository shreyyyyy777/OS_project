"""Project-owned, explainable action policy; intentionally not a LAPS P.M.T."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .features import interactive_hint, number, resource_behavior
from .pressure import Pressure


@dataclass(frozen=True)
class PolicyConfig:
    cpu_pressure_threshold: float = 0.65
    protected_process_names: tuple[str, ...] = ()
    protect_descendants: bool = True
    memory_pressure_threshold: float = 0.75
    candidate_cpu_percent: float = 5.0
    nice_increment: int = 5
    severe_nice_increment: int = 10
    severe_pressure_threshold: float = 0.85
    max_nice: int = 19
    enable_cgroup_suggestions: bool = False
    cgroup_cpu_weight: int = 50
    enable_cpu_limit_suggestions: bool = False
    cpu_limit_quota_percent: int = 50


@dataclass(frozen=True)
class Decision:
    kind: str
    target_nice: int | None
    cgroup_cpu_weight: int | None
    cpu_limit_quota_percent: int | None
    action_code: str
    reason: str
    safety_checks: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["safety_checks"] = list(self.safety_checks)
        return data


def no_op(reason: str, *checks: str) -> Decision:
    return Decision("no_op", None, None, None, "NO_ACTION", reason, checks)


def decide(record: dict[str, Any], workload_label: str, pressure: Pressure, config: PolicyConfig) -> Decision:
    process_name = str(record.get("process_name") or "").lower()
    protected = {name.lower() for name in config.protected_process_names}
    if process_name in protected:
        return no_op("configured foreground process is protected", "protect_configured_foreground")
    if bool(record.get("is_protected_descendant")):
        return no_op(
            "descendant of configured foreground process is protected",
            "protect_configured_foreground_descendant",
        )
    """Return a proposed action only when it has a narrow, auditable rationale.

    A prediction is insufficient on its own. The policy requires measured host
    pressure, an observed resource-heavy background candidate, and safeguards.
    """
    if interactive_hint(record) or workload_label == "interactive":
        return no_op("likely foreground or terminal-bound process", "protect_interactive")
    if workload_label not in {"background_cpu", "background_io", "background_network", "background_mixed", "memory_heavy", "idle"}:
        return no_op("unrecognised workload class", "require_known_workload_class")
    # ``score`` is a blended health indicator; CPU demotion should respond to
    # actual CPU/load contention even when memory is comfortably available.
    if pressure.cpu_component < config.cpu_pressure_threshold:
        return no_op("CPU/load pressure below action threshold", "require_cpu_pressure")
    if workload_label == "memory_heavy" and pressure.memory_component >= config.memory_pressure_threshold:
        return no_op("memory pressure detected; v1 has no memory-throttling action", "avoid_unsafe_memory_action")
    if workload_label in {"background_io", "background_network", "background_mixed"}:
        return no_op("non-CPU background work observed; v1 avoids I/O/network throttling", "avoid_unimplemented_io_action")
    if workload_label != "background_cpu" or number(record, "proc_cpu_percent") < config.candidate_cpu_percent:
        return no_op("no CPU-heavy background candidate", "require_cpu_candidate")
    current_nice = int(number(record, "nice"))
    increment = config.severe_nice_increment if pressure.score >= config.severe_pressure_threshold else config.nice_increment
    target_nice = min(config.max_nice, max(current_nice, current_nice + increment))
    if target_nice <= current_nice:
        return no_op("process is already at the configured lowest priority", "never_raise_priority")
    if config.enable_cpu_limit_suggestions and pressure.score >= config.severe_pressure_threshold:
        return Decision("cgroup_cpu_limit", None, None, config.cpu_limit_quota_percent, "CPU_LIMIT", "critical CPU pressure and CPU-heavy background process", ("protect_interactive", "require_pressure", "require_explicit_cgroup_guard"))
    if config.enable_cgroup_suggestions and pressure.score >= config.severe_pressure_threshold:
        return Decision(
            "cgroup_cpu_weight",
            None,
            config.cgroup_cpu_weight,
            None,
            "REDUCE_CPU_WEIGHT",
            "high CPU pressure and CPU-heavy background process",
            ("protect_interactive", "require_pressure", "require_explicit_cgroup_guard"),
        )
    return Decision(
        "nice",
        target_nice,
        None,
        None,
        "NICE_PLUS_10" if increment >= 10 else "NICE_PLUS_5",
        "CPU pressure and CPU-heavy background process",
        ("protect_interactive", "require_pressure", "only_lower_priority"),
    )


def classification_summary(label: str, confidence: float, source: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "workload_label": label,
        "resource_behavior": resource_behavior(label),
        "role": "foreground" if interactive_hint(record) or label == "interactive" else "background",
        "confidence": confidence,
        "source": source,
    }
