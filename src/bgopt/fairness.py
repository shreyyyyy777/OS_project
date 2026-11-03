"""Per-process intervention ledger: cooldown, bounded duration, and restoration.

The ledger is controller-local by design.  It never treats a dry-run proposal as
an applied OS change; it only prevents the decision engine from repeatedly
recommending escalations during a single observation run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .policy import Decision, no_op
from .pressure import Pressure


@dataclass(frozen=True)
class FairnessConfig:
    cooldown_seconds: float = 30.0
    max_intervention_seconds: float = 300.0
    max_interventions_per_pid: int = 3
    restoration_cpu_threshold: float = 0.35


@dataclass
class InterventionState:
    original_nice: int
    process_create_time: float
    first_intervention_at: float
    last_intervention_at: float
    intervention_count: int = 0
    restoration_proposed: bool = False


@dataclass
class InterventionLedger:
    config: FairnessConfig = field(default_factory=FairnessConfig)
    states: dict[int, InterventionState] = field(default_factory=dict)

    def constrain(self, record: dict[str, Any], pressure: Pressure, proposal: Decision, now: float) -> Decision:
        pid = int(record["pid"])
        state = self.states.get(pid)
        if state is not None and state.process_create_time and float(record.get("process_create_time", 0.0)) not in {0.0, state.process_create_time}:
            # PID was recycled during this controller run; never restore or
            # constrain the new process based on the old process's history.
            self.states.pop(pid, None)
            state = None
        if state is not None:
            elapsed = now - state.first_intervention_at
            if (pressure.cpu_component <= self.config.restoration_cpu_threshold or elapsed >= self.config.max_intervention_seconds) and not state.restoration_proposed:
                state.restoration_proposed = True
                reason = "contention cleared; restore original priority" if pressure.cpu_component <= self.config.restoration_cpu_threshold else "maximum intervention duration reached; restore original priority"
                return Decision("restore_nice", state.original_nice, None, None, "RESTORE_PRIORITY", reason, ("same_pid_only", "explicit_restore_guard", "bounded_intervention"))
            if state.restoration_proposed:
                return no_op("restoration already proposed; await explicit application/result", "avoid_repeated_restore")
            if proposal.kind != "no_op" and now - state.last_intervention_at < self.config.cooldown_seconds:
                return no_op("per-process intervention cooldown active", "cooldown")
            if proposal.kind != "no_op" and state.intervention_count >= self.config.max_interventions_per_pid:
                return no_op("per-process intervention budget reached", "anti_starvation_budget")
        return proposal

    def record_proposal(self, record: dict[str, Any], decision: Decision, now: float) -> None:
        if decision.kind in {"no_op", "restore_nice"}:
            return
        pid = int(record["pid"])
        state = self.states.get(pid)
        if state is None:
            state = InterventionState(int(record.get("nice", 0)), float(record.get("process_create_time", 0.0)), now, now)
            self.states[pid] = state
        state.last_intervention_at = now
        state.intervention_count += 1

    def as_dict(self, pid: int) -> dict[str, Any] | None:
        state = self.states.get(pid)
        if state is None:
            return None
        return {"original_nice": state.original_nice, "process_create_time": state.process_create_time, "first_intervention_at": state.first_intervention_at, "last_intervention_at": state.last_intervention_at, "intervention_count": state.intervention_count, "restoration_proposed": state.restoration_proposed}
