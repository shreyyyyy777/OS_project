"""Observe → classify → pressure estimate → guarded action decision loop."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .actions import ActionApplier
from .features import heuristic_classify
from .fairness import FairnessConfig, InterventionLedger
from .model import load as load_model
from .model import predict
from .policy import PolicyConfig, classification_summary, decide
from .pressure import estimate_pressure
from .telemetry import TelemetryCollector


@dataclass
class ControllerConfig:
    current_user_only: bool = True
    exclude_process_names: tuple[str, ...] = ()
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)


def load_config(path: str | Path) -> ControllerConfig:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 only
        import tomli as tomllib
    config_path = Path(path)
    loaded = tomllib.loads(config_path.read_text(encoding="utf-8"))
    collection = loaded.get("collection", {})
    policy_data = loaded.get("policy", {})
    fairness_data = loaded.get("fairness", {})
    allowed = set(PolicyConfig.__dataclass_fields__)
    policy = PolicyConfig(**{key: value for key, value in policy_data.items() if key in allowed})
    fairness_allowed = set(FairnessConfig.__dataclass_fields__)
    fairness = FairnessConfig(**{key: value for key, value in fairness_data.items() if key in fairness_allowed})
    return ControllerConfig(
        current_user_only=bool(collection.get("current_user_only", True)),
        exclude_process_names=tuple(str(name) for name in collection.get("exclude_process_names", [])),
        policy=policy,
        fairness=fairness,
    )



def annotate_protected_descendants(
    records: list[dict[str, Any]],
    protected_process_names: tuple[str, ...],
) -> None:
    """Mark records descended from a configured protected process in this snapshot."""
    protected_names = {str(name).lower() for name in protected_process_names}
    if not protected_names:
        return

    parents: dict[int, int] = {}
    protected_pids: set[int] = set()
    for record in records:
        try:
            pid = int(record["pid"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            parents[pid] = int(record.get("ppid") or 0)
        except (TypeError, ValueError):
            parents[pid] = 0
        if str(record.get("process_name") or "").lower() in protected_names:
            protected_pids.add(pid)

    for record in records:
        record["is_protected_descendant"] = False
        try:
            ancestor_pid = parents.get(int(record["pid"]), 0)
        except (KeyError, TypeError, ValueError):
            continue
        seen: set[int] = set()
        while ancestor_pid and ancestor_pid not in seen:
            if ancestor_pid in protected_pids:
                record["is_protected_descendant"] = True
                break
            seen.add(ancestor_pid)
            ancestor_pid = parents.get(ancestor_pid, 0)

class Controller:
    def __init__(self, config: ControllerConfig, model_path: str | None = None) -> None:
        self.config = config
        self.collector = TelemetryCollector(config.current_user_only, config.exclude_process_names)
        self.model = load_model(model_path) if model_path else None
        self.applier = ActionApplier()
        self.ledger = InterventionLedger(config.fairness)

    def run_once(
        self,
        log_path: str | Path,
        *,
        dry_run: bool = True,
        allow_pid: int | None = None,
        allow_cgroup: bool = False,
        cgroup_path: str | None = None,
        allow_restore: bool = False,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        records = self.collector.collect()
        if self.config.policy.protect_descendants:
            annotate_protected_descendants(records, self.config.policy.protected_process_names)
        for record in records:
            if int(record["pid"]) == os.getpid():
                continue
            if self.model:
                label, confidence, source = predict(self.model, record)
            else:
                label, confidence, source = heuristic_classify(record)
            pressure = estimate_pressure(record)
            now = time.time()
            decision = self.ledger.constrain(record, pressure, decide(record, label, pressure, self.config.policy), now)
            result = self.applier.apply(
                decision,
                record,
                dry_run=dry_run,
                allow_pid=allow_pid,
                allow_cgroup=allow_cgroup,
                cgroup_path=cgroup_path,
                allow_restore=allow_restore,
            )
            self.ledger.record_proposal(record, decision, now)
            events.append({
                "event_timestamp": now,
                "mode": "dry_run" if dry_run else "apply",
                "telemetry": record,
                "classification": classification_summary(label, confidence, source, record),
                "pressure": pressure.as_dict(),
                "decision": decision.as_dict(),
                "result": result.as_dict(),
                "fairness": self.ledger.as_dict(int(record["pid"])),
            })
        destination = Path(log_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        return events

    def run(
        self,
        log_path: str | Path,
        *,
        interval: float,
        iterations: int,
        dry_run: bool = True,
        allow_pid: int | None = None,
        allow_cgroup: bool = False,
        cgroup_path: str | None = None,
        allow_restore: bool = False,
    ) -> int:
        total = 0
        for index in range(iterations):
            total += len(self.run_once(log_path, dry_run=dry_run, allow_pid=allow_pid, allow_cgroup=allow_cgroup, cgroup_path=cgroup_path, allow_restore=allow_restore))
            if index + 1 < iterations:
                time.sleep(max(interval, 0.1))
        return total
