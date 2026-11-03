"""Narrowly guarded execution of decisions; dry-run is the default contract."""

from __future__ import annotations

import getpass
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .policy import Decision


@dataclass(frozen=True)
class ApplyResult:
    status: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class ActionApplier:
    """Never kills processes and never acts without a PID-specific opt-in."""

    def apply(
        self,
        decision: Decision,
        record: dict[str, Any],
        *,
        dry_run: bool = True,
        allow_pid: int | None = None,
        allow_cgroup: bool = False,
        cgroup_path: str | None = None,
        allow_restore: bool = False,
    ) -> ApplyResult:
        pid = int(record["pid"])
        if decision.kind == "no_op":
            return ApplyResult("skipped", decision.reason)
        if dry_run:
            return ApplyResult("would_apply", f"dry-run: would apply {decision.kind} to pid {pid}")
        if allow_pid != pid:
            return ApplyResult("blocked", "real actions require --allow-pid for this exact PID")
        if pid == os.getpid():
            return ApplyResult("blocked", "refusing to change the controller process")
        try:
            import psutil

            process = psutil.Process(pid)
            if process.username() != getpass.getuser():
                return ApplyResult("blocked", "refusing to change a process owned by another user")
            expected_start = float(record.get("process_create_time", 0.0) or 0.0)
            if expected_start and abs(float(process.create_time()) - expected_start) > 0.01:
                return ApplyResult("blocked", "PID identity changed since observation; refusing recycled PID")
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as error:
            return ApplyResult("blocked", f"cannot validate target process: {error}")
        if decision.kind == "nice":
            return self._apply_nice(process, decision)
        if decision.kind == "restore_nice":
            return self._restore_nice(process, decision, allow_restore)
        if decision.kind == "cgroup_cpu_weight":
            return self._apply_cgroup(pid, decision, allow_cgroup, cgroup_path)
        if decision.kind == "cgroup_cpu_limit":
            return self._apply_cgroup(pid, decision, allow_cgroup, cgroup_path, limit=True)
        return ApplyResult("blocked", f"unknown action kind: {decision.kind}")

    @staticmethod
    def _apply_nice(process: Any, decision: Decision) -> ApplyResult:
        target = decision.target_nice
        if target is None:
            return ApplyResult("blocked", "nice decision has no target")
        current = int(process.nice())
        if target <= current:
            return ApplyResult("blocked", "refusing to raise process priority")
        try:
            process.nice(target)
        except (OSError, PermissionError) as error:
            return ApplyResult("failed", f"nice change failed: {error}")
        return ApplyResult("applied", f"increased nice from {current} to {target}")

    @staticmethod
    def _restore_nice(process: Any, decision: Decision, allow_restore: bool) -> ApplyResult:
        if not allow_restore:
            return ApplyResult("blocked", "priority restoration requires --allow-restore")
        target = decision.target_nice
        if target is None or target < 0:
            return ApplyResult("blocked", "invalid restoration target")
        current = int(process.nice())
        if target >= current:
            return ApplyResult("blocked", "restoration target does not raise priority")
        try:
            process.nice(target)
        except (OSError, PermissionError) as error:
            return ApplyResult("failed", f"priority restoration failed: {error}")
        return ApplyResult("applied", f"restored nice from {current} to {target}")

    @staticmethod
    def _apply_cgroup(pid: int, decision: Decision, allow_cgroup: bool, cgroup_path: str | None, limit: bool = False) -> ApplyResult:
        if not allow_cgroup:
            return ApplyResult("blocked", "cgroup changes require --allow-cgroup")
        if platform.system() != "Linux":
            return ApplyResult("blocked", "cgroup CPU weights are Linux-only")
        if not cgroup_path:
            return ApplyResult("blocked", "cgroup action requires --cgroup-path to an existing dedicated cgroup")
        root = Path("/sys/fs/cgroup")
        target = Path(cgroup_path)
        try:
            target = target.resolve(strict=True)
            root = root.resolve(strict=True)
            target.relative_to(root)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            return ApplyResult("blocked", f"invalid cgroup path: {error}")
        if target == root:
            return ApplyResult("blocked", "refusing to use the root cgroup; provide a dedicated existing leaf")
        controller_file, weight_file, procs_file = root / "cgroup.controllers", target / "cpu.weight", target / "cgroup.procs"
        if not controller_file.exists() or "cpu" not in controller_file.read_text(encoding="utf-8").split():
            return ApplyResult("blocked", "cgroup v2 CPU controller is unavailable")
        if not weight_file.is_file() or not procs_file.is_file():
            return ApplyResult("blocked", "target is not an existing cgroup-v2 leaf with cpu.weight and cgroup.procs")
        value_file, value = (target / "cpu.max", decision.cpu_limit_quota_percent) if limit else (weight_file, decision.cgroup_cpu_weight)
        if limit:
            if not value_file.is_file():
                return ApplyResult("blocked", "target cgroup does not expose cpu.max")
            if value is None or not 1 <= value <= 100:
                return ApplyResult("blocked", "invalid CPU limit percentage")
        elif value is None or not 1 <= value <= 10000:
            return ApplyResult("blocked", "invalid cgroup CPU weight")
        try:
            if limit:
                value_file.write_text(f"{int(value) * 1000} 100000\n", encoding="utf-8")
            else:
                value_file.write_text(f"{value}\n", encoding="utf-8")
            procs_file.write_text(f"{pid}\n", encoding="utf-8")
        except (OSError, PermissionError) as error:
            return ApplyResult("failed", f"cgroup change failed: {error}")
        description = f"set cpu.max to {value}%" if limit else f"set cpu.weight={value}"
        return ApplyResult("applied", f"{description} and moved pid {pid} into {target}")
