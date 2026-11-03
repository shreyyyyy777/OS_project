"""Portable, current-user process telemetry collection using psutil only."""

from __future__ import annotations

from pathlib import Path
import getpass
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from .features import interactive_hint


SHELL_NAMES = {"bash", "zsh", "fish", "sh", "dash", "ksh", "tcsh", "csh", "pwsh", "powershell"}
def terminal_foreground_process_group(pid: int) -> int | None:
    """Return the foreground process-group ID for a process's controlling TTY."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        end_comm = stat.rfind(")")
        if end_comm < 0:
            return None
        fields = stat[end_comm + 2:].split()
        return int(fields[5])
    except (OSError, ValueError, IndexError):
        return None

@dataclass
class _PriorCounters:
    timestamp: float
    read_bytes: int = 0
    write_bytes: int = 0
    context_switches: int = 0


@dataclass
class TelemetryCollector:
    """Collect a best-effort snapshot. Access errors skip only one process."""

    current_user_only: bool = True
    exclude_process_names: tuple[str, ...] = ()
    _prior: dict[int, _PriorCounters] = field(default_factory=dict)
    _prior_disk_bytes: int | None = None
    _prior_disk_time: float | None = None
    _prior_network_bytes: int | None = None
    _prior_network_time: float | None = None

    def __post_init__(self) -> None:
        # Prime psutil's non-blocking CPU gauges. The first sample may still be
        # near zero, which is preferable to delaying the controller.
        import psutil

        psutil.cpu_percent(interval=None)
        self._username = getpass.getuser()

    def _host_metrics(self, now: float) -> dict[str, float]:
        import psutil

        # Some constrained macOS environments deny individual host counters.
        # Telemetry is observational, so an unavailable optional counter should
        # be represented as zero and logged through the remaining signals—not
        # terminate the collector or controller.
        try:
            vm = psutil.virtual_memory()
            memory_percent = float(vm.percent)
            memory_available_mb = float(vm.available) / (1024 * 1024)
        except (OSError, ValueError):
            memory_percent, memory_available_mb = 0.0, 0.0
        try:
            swap_percent = float(psutil.swap_memory().percent)
        except (OSError, ValueError):
            swap_percent = 0.0
        cpu_count = max(psutil.cpu_count(logical=True) or 1, 1)
        try:
            load_1 = os.getloadavg()[0]
        except (AttributeError, OSError):
            load_1 = 0.0
        try:
            disk = psutil.disk_io_counters()
        except (OSError, ValueError):
            disk = None
        disk_bytes = int((disk.read_bytes + disk.write_bytes) if disk else 0)
        if self._prior_disk_bytes is None or self._prior_disk_time is None:
            disk_mb_s = 0.0
        else:
            elapsed = max(now - self._prior_disk_time, 0.001)
            disk_mb_s = max(disk_bytes - self._prior_disk_bytes, 0) / elapsed / (1024 * 1024)
        self._prior_disk_bytes, self._prior_disk_time = disk_bytes, now
        try:
            net = psutil.net_io_counters()
            network_bytes = int((net.bytes_sent + net.bytes_recv) if net else 0)
        except (OSError, ValueError):
            network_bytes = 0
        if self._prior_network_bytes is None or self._prior_network_time is None:
            network_mb_s = 0.0
        else:
            network_mb_s = max(network_bytes - self._prior_network_bytes, 0) / max(now - self._prior_network_time, 0.001) / (1024 * 1024)
        self._prior_network_bytes, self._prior_network_time = network_bytes, now
        return {
            "host_cpu_percent": psutil.cpu_percent(interval=None),
            "host_load_1_norm": load_1 / cpu_count,
            "host_load_1": load_1,
            "host_memory_percent": memory_percent,
            "host_memory_available_mb": memory_available_mb,
            "host_swap_percent": swap_percent,
            "host_disk_busy_mb_s": disk_mb_s,
            "host_network_mb_s": network_mb_s,
            "host_platform": platform.system().lower(),
        }

    @staticmethod
    def _safe_open_file_count(proc: Any) -> int:
        try:
            if hasattr(proc, "num_fds"):
                return int(proc.num_fds())
            return len(proc.open_files())
        except (OSError, ValueError):
            return 0

    def _rates(self, pid: int, now: float, read_bytes: int, write_bytes: int, ctx: int) -> tuple[float, float, float]:
        prior = self._prior.get(pid)
        self._prior[pid] = _PriorCounters(now, read_bytes, write_bytes, ctx)
        if prior is None:
            return 0.0, 0.0, 0.0
        elapsed = max(now - prior.timestamp, 0.001)
        return (
            max(read_bytes - prior.read_bytes, 0) / elapsed / (1024 * 1024),
            max(write_bytes - prior.write_bytes, 0) / elapsed / (1024 * 1024),
            max(ctx - prior.context_switches, 0) / elapsed,
        )

    def collect(self, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        import psutil

        now = time.time()
        host = self._host_metrics(now)
        rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        excluded = {name.lower() for name in self.exclude_process_names}
        attrs = ["pid", "ppid", "name", "username", "create_time", "status", "nice"]
        try:
            processes = psutil.process_iter(attrs)
            # Materialise inside the try: some sandboxes deny the pids() call
            # only when iteration starts.
            process_list = list(processes)
        except (psutil.Error, OSError, PermissionError):
            # A privacy-restricted host can still often observe the caller. It
            # is preferable to return a visibly limited sample than to crash.
            process_list = [psutil.Process(os.getpid())]
        for proc in process_list:
            try:
                info = getattr(proc, "info", None) or proc.as_dict(attrs=attrs)
                pid = int(info["pid"])
                seen.add(pid)
                name = str(info.get("name") or "")
                if name.lower() in excluded:
                    continue
                username = str(info.get("username") or "")
                if self.current_user_only and username != self._username:
                    continue
                with proc.oneshot():
                    io = proc.io_counters() if hasattr(proc, "io_counters") else None
                    ctx = proc.num_ctx_switches() if hasattr(proc, "num_ctx_switches") else None
                    terminal = proc.terminal()
                    parent_name = ""
                    try:
                        parent_name = proc.parent().name() if proc.parent() else ""
                    except (psutil.Error, OSError, ValueError):
                        pass
                    terminal_pgid = terminal_foreground_process_group(pid)
                    process_pgid = os.getpgid(pid)
                    is_terminal_foreground = bool(
                        terminal
                        and terminal_pgid is not None
                        and process_pgid == terminal_pgid
                    )

                    read_b = int(getattr(io, "read_bytes", 0) or 0)
                    write_b = int(getattr(io, "write_bytes", 0) or 0)
                    context_switch_total = int((getattr(ctx, "voluntary", 0) or 0) + (getattr(ctx, "involuntary", 0) or 0))
                    read_rate, write_rate, ctx_rate = self._rates(pid, now, read_b, write_b, context_switch_total)
                    row: dict[str, Any] = {
                        **host,
                        "timestamp": now,
                        "pid": pid,
                        "ppid": int(info.get("ppid") or 0),
                        "process_name": name,
                        "username": username,
                        "status": str(info.get("status") or ""),
                        "nice": int(info.get("nice") or 0),
                        "parent_name": parent_name,
                        "proc_cpu_percent": float(proc.cpu_percent(interval=None)),
                        "proc_memory_percent": float(proc.memory_percent()),
                        "proc_rss_mb": float(proc.memory_info().rss) / (1024 * 1024),
                        "proc_threads": int(proc.num_threads()),
                        "proc_open_files": self._safe_open_file_count(proc),
                        "proc_read_bytes": read_b,
                        "proc_write_bytes": write_b,
                        "proc_read_mb_s": read_rate,
                        "proc_write_mb_s": write_rate,
                        "proc_ctx_switches_s": ctx_rate,
                        "proc_age_s": max(now - float(info.get("create_time") or now), 0.0),
                        "process_create_time": float(info.get("create_time") or 0.0),
                        "is_terminal_bound": bool(terminal),
                        "parent_is_shell": parent_name.lower() in SHELL_NAMES,
            "is_terminal_foreground": is_terminal_foreground,
                    }

                    try:
                        cpu_times = proc.cpu_times()
                        row["proc_user_cpu_seconds"] = float(getattr(cpu_times, "user", 0.0))
                        row["proc_system_cpu_seconds"] = float(getattr(cpu_times, "system", 0.0))
                    except (psutil.Error, OSError):
                        row["proc_user_cpu_seconds"] = 0.0
                        row["proc_system_cpu_seconds"] = 0.0
                    row["voluntary_context_switches"] = int(getattr(ctx, "voluntary", 0) or 0)
                    row["involuntary_context_switches"] = int(getattr(ctx, "involuntary", 0) or 0)
                    row["is_interactive_hint"] = interactive_hint(row)
                    if context:
                        row.update({key: value for key, value in context.items() if value is not None})
                    rows.append(row)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError, ValueError):
                continue
        # Forget exited PIDs so a recycled PID never receives stale I/O rates.
        self._prior = {pid: prior for pid, prior in self._prior.items() if pid in seen}
        return rows


def current_process_ids(rows: Iterable[dict[str, Any]]) -> set[int]:
    return {int(row["pid"]) for row in rows if "pid" in row}
