"""Reproducible, non-invasive measurement harness for Experiment 001."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import json
import math
import os
import subprocess
from dataclasses import dataclass
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import psutil

from .experiments import BASELINES
from .controller import Controller, load_config



def _append_worker_event(log_path: str, event: str, **fields: object) -> None:
    payload = {"worker_event": event, "timestamp": time.time(), **fields}
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _controller_worker(mode: str, log_path: str, allow_pid: int, interval_s: float, stop_event: object, invocation_count: object) -> None:
    """Run controller iterations outside the scheduler-lag probe process."""
    model_path = "models/bgopt_native_workload_classifier_v4.joblib" if mode == "bgopt_ml_assisted" else None
    controller = Controller(load_config("config/example.toml"), model_path=model_path)
    _append_worker_event(log_path, "started", mode=mode, allow_pid=allow_pid, interval_s=interval_s)
    try:
        while not stop_event.is_set():
            started = time.monotonic()
            try:
                controller.run_once(log_path, dry_run=False, allow_pid=allow_pid)
            except Exception as exc:
                _append_worker_event(log_path, "error", error=type(exc).__name__, message=str(exc))
            with invocation_count.get_lock():
                invocation_count.value += 1
            stop_event.wait(max(0.0, interval_s - (time.monotonic() - started)))
    finally:
        _append_worker_event(log_path, "stopped")


class EvaluationSafetyError(RuntimeError):
    """Raised when a requested evaluation would change system state."""

@dataclass(frozen=True)
class WorkloadIdentity:
    pid: int
    create_time: float
    executable: str

    def validate(self, record: dict[str, Any]) -> tuple[bool, str]:
        if int(record.get("pid", -1)) != self.pid:
            return False, "PID does not match scoped workload"
        observed = float(record.get("process_create_time", 0.0) or 0.0)
        if not observed or abs(observed - self.create_time) > 0.01:
            return False, "process create_time does not match scoped workload"
        try:
            current = psutil.Process(self.pid)
            if abs(float(current.create_time()) - self.create_time) > 0.01:
                return False, "PID was reused"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False, "scoped workload identity cannot be verified"
        return True, "accepted"

def validate_scoped_target(identity: WorkloadIdentity, record: dict[str, Any]) -> tuple[bool, str]:
    return identity.validate(record)

def percentile(values: Sequence[float], fraction: float) -> float:
    """Return an interpolated percentile without a heavyweight dependency."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarise_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate probe and host samples into stable, report-ready metrics."""
    def values(name: str) -> list[float]:
        return [sample[name] for sample in samples]

    if not samples:
        return {"sample_count": 0}
    lag = values("scheduler_lag_ms")
    service = values("probe_service_ms")
    cpu = values("cpu_percent")
    load = values("load1_per_cpu")
    memory = values("memory_percent")
    swap = values("swap_percent")
    return {
        "sample_count": len(samples),
        "foreground_latency_ms_p50": percentile(lag, 0.50),
        "foreground_latency_ms_p95": percentile(lag, 0.95),
        "foreground_latency_ms_max": max(lag),
        "foreground_probe_service_ms_p95": percentile(service, 0.95),
        "host_cpu_percent_mean": fmean(cpu),
        "host_cpu_percent_max": max(cpu),
        "host_load1_per_cpu_mean": fmean(load),
        "host_load1_per_cpu_max": max(load),
        "memory_percent_mean": fmean(memory),
        "memory_percent_max": max(memory),
        "swap_percent_mean": fmean(swap),
        "swap_percent_max": max(swap),
    }


def append_result(path: str | Path, result: dict[str, Any]) -> None:
    """Append one immutable evaluation record; existing experiment data is preserved."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")


def _ensure_safe_mode(mode: str) -> None:
    if mode not in BASELINES:
        raise EvaluationSafetyError(f"unknown experiment mode: {mode}")


def run_evaluation(
    *,
    mode: str,
    repetition: int,
    workload: str | Path,
    output: str | Path,
    probe_interval_ms: float = 50.0,
    workload_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Measure one naturally completing Linux-default workload run.

    The foreground probe performs the same small SHA-256 operation at fixed
    deadlines.  Its scheduler lag is a repeatable responsiveness proxy.  This
    function never changes a priority, cgroup, scheduler setting, or process
    lifetime: the child workload is only awaited until it exits naturally.
    """
    _ensure_safe_mode(mode)
    if repetition < 1:
        raise ValueError("repetition must be at least 1")
    if probe_interval_ms <= 0:
        raise ValueError("probe_interval_ms must be positive")
    command = list(workload_command) if workload_command is not None else [str(Path(workload))]
    if workload_command is None and not Path(workload).is_file():
        raise FileNotFoundError(workload)

    interval_s = probe_interval_ms / 1000.0
    psutil.cpu_percent(interval=None)
    started_wall = datetime.now(UTC).isoformat()
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    workload_proc = psutil.Process(process.pid)
    identity = WorkloadIdentity(process.pid, float(workload_proc.create_time()), str(Path(command[0]).resolve()))
    original_nice = int(workload_proc.nice())
    static_applied = None
    controller = None
    controller_worker = None
    controller_stop = None
    controller_counter = None
    controller_invocations = 0
    controller_log = str(Path(output).with_suffix('.controller.jsonl'))
    if mode == "static_nice_plus_5":
        ok, reason = validate_scoped_target(identity, {"pid": process.pid, "process_create_time": identity.create_time})
        if not ok: raise EvaluationSafetyError(reason)
        workload_proc.nice(original_nice + 5)
        static_applied = int(workload_proc.nice())
    elif mode in {"bgopt_heuristic", "bgopt_ml_assisted"}:
        model_path = "models/bgopt_native_workload_classifier_v4.joblib" if mode == "bgopt_ml_assisted" else None
        controller_stop = mp.Event()
        controller_counter = mp.Value("i", 0)
        controller_worker = mp.Process(target=_controller_worker, args=(mode, controller_log, identity.pid, interval_s, controller_stop, controller_counter), daemon=True)
        controller_worker.start()
    samples: list[dict[str, float]] = []
    deadline = started
    probe_payload = b"bgopt-experiment-001-interactive-probe"
    while process.poll() is None:
        deadline += interval_s
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        observed = time.monotonic()
        service_started = time.monotonic()
        hashlib.sha256(probe_payload).digest()
        service_ms = (time.monotonic() - service_started) * 1000.0
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        load1 = os.getloadavg()[0] / max(os.cpu_count() or 1, 1)
        samples.append({
            "scheduler_lag_ms": max(0.0, (observed - deadline) * 1000.0),
            "probe_service_ms": service_ms,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "load1_per_cpu": load1,
            "memory_percent": memory.percent,
            "swap_percent": swap.percent,
        })
    if controller_stop is not None:
        controller_stop.set()
    if controller_worker is not None:
        controller_worker.join(timeout=5.0)
        if controller_worker.is_alive():
            controller_worker.terminate()
            controller_worker.join(timeout=2.0)
    if controller_counter is not None:
        controller_invocations = controller_counter.value
    completed = time.monotonic()
    if mode == "static_nice_plus_5" and psutil.pid_exists(identity.pid):
        current = psutil.Process(identity.pid)
        if abs(float(current.create_time()) - identity.create_time) > 0.01:
            raise EvaluationSafetyError("workload identity changed before restoration")
        current.nice(original_nice)
    metrics = summarise_samples(samples)
    duration_s = completed - started
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "fg_web_bg_compile_001",
        "scenario": "scripted_foreground_probe_plus_compile_workload",
        "mode": mode,
        "repetition": repetition,
        "recorded_at_utc": started_wall,
        "run_duration_s": duration_s,
        "foreground_metric": "fixed_interval_scheduler_lag_ms",
        "foreground_latency_ms": metrics.get("foreground_latency_ms_p95", 0.0),
        "background_metric": "natural_workload_completion_s",
        "background_completion_s": duration_s,
        "background_exit_code": process.returncode,
        "background_throughput_runs_per_s": 1.0 / duration_s if duration_s else 0.0,
        "host_metrics": metrics,
        "workload_identity": {"pid": identity.pid, "create_time": identity.create_time, "executable": identity.executable},
        "bgopt": {
            "controller_invoked": controller_worker is not None,
            "controller_invocations": controller_invocations,
            "decision_count": 0,
            "intervention_count": 0,
            "dry_run": False,
            "controller_log": controller_log if controller_worker is not None else None,
            "controller_interval_ms": probe_interval_ms if controller_worker is not None else None,
        },
        "treatment": {"original_nice": original_nice, "requested_nice": original_nice + 5 if mode == "static_nice_plus_5" else None, "applied_nice": static_applied},
        "safety": {
            "real_priority_changes": mode != "linux_default",
            "cgroup_changes": False,
            "scheduler_changes": False,
            "processes_killed": False,
        },
    }
    append_result(output, result)
    return result
