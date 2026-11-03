import json
import multiprocessing as mp
import time
import sys
import tempfile
import unittest
from pathlib import Path

from bgopt.evaluation import EvaluationSafetyError, WorkloadIdentity, _controller_worker, percentile, run_evaluation, summarise_samples, validate_scoped_target


class EvaluationTests(unittest.TestCase):
    def test_sample_summary_has_latency_and_pressure_metrics(self) -> None:
        summary = summarise_samples([
            {"scheduler_lag_ms": 1.0, "probe_service_ms": 0.1, "cpu_percent": 10.0, "load1_per_cpu": 0.2, "memory_percent": 40.0, "swap_percent": 0.0},
            {"scheduler_lag_ms": 5.0, "probe_service_ms": 0.2, "cpu_percent": 30.0, "load1_per_cpu": 0.4, "memory_percent": 42.0, "swap_percent": 1.0},
        ])
        self.assertEqual(summary["sample_count"], 2)
        self.assertGreater(summary["foreground_latency_ms_p95"], 4.0)
        self.assertEqual(summary["host_cpu_percent_max"], 30.0)
        self.assertEqual(percentile([1.0, 5.0], 0.5), 3.0)

    def test_unknown_mode_is_refused_before_any_workload_starts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(EvaluationSafetyError):
                run_evaluation(
                    mode="unknown_mode",
                    repetition=1,
                    workload="does-not-matter",
                    output=Path(directory) / "results.jsonl",
                )

    def test_linux_default_writes_append_only_structured_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            result = run_evaluation(
                mode="linux_default",
                repetition=1,
                workload="unused",
                output=output,
                probe_interval_ms=10.0,
                workload_command=(sys.executable, "-c", "import time; time.sleep(0.06)"),
            )
            saved = json.loads(output.read_text().strip())
            self.assertEqual(saved["mode"], "linux_default")
            self.assertEqual(saved["background_exit_code"], 0)
            self.assertFalse(saved["safety"]["real_priority_changes"])
            self.assertFalse(saved["bgopt"]["controller_invoked"])
            self.assertEqual(result["host_metrics"]["sample_count"], saved["host_metrics"]["sample_count"])


class WorkloadIdentityTests(unittest.TestCase):
    def test_matching_pid_and_create_time_is_accepted(self):
        import os, psutil
        pid=os.getpid(); ct=psutil.Process(pid).create_time()
        ok, reason=validate_scoped_target(WorkloadIdentity(pid,ct,sys.executable), {"pid":pid,"process_create_time":ct})
        self.assertTrue(ok); self.assertEqual(reason, "accepted")

    def test_wrong_pid_is_rejected(self):
        import os, psutil
        pid=os.getpid(); ct=psutil.Process(pid).create_time()
        ok, reason=validate_scoped_target(WorkloadIdentity(pid,ct,sys.executable), {"pid":pid+1,"process_create_time":ct})
        self.assertFalse(ok); self.assertIn("PID does not match", reason)

    def test_wrong_create_time_is_rejected(self):
        import os, psutil
        pid=os.getpid(); ct=psutil.Process(pid).create_time()
        ok, reason=validate_scoped_target(WorkloadIdentity(pid,ct,sys.executable), {"pid":pid,"process_create_time":ct+1000})
        self.assertFalse(ok); self.assertIn("create_time", reason)


class ControllerWorkerTests(unittest.TestCase):
    def test_slow_controller_does_not_block_independent_probe_and_stops(self):
        class SlowController:
            def __init__(self, *args, **kwargs):
                pass
            def run_once(self, *args, **kwargs):
                time.sleep(0.15)

        import bgopt.evaluation as evaluation
        original_controller = evaluation.Controller
        original_config = evaluation.load_config
        evaluation.Controller = SlowController
        evaluation.load_config = lambda path: object()
        try:
            with tempfile.TemporaryDirectory() as directory:
                stop_event = mp.Event()
                counter = mp.Value("i", 0)
                log_path = str(Path(directory) / "worker.jsonl")
                worker = mp.Process(target=_controller_worker, args=("bgopt_heuristic", log_path, 12345, 0.01, stop_event, counter))
                worker.start()
                probe_samples = []
                deadline = time.monotonic() + 0.22
                while time.monotonic() < deadline:
                    probe_samples.append(time.monotonic())
                    time.sleep(0.05)
                stop_event.set()
                worker.join(timeout=2.0)
                self.assertFalse(worker.is_alive())
                self.assertGreaterEqual(len(probe_samples), 4)
                self.assertGreaterEqual(counter.value, 1)
        finally:
            evaluation.Controller = original_controller
            evaluation.load_config = original_config


    def test_ml_worker_runs_and_stops_independently(self):
        class NoopController:
            def __init__(self, *args, **kwargs):
                pass
            def run_once(self, *args, **kwargs):
                time.sleep(0.02)

        import bgopt.evaluation as evaluation
        original_controller = evaluation.Controller
        original_config = evaluation.load_config
        evaluation.Controller = NoopController
        evaluation.load_config = lambda path: object()
        try:
            with tempfile.TemporaryDirectory() as directory:
                stop_event = mp.Event()
                counter = mp.Value("i", 0)
                worker = mp.Process(target=_controller_worker, args=("bgopt_ml_assisted", str(Path(directory) / "ml-worker.jsonl"), 12345, 0.01, stop_event, counter))
                worker.start()
                time.sleep(0.08)
                stop_event.set()
                worker.join(timeout=2.0)
                self.assertFalse(worker.is_alive())
                self.assertGreaterEqual(counter.value, 1)
        finally:
            evaluation.Controller = original_controller
            evaluation.load_config = original_config
