import json
import tempfile
import unittest
from pathlib import Path

from bgopt.actions import ActionApplier
from bgopt.controller import Controller, ControllerConfig, load_config
from bgopt.fairness import InterventionLedger
from bgopt.policy import PolicyConfig


class _StaticCollector:
    def __init__(self, process_name="batch-job", records=None):
        self.process_name = process_name
        self.records = records

    def collect(self):
        if self.records is not None:
            return [dict(record) for record in self.records]
        return [{
            "pid": 424242,
            "ppid": 1,
            "process_name": self.process_name,
            "nice": 0,
            "proc_cpu_percent": 80.0,
            "proc_memory_percent": 1.0,
            "proc_rss_mb": 100.0,
            "proc_read_mb_s": 0.0,
            "proc_write_mb_s": 0.0,
            "is_terminal_bound": False,
            "parent_is_shell": False,
            "is_interactive_hint": False,
            "host_cpu_percent": 99.0,
            "host_load_1_norm": 1.2,
            "host_memory_percent": 30.0,
            "host_swap_percent": 0.0,
            "host_disk_busy_mb_s": 0.0,
        }]


class ControllerTests(unittest.TestCase):
    def test_cpu_candidate_is_logged_as_a_dry_run_proposal(self) -> None:
        controller = Controller.__new__(Controller)
        controller.config = ControllerConfig(policy=PolicyConfig())
        controller.collector = _StaticCollector()
        controller.model = None
        controller.applier = ActionApplier()
        controller.ledger = InterventionLedger()
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "audit.jsonl"
            events = controller.run_once(log, dry_run=True)
            stored = json.loads(log.read_text(encoding="utf-8"))
        self.assertEqual(len(events), 1)
        self.assertEqual(stored["decision"]["kind"], "nice")
        self.assertEqual(stored["result"]["status"], "would_apply")
    def test_protected_process_cannot_receive_optimization(self) -> None:
        controller = Controller.__new__(Controller)
        controller.config = ControllerConfig(
            policy=PolicyConfig(protected_process_names=("firefox",))
        )
        controller.collector = _StaticCollector("firefox")
        controller.model = None
        controller.applier = ActionApplier()
        controller.ledger = InterventionLedger()

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "audit.jsonl"
            events = controller.run_once(log, dry_run=True)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["decision"]["action_code"], "NO_ACTION")
        self.assertEqual(
            events[0]["decision"]["reason"],
            "configured foreground process is protected",
        )
    def test_background_shell_child_is_not_interactive(self) -> None:
        record = {
            "is_terminal_bound": True,
            "parent_is_shell": True,
            "is_terminal_foreground": False,
        }

        from bgopt.features import interactive_hint

        self.assertFalse(interactive_hint(record))


    def test_protected_process_descendant_cannot_receive_optimization(self) -> None:
        firefox = _StaticCollector("firefox").collect()[0]
        firefox.update({"pid": 100, "ppid": 1})
        helper = _StaticCollector("Isolated Web Co").collect()[0]
        helper.update({"pid": 101, "ppid": 100})

        controller = Controller.__new__(Controller)
        controller.config = ControllerConfig(
            policy=PolicyConfig(protected_process_names=("firefox",))
        )
        controller.collector = _StaticCollector(records=[firefox, helper])
        controller.model = None
        controller.applier = ActionApplier()
        controller.ledger = InterventionLedger()

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "audit.jsonl"
            events = controller.run_once(log, dry_run=True)

        decisions = {
            event["telemetry"]["process_name"]: event["decision"]
            for event in events
        }
        helper_event = next(
            event for event in events
            if event["telemetry"]["process_name"] == "Isolated Web Co"
        )
        self.assertEqual(decisions["firefox"]["action_code"], "NO_ACTION")
        self.assertEqual(decisions["Isolated Web Co"]["action_code"], "NO_ACTION")
        self.assertEqual(
            decisions["Isolated Web Co"]["reason"],
            "descendant of configured foreground process is protected",
        )
        self.assertTrue(helper_event["telemetry"]["is_protected_descendant"])

    def test_terminal_foreground_process_remains_interactive(self) -> None:
        record = {
            "is_terminal_bound": True,
            "parent_is_shell": True,
            "is_terminal_foreground": True,
        }

        from bgopt.features import interactive_hint

        self.assertTrue(interactive_hint(record))

    def test_protected_descendant_setting_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.write_text("[policy]\nprotect_descendants = false\n", encoding="utf-8")
            loaded = load_config(config)

        self.assertFalse(loaded.policy.protect_descendants)
