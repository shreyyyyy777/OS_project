import unittest

from bgopt.policy import PolicyConfig, decide
from bgopt.pressure import Pressure


HIGH_CPU_PRESSURE = Pressure(0.9, "high", 0.95, 0.2, 0.0, 0.0, ("high_cpu_or_load",))


class PolicyTests(unittest.TestCase):
    def test_terminal_bound_process_is_never_changed(self) -> None:
        decision = decide(
            {"is_terminal_bound": True, "proc_cpu_percent": 99, "nice": 0},
            "background_cpu",
            HIGH_CPU_PRESSURE,
            PolicyConfig(),
        )
        self.assertEqual(decision.kind, "no_op")
        self.assertIn("protect_interactive", decision.safety_checks)

    def test_cpu_background_process_gets_lower_priority_proposal(self) -> None:
        decision = decide(
            {"is_terminal_bound": False, "parent_is_shell": False, "proc_cpu_percent": 70, "nice": 0},
            "background_cpu",
            HIGH_CPU_PRESSURE,
            PolicyConfig(),
        )
        self.assertEqual(decision.kind, "nice")
        self.assertEqual(decision.target_nice, 10)
        self.assertEqual(decision.action_code, "NICE_PLUS_10")

    def test_low_pressure_is_noop(self) -> None:
        low = Pressure(0.1, "low", 0.1, 0.1, 0.0, 0.0, ("normal_resource_pressure",))
        decision = decide({"proc_cpu_percent": 70, "nice": 0}, "background_cpu", low, PolicyConfig())
        self.assertEqual(decision.kind, "no_op")
