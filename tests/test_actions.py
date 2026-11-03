import unittest

from bgopt.actions import ActionApplier
from bgopt.policy import Decision


class ActionTests(unittest.TestCase):
    def test_cpu_limit_is_dry_run_only_without_explicit_apply(self) -> None:
        decision = Decision("cgroup_cpu_limit", None, None, 50, "CPU_LIMIT", "test", ())
        result = ActionApplier().apply(decision, {"pid": 12345}, dry_run=True)
        self.assertEqual(result.status, "would_apply")

