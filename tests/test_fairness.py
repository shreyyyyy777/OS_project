import unittest

from bgopt.fairness import FairnessConfig, InterventionLedger
from bgopt.policy import Decision
from bgopt.pressure import Pressure


class FairnessTests(unittest.TestCase):
    def test_cooldown_prevents_repeated_proposal(self) -> None:
        ledger = InterventionLedger(FairnessConfig(cooldown_seconds=60))
        record = {"pid": 99, "nice": 0}
        proposal = Decision("nice", 5, None, None, "NICE_PLUS_5", "test", ())
        pressure = Pressure(.9, "high", .9, .1, 0, 0, ())
        ledger.record_proposal(record, proposal, 100.0)
        constrained = ledger.constrain(record, pressure, proposal, 110.0)
        self.assertEqual(constrained.kind, "no_op")
        self.assertIn("cooldown", constrained.safety_checks)

    def test_contention_clearance_proposes_restoration(self) -> None:
        ledger = InterventionLedger(FairnessConfig(restoration_cpu_threshold=.4))
        record = {"pid": 99, "nice": 0}
        proposal = Decision("nice", 5, None, None, "NICE_PLUS_5", "test", ())
        ledger.record_proposal(record, proposal, 100.0)
        low = Pressure(.1, "low", .2, .1, 0, 0, ())
        restore = ledger.constrain(record, low, proposal, 140.0)
        self.assertEqual(restore.kind, "restore_nice")
        self.assertEqual(restore.target_nice, 0)
