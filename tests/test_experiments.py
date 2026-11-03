import json
import tempfile
import unittest
from pathlib import Path

from bgopt.experiments import outcome_template, plan, summarise_outcomes


class ExperimentTests(unittest.TestCase):
    def test_plan_and_empty_outcome_template_do_not_fabricate_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "scenario.toml"
            config.write_text("[experiment]\nid='e1'\nscenario='fg_bg'\nrepetitions=2\n", encoding="utf-8")
            report = plan(config)
            self.assertEqual(len(report["runs"]), 8)
            planned = root / "plan.json"; planned.write_text(json.dumps(report), encoding="utf-8")
            output = root / "outcomes.csv"
            self.assertEqual(outcome_template(planned, output), 8)
            summary = summarise_outcomes(output)
        self.assertEqual(summary["metrics"], {})
