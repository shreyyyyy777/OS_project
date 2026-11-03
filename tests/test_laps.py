import tempfile
import unittest
from pathlib import Path

from bgopt.laps import inspect_artifact


class LapsInspectionTests(unittest.TestCase):
    def test_inventory_reads_feature_map_without_claiming_model_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("notes\n", encoding="utf-8")
            (root / "feature_map.txt").write_text("cpu_util\nmem_free\n", encoding="utf-8")
            report = inspect_artifact(root)
        self.assertEqual(report["laps_feature_count"], 2)
        self.assertIn("semantic crosswalk", report["compatibility_note"])

