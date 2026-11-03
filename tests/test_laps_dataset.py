import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from bgopt.laps_dataset import profile_dataset, scene_reference


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is not installed")
class LapsDatasetTests(unittest.TestCase):
    def test_profile_reports_shape_and_scene_boundary(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["cpu_usage_mean", "focus_process_cpu"]
            (root / "feature_names.json").write_text(json.dumps(names), encoding="utf-8")
            with (root / "features.pkl").open("wb") as handle:
                pickle.dump(pd.DataFrame([[1.0, 2.0]], columns=names), handle)
            with (root / "metadata.pkl").open("wb") as handle:
                pickle.dump(pd.DataFrame([{"label": "game_play_kernel_compile", "vm_identifier": "vm_1", "group": "g"}]), handle)
            feature_map = root / "feature_map.txt"
            feature_map.write_text("cpu_usage_mean\nfocus_process_cpu\n", encoding="utf-8")
            report = profile_dataset(root, feature_map)
        self.assertEqual(report["feature_shape"], [1, 2])
        self.assertEqual(report["label_count"], 1)
        self.assertIn("cannot train BGOpt", report["reuse_boundary"])

    def test_scene_reference_keeps_the_label_separate_from_an_action(self) -> None:
        reference = scene_reference("web_browsing_network_intensive")
        self.assertEqual(reference["foreground_context"], "web_browsing")
        self.assertEqual(reference["provisional_resource_hint"], "network_intensive")
        self.assertTrue(reference["human_review_required"])

