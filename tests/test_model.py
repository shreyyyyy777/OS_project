import tempfile
import unittest
from pathlib import Path

from bgopt.features import MODEL_FEATURES
from bgopt.model import load, predict, train


@unittest.skipUnless(__import__("importlib").util.find_spec("pandas"), "ML dependencies are not installed")
class ModelTests(unittest.TestCase):
    def test_train_load_and_predict(self) -> None:
        import pandas as pd

        rows = []
        for label, cpu, terminal in (("background_cpu", 70.0, 0.0), ("interactive", 1.0, 1.0)):
            for index in range(6):
                row = {feature: 0.0 for feature in MODEL_FEATURES}
                row.update({"proc_cpu_percent": cpu + index, "host_cpu_percent": 85.0, "is_terminal_bound": terminal})
                row["workload_label"] = label
                rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, model = root / "training.csv", root / "model.joblib"
            pd.DataFrame(rows).to_csv(dataset, index=False)
            train(dataset, model)
            label, confidence, source = predict(load(model), rows[0])
        self.assertIn(label, {"background_cpu", "interactive"})
        self.assertGreaterEqual(confidence, 0.0)
        self.assertEqual(source, "model")

    def test_grouped_split_is_used_for_independent_experiments(self) -> None:
        import pandas as pd

        rows = []
        for group in range(6):
            for label, cpu in (("background_cpu", 70.0), ("interactive", 1.0)):
                for index in range(3):
                    row = {feature: 0.0 for feature in MODEL_FEATURES}
                    row.update({"proc_cpu_percent": cpu + index, "host_cpu_percent": 85.0, "experiment_id": f"experiment_{group}", "workload_label": label})
                    rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dataset, model = root / "training.csv", root / "model.joblib"
            pd.DataFrame(rows).to_csv(dataset, index=False)
            report = train(dataset, model)
        self.assertEqual(report["split_strategy"], "grouped")
        self.assertEqual(report["group_column"], "experiment_id")
