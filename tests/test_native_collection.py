import runpy
import unittest
from pathlib import Path

from bgopt.features import MODEL_FEATURES


class NativeCollectionTests(unittest.TestCase):
    def test_protocol_uses_runtime_schema_and_explicit_workload_classes(self) -> None:
        module = runpy.run_path(Path(__file__).parents[1] / "tools" / "collect_bgopt_native_training.py")
        self.assertEqual(set(module["SPECS"]), {"idle", "interactive", "background_cpu", "memory_heavy", "io_heavy"})
        self.assertTrue(MODEL_FEATURES)
        self.assertNotIn("laps", Path(module["__file__"]).read_text().lower())
