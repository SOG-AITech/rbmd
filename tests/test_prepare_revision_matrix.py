"""Check alpha mode selection in the four-way input generator."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_revision_matrix.py"


class RevisionMatrixTest(unittest.TestCase):
    def generate(self, coulomb, flags=(), error=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "init_configuration": {"read_data": {"file": "fixture.data"}},
                "hyper_parameters": {"force_field": {"type": "LJ/CUT/COUL/LONG"},
                                     "neighbor": {"r_core": 2}, "coulomb": coulomb},
                "execution": {},
            }
            (root / "input.json").write_text(json.dumps(config))
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root / "input.json"), str(root / "matrix"),
                 "--kmax", "4", "4", "4", "--rbe-samples", "32", "--rbl-samples", "16", *flags],
                capture_output=True, text=True)
            if error:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertFalse((root / "matrix").exists())
                return
            self.assertEqual(result.returncode, 0, result.stderr)
            configs = [json.loads(p.read_text()) for p in (root / "matrix").glob("*/rbmd.json")]
            self.assertEqual(len(configs), 4)
            return [c["hyper_parameters"]["coulomb"] for c in configs]

    def test_default_and_explicit_false_are_automatic(self):
        for extra in ({}, {"debug_alpha": False}):
            for c in self.generate({"accuracy": 1e-4, **extra}):
                self.assertNotIn("alpha", c)
                self.assertFalse(c.get("debug_alpha", False))
                self.assertEqual(c["accuracy"], 1e-4)

    def test_explicit_debug_override_reaches_all_four_cases(self):
        for c in self.generate({"accuracy": 1e-4, "alpha": 0.9},
                               ["--debug-alpha", "--alpha", "0.25"]):
            self.assertTrue(c["debug_alpha"])
            self.assertEqual(c["alpha"], 0.25)

    def test_both_debug_arguments_are_required(self):
        for flags in (["--alpha", "0.25"], ["--debug-alpha"]):
            self.generate({"accuracy": 1e-4}, flags, "must be supplied together")

    def test_invalid_manual_values_are_rejected(self):
        for value in ("0", "-1", "nan", "inf"):
            self.generate({}, ["--debug-alpha", "--alpha", value], "finite and positive")

    def test_ambiguous_base_and_missing_accuracy_are_rejected(self):
        self.generate({"accuracy": 1e-4, "alpha": 0.3}, error="remove manual alpha")
        self.generate({"accuracy": 1e-4, "debug_alpha": True}, error="remove manual alpha")
        for accuracy in (None, 0, True, "1e-4"):
            self.generate({"accuracy": accuracy}, error="finite positive coulomb.accuracy")


if __name__ == "__main__":
    unittest.main()
