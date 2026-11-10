"""Host-only statistical regression tests; no CUDA/HIP or RBMD executable needed."""
import csv
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("force_analysis", Path(__file__).resolve().parents[1] / "common" / "analyze_force_samples.py")
analysis = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(Path(spec.origin).parent))
spec.loader.exec_module(analysis)
sys.path.pop(0)


class ForceStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, rows):
        path = Path(self.tmp.name) / name
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample", "id", "x", "y", "z", "fx", "fy", "fz"])
            w.writerows(rows)
        return str(path)

    def test_vector_rms_variance_bias_and_rank_migration(self):
        # Atom 0 changes rank between samples. Mean force (2,0,0), ref (0,0,0).
        a = self.write("a.csv", [(1, 0, 0, 0, 0, 1, 0, 0), (2, 1, 1, 0, 0, 0, 4, 0)])
        b = self.write("b.csv", [(1, 1, 1, 0, 0, 0, 2, 0), (2, 0, 0, 0, 0, 3, 0, 0)])
        ref = self.write("ref.csv", [(0, 0, 0, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0, 1, 0)])
        result, _, _ = analysis.analyze([a, b], [ref])
        self.assertEqual(result["n_atoms"], 2)
        self.assertAlmostEqual(result["rms_force_error"], math.sqrt(5))
        self.assertAlmostEqual(result["rms_bias"], 2)
        self.assertAlmostEqual(result["variance_population"], 1)
        self.assertAlmostEqual(result["variance_unbiased"], 2)
        self.assertAlmostEqual(result["decomposition_residual"], 0)

    def test_missing_or_duplicate_owned_ids_rejected(self):
        for name, rows in [
            ("missing", [(1, 0, 0, 0, 0, 1, 0, 0), (1, 1, 1, 0, 0, 1, 0, 0), (2, 0, 0, 0, 0, 1, 0, 0)]),
            ("duplicate", [(1, 0, 0, 0, 0, 1, 0, 0), (1, 0, 0, 0, 0, 2, 0, 0)])]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                analysis.analyze([self.write(name + ".csv", rows)])

    def test_motion_is_not_sampling_variance(self):
        rows = [(1, 0, 0, 0, 0, 1, 0, 0), (2, 0, .1, 0, 0, 2, 0, 0)]
        with self.assertRaisesRegex(ValueError, "moved"):
            analysis.analyze([self.write("moving.csv", rows)])

    def test_one_sample_and_zero_reference_have_no_unbiased_variance_or_relative_error(self):
        sample = self.write("one.csv", [(1, 0, 0, 0, 0, 1, 0, 0)])
        ref = self.write("zero.csv", [(1, 1, 0, 0, 0, 0, 0, 0)])
        result, _, _ = analysis.analyze([sample], [ref], id_offset=-1)
        self.assertIsNone(result["variance_unbiased"])
        self.assertIsNone(result["relative_rms_force_error"])
        self.assertEqual(result["rms_force_error"], 1)

    def test_unordered_nonfinite_and_wrong_reference_rejected(self):
        for name, rows in [
            ("unordered", [(2, 0, 0, 0, 0, 1, 0, 0), (1, 0, 0, 0, 0, 1, 0, 0)]),
            ("nonfinite", [(1, 0, 0, 0, 0, float("nan"), 0, 0)])]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                analysis.analyze([self.write(name + ".csv", rows)])
        sample = self.write("valid.csv", [(1, 0, 0, 0, 0, 1, 0, 0)])
        wrong = self.write("wrong.csv", [(1, 0, 2, 0, 0, 1, 0, 0)])
        with self.assertRaisesRegex(ValueError, "position mismatch"):
            analysis.analyze([sample], [wrong])

    def dump_frame(self, step, atoms, fields="id x y z fx fy fz"):
        return (f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(atoms)}\n"
                "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
                f"ITEM: ATOMS {fields}\n" + "".join(line + "\n" for line in atoms))

    def test_lammps_reference_cli_timestep_columns_and_id_offset(self):
        import json
        sample = self.write("samples.csv", [(1, 0, 0, 0, 0, 1, 0, 0),
                                            (1, 1, 1, 0, 0, 0, 3, 0),
                                            (2, 1, 1, 0, 0, 0, 5, 0),
                                            (2, 0, 0, 0, 0, 3, 0, 0)])
        dump = Path(self.tmp.name) / "reference.dump"
        fields = "fy id z fx x fz y type"
        dump.write_text(self.dump_frame(0, ["0 1 0 9 0 0 0 1", "9 2 0 0 1 0 0 1"], fields)
                        + self.dump_frame(10, ["2 2 0 0 1 0 0 1", "0 1 0 0 0 0 0 1"], fields))
        output = Path(self.tmp.name) / "result.json"
        subprocess.run([sys.executable, str(spec.origin), sample, "--reference", str(dump),
                        "--reference-sample", "10", "--reference-id-offset", "-1",
                        "--output", str(output)], check=True, capture_output=True, text=True)
        result = json.loads(output.read_text())
        self.assertEqual(result["reference_sample"], 10)
        self.assertTrue(result["reference_positions_checked"])
        self.assertAlmostEqual(result["rms_force_error"], math.sqrt(5))
        self.assertAlmostEqual(result["relative_rms_force_error"], math.sqrt(2.5))
        self.assertAlmostEqual(result["rms_bias"], 2)
        self.assertAlmostEqual(result["decomposition_residual"], 0)
        with self.assertRaisesRegex(ValueError, "no atoms"):
            analysis.analyze([sample], [str(dump)], reference_sample=99, id_offset=-1)

    def test_lammps_reference_default_frame_and_invalid_inputs(self):
        sample = self.write("sample.csv", [(1, 0, 0, 0, 0, 1, 0, 0)])
        dump = Path(self.tmp.name) / "reference.dump"
        good = self.dump_frame(0, ["0 0 0 0 0 0 0"])
        dump.write_text(good + self.dump_frame(5, ["0 0 0 0 8 0 0"]))
        result, _, _ = analysis.analyze([sample], [str(dump)])
        self.assertEqual(result["rms_force_error"], 1)
        for name, content, message in [
            ("truncated", good.rsplit("\n", 2)[0] + "\n", "malformed"),
            ("scaled", good.replace("id x y z", "id xs ys zs"), "requires"),
            ("nan", self.dump_frame(0, ["0 0 0 0 nan 0 0"]), "non-finite"),
            ("duplicate", self.dump_frame(0, ["0 0 0 0 0 0 0"] * 2), "Duplicate"),
            ("position", self.dump_frame(0, ["0 1 0 0 0 0 0"]), "position mismatch"),
            ("ids", self.dump_frame(0, ["2 0 0 0 0 0 0"]), "missing atom"),
            ("repeated_step", good + good, "strictly increasing"),
        ]:
            with self.subTest(name=name):
                dump.write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    analysis.analyze([sample], [str(dump)])


if __name__ == "__main__":
    unittest.main()
