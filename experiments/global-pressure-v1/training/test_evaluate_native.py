"""Focused witness/population/admission checks; no source-case solver execution."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

spec = importlib.util.spec_from_file_location("native_eval", Path(__file__).with_name("evaluate_native.py"))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class NativeWitnessTests(unittest.TestCase):
    def test_population_is_all16_trajectory_endpoints_in_whole_history_order(self):
        cases = native.case_population()
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({c["case_id"] for c in cases}), 32)
        self.assertEqual([c["frame"] for c in cases], [0, 23] * 16)
        self.assertEqual(cases[0]["case_id"], "smoke-train-100000:frame-000")
        self.assertEqual(cases[-1]["case_id"], "smoke-validation-100103:frame-023")
        self.assertEqual(cases[-1]["row_index"], 383)

    def witness(self, directory, *, rows=None, pressure=None, extra=False):
        path = Path(directory) / "predictions.npz"
        arrays = {"pressure": np.zeros((32, 16, 16, 16), dtype="<f8") if pressure is None else pressure,
                  "row_indices": np.array([c["row_index"] for c in native.case_population()], dtype="<i4")
                  if rows is None else rows}
        if extra:
            arrays["text"] = np.array(["forbidden"])
        np.savez(path, **arrays)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_actual_numeric_witness_keeps_nonfinite_prediction_for_visible_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            pressure = np.zeros((32, 16, 16, 16), dtype="<f8")
            pressure[0, 0, 0, 1] = np.nan
            path, digest = self.witness(directory, pressure=pressure)
            actual = native.read_witness(path, digest)
            self.assertTrue(np.isnan(actual[0, 0, 0, 1]))

    def test_relabelled_population_and_executable_extra_inventory_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = np.array([c["row_index"] for c in native.case_population()], dtype="<i4")
            rows[1] = rows[0]
            path, digest = self.witness(directory, rows=rows)
            with self.assertRaisesRegex(ValueError, "population/order"):
                native.read_witness(path, digest)
            path, digest = self.witness(directory, extra=True)
            with self.assertRaisesRegex(ValueError, "inventory"):
                native.read_witness(path, digest)

    def test_drifted_actual_witness_and_wrong_shape_fail_before_native_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self.witness(directory)
            with path.open("ab") as stream:
                stream.write(b"drift")
            with self.assertRaisesRegex(ValueError, "identity/size"):
                native.read_witness(path, digest)
            path, digest = self.witness(directory, pressure=np.zeros((31, 16, 16, 16), dtype="<f8"))
            with self.assertRaisesRegex(ValueError, "dtype/shape"):
                native.read_witness(path, digest)

    def test_duplicate_json_and_symlink_inputs_are_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            native.strict_json('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            native.strict_json('{"a":1e999}')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.write_bytes(b"x")
            link = Path(directory) / "link"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "regular file"):
                native.regular(link)

    def test_forged_huge_shape_is_refused_before_numpy_can_allocate(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.witness(directory)
            with zipfile.ZipFile(path) as archive:
                rows = archive.read("row_indices.npy")
                pressure = archive.read("pressure.npy")
            pressure = pressure.replace(b"(32, 16, 16, 16)", b"(99, 99, 99, 99)", 1)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("pressure.npy", pressure)
                archive.writestr("row_indices.npy", rows)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(np, "load", side_effect=AssertionError("NumPy allocation was reached")):
                with self.assertRaisesRegex(ValueError, "NPY header dtype/shape"):
                    native.read_witness(path, digest)


if __name__ == "__main__":
    unittest.main()
