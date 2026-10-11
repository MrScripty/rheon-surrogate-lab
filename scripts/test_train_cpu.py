import unittest
import tempfile
from pathlib import Path
import numpy as np
import train_cpu as model


class ModelContracts(unittest.TestCase):
    def test_gradient_including_cell_zero_gauge(self):
        rng = np.random.default_rng(71)
        patches = rng.normal(size=(2, 8, 108))
        target = rng.normal(size=(2, 8))
        target[:, 0] = 0
        params = model.initialize(hidden=3)
        _, gradients = model.loss_and_gradients(params, patches, target)
        for key, positions in {"kernel": [(0, 0), (50, 1), (107, 2)], "bias": [(0,), (2,)], "output": [(0,), (2,)]}.items():
            for position in positions:
                original = params[key][position]
                params[key][position] = original + 1e-6
                plus = model.loss_and_gradients(params, patches, target)[0]
                params[key][position] = original - 1e-6
                minus = model.loss_and_gradients(params, patches, target)[0]
                params[key][position] = original
                self.assertAlmostEqual((plus - minus) / 2e-6, gradients[key][position], places=7)
        self.assertTrue(np.all(model.forward(params, patches)[0][:, 0] == 0))

    def test_normalization_ignores_holdout_and_check(self):
        cases = [{"id": str(i), "split": split, "rhs": np.arange(8, dtype=float).reshape(2, 2, 2) + i,
                  "pressure": np.arange(8, dtype=float).reshape(2, 2, 2) * (i + 1)}
                 for i, split in enumerate(("train", "train", "validation", "test", "check"))]
        expected = model.fit_normalization(cases)
        for case in cases[2:]:
            case["rhs"] *= 1e10
            case["pressure"] *= 1e10
        self.assertEqual(expected, model.fit_normalization(cases))
        self.assertEqual(expected["fit_case_ids"], ["0", "1"])

    def test_x_fastest_encoding_and_padding(self):
        rhs = np.arange(24, dtype=float).reshape(2, 3, 4)
        cases = [{"rhs": rhs, "pressure": rhs}]
        patches, target = model.encode(cases, {"rhs_mean": 0, "rhs_std": 1, "pressure_rms": 1})
        self.assertEqual(patches.shape, (1, 24, 108))
        np.testing.assert_array_equal(patches[0, :, 13], rhs.ravel())
        np.testing.assert_array_equal(target[0], rhs.ravel())
        self.assertEqual(patches[0, 0, 0], 0)  # Outside-domain padding.
        self.assertEqual(patches[0, 0, 14], 1)  # Immediate +x neighbor.

    def test_small_optimization_reduces_loss(self):
        rng = np.random.default_rng(9)
        patches = rng.normal(size=(1, 8, 108))
        target = np.arange(8, dtype=float).reshape(1, 8) / 8
        history = model.train(model.initialize(hidden=4), patches, target, 100)
        self.assertLess(history[-1]["normalized_mse"], history[0]["normalized_mse"] * 0.01)

    def test_default_consumer_skips_test_before_reading_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.arange(8, dtype="<f8")
            values.tofile(root / "field.bin")
            array = {"path": "field.bin", "sha256": model.digest(root / "field.bin"),
                     "dtype": "<f8", "shape_xyz": [2, 2, 2]}
            manifest = {"cases": [
                {"spec": {"id": "train", "split": "train"}, "arrays": {"rhs": array, "pressure": array}},
                {"spec": {"id": "test", "split": "test"}, "arrays": {
                    "rhs": {**array, "path": "not-readable-test.bin"}, "pressure": array}},
            ]}
            self.assertEqual([c["id"] for c in model.load_cases(root, manifest)], ["train"])
            with self.assertRaises(FileNotFoundError):
                model.load_cases(root, manifest, include_test=True)

    def test_radius_one_model_cannot_see_remote_rhs_change(self):
        zero = np.zeros((6, 7, 8))
        remote = zero.copy()
        remote[4, 5, 6], remote[4, 5, 7] = -1, 1
        norm = {"rhs_mean": 0, "rhs_std": 1, "pressure_rms": 1}
        cases = [{"rhs": rhs, "pressure": zero} for rhs in (zero, remote)]
        patches, _ = model.encode(cases, norm)
        index = 3 + 8 * (3 + 7 * 2)
        np.testing.assert_array_equal(patches[0, index], patches[1, index])
        np.testing.assert_array_equal(patches[0, 0], patches[1, 0])
        predictions = model.forward(model.initialize(), patches)[0]
        self.assertEqual(predictions[0, index], predictions[1, index])


if __name__ == "__main__":
    unittest.main()
