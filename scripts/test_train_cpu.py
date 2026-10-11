import unittest
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


if __name__ == "__main__":
    unittest.main()
