import unittest
import numpy as np
import global_candidate as candidate


class GlobalContracts(unittest.TestCase):
    def test_basis_and_linear_zero_gauge_contract(self):
        basis = candidate.cosine_basis([4, 3, 2])
        np.testing.assert_allclose(basis.T @ basis, np.eye(23), atol=1e-14)
        np.testing.assert_allclose(basis.sum(axis=0), 0, atol=1e-14)
        rng = np.random.default_rng(29)
        rhs = rng.normal(size=(2, 24))
        gain = rng.normal(size=23)
        p = candidate.predict(gain, rhs, basis)
        np.testing.assert_array_equal(candidate.predict(gain, np.zeros_like(rhs), basis), 0)
        np.testing.assert_array_equal(p[:, 0], 0)
        np.testing.assert_array_equal(candidate.predict(gain, -rhs, basis), -p)
        np.testing.assert_array_equal(candidate.predict(gain, 2*rhs, basis), 2*p)

    def test_relative_loss_gradient_with_gauge_and_full_operator(self):
        rng = np.random.default_rng(73)
        basis = candidate.cosine_basis([3, 2, 2])
        factor = rng.normal(size=(12, 12))
        operator = factor.T @ factor
        rhs = rng.normal(size=(3, 12))
        target = rng.normal(size=(3, 12))
        target -= target[:, :1]
        log_gain = rng.normal(size=11) * 0.1
        _, gradient, _ = candidate.loss_gradient(log_gain, rhs, target, basis, operator)
        for index in (0, 4, 10):
            original = log_gain[index]
            log_gain[index] = original + 1e-6
            plus = candidate.loss_gradient(log_gain, rhs, target, basis, operator)[0]
            log_gain[index] = original - 1e-6
            minus = candidate.loss_gradient(log_gain, rhs, target, basis, operator)[0]
            log_gain[index] = original
            np.testing.assert_allclose((plus-minus)/2e-6, gradient[index], rtol=1e-7, atol=1e-7)
        # Scaling individual cases cannot change their relative contribution.
        scale = np.array([0.1, 3, 10])[:, None]
        loss, grad, _ = candidate.loss_gradient(log_gain, rhs, target, basis, operator)
        scaled, scaled_grad, _ = candidate.loss_gradient(log_gain, rhs*scale, target*scale, basis, operator)
        np.testing.assert_allclose(loss, scaled, rtol=1e-14)
        np.testing.assert_allclose(grad, scaled_grad, rtol=1e-13)


if __name__ == "__main__":
    unittest.main()
