"""Analytic low modes and exact zero-weight parity for the training-only change."""
import copy
import math
import unittest
from unittest.mock import patch
import torch
import band_feedback
import model
import physical_residual
import policy
import test_residual


class BandFeedback(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.spec = {'weight': .1, 'max_mode': 3, 'signal_floor_fraction': .1, 'stages': 'first-and-final'}
        old = policy.decode(policy.read(policy.LAB / 'experiments/water-early-pressure-v1/configs/control.json'))
        self.old = old

    def configuration(self, weight):
        return {**copy.deepcopy(self.old), 'schema': 'rheon-water-residual-experiment-v10',
                'band_feedback': {**self.spec, 'weight': weight}}

    def test_odd_rectangular_cosine_has_known_orthonormal_coefficient(self):
        wave = torch.tensor([2 * math.cos(math.pi * (x + .5) / 5) for x in range(5)], dtype=torch.float64)
        field = wave.reshape(1, 1, 1, 1, 5).expand(1, 1, 9, 7, 5)
        coefficients = band_feedback.coefficients(field, 3)
        self.assertAlmostEqual(float(coefficients[0, 0]), 2 * math.sqrt(9 * 7 * 5 / 2), places=11)
        self.assertLess(float(coefficients[0, 1:].abs().max()), 1e-12)
        constant = torch.full_like(field, 123.)
        self.assertLess(float(band_feedback.coefficients(constant, 3).abs().max()), 1e-11)

    def test_high_only_target_penalizes_spurious_low_mode_with_bounded_floor(self):
        x = torch.arange(9, dtype=torch.float64) + .5
        high = torch.cos(math.pi * 7 / 9 * x).reshape(1, 1, 1, 1, 9).expand(1, 1, 9, 7, 9)
        low = torch.cos(math.pi / 9 * x).reshape(1, 1, 1, 1, 9).expand_as(high)
        sample = {'reference': high, 'base': torch.zeros_like(high)}
        value = band_feedback.loss(high + .01 * low, sample, self.spec)
        self.assertAlmostEqual(float(value), .01, places=11)
        self.assertEqual(float(band_feedback.loss(high, sample, self.spec)), 0.)
        shifted = {'reference': high + 100, 'base': sample['base']}
        self.assertAlmostEqual(float(band_feedback.loss(high + .01 * low + 100, shifted, self.spec)), float(value), places=10)

    def test_zero_weight_preserves_loss_gradients_and_inference_exactly(self):
        fixture = test_residual.Residual()
        fixture.setUp()
        sample, norm = fixture.sample, fixture.norm
        model.select(self.old)
        torch.manual_seed(19)
        network = model.PressureCorrectionNet()
        before = model.training_loss(network, sample, norm)
        before.sum().backward()
        gradients = [p.grad.clone() for p in network.parameters()]
        prediction = model.predict_pressure(network, sample, norm)
        network.zero_grad(set_to_none=True)
        policy.experiment(self.configuration(0))
        model.select(self.configuration(0))
        after = model.training_loss(network, sample, norm)
        after.sum().backward()
        self.assertTrue(torch.equal(before, after))
        for expected, parameter in zip(gradients, network.parameters()):
            self.assertTrue(torch.equal(expected, parameter.grad))
        model.select(self.configuration(.1))
        self.assertTrue(torch.equal(prediction, model.predict_pressure(network, sample, norm)))

    def test_first_and_final_feedback_reuses_predictions_and_backpropagates(self):
        model.select(self.configuration(.1))
        x = torch.arange(9, dtype=torch.float64) + .5
        reference = torch.cos(math.pi / 9 * x).reshape(1, 1, 1, 1, 9)
        sample = {'reference': reference, 'base': torch.zeros_like(reference)}
        a, b = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (2., 3.))
        def predict(_network, _sample, _norm, _owner, _correction, intermediates):
            intermediates.extend([a * reference, b * reference])
            return b * reference
        with patch.object(physical_residual, 'predict_pressure', side_effect=predict) as forward, \
             patch.object(physical_residual, 'intermediate_contraction', return_value=torch.tensor([0.], dtype=torch.float64)):
            value = model.training_loss(None, sample, None, criterion=lambda p: p.flatten(1).sum(1) * 0)
        self.assertAlmostEqual(float(value.detach()), .25, places=12)
        value.sum().backward()
        self.assertAlmostEqual(float(a.grad), .1, places=12)
        self.assertAlmostEqual(float(b.grad), .2, places=12)
        self.assertEqual(forward.call_count, 1)

    def test_configuration_rejects_unbounded_or_unknown_feedback(self):
        for key, value in [('weight', True), ('weight', -1), ('max_mode', 0), ('max_mode', 5),
                           ('signal_floor_fraction', 0), ('signal_floor_fraction', float('nan')),
                           ('stages', 'arbitrary')]:
            config = self.configuration(.1)
            config['band_feedback'][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'band_feedback'):
                policy.experiment(config)

    def test_longer_finite_spectral_deadline_preserves_historical_cpu_policy(self):
        config=self.configuration(.1);config['resources']['wall_seconds']=1800
        self.assertEqual(policy.experiment(config)['resources']['wall_seconds'],1800)
        prior=copy.deepcopy(self.old);prior['resources']['wall_seconds']=1800
        with self.assertRaisesRegex(ValueError,'resources'):policy.experiment(prior)
        for seconds in (True,29,1801):
            config['resources']['wall_seconds']=seconds
            with self.assertRaisesRegex(ValueError,'resources'):policy.experiment(config)
