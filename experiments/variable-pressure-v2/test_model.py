import unittest
from unittest.mock import Mock

import numpy as np
import torch

import data
import model
import policy
import train


def normalization():
    return {'schema': 'rheon-cubic-pressure-normalization-v2', 'channels': model.CHANNELS, 'scales':
            {name: {'value': 1.0} for name in ('residual', 'pressure', 'rhs', 'rho', 'dt')}}


def sample(shape, h=.0625):
    return {'base': torch.zeros((1, 1, *shape), dtype=torch.float64),
            'native_input_residual': torch.ones((1, 1, *shape), dtype=torch.float64),
            'cell_size_m': torch.tensor([h], dtype=torch.float64),
            'rho': torch.tensor([1.25], dtype=torch.float64), 'dt': torch.tensor([1/60], dtype=torch.float64)}


class VariableModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model.configure_cpu()

    def test_arbitrary_positive_shapes_keep_every_cell_and_backpropagate(self):
        network = model.PressureCorrectionNet()
        for shape in [(1,1,1), (2,3,5), (7,9,11), (12,16,24), (1,2,257), (16,16,16)]:
            with self.subTest(shape=shape):
                network.zero_grad(set_to_none=True)
                values = torch.randn((1,7,*shape), dtype=torch.float32, requires_grad=True)
                result = network(values)
                self.assertEqual(result.shape, (1,1,*shape))
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(values.grad).all())
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters()))

    def test_only_spacing_columns_are_removed_from_seeded_initialization(self):
        torch.manual_seed(127); previous = model.frozen.PressureCorrectionNet()
        torch.manual_seed(127); current = model.PressureCorrectionNet()
        self.assertEqual(model.parameter_count(current), 371729)
        self.assertEqual(current.encoders[0].layers[0].in_channels, 7)
        for name, value in current.state_dict().items():
            old = previous.state_dict()[name]
            expected = old[:, model.RETAINED_CHANNELS] if name == 'encoders.0.layers.0.weight' else old
            self.assertTrue(torch.equal(value, expected), name)
        state = sample((5,3,7))
        # The old ten-channel preprocessor must never be invoked, even temporarily.
        from unittest.mock import patch
        with patch.object(model.frozen, 'build_inputs', side_effect=AssertionError('No spacing maps')):
            inputs = model.build_inputs(state['native_input_residual'], state['base'], state['cell_size_m'], state['rho'], state['dt'], normalization())
        self.assertEqual(inputs.shape, (1,7,5,3,7))
        self.assertEqual(model.CHANNELS, ['native_residual','previous_pressure','x','y','z','rho','dt'])

    def test_physical_coordinates_and_cubic_spacing_before_inference(self):
        state = sample((3,2,5), .125)
        values = model.build_inputs(state['native_input_residual'], state['base'], state['cell_size_m'], state['rho'], state['dt'], normalization())
        self.assertAlmostEqual(float(values[0,2,0,0,-1]), .5625)
        self.assertAlmostEqual(float(values[0,3,0,-1,0]), .1875)
        self.assertAlmostEqual(float(values[0,4,-1,0,0]), .3125)
        network = Mock()
        state['cell_size_m'] = torch.tensor([[.125,.25,.125]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'cell_size N'):
            model.predict_pressure(network, state, normalization())
        network.assert_not_called()

    def test_independent_hand_operator_on_cubic_thin_domain(self):
        p = torch.tensor([[[[[0.,2.,5.]]]]], dtype=torch.float64)
        h = torch.tensor([.25], dtype=torch.float64)
        result = model.full_operator(p, h, torch.tensor([1.25], dtype=torch.float64))
        torch.testing.assert_close(result.flatten(), torch.tensor([-.4,-.2,.6], dtype=torch.float64), rtol=0, atol=1e-15)
        h = torch.tensor([[.25,.5,.25]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'cell_size N'):
            model.full_operator(p, h, torch.tensor([1.25], dtype=torch.float64))

    def test_zero_residual_exact_bypass_and_pressure_output_gauge(self):
        state = sample((5,3,7)); state['base'].fill_(.125); state['base'][0,0,0,0,0] = 0
        state['native_input_residual'].zero_(); network = Mock()
        self.assertTrue(torch.equal(model.predict_pressure(network, state, normalization()), state['base']))
        network.assert_not_called()
        state['native_input_residual'].fill_(1)
        pressure = model.predict_pressure(model.PressureCorrectionNet(), state, normalization())
        self.assertEqual(pressure.shape, state['base'].shape)
        self.assertEqual(pressure.dtype, torch.float64)
        self.assertEqual(float(pressure[0,0,0,0,0].detach()), 0)

    def test_resource_invalid_shape_and_nonfinite_rejection(self):
        network = model.PressureCorrectionNet()
        for values in (torch.zeros((1,7,1,1,0)), torch.zeros((1,7,41,41,41)), torch.zeros((1,7,2,2,2), dtype=torch.float64), torch.full((1,7,2,2,2), float('nan'))):
            with self.assertRaises(ValueError):
                network(values)

    def test_train_only_rms_and_order_exclude_validation(self):
        parents = []
        for index, shape in enumerate([(24,2,3,4), (24,3,2,5), (24,4,4,4)]):
            fields = {name: np.full(shape, index+1., dtype='<f8') for name in data.FIELDS}
            fields['base'][:,0,0,0] = 0; fields['reference'][:,0,0,0] = 0
            parents.append({'entry': {'split': 'validation' if index==2 else 'train', 'trajectory_id': str(index), 'dt_s': 1/60},
                            'cell_size_m': .125 if index==0 else .0625, 'counts_xyz': list(shape[:0:-1]), 'arrays': fields})
        first = data.fit_normalization(parents)
        parents[2]['arrays']['base'].fill(1e12)
        self.assertEqual(first, data.fit_normalization(parents))
        self.assertEqual(set(first['scales']), {'residual','pressure','rhs','rho','dt'})
        config = {'training': {'seed': 71, 'updates': 5, 'batch_size': 2}}
        order = train.order(parents, config)
        self.assertTrue(np.array_equal(order, train.order(parents, config)))
        self.assertTrue(set(order[:,:,0].flatten()) <= {0,1})

    def test_heterogeneous_batch_matches_mean_per_frame_gradient(self):
        import copy
        parents = []
        for shape in [(24,2,3,4), (24,3,2,5)]:
            fields = {name: np.full(shape, .1, dtype='<f8') for name in data.FIELDS}
            fields['base'][:,0,0,0] = 0; fields['reference'][:,0,0,0] = 0
            parents.append({'entry': {'dt_s': 1/60}, 'cell_size_m': .0625, 'arrays': fields})
        actual = model.PressureCorrectionNet(); expected = copy.deepcopy(actual)
        optimizer = torch.optim.Adam(actual.parameters(), lr=.001)
        expected_optimizer = torch.optim.Adam(expected.parameters(), lr=.001)
        shapes = []
        hook = actual.register_forward_pre_hook(lambda _network, args: shapes.append(tuple(args[0].shape[2:])))
        loss, gradient = train.one_update(actual, optimizer, parents, [(0,3),(1,5)], normalization())
        hook.remove()
        expected_optimizer.zero_grad(set_to_none=True)
        for parent, frame in [(0,3),(1,5)]:
            state = data.sample(parents[parent], frame)
            (model.pressure_loss(model.predict_pressure(expected, state, normalization()), state, normalization()).mean()/2).backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0, error_if_nonfinite=True)
        expected_optimizer.step()
        self.assertEqual(shapes, [(2,3,4),(3,2,5)])
        self.assertTrue(all(torch.equal(value, expected.state_dict()[name]) for name,value in actual.state_dict().items()))
        self.assertTrue(np.isfinite([loss,gradient]).all())

    def test_packet_rejects_unequal_spacing_before_numeric_load(self):
        from unittest.mock import patch
        config = policy.experiment(policy.decode(policy.read(policy.HERE/'configs/development.json')))
        parents = []
        for capture in config['captures']:
            for entry in capture['experiment']['trajectories']:
                counts = policy.dimensions(capture['experiment']['geometries_extents_m'][entry['geometry']], capture['cells_per_metre'])
                parents.append({'entry': entry, 'counts_xyz': counts, 'cell_size_m': 1/capture['cells_per_metre'], 'frames': []})
        parents[0]['cell_size_m'] *= 2
        raw = b'not a numeric packet'
        manifest = {'schema': 'rheon-variable-pressure-packet-v2', 'dataset_sha256': policy.sha(raw), 'experiment': config, 'parents': parents, 'arrays': {}}
        with patch.object(data, 'helpers', side_effect=AssertionError('Numeric decoding must not start')):
            with self.assertRaisesRegex(ValueError, 'physical_lineage'):
                data.load(raw, manifest)


if __name__ == '__main__':
    unittest.main()
