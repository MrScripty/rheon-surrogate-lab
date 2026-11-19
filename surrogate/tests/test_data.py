import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from surrogate import data
from surrogate.fields import check_staggering, divergence
from surrogate.forcing import velocity_increment
from surrogate.tests.support import sealed_state

LEGACY = Path(__file__).resolve().parents[2] / 'experiments' / 'full-ai-next-priorities-v1'
SCENE = {'seed': 1550351134, 'counts': [6, 5, 4], 'cells_per_metre': 8, 'steps': 12, 'warmup_steps': 0,
         'center': [0.31, 0.53, 0.47], 'acceleration': 0.126, 'radius': 0.3, 'swirl': -0.32,
         'fluctuation': 0.115, 'frequency': 1.93, 'sgs': False}


def write_bank(root, dense_states=6, sparse_steps=(0, 1, 2, 5, 6)):
    """Two tiny recorded scenes on disk, one in each index layout; fields follow a known decay."""
    grid = (4, 5, 6)   # z, y, x for counts [6, 5, 4]
    start = sealed_state(1, grid, dtype=torch.float32).map(lambda tensor: tensor[0])

    def save(path, step):
        factor = 0.9 ** step
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, pressure=(start.pressure.double() * factor).numpy().reshape(-1),
                 vx=(start.u * factor).numpy().reshape(-1), vy=(start.v * factor).numpy().reshape(-1),
                 vz=(start.w * factor).numpy().reshape(-1), tracer=(start.tracer * factor).numpy().reshape(-1),
                 rhs=np.full(4 * 5 * 6, float(step)))

    records = []
    for name, steps in (('dense-scene', range(dense_states)), ('sparse-scene', sparse_steps)):
        directory = root / 'runs' / name
        for step in steps:
            save(directory / f'state-{step:03d}.npz', step)
        (directory / 'receipt.json').write_text(json.dumps({'scene': SCENE}))
        record = {'identifier': name, 'split': 'train' if name == 'dense-scene' else 'development',
                  'counts_xyz': SCENE['counts'], 'cells_per_metre': 8, 'receipt': f'runs/{name}/receipt.json'}
        if name == 'dense-scene':
            record['states'] = dense_states
        else:
            record.update(profile='continuous', reference_substeps=1,
                          states=[{'step': step, 'path': f'runs/{name}/state-{step:03d}.npz'} for step in steps])
        records.append(record)
    (root / 'bank.json').write_text(json.dumps({'trajectories': records}))
    return start


def write_definition(directory, steps=5):
    """A dataset definition `unit-v1` whose two scenes are the ones write_bank records."""
    keys = ('seed', 'counts', 'cells_per_metre', 'center', 'acceleration', 'radius', 'swirl', 'fluctuation', 'frequency')
    rows = [{'id': name, 'split': split, 'group': name, **{key: SCENE[key] for key in keys}}
            for name, split in (('dense-scene', 'train'), ('sparse-scene', 'development'))]
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / 'unit-v1.json').write_text(json.dumps(
        {'id': 'unit-v1', 'label': 'Unit dataset', 'description': 'Two synthetic scenes.',
         'scene_family': 'sinusoidal-forcing', 'solver': 'none', 'steps': steps, 'scenes': rows}))


class Banks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.start = write_bank(self.root)
        self.dense, self.sparse = data.load_bank('bank.json', self.root)

    def tearDown(self):
        self.directory.cleanup()

    def test_both_index_layouts_load_with_the_documented_staggering(self):
        self.assertEqual((self.dense.grid, self.dense.cells, self.dense.spacing), ((4, 5, 6), 120, 0.125))
        state = self.dense.state(0)
        check_staggering(state)
        self.assertEqual((state.u.dtype, state.pressure.dtype), (torch.float32, torch.float64))
        self.assertTrue(torch.equal(state.u, self.start.u))
        self.assertLess(divergence(state.u, state.v, state.w).abs().max().item(), 1e-5)
        self.assertEqual(self.sparse.attributes, {'profile': 'continuous', 'reference_substeps': 1})
        self.assertTrue(torch.allclose(self.sparse.state(5).tracer, self.start.tracer * 0.9 ** 5))

    def test_starts_need_every_following_state(self):
        self.assertEqual(self.dense.starts(2), [0, 1, 2, 3])
        self.assertEqual(self.sparse.starts(2), [0])       # 0,1,2 are saved; 5,6 are only two
        self.assertEqual(self.sparse.starts(1), [0, 1, 5])
        self.assertEqual(len(data.windows([self.dense, self.sparse], 2)), 5)

    def test_batch_stacks_windows_and_matches_forcing_and_constants(self):
        start, forcing, future, scene = data.batch([(self.dense, 0), (self.dense, 2)], 3)
        self.assertEqual(tuple(start.u.shape), (2, 4, 5, 7))
        self.assertEqual(tuple(future.tracer.shape), (2, 3, 4, 5, 6))
        self.assertEqual(tuple(forcing[2].shape), (2, 3, 5, 5, 6))
        self.assertTrue(torch.allclose(future.u[1, 0], self.start.u * 0.9 ** 3))
        self.assertTrue(torch.equal(forcing[0][1, 0], velocity_increment(SCENE, 2)[0].float()))
        self.assertEqual(scene.spacing.tolist(), [0.125, 0.125])
        self.assertAlmostEqual(scene.timestep[0].item(), 1 / 60)
        self.assertEqual(start.pressure.dtype, torch.float32)

    def test_the_same_trajectory_in_two_banks_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'more than once'):
            data.load_banks(['bank.json', 'bank.json'], self.root)

    def test_right_hand_sides_come_from_the_state_files(self):
        self.assertEqual(self.dense.right_hand_sides(1, 4)[:, 0, 0, 0].tolist(), [1.0, 2.0, 3.0, 4.0])


class Forcing(unittest.TestCase):
    def test_no_forcing_through_walls_and_shapes_follow_the_grid(self):
        fu, fv, fw = velocity_increment(SCENE, 3)
        self.assertEqual((tuple(fu.shape), tuple(fv.shape), tuple(fw.shape)), ((4, 5, 7), (4, 6, 6), (5, 5, 6)))
        for wall in (fu[..., 0], fu[..., -1], fv[:, 0], fv[:, -1], fw[0], fw[-1]):
            self.assertEqual(wall.abs().max().item(), 0.0)
        self.assertGreater(fu.abs().max().item(), 1e-5)

    def test_scheduled_scene_follows_its_schedule_and_sums_substeps(self):
        frames = [{'center': [0.5, 0.5, 0.5], 'pulse': 0.0 if step == 1 else 1.0} for step in range(12)]
        scheduled = {**SCENE, 'forcing_schedule': frames, 'fluctuation': 0.0}
        self.assertEqual(velocity_increment(scheduled, 1)[0].abs().max().item(), 0.0)   # pulse off
        whole = velocity_increment(scheduled, 2)
        # With no time-dependent term, eight substeps of an eighth each add up to the same increment.
        split = velocity_increment({**scheduled, 'reference_substeps': 8, 'forcing_control_substeps': 1}, 2)
        self.assertTrue(torch.allclose(whole[0], split[0], rtol=1e-12, atol=0))

    def test_step_outside_the_recording_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            velocity_increment(SCENE, 12)

    @unittest.skipUnless((LEGACY / 'temporal_features.py').is_file(), 'the earlier implementation is not present')
    def test_matches_the_earlier_implementation_for_sinusoidal_scenes(self):
        sys.path.insert(0, str(LEGACY))
        try:
            from temporal_features import force_features
        finally:
            sys.path.remove(str(LEGACY))
        nx, ny, nz = SCENE['counts']
        for step in (0, 7, 11):
            earlier = force_features(SCENE, step, 1, 1.0)[0]
            for axis, shape in enumerate(((nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx))):
                expected = earlier[axis][:shape[0], :shape[1], :shape[2]]
                actual = velocity_increment(SCENE, step)[axis].float()
                self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-10))


if __name__ == '__main__':
    unittest.main()
