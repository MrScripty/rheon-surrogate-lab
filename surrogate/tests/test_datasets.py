import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from surrogate import data, datasets
from surrogate.datasets import __main__ as cli
from surrogate.datasets import generate
from surrogate.solver import PROBES, REPOSITORY, pressure_flux, spectral_correction
from surrogate.tests.support import sealed_state

# sha256 of every diversity-v1 scene built from its definition (canonical JSON, definition order).
# The scenes were checked equal to the 224 the lab's reference data was simulated from; if this
# value changes, the definition no longer describes that data.
DIVERSITY_SCENES_SHA256 = 'ada00398842f1e84112a6cdef9db2eddc878dee909a19a43de080e8622ac5caf'


class FakeProbe:
    """Stands in for the solver: every step scales the state by 0.9."""

    def __init__(self, scene, binary=None, density=1000.0):
        nx, ny, nz = scene['counts']
        start = sealed_state(1, (nz, ny, nx), dtype=torch.float32).map(lambda tensor: tensor[0])
        self.current = replace(start, pressure=start.pressure.double())   # the solver's own precisions
        self.step = 0
        self.fail_at = scene.get('fail_at')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def state(self):
        return self.step, self.current

    def advance_reference(self):
        self.step += 1
        self.current = self.current.map(lambda tensor: tensor * 0.9)
        labels = {name: self.current.pressure.double() for name in ('rhs', 'native_input_residual', 'pressure_base')}
        return {'advanced': self.step != self.fail_at, 'step': self.step, 'quality': 'rejected'}, labels


class Definitions(unittest.TestCase):
    def test_every_definition_loads_and_keeps_groups_in_one_split(self):
        self.assertEqual(datasets.available(), ['diversity-v1', 'sinusoidal-v1'])
        counts = {name: len(datasets.load(name)['scenes']) for name in datasets.available()}
        self.assertEqual(counts, {'diversity-v1': 224, 'sinusoidal-v1': 112})

    def test_diversity_scenes_are_exactly_the_ones_the_reference_data_was_simulated_from(self):
        definition = datasets.load('diversity-v1')
        digest = hashlib.sha256()
        for row in definition['scenes']:
            digest.update(json.dumps(datasets.scene(definition, row), sort_keys=True, separators=(',', ':')).encode())
        self.assertEqual(digest.hexdigest(), DIVERSITY_SCENES_SHA256)

    def test_scene_rows_become_solver_scenes(self):
        diversity = datasets.load('diversity-v1')
        built = datasets.scene(diversity, diversity['scenes'][0])
        self.assertEqual((built['counts'], built['steps'], len(built['forcing_schedule'])), ([8, 8, 8], 240, 240))
        self.assertNotIn('reference_substeps', built)
        substepped = next(row for row in diversity['scenes'] if row.get('solver_substeps'))
        self.assertEqual(datasets.scene(diversity, substepped)['reference_substeps'], 8)
        sinusoidal = datasets.load('sinusoidal-v1')
        built = datasets.scene(sinusoidal, sinusoidal['scenes'][0])
        self.assertEqual((built['steps'], built['sgs'], 'forcing_schedule' in built), (128, False, False))

    def test_a_group_in_two_splits_is_refused(self):
        broken = json.loads((datasets.DEFINITIONS / 'sinusoidal-v1.json').read_text())
        broken['scenes'][0]['split'] = 'development'
        broken['scenes'][0]['group'] = broken['scenes'][1]['group']
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(datasets, 'DEFINITIONS', Path(directory)):
            (Path(directory) / 'broken.json').write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, 'more than one split'):
                datasets.load('broken')


class SpectralSolve(unittest.TestCase):
    def test_solves_the_sealed_box_pressure_equation(self):
        generator = torch.Generator().manual_seed(2)
        pressure = torch.randn(5, 6, 7, generator=generator, dtype=torch.float64)
        spacing, density = 0.125, 1000.0
        right_hand_side = pressure_flux(pressure, spacing, density)
        solved = spectral_correction(right_hand_side, spacing, density)
        self.assertEqual(solved[0, 0, 0].item(), 0.0)
        self.assertLess((solved - (pressure - pressure[0, 0, 0])).abs().max().item(), 1e-9)

    def test_refuses_a_right_hand_side_that_does_not_sum_to_zero(self):
        with self.assertRaisesRegex(ValueError, 'no solution'):
            spectral_correction(torch.ones(4, 4, 4, dtype=torch.float64), 0.1, 1000.0)


@mock.patch.object(generate, 'probe_for', lambda scene: None)
@mock.patch.object(generate, 'RheonProbe', FakeProbe)
class Generation(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.messages = []

    def run_generate(self, **options):
        return generate.generate('sinusoidal-v1', self.root, report=self.messages.append, **options)

    def test_writes_dense_states_that_the_loader_reads(self):
        scenes = ['temporal-g2-cpm8-r0', 'temporal-g2-cpm8-r1']
        bank = self.run_generate(scenes=scenes)
        first, second = data.load_bank(bank)
        self.assertEqual((first.id, first.split, first.grid, len(first.paths)), (scenes[0], 'train', (8, 6, 15), 129))
        self.assertEqual(first.starts(128), [0])
        self.assertTrue(torch.allclose(first.state(10).u, first.state(0).u * 0.9 ** 10, rtol=1e-5))
        self.assertEqual(first.state(3).pressure.dtype, torch.float64)
        self.assertEqual(tuple(first.right_hand_sides(0, 2).shape), (2, 8, 6, 15))
        self.assertEqual(first.forcing(0, 1)[0].shape, (1, 8, 6, 16))
        receipt = json.loads((bank.parent / scenes[0] / 'receipt.json').read_text())
        self.assertEqual((receipt['dataset'], receipt['solver'], receipt['states']),
                         ('sinusoidal-v1', 'rheon-reference-v1', 129))

    def test_a_second_run_only_does_what_is_missing(self):
        self.run_generate(scenes=['temporal-g2-cpm8-r0'])
        self.messages.clear()
        bank = self.run_generate(scenes=['temporal-g2-cpm8-r0', 'temporal-g2-cpm8-r1'])
        self.assertEqual(len(self.messages), 1)
        self.assertIn('temporal-g2-cpm8-r1', self.messages[0])
        self.assertEqual(len(data.load_bank(bank)), 2)

    def test_a_rejected_step_leaves_no_scene_behind(self):
        original = datasets.scene
        with mock.patch.object(generate, 'scene', lambda definition, row: {**original(definition, row), 'fail_at': 5}):
            with self.assertRaisesRegex(RuntimeError, 'rejected step 5'):
                self.run_generate(scenes=['temporal-g2-cpm8-r0'])
        self.assertFalse((self.root / 'sinusoidal-v1' / 'temporal-g2-cpm8-r0').exists())
        self.assertFalse((self.root / 'sinusoidal-v1' / 'bank.json').exists())

    def test_reserved_scenes_need_an_explicit_decision(self):
        with self.assertRaisesRegex(ValueError, 'final checks only'):
            self.run_generate(splits=['reserved'])
        with self.assertRaisesRegex(ValueError, 'Not in this dataset'):
            self.run_generate(scenes=['no-such-scene'])

    def test_plan_counts_what_is_left_and_what_is_not_wired(self):
        diversity = datasets.load('diversity-v1')
        summary = generate.plan(diversity, generate.select(diversity, ['development']), self.root)['development']
        self.assertEqual((summary['scenes'], summary['done']), (24, 0))
        self.assertEqual(summary['scenes'] - summary['not_wired'], summary['states'] // 241)
        self.assertEqual(summary['estimated_bytes'], summary['cell_states'] * generate.BYTES_PER_CELL_STATE)
        self.run_generate(scenes=['temporal-g2-cpm8-r0'])
        sinusoidal = datasets.load('sinusoidal-v1')
        self.assertEqual(generate.plan(sinusoidal, generate.select(sinusoidal, ['train']), self.root)['train']['done'], 1)

    def test_command_line_lists_and_plans(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cli.main(['list'])
            cli.main(['plan', 'diversity-v1', '--split', 'reserved', '--root', str(self.root)])
        text = output.getvalue()
        self.assertIn('148 train, 24 development, 52 reserved', text)
        self.assertIn('reserved       52 scenes: 0 on disk', text)


@unittest.skipUnless((REPOSITORY / PROBES['scheduled']).is_file(), 'the Rheon probe is not built here')
class RealSolver(unittest.TestCase):
    def test_generates_a_small_scene_with_the_real_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = generate.generate('diversity-v1', directory, scenes=['diversity-train-g000-cpm8'],
                                     report=lambda message: None)
            (trajectory,) = data.load_bank(bank)
            self.assertEqual((trajectory.grid, len(trajectory.paths)), ((8, 8, 8), 241))
            state = trajectory.state(240)
            self.assertGreater(state.u.abs().max().item(), 0.0)
            stored = REPOSITORY / 'runs/full-ai-data-diversity-v1/train-capture-01/diversity-train-g000-cpm8/state-240.npz'
            if stored.is_file():   # the lab's earlier recording of the same scene
                import numpy as np
                with np.load(stored) as archive:
                    self.assertTrue(torch.equal(state.u, torch.from_numpy(archive['vx'].reshape(8, 8, 9))))


if __name__ == '__main__':
    unittest.main()
