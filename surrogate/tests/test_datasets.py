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
from surrogate.bench import suites
from surrogate.datasets import __main__ as cli
from surrogate.datasets import generate, link
from surrogate.solver import PROBE, REPOSITORY, pressure_flux, spectral_correction
from surrogate.tests.support import sealed_state
from surrogate.tests.test_data import SCENE, write_bank, write_definition

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
        self.assertEqual((receipt['dataset'], receipt['origin'], receipt['states']), ('sinusoidal-v1', 'generated', 129))

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
        self.assertEqual(data.load_bank(self.root / 'sinusoidal-v1' / 'bank.json'), [])

    def test_reserved_scenes_need_an_explicit_decision(self):
        with self.assertRaisesRegex(ValueError, 'final checks only'):
            self.run_generate(splits=['reserved'])
        with self.assertRaisesRegex(ValueError, 'Not in this dataset'):
            self.run_generate(scenes=['no-such-scene'])

    def test_plan_counts_what_is_left_to_run(self):
        diversity = datasets.load('diversity-v1')
        summary = generate.plan(diversity, generate.select(diversity, ['development']), self.root)['development']
        self.assertEqual((summary['scenes'], summary['complete'], summary['states']), (24, 0, 24 * 241))
        self.assertEqual(summary['estimated_bytes'], summary['cell_states'] * generate.BYTES_PER_CELL_STATE)
        self.run_generate(scenes=['temporal-g2-cpm8-r0'])
        sinusoidal = datasets.load('sinusoidal-v1')
        after = generate.plan(sinusoidal, generate.select(sinusoidal, ['train']), self.root)['train']
        self.assertEqual((after['complete'], after['partial']), (1, 0))

    def test_command_line_lists_and_plans(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cli.main(['list'])
            cli.main(['plan', 'diversity-v1', '--split', 'reserved', '--root', str(self.root)])
        text = output.getvalue()
        self.assertIn('148 train, 24 development, 52 reserved', text)
        self.assertIn('reserved       52 scenes: 0 complete on disk, 52 to run', text)

    def plan_output(self, *arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(['plan', *arguments, '--root', str(self.root)]), 0)
        return output.getvalue()

    def test_command_line_plans_as_json(self):
        self.run_generate(scenes=['temporal-g2-cpm8-r0'])
        for selection, splits, chosen in (
                (['sinusoidal-v1', '--split', 'train', '--split', 'development'], ['train', 'development'], None),
                (['diversity-v1', '--split', 'reserved'], ['reserved'], None),
                (['sinusoidal-v1', '--scene', 'temporal-g2-cpm8-r0'], None, ['temporal-g2-cpm8-r0'])):
            with self.subTest(selection=selection):
                printed = self.plan_output(*selection, '--json')
                self.assertEqual(printed.count('\n'), 1)
                figures = json.loads(printed)
                self.assertEqual(set(figures), {'disk_bytes', 'items', 'complete', 'to_run', 'partial',
                                                'states', 'cell_states'})
                self.assertTrue(all(type(value) is int for value in figures.values()))
                self.assertNotIn('wall_seconds', figures)
                text = self.plan_output(*selection).splitlines()
                lines = text[1:-1]
                counts = [int(line.split()[1]) for line in lines]
                complete = [int(line.split('scenes: ')[1].split()[0]) for line in lines]
                states = [int(line.split('-> ')[1].split()[0]) for line in lines]
                gigabytes = [float(line.split('about ')[1].split()[0]) for line in lines]
                self.assertEqual(figures['items'], sum(counts))
                self.assertEqual(figures['complete'], sum(complete))
                self.assertEqual(figures['to_run'], sum(counts) - sum(complete))
                self.assertEqual(figures['states'], sum(states))
                self.assertAlmostEqual(figures['disk_bytes'] / 1e9, sum(gigabytes), delta=0.05 * len(lines))
                self.assertEqual(f"{figures['disk_bytes'] / 1e9:.1f}", text[-1].split('about ')[1].split()[0])
                definition = datasets.load(selection[0])
                summary = generate.plan(definition, generate.select(definition, splits, chosen), self.root)
                self.assertEqual(figures['disk_bytes'], sum(entry['estimated_bytes'] for entry in summary.values()))
        figures = json.loads(self.plan_output('sinusoidal-v1', '--scene', 'temporal-g2-cpm8-r0', '--json'))
        self.assertEqual((figures['items'], figures['complete'], figures['disk_bytes']), (1, 1, 0))


class Linking(unittest.TestCase):
    """Recordings that already exist are used in place, and are never simulated again."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.old = Path(self.directory.name) / 'lab'          # stands in for the repository with its runs/
        self.root = Path(self.directory.name) / 'data'
        write_bank(self.old)                                   # dense-scene: steps 0..5; sparse-scene: 0,1,2,5,6
        write_definition(Path(self.directory.name) / 'definitions', steps=5)
        patcher = mock.patch.object(datasets, 'DEFINITIONS', Path(self.directory.name) / 'definitions')
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.directory.cleanup)
        self.messages = []

    def link(self):
        return link.link('unit-v1', ['bank.json'], self.root, repository=self.old, report=self.messages.append)

    def test_links_complete_and_partial_scenes_without_copying(self):
        dense, sparse = data.load_bank(self.link())
        self.assertEqual((dense.id, sorted(dense.paths), dense.starts(5)), ('dense-scene', [0, 1, 2, 3, 4, 5], [0]))
        self.assertEqual((sorted(sparse.paths), sparse.starts(1)), ([0, 1, 2, 5], [0, 1]))
        self.assertTrue(dense.paths[3].is_symlink())
        self.assertTrue(torch.equal(dense.state(3).u, data.load_bank('bank.json', self.old)[0].state(3).u))
        self.assertIn('linked 2 scenes, 0 already complete; 1 of 2 scenes are complete on disk, 1 have some steps',
                      self.messages[0])
        definition = datasets.load('unit-v1')
        summary = generate.plan(definition, definition['scenes'], self.root)
        self.assertEqual((summary['train']['complete'], summary['development']['partial']), (1, 1))

    def test_a_recording_of_a_different_scene_is_refused(self):
        receipt = self.old / 'runs/dense-scene/receipt.json'
        receipt.write_text(json.dumps({'scene': {**SCENE, 'swirl': 0.5}}))
        with self.assertRaisesRegex(ValueError, 'different scene'):
            self.link()

    @mock.patch.object(generate, 'RheonProbe', FakeProbe)
    def test_generation_skips_linked_complete_scenes_and_completes_partial_ones(self):
        self.link()
        generated = []
        bank = generate.generate('unit-v1', self.root, splits=['train', 'development'], report=generated.append)
        self.assertEqual(len(generated), 1)
        self.assertIn('sparse-scene', generated[0])
        dense, sparse = data.load_bank(bank)
        self.assertTrue(dense.paths[0].is_symlink())           # still the original recording
        self.assertEqual((sorted(sparse.paths), sparse.paths[0].is_symlink()), ([0, 1, 2, 3, 4, 5], False))
        self.link()
        self.assertIn('linked 0 scenes, 2 already complete', self.messages[-1])

    def test_a_suite_needs_every_scene_of_its_split_and_says_how_to_get_them(self):
        unit = {'label': 'Unit', 'population': 'consumed', 'description': 'x', 'dataset': 'unit-v1',
                'split': 'development', 'starts': [0], 'intervals': 2, 'velocity_error_limit': 0.1}
        with mock.patch.dict(suites.SPECIFICATIONS, {'unit': unit}):
            with self.assertRaisesRegex(FileNotFoundError, 'python -m surrogate.datasets'):
                suites.build('unit', self.root)
            self.link()
            (case,) = list(suites.build('unit', self.root).cases)
            self.assertEqual((case.id, case.intervals), ('sparse-scene.start-0', 2))


@unittest.skipUnless((REPOSITORY / PROBE).is_file(), 'the Rheon reference probe is not built here')
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
