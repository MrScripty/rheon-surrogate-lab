import json
import tempfile
import unittest
from pathlib import Path

import torch

from surrogate import data
from surrogate.bench import metrics, packet, suites
from surrogate.bench.runner import (TIMING_SAMPLES, TIMING_WARMUPS, Case, Subject, Suite, Unsupported, run,
                                    run_case)
from surrogate.bench.subjects import LearnedStepper, Persistence
from surrogate.checkpoint import save_stepper
from surrogate.models import DivergenceFreeStepper, StepperConfig
from surrogate.tests.support import forcing, scene, sealed_state
from surrogate.tests.test_data import write_bank

GRID = (4, 5, 6)
DECAY = 0.9


def decaying_case(intervals=5, identity='decay.start-0'):
    """A reference whose every field shrinks by DECAY each interval, so errors are known exactly."""
    start = sealed_state(1, GRID, dtype=torch.float32)
    factors = DECAY ** torch.arange(intervals + 1, dtype=torch.float32)
    reference = start.map(lambda tensor: tensor[0] * factors.reshape(-1, 1, 1, 1))
    return Case(id=identity, reference=reference,
                forcing=tuple(f[0] for f in forcing(1, intervals, GRID, dtype=torch.float32)),
                scene=scene(1, dtype=torch.float32), attributes={'scene': 'decay', 'grid_x': GRID[2]})


class Replay(Subject):
    """Returns the reference itself, `outputs_per_pass` states at a time."""
    label = 'Reference replay'

    def __init__(self, case, outputs_per_pass=1, poison_at=None):
        self.id = f'replay-k{outputs_per_pass}' + ('' if poison_at is None else '-poisoned')
        self.case, self.outputs_per_pass, self.poison_at = case, outputs_per_pass, poison_at
        self.begun = self.ended = 0

    def begin(self, case):
        self.begun += 1

    def end(self):
        self.ended += 1

    def advance(self, state, forcing, scene, step):
        outputs = []
        for index in range(step + 1, step + 1 + self.outputs_per_pass):
            output = self.case.reference.map(lambda tensor: tensor[index:index + 1].clone())
            if index == self.poison_at:
                output.u[0, 0, 0, 0] = float('nan')
            outputs.append(output)
        return outputs


class Broken(Subject):
    id, label = 'broken', 'Always raises'

    def advance(self, state, forcing, scene, step):
        raise RuntimeError('no kernel\nfor this grid')


class Picky(Subject):
    id, label = 'picky', 'Runs nothing'

    def begin(self, case):
        raise Unsupported('this grid has no kernel')


class SelfTimed(Persistence):
    id, label = 'self-timed', 'Reports its own time'

    def own_seconds(self):
        return 0.002


class RunCase(unittest.TestCase):
    def test_persistence_error_matches_the_known_decay(self):
        result = run_case(Persistence(), decaying_case(), velocity_error_limit=0.25)
        values = result['values']
        self.assertIsNone(result['failure'])
        final = (1 - DECAY ** 5) / DECAY ** 5
        for name in ('velocity', 'pressure', 'tracer'):
            self.assertAlmostEqual(values[f'{name}_error_final'], final, places=5)
            self.assertAlmostEqual(values[f'{name}_error_worst'], final, places=5)
        self.assertAlmostEqual(values['energy_ratio_final'], DECAY ** -10, places=4)
        # errors are 0.111, 0.235, 0.372, ...: two outputs stay within a limit of 0.25
        self.assertEqual(values['valid_outputs'], 2)
        self.assertLess(values['divergence_worst'], 1e-3)
        self.assertGreater(values['time_per_output_ms'], 0.0)
        self.assertAlmostEqual(values['simulated_seconds_per_second'] * values['time_per_output_ms'],
                               1000.0 / 60.0, places=3)

    def test_exact_subject_scores_zero_and_multi_output_passes_score_every_output(self):
        case = decaying_case(intervals=5)
        values = run_case(Replay(case, outputs_per_pass=2), case, 0.25)['values']
        self.assertEqual(values['velocity_error_worst'], 0.0)
        self.assertEqual(values['valid_outputs'], 4)   # two passes of two; the fifth interval is unused

    def test_non_finite_output_stops_the_rollout_and_keeps_what_was_measured(self):
        case = decaying_case()
        result = run_case(Replay(case, poison_at=3), case, 0.25)
        self.assertEqual(result['failure'], 'A field was not finite at output 3.')
        self.assertEqual(result['values']['valid_outputs'], 2)
        self.assertEqual(result['values']['velocity_error_worst'], 0.0)

    def test_exception_becomes_a_failure_with_no_values(self):
        result = run_case(Broken(), decaying_case(), 0.25)
        self.assertIn('RuntimeError: no kernel', result['failure'])
        self.assertIsNone(result['values']['velocity_error_worst'])
        self.assertIsNone(result['values']['time_per_output_ms'])
        self.assertEqual(result['values']['valid_outputs'], 0)

    def test_case_shorter_than_one_pass_does_not_apply(self):
        case = decaying_case(intervals=2)
        result = run_case(Replay(case, outputs_per_pass=3), case, 0.25)
        self.assertEqual(result, {'values': dict.fromkeys(metrics.IDS), 'failure': None})

    def test_unsupported_case_does_not_apply_and_is_not_a_failure(self):
        result = run_case(Picky(), decaying_case(), 0.25)
        self.assertEqual(result, {'values': dict.fromkeys(metrics.IDS), 'failure': None})

    def test_each_case_is_begun_and_ended_once(self):
        case = decaying_case()
        for subject in (Replay(case), Replay(case, poison_at=2)):
            run_case(subject, case, 0.25)
            self.assertEqual((subject.begun, subject.ended), (1, 1))

    def test_timing_repeats_the_first_pass_and_prefers_the_subjects_own_clock(self):
        calls = []

        class Counting(Persistence):
            def advance(self, state, forcing, scene, step):
                calls.append(step)
                return [state]

        case = Case(**{**decaying_case(intervals=2).__dict__, 'start': 40})
        run_case(Counting(), case, 0.25)
        self.assertEqual(calls, [40] * (TIMING_WARMUPS + TIMING_SAMPLES) + [40, 41])
        values = run_case(SelfTimed(), case, 0.25)['values']
        self.assertEqual(values['time_per_output_ms'], 2.0)

    def test_equation_residual_uses_the_recorded_right_hand_side(self):
        case = decaying_case(intervals=2)
        spacing, density = case.scene.spacing.item(), case.scene.density.item()
        exact = torch.stack([metrics.pressure_operator(p.double(), spacing, density) for p in case.reference.pressure])
        with_rhs = Case(**{**case.__dict__, 'right_hand_sides': exact})
        self.assertIsNone(run_case(Replay(case), case, 0.25)['values']['equation_error_worst'])
        self.assertLess(run_case(Replay(case), with_rhs, 0.25)['values']['equation_error_worst'], 1e-12)
        # Persistence keeps the start pressure, which is 1/DECAY times the first output's.
        self.assertAlmostEqual(run_case(Persistence(), with_rhs, 0.25)['values']['equation_error_worst'],
                               1 / DECAY ** 2 - 1, places=5)

    def test_error_against_a_fluid_at_rest_is_zero_or_undefined_never_infinite(self):
        still = sealed_state(1, GRID, dtype=torch.float32).map(torch.zeros_like)
        moving = sealed_state(1, GRID, dtype=torch.float32)
        compare = lambda predicted: metrics.compare(predicted.map(lambda t: t[0]), still.map(lambda t: t[0]), 0.1, 1000.0)
        self.assertEqual(compare(still)['velocity_error'], 0.0)
        self.assertIsNone(compare(moving)['velocity_error'])
        self.assertIsNone(compare(moving)['energy_ratio'])
        summary = metrics.summarise([compare(moving), compare(still)], 0.25)
        self.assertEqual((summary['velocity_error_worst'], summary['valid_outputs']), (0.0, 2))


class SuiteCases(unittest.TestCase):
    def test_a_case_carries_what_solvers_and_the_equation_metric_need(self):
        with tempfile.TemporaryDirectory() as directory:
            write_bank(Path(directory))
            dense, _ = data.load_bank('bank.json', directory)
            case = suites.case(dense, 1, 3)
        self.assertEqual((case.id, case.start, case.intervals), ('dense-scene.start-1', 1, 3))
        self.assertEqual(case.parameters['seed'], dense.scene['seed'])
        self.assertEqual(case.right_hand_sides[:, 0, 0, 0].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual((case.attributes['grid_x'], case.attributes['cells_per_metre']), (6, 8))
        self.assertEqual(tuple(case.reference.u.shape), (4, 4, 5, 7))


class RunSuite(unittest.TestCase):
    def suite(self):
        cases = [decaying_case(identity='decay.start-0'), decaying_case(intervals=3, identity='decay.start-8')]
        return Suite(id='unit', label='Unit suite', population='training', description='Synthetic decay.',
                     velocity_error_limit=0.25, cases=cases), cases

    def test_result_document_is_complete_and_becomes_a_packet(self):
        suite, cases = self.suite()
        document = run([Persistence(), Broken(), Replay(cases[0], outputs_per_pass=4)], suite)
        self.assertEqual(len(document['results']), 6)
        self.assertRegex(document['git_commit'], r'^[0-9a-f]{40}$')
        self.assertEqual(document['subjects'][0], {'id': 'persistence', 'label': 'No change (persistence)',
                                                   'outputs_per_pass': 1, 'kind': 'baseline'})
        self.assertEqual(json.loads(json.dumps(document)), document)   # plain JSON types only
        (raw,) = packet.build(document)
        states = [o['state'] for o in json.loads(raw)['observations']]
        count = len(metrics.IDS)
        # Broken fails both cases except its valid_outputs count. The K=4 replay does not fit the short
        # case, and the three rollouts that did run have no recorded right-hand side for the equation metric.
        self.assertEqual(states.count('failed'), 2 * (count - 1))
        self.assertEqual(states.count('unsupported'), count + 3)

    def test_checkpoint_subject_runs_a_saved_model(self):
        suite, _ = self.suite()
        model = DivergenceFreeStepper(StepperConfig(outputs_per_pass=2, widths=(8, 12), blocks_per_level=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_stepper(path, model, name='unit-model', label='Unit model', updates=0, git_commit='0' * 40)
            subject = LearnedStepper(path, 'cpu')
            document = run([subject], suite)
        self.assertEqual(subject.outputs_per_pass, 2)
        self.assertEqual(document['subjects'][0]['kind'], 'learned')
        # An untrained stepper returns its input, so it scores exactly like persistence.
        reference = run_case(Persistence(), suite.cases[0], 0.25)['values']
        measured = document['results'][0]['values']
        self.assertAlmostEqual(measured['velocity_error_final'], (1 - DECAY ** 4) / DECAY ** 4, places=5)
        self.assertLess(measured['velocity_error_final'], reference['velocity_error_final'])


if __name__ == '__main__':
    unittest.main()
