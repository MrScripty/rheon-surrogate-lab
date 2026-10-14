"""Protocol/publication fixtures only; these tests never invoke the native solver."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

import evaluate


def quality():
    return {'qualified': True, 'failures': [], 'residual_l2': 0., 'residual_max': 0.,
            'rhs_l2': 1., 'relative_residual': 0., 'predicted_divergence_max_s_inv': 0.,
            'actual_divergence_max_s_inv': 0., 'actual_divergence_rms_s_inv': 0.,
            'net_flux_m3_s': 0., 'wall_normal_flux_exact_zero': True, 'courant': 0., 'gauge_cell': 0.}


def attempt(iterations=5, error=None):
    return {'qualified': error is None, 'solver_error': error, 'iterations': iterations,
            'solver_s': .01, 'verification_s': .001, 'quality': quality(), 'iteration_limit': 3000}


def result():
    policy = {'qualified': True, 'outcome': 'direct_acceptance', 'proposal_rejection': None,
              'warm': None, 'reset': None, 'any_pcg': False, 'reset_fallback': False, 'elapsed_s': .002}
    return {'schema': 'rheon-global-pressure-teacher-diagnostic-v1', 'state_advanced': False,
            'reference': quality(), 'raw_learned': quality(), 'raw_temporal': quality(), 'raw_zero': quality(),
            'learned_jacobi': dict(policy), 'temporal_jacobi': dict(policy),
            'temporal_full_budget': attempt(4), 'zero_jacobi': attempt(6), 'native_elapsed_s': .03}


class AdmissionFixture:
    def __init__(self):
        self.verified = False
        self.calls = 0
        self.identity = {'fixture': True}

    def admit(self, paths, deadline):
        self.calls += 1
        population = [{'case_id': f'fixture-{i}', 'trajectory_id': f'fixture-parent-{i//2}',
                       'frame': 0 if i % 2 == 0 else 23, 'split': 'train' if i < 24 else 'validation'}
                      for i in range(32)]
        return {'population': population, 'requests': [{'fixture': i} for i in range(32)],
                'inference': [.01]*32, 'identity': self.identity, 'proof': {'fixture': True}}

    def validate_result(self, request, value, case):
        if value.get('state_advanced') is not False:
            raise ValueError('fixture_state_advanced')

    def verify_unchanged(self, admitted, deadline):
        self.verified = True


class ProcessFixture:
    def __init__(self, failed_index=None):
        self.calls = []
        self.failed_index = failed_index

    def native_call(self, binary, mode, request, deadline, timeout):
        self.calls.append((mode, request))
        if mode == '--identity':
            return {'fixture': True}, {'wall_s': .001, 'drained': True}
        value = result()
        if request['fixture'] == self.failed_index:
            value['learned_jacobi'].update(qualified=False, outcome='terminal_failure',
                reset=attempt(3000, 'pressure iteration limit 3000, full residual maximum 1.0'),
                any_pcg=True, reset_fallback=True)
        return value, {'wall_s': .04, 'drained': True}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        evaluate.STARTED = time.monotonic()
        self.output = Path('/tmp') / ('gn-native-test-' + uuid.uuid4().hex)
        self.paths = {name: Path('/tmp') / ('fixture-' + name) for name in (
            'training_artifacts', 'restored_artifacts', 'capture', 'packet', 'binary')}
        self.freeze = {'input_paths': {k: str(v) for k, v in self.paths.items()}, 'source_sha256': {}}

    def tearDown(self):
        if self.output.exists():
            shutil.rmtree(self.output)

    def test_verify_never_invokes_native_or_creates_model_outputs(self):
        admission = AdmissionFixture()
        process = ProcessFixture()
        summary = evaluate.execute('verify', self.paths, self.output, self.freeze,
                                   admission_api=admission, process_api=process)
        self.assertEqual(process.calls, [])
        self.assertEqual(summary['native_calls'], 0)
        self.assertEqual(list(p.name for p in self.output.iterdir()), ['input-proof.json'])
        self.assertGreaterEqual(summary['workflow_elapsed_s_through_publication'], 0.)

    def test_exact32_order_identity_brackets_and_durable_report(self):
        admission = AdmissionFixture()
        process = ProcessFixture()
        summary = evaluate.execute('evaluate', self.paths, self.output, self.freeze,
                                   admission_api=admission, process_api=process)
        self.assertEqual(len(process.calls), 34)
        self.assertEqual(process.calls[0][0], '--identity')
        self.assertEqual(process.calls[-1][0], '--identity')
        self.assertEqual([r['fixture'] for m, r in process.calls if m == '--evaluate'], list(range(32)))
        self.assertTrue(admission.verified)
        self.assertEqual(summary['cases'], 32)
        report = json.loads((self.output/'evaluation.json').read_text())
        self.assertEqual(report['summaries']['all']['methods']['temporal_full_budget']['completed_iterations_known']['sum'], 128)
        self.assertEqual(report['summaries']['split']['validation']['cases'], 8)
        self.assertEqual(report['summaries']['frame']['23']['cases'], 16)
        self.assertFalse(report['state_advanced'])
        self.assertEqual(len((self.output/'cases.jsonl').read_text().splitlines()), 32)
        self.assertGreaterEqual(summary['workflow_elapsed_s_through_publication'], report['workflow_elapsed_s_before_final_publication'])
        self.assertIn('standalone candidate enclosing wall/speedup', report['unavailable'])

    def test_terminal_failure_journaled_no_retry_or_later_cases(self):
        process = ProcessFixture(failed_index=4)
        with self.assertRaisesRegex(ValueError, 'terminal_numerical_failure:fixture-4'):
            evaluate.execute('evaluate', self.paths, self.output, self.freeze,
                             admission_api=AdmissionFixture(), process_api=process)
        self.assertEqual([r['fixture'] for m, r in process.calls if m == '--evaluate'], list(range(5)))
        self.assertEqual(len((self.output/'cases.jsonl').read_text().splitlines()), 5)
        self.assertFalse((self.output/'evaluation.json').exists())
        failure = json.loads((self.output/'failure.json').read_text())
        self.assertEqual(failure['observed_responses'], 5)
        self.assertFalse(failure['automatic_retry'])

    def test_frozen_locator_rejected_before_admission(self):
        admission = AdmissionFixture()
        self.freeze['input_paths']['capture'] = '/tmp/other'
        with self.assertRaisesRegex(ValueError, 'frozen_input_locator'):
            evaluate.execute('verify', self.paths, self.output, self.freeze,
                             admission_api=admission, process_api=ProcessFixture())
        self.assertEqual(admission.calls, 0)
        self.assertFalse(self.output.exists())

    def test_outputs_refuse_existing_path_symlink_and_overwrite(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh_direct_tmp_output_required'):
            evaluate.Outputs(self.output, time.monotonic()+5)
        self.output.rmdir()
        self.output.symlink_to('/tmp')
        with self.assertRaisesRegex(ValueError, 'fresh_direct_tmp_output_required'):
            evaluate.Outputs(self.output, time.monotonic()+5)
        self.output.unlink()
        out = evaluate.Outputs(self.output, time.monotonic()+5)
        try:
            out.write('input-proof.json', {'fixture': True})
            with self.assertRaisesRegex(ValueError, 'output_already_published'):
                out.write('input-proof.json', {'fixture': False})
            with self.assertRaisesRegex(ValueError, 'output_file_limit:population.json'):
                out.write('population.json', {'too_large': 'x' * (64*1024)})
        finally:
            out.close()
        self.assertEqual(json.loads((self.output/'input-proof.json').read_text()), {'fixture': True})

    def test_breakdown_index_not_labeled_completed_and_wasted_warm_charged(self):
        record = {'split': 'validation', 'frame': 23, 'trajectory_id': 'fixture',
                  'recorded_model_inference_s': .02, 'process': {'wall_s': .05}, **result()}
        record['learned_jacobi'].update(outcome='temporal_reset_fallback', warm=attempt(32,
            'pressure iteration limit 32, full residual maximum 1'), reset=attempt(90),
            any_pcg=True, reset_fallback=True)
        summary = evaluate.aggregate([record])['methods']['learned_jacobi']
        self.assertEqual(summary['completed_iterations_known']['sum'], 122)
        self.assertEqual(summary['wasted_warm_completed_iterations_known']['sum'], 32)
        self.assertIsNone(evaluate.completed_iterations(attempt(4, 'pressure arithmetic broke down at iteration 4')))
        self.assertEqual(evaluate.completed_iterations(attempt(7, 'pressure solve cancelled after 7 iterations')), 7)
        self.assertEqual(summary['resets'], 1)

    def test_post_fsync_publication_failure_revokes_complete_authority(self):
        original = evaluate.Outputs.write
        def failed_write(owner, name, value):
            original(owner, name, value)
            if name == 'evaluation.json':
                raise ValueError('injected_post_fsync_deadline')
        with patch.object(evaluate.Outputs, 'write', failed_write):
            with self.assertRaisesRegex(ValueError, 'injected_post_fsync_deadline'):
                evaluate.execute('evaluate', self.paths, self.output, self.freeze,
                                 admission_api=AdmissionFixture(), process_api=ProcessFixture())
        self.assertFalse((self.output/'evaluation.json').exists())
        self.assertTrue((self.output/'evaluation.incomplete.json').is_file())
        failure = json.loads((self.output/'failure.json').read_text())
        self.assertEqual(failure['reason'], 'injected_post_fsync_deadline')

    def test_existing_output_rejected_before_reading_admission(self):
        self.output.mkdir()
        admission = AdmissionFixture()
        with self.assertRaisesRegex(ValueError, 'fresh_direct_tmp_output_required'):
            evaluate.execute('verify', self.paths, self.output, self.freeze,
                             admission_api=admission, process_api=ProcessFixture())
        self.assertEqual(admission.calls, 0)

    def test_bootstrap_oversized_regular_file_rejected_before_read(self):
        self.output.write_bytes(b'x' * 2048)
        before = evaluate.SOURCE_BYTES_READ
        with self.assertRaisesRegex(ValueError, 'source_file_bound'):
            evaluate.bounded_source_bytes(self.output, 1024)
        self.assertEqual(evaluate.SOURCE_BYTES_READ, before)
        self.output.unlink()

    def test_watchdog_restored_and_expired_work_not_dispatched(self):
        evaluate.STARTED = time.monotonic() - 600
        process = ProcessFixture()
        with self.assertRaisesRegex(ValueError, 'workflow_deadline'):
            evaluate.execute('evaluate', self.paths, self.output, self.freeze,
                             admission_api=AdmissionFixture(), process_api=process)
        self.assertEqual(process.calls, [])
        self.assertEqual(evaluate.signal.getitimer(evaluate.signal.ITIMER_REAL), (0., 0.))


if __name__ == '__main__':
    unittest.main()
