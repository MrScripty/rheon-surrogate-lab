import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import live_migration as live


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class LiveMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.migration = self.root / 'runs/dependency-migration-v1'
        self.cohort = self.migration / 'numerical-qualification-v1/new-cohort-01'
        self.case = self.cohort / 'worker-a/scene-00'
        self.case.mkdir(parents=True)
        self.frame = self.case / 'state-0016.npz'
        self.pressure = np.arange(24, dtype='<f8').reshape(1, 1, 2, 3, 4)
        np.savez(self.frame, prediction=self.pressure, base=self.pressure,
                 native_input_residual=self.pressure, cell_size_m=np.array([0.125]),
                 dt=np.array([1/60]), rho=np.array([1000.0]))
        self.receipt = {'status': 'passed', 'metrics': {'scene': 0}, 'captures': [
            {'step': 16, 'path': str(self.frame), 'sha256': hashlib.sha256(self.frame.read_bytes()).hexdigest()}]}
        write(self.case / 'receipt.json', self.receipt)
        write(self.case / 'native/scene.json', {'counts': [4, 3, 2], 'cells_per_metre': 8,
              'steps': 96, 'seed': 123, 'acceleration': 0.2})
        write(self.case / 'native/receipt.json', {'status': 'completed', 'native_process': {'reaped': True}})
        self.output = self.migration / 'live-migration'

    def preview(self):
        value, digest = live.read_json(self.case / 'receipt.json')
        return live.preview_from_receipt(self.root, self.case / 'receipt.json', value, digest)

    def test_pressure_only_preview_exact_geometry_clock_provenance_no_invented_tracer_or_force(self):
        preview = self.preview()
        self.assertEqual(preview['pressure'], self.pressure.reshape(-1).tolist())
        self.assertEqual(preview['counts_xyz'], [4, 3, 2])
        self.assertEqual(preview['extents_m'], [0.5, 0.375, 0.25])
        self.assertEqual(preview['time_s'], 16/60)
        self.assertEqual(preview['source']['scene'], 0)
        self.assertEqual(preview['source']['sha256'], self.receipt['captures'][0]['sha256'])
        self.assertEqual(preview['source']['authority'], 'external_observation')
        self.assertNotIn('tracer', preview)
        self.assertNotIn('forcing', preview)

    def test_current_scene_nominal_forcing_is_exact_and_source_is_unchanged(self):
        # Exact completed current Q7 scene-44 parameters; numerical arrays are
        # small contract fixtures, not a new simulation or accepted result.
        scene = {'seed': 2026101902, 'counts': [21, 25, 18], 'cells_per_metre': 8,
                 'steps': 128, 'warmup_steps': 15, 'center': [0.528774, 0.671546, 0.393733],
                 'acceleration': 0.265913, 'radius': 0.308746, 'swirl': -0.783611,
                 'fluctuation': 0.309747, 'frequency': 1.582985, 'sgs': False}
        path = self.case / 'native/scene.json'
        write(path, scene)
        raw = path.read_bytes()
        pressure = np.arange(21 * 25 * 18, dtype='<f8').reshape(1, 1, 18, 25, 21)
        np.savez(self.frame, prediction=pressure, cell_size_m=np.array([0.125]),
                 dt=np.array([1/60]), rho=np.array([1000.]))
        self.receipt['captures'][0]['sha256'] = hashlib.sha256(self.frame.read_bytes()).hexdigest()
        write(self.case / 'receipt.json', self.receipt)
        source = self.preview()['source']
        self.assertEqual(source['forcing_kind'], 'rheon-diagnostic-scene-v1')
        self.assertEqual(source['forcing_parameters'], scene)
        self.assertEqual(source['forcing_scene_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(source['forcing_units']['acceleration'], 'm/s^2')
        self.assertEqual(source['forcing_units']['frequency'], 'Hz')
        self.assertEqual(source['forcing_units']['center'], 'normalized coordinates')
        self.assertEqual(source['forcing_units']['radius'], 'fraction of minimum domain extent')
        self.assertIn('actual force field not captured', source['forcing_scope'])
        self.assertNotIn('forcing', self.preview())
        self.assertEqual(path.read_bytes(), raw)
        scene['center'] = [2., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'center'):
            live.current_forcing_metadata(scene, str(path), source['forcing_scene_sha256'])

    def test_scene_changed_during_projection_cannot_publish_metadata(self):
        original = live.read_json
        reads = 0
        scene_path = self.case / 'native/scene.json'
        before = scene_path.read_bytes()
        def changed(path):
            nonlocal reads
            value, digest = original(path)
            if path == scene_path:
                reads += 1
                if reads == 2:
                    return value, '0' * 64
            return value, digest
        with patch.object(live, 'read_json', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'Source scene changed'):
                self.preview()
        self.assertEqual(scene_path.read_bytes(), before)

    def test_changed_bytes_failed_case_unreaped_case_and_path_escape_refused(self):
        original = self.frame.read_bytes()
        self.frame.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            self.preview()
        self.frame.write_bytes(original)
        self.receipt['status'] = 'failed'
        write(self.case / 'receipt.json', self.receipt)
        with self.assertRaisesRegex(ValueError, 'Failed/incomplete'):
            self.preview()
        self.receipt['status'] = 'passed'
        self.receipt['captures'][0]['path'] = str(self.root.parent / 'escape.npz')
        write(self.case / 'receipt.json', self.receipt)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.preview()
        self.receipt['captures'][0]['path'] = str(self.frame)
        write(self.case / 'receipt.json', self.receipt)
        write(self.case / 'native/receipt.json', {'status': 'completed', 'native_process': {'reaped': False}})
        with self.assertRaisesRegex(ValueError, 'reaped'):
            self.preview()

    def test_header_allocation_bomb_refused_before_numpy_load(self):
        member = io.BytesIO()
        np.lib.format.write_array_header_1_0(member, {'descr': '<f8', 'fortran_order': False, 'shape': (10**12,)})
        with zipfile.ZipFile(self.frame, 'w') as archive:
            archive.writestr('prediction.npy', member.getvalue())
        self.receipt['captures'][0]['sha256'] = hashlib.sha256(self.frame.read_bytes()).hexdigest()
        write(self.case / 'receipt.json', self.receipt)
        with patch.object(live.np, 'load', side_effect=AssertionError('must validate headers first')):
            with self.assertRaisesRegex(ValueError, 'shape/dtype'):
                self.preview()

    def test_only_latest_frame_loaded_and_unchanged_frame_reused(self):
        publisher = live.Publisher(self.root, self.cohort, self.output)
        with patch.object(live, 'preview_from_receipt', wraps=live.preview_from_receipt) as reader:
            publisher.poll()
            publisher.poll()
            self.assertEqual(reader.call_count, 1)
        activity = json.loads((self.output / 'activity.json').read_text())
        frame = json.loads((self.output / 'frame.json').read_text())
        self.assertEqual(frame['sequence'], activity['sequence'])
        self.assertEqual(activity['progress'], {'completed': 1, 'total': 74})
        self.assertIn('creates no Tuldok Run', activity['message'])
        self.assertNotIn('tracer', frame)

    def test_new_invalid_preview_retains_prior_actual_frame_with_visible_failure(self):
        publisher = live.Publisher(self.root, self.cohort, self.output)
        publisher.poll()
        first = json.loads((self.output / 'frame.json').read_text())
        self.receipt['captures'][0]['sha256'] = '0' * 64
        write(self.case / 'receipt.json', self.receipt)
        publisher.poll()
        frame = json.loads((self.output / 'frame.json').read_text())
        activity = json.loads((self.output / 'activity.json').read_text())
        self.assertEqual(frame['source'], first['source'])
        self.assertEqual(frame['pressure'], first['pressure'])
        self.assertIn('New field preview unavailable', activity['message'])

    def test_failed_receipt_before_metric_publication_remains_visible(self):
        write(self.cohort / 'worker-b/scene-01/receipt.json', {'status': 'failed', 'error': 'actual negative fixture'})
        result = live.Publisher(self.root, self.cohort, self.output).poll()
        self.assertEqual(result['failed_scene_receipts'], 1)
        self.assertEqual(result['passed_scene_receipts'], 1)
        self.assertEqual(result['unreadable_receipts'], 0)
        message = json.loads((self.output / 'activity.json').read_text())['message']
        self.assertIn('1 failed', message)

    def test_process_has_single_owner_and_sigterm_publishes_valid_idle_state(self):
        command = [sys.executable, '-B', str(Path(live.__file__).resolve()), '--root', str(self.root), '--interval', '1']
        environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        child = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=environment)
        try:
            deadline = time.monotonic() + 5
            while not (self.output / 'activity.json').exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIsNone(child.poll())
            second = subprocess.run(command + ['--once'], capture_output=True, env=environment, timeout=5)
            self.assertEqual(second.returncode, 2)
            self.assertIn(b'already owns', second.stderr)
            child.terminate()
            _, error = child.communicate(timeout=5)
            self.assertEqual(child.returncode, 0, error)
            activity = json.loads((self.output / 'activity.json').read_text())
            self.assertEqual(activity['state'], 'idle')
            self.assertIn('not migration acceptance', activity['message'])
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)

    def publisher_with_full_stdout(self, once=False):
        import resource
        def limits():
            resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        log = self.root / 'full-publisher.log'
        log.write_bytes(b'x' * (2 * 1024 * 1024))
        command = [sys.executable, '-B', str(Path(live.__file__).resolve()),
                   '--root', str(self.root), '--interval', '1']
        if once:
            command.append('--once')
        environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        with log.open('ab') as output:
            child = subprocess.Popen(command, stdout=output, stderr=output,
                                     env=environment, preexec_fn=limits)
        def reap():
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
        self.addCleanup(reap)
        return child, log

    def test_full_stdout_watch_keeps_actual_progress_and_refreshes_changed_receipts(self):
        child, log = self.publisher_with_full_stdout()
        activity_path = self.output / 'activity.json'
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if activity_path.exists() and json.loads(activity_path.read_text())['sequence'] >= 3:
                break
            self.assertIsNone(child.poll())
            time.sleep(0.02)
        time.sleep(0.1)
        activity = json.loads(activity_path.read_text())
        self.assertEqual(activity['state'], 'running')
        self.assertEqual(activity['progress'], {'completed': 1, 'total': 74})
        before = json.loads((self.output / 'frame.json').read_text())
        write(self.migration / 'ops/m3-complete.json', {
            'schema': 'rheon-tuldok-migration-completion-v1', 'status': 'complete_fixture',
            'm3_service_switched': True, 'm3_complete': True, 'ordinary_admissions_open': False})
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            activity = json.loads(activity_path.read_text())
            if 'Final M3 completion: status=complete_fixture' in activity['message']:
                break
            self.assertIsNone(child.poll())
            time.sleep(0.02)
        self.assertIn('Final M3 completion: status=complete_fixture', activity['message'])
        self.assertEqual(activity['progress'], {'completed': 1, 'total': 74})
        after = json.loads((self.output / 'frame.json').read_text())
        self.assertEqual(after['pressure'], before['pressure'])
        self.assertEqual(after['source'], before['source'])
        self.assertEqual(log.stat().st_size, 2 * 1024 * 1024)
        self.assertIsNone(child.poll())

    def test_full_stdout_once_fails_transport_without_relabelling_valid_observation(self):
        child, log = self.publisher_with_full_stdout(once=True)
        self.assertNotEqual(child.wait(timeout=5), 0)
        activity = json.loads((self.output / 'activity.json').read_text())
        self.assertEqual(activity['state'], 'running')
        self.assertEqual(activity['progress'], {'completed': 1, 'total': 74})
        self.assertNotIn('Observation unavailable', activity['phase'])
        self.assertEqual(log.stat().st_size, 2 * 1024 * 1024)

    @unittest.skipUnless(os.environ.get('RHEON_ACTIVITY_CONSUMER_ROOT'),
                         'requires the selected deployed activity decoder and public status root')
    def test_all_current_migration_statuses_pass_actual_deployed_activity_decoder(self):
        consumer = Path(os.environ['RHEON_ACTIVITY_CONSUMER_ROOT']).resolve()
        actual = live.status_summary(Path(os.environ['RHEON_ACTIVITY_STATUS_ROOT']))
        self.assertGreaterEqual(len(actual['m3_steps']), 9)
        self.assertEqual(len(actual['smoke_journal']), 14)
        self.assertEqual(actual['typed_pressure_live']['status'], 'machine_qualified_browser_pending')
        # Only scene zero supplies the small numerical preview. The other
        # bounded receipt fixtures exercise the full declared74-case count.
        for scene in range(1, 74):
            path = self.cohort / f'worker-b/scene-{scene:02}/receipt.json'
            write(path, {'status': 'passed', 'metrics': {'scene': scene}})
            os.utime(path, ns=(0, 0))
        with patch.object(live, 'status_summary', return_value=actual):
            result = live.Publisher(self.root, self.cohort, self.output).poll()
        self.assertEqual(result['passed_scene_receipts'], 74)
        publication = self.output / 'activity.json'
        activity = json.loads(publication.read_text())
        self.assertLessEqual(len(activity['message']), 1000)
        self.assertIn('74/74', activity['message'])
        self.assertIn('0 failed, 0 unreadable', activity['message'])
        self.assertIn('machine_qualified_browser_pending', activity['message'])
        code = ('import sys,json;sys.path.insert(0,sys.argv[1]);'
                'import project_activity;'
                'value=project_activity.activity(json.load(open(sys.argv[2])));'
                'print(json.dumps({k:value[k] for k in ("available","authority","progress")}))')
        decoded = subprocess.run([sys.executable, '-B', '-c', code, str(consumer), str(publication)],
                                 capture_output=True, text=True, timeout=5)
        self.assertEqual(decoded.returncode, 0, decoded.stderr)
        projection = json.loads(decoded.stdout)
        self.assertTrue(projection['available'])
        self.assertEqual(projection['authority'], 'external_observation')
        self.assertEqual(projection['progress'], {'completed': 74, 'total': 74})
        print(json.dumps({'activity_consumer': str(consumer / 'project_activity.py'),
                          'consumer_sha256': hashlib.sha256((consumer / 'project_activity.py').read_bytes()).hexdigest(),
                          'message_characters': len(activity['message']),
                          'actual_m3_receipts': len(actual['m3_steps']),
                          'actual_smoke_journals': len(actual['smoke_journal']),
                          'available': projection['available']}))

    def test_private_ops_never_read_and_historical_output_root_refused(self):
        private = self.migration / 'ops/live-selection.private.json'
        write(private, {'secret': 'must not publish'})
        original = live.read_json
        def safe(path):
            self.assertNotIn('private', str(path))
            return original(path)
        with patch.object(live, 'read_json', side_effect=safe):
            result = live.Publisher(self.root, self.cohort, self.output).poll()
        self.assertEqual(result['m3_cutover'], {'observation': 'absent'})
        with self.assertRaisesRegex(ValueError, 'dedicated'):
            live.Publisher(self.root, self.cohort, self.root / 'runs/live-project')

    def test_old_schema_pass_cannot_stand_in_for_absent_or_current_catalog_progress(self):
        ops = self.migration / 'ops'
        write(ops / 'schema-rehearsal-receipt.json', {'status': 'passed'})
        self.assertEqual(live.status_summary(self.migration)['catalog_rehearsal'], {'observation': 'absent'})
        write(ops / 'rehearsal-status.json', {'schema': 'rheon-migration-rehearsal-progress-v1',
              'schema_only_restore02': 'passed', 'prepare03': 'passed_superseded',
              'current_stage': 'runs/dependency-migration-v1/domain-stage-04',
              'prepare04': 'running_read_only_historical_row_evidence',
              'publication': 'corrected04_pending_independent_review', 'cold_read': 'pending',
              'completed_roundtrip': 'pending', 'browser': 'pending'})
        result = live.Publisher(self.root, self.cohort, self.output).poll()
        self.assertEqual(result['catalog_rehearsal']['prepare04'], 'running_read_only_historical_row_evidence')
        message = json.loads((self.output / 'activity.json').read_text())['message']
        self.assertEqual(result['catalog_rehearsal']['publication'], 'corrected04_pending_independent_review')
        self.assertEqual(result['catalog_rehearsal']['cold_read'], 'pending')
        self.assertIn('Current catalog rehearsal: recorded', message)
        self.assertNotIn('schema rehearsal: passed', message)
        self.assertNotIn('passed_superseded', message)

    def test_current_m3_and_smoke_journals_are_projected_without_sensitive_payloads(self):
        ops = self.migration / 'ops'
        smoke = self.migration / 'typed-pressure-live-01'
        write(ops / 'm3-app-launch.json', {'schema': 'rheon-m3-orchestrator-v1',
              'status': 'current_service_launched_smoke_pending', 'command': ['must-not-publish']})
        write(ops / 'm3-cutover.json', {'schema': 'rheon-tuldok-migration-cutover-v1',
              'm3_service_switched': True, 'm3_complete': False, 'ordinary_admissions_open': False})
        write(ops / 'typed-pressure-smoke-launcher.json', {'format': 'rheon_typed_pressure_smoke_launcher_v1',
              'status': 'qualified_client_dispatch', 'commands': ['must-not-publish']})
        write(smoke / 'mutations/09-admit-training.json', {'phase': 'admit-training',
              'status': 'acknowledged', 'arguments': {'token': 'must-not-publish'},
              'result': {'token': 'must-not-publish'}})
        result = live.status_summary(self.migration)
        self.assertEqual(result['smoke_journal']['09-admit-training']['status'], 'acknowledged')
        self.assertEqual(result['typed_pressure_live'], {'observation': 'absent'})
        self.assertEqual(result['m3_completion'], {'observation': 'absent'})
        write(smoke / 'receipt.json', {'format': 'rheon_typed_pressure_qualification_v1',
              'status': 'machine_qualified_browser_pending', 'scientific_acceptance': 'not_asserted'})
        result = live.Publisher(self.root, self.cohort, self.output).poll()
        self.assertEqual(result['typed_pressure_live']['status'], 'machine_qualified_browser_pending')
        self.assertFalse(result['m3_cutover']['m3_complete'])
        message = json.loads((self.output / 'activity.json').read_text())['message']
        self.assertIn('machine_qualified_browser_pending', message)
        self.assertEqual(result['smoke_journal']['09-admit-training']['status'], 'acknowledged')
        self.assertIn('smoke journal: 1/14 recorded', message)
        self.assertIn('m3_complete=false', message)
        self.assertIn('Migration acceptance is not asserted', message)
        self.assertNotIn('must-not-publish', json.dumps(result) + message)

    def test_final_completion_and_bookend_reconcile_keep_their_own_receipt_schemas(self):
        ops = self.migration / 'ops'
        cutover = {'schema': 'rheon-tuldok-migration-cutover-v1',
                   'm3_service_switched': True, 'm3_complete': False, 'ordinary_admissions_open': False}
        write(ops / 'm3-cutover.json', cutover)
        write(ops / 'm3-fresh-bookend.json', {'schema': 'rheon-fresh-stopped-original-bookend-v1',
              'status': 'exact_backup_revalidated'})
        write(ops / 'm3-domain-reconcile.json', {'schema': 'rheon-m3-orchestrator-v1',
              'status': 'verified-completed'})
        completion = {'schema': 'rheon-tuldok-migration-completion-v1', 'status': 'completed',
                      'm3_service_switched': True, 'm3_complete': True, 'ordinary_admissions_open': True}
        write(ops / 'm3-complete.json', completion)
        result = live.Publisher(self.root, self.cohort, self.output).poll()
        self.assertEqual(result['m3_steps']['m3-fresh-bookend']['status'], 'exact_backup_revalidated')
        self.assertEqual(result['m3_steps']['m3-domain-reconcile']['status'], 'verified-completed')
        self.assertFalse(result['m3_cutover']['m3_complete'])
        self.assertTrue(result['m3_completion']['m3_complete'])
        self.assertTrue(result['m3_completion']['ordinary_admissions_open'])
        self.assertEqual(json.loads((ops / 'm3-cutover.json').read_text()), cutover)
        message = json.loads((self.output / 'activity.json').read_text())['message']
        self.assertIn('Final M3 completion: status=completed', message)
        self.assertIn('m3_complete=false', message)
        self.assertIn('m3_complete=true', message)
        write(ops / 'm3-complete.json', {'schema': 'rheon-tuldok-migration-cutover-v1', 'status': 'completed'})
        self.assertEqual(live.status_summary(self.migration)['m3_completion'], {'observation': 'unreadable'})
        write(ops / 'm3-complete.json', {'schema': 'rheon-tuldok-migration-completion-v1', 'status': 'completed'})
        self.assertEqual(live.status_summary(self.migration)['m3_completion']['m3_complete'], 'not_recorded')

    def test_malformed_missing_and_changed_status_do_not_reuse_an_old_pass(self):
        path = self.migration / 'ops/rehearsal-status.json'
        write(path, {'schema': 'rheon-migration-rehearsal-progress-v1', 'publication': 'passed'})
        publisher = live.Publisher(self.root, self.cohort, self.output)
        publisher.poll()
        before = json.loads((self.output / 'frame.json').read_text())
        with patch.object(live, 'preview_from_receipt', side_effect=AssertionError('unchanged capture must be reused')):
            for raw in ('{', '{"schema":"wrong","publication":"passed"}',
                        '{"schema":"rheon-migration-rehearsal-progress-v1","publication":true}',
                        '{"schema":"rheon-migration-rehearsal-progress-v1","publication":"' + 'x'*161 + '"}'):
                with self.subTest(raw=raw):
                    path.write_text(raw)
                    self.assertEqual(publisher.poll()['catalog_rehearsal'], {'observation': 'unreadable'})
                    message = json.loads((self.output / 'activity.json').read_text())['message']
                    self.assertIn('Current catalog rehearsal: unreadable', message)
                    self.assertNotIn('publication=passed', message)
            path.unlink()
            self.assertEqual(publisher.poll()['catalog_rehearsal'], {'observation': 'absent'})
        after = json.loads((self.output / 'frame.json').read_text())
        self.assertEqual(after['source'], before['source'])
        self.assertEqual(after['pressure'], before['pressure'])

    def test_status_reads_have_exact_paths_and_existing_json_bound(self):
        smoke = self.migration / 'typed-pressure-live-01'
        write(smoke / 'mutations/99-unknown.json', {'status': 'must-not-read'})
        write(smoke / 'private/read-agent.json', {'token': 'must-not-read'})
        original = live.read_json
        paths = []
        def selected(path):
            paths.append(path)
            self.assertNotIn('private', str(path))
            self.assertNotIn('99-unknown', str(path))
            return original(path)
        with patch.object(live, 'read_json', side_effect=selected):
            summary = live.status_summary(self.migration)
        self.assertFalse(summary['smoke_journal'])
        self.assertEqual(len(paths), 30)
        path = self.migration / 'ops/rehearsal-status.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b' ' * (live.JSON_LIMIT + 1))
        self.assertEqual(live.status_summary(self.migration)['catalog_rehearsal'], {'observation': 'unreadable'})
        path.unlink()
        path.symlink_to(smoke / 'private/read-agent.json')
        self.assertEqual(live.status_summary(self.migration)['catalog_rehearsal'], {'observation': 'unreadable'})

    def test_nonfinite_pressure_geometry_mismatch_and_duplicate_receipts_refused(self):
        changed = self.pressure.copy()
        changed.flat[0] = np.nan
        np.savez(self.frame, prediction=changed, cell_size_m=np.array([0.125]),
                 dt=np.array([1/60]), rho=np.array([1000.]))
        self.receipt['captures'][0]['sha256'] = hashlib.sha256(self.frame.read_bytes()).hexdigest()
        write(self.case / 'receipt.json', self.receipt)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.preview()
        write(self.cohort / 'worker-b/scene-00/receipt.json', self.receipt)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            live.scene_receipts(self.cohort)


if __name__ == '__main__':
    unittest.main()
