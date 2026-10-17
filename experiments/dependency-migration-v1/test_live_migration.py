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

    def test_private_ops_never_read_and_historical_output_root_refused(self):
        private = self.migration / 'ops/live-selection.private.json'
        write(private, {'secret': 'must not publish'})
        original = live.read_json
        def safe(path):
            self.assertNotIn('private', str(path))
            return original(path)
        with patch.object(live, 'read_json', side_effect=safe):
            result = live.Publisher(self.root, self.cohort, self.output).poll()
        self.assertEqual(result['migration_authority'], 'pending')
        with self.assertRaisesRegex(ValueError, 'dedicated'):
            live.Publisher(self.root, self.cohort, self.root / 'runs/live-project')

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
