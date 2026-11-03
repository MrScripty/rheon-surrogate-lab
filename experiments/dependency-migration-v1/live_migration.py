"""Read completed migration receipts and publish bounded external observations.

This module never imports a solver, model, catalog or execution owner.
"""
import argparse
from collections import Counter
import hashlib
import fcntl
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import signal
import threading
import time
import zipfile

import numpy as np

JSON_LIMIT = 2 * 1024 * 1024
NPZ_LIMIT = 4 * 1024 * 1024
EXPANDED_LIMIT = 8 * 1024 * 1024
CELL_LIMIT = 65536
SCENE_LIMIT = 128


def observer_type():
    path = Path(__file__).resolve().parents[1] / 'live-project-v1/observer.py'
    spec = importlib.util.spec_from_file_location('migration_observation_owner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Observer


def read_json(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(JSON_LIMIT + 1)
    if len(raw) > JSON_LIMIT:
        raise ValueError('Observation JSON exceeds 2 MiB')
    value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(value, dict):
        raise ValueError('Observation JSON must be an object')
    return value, hashlib.sha256(raw).hexdigest()


def contained(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Observation source escapes the selected root')
    return path


def scene_receipts(cohort):
    """Read only bounded worker/scene receipts; never descend into catalogs."""
    paths = sorted(itertools.islice(cohort.glob('worker-*/scene-*/receipt.json'), SCENE_LIMIT + 1))
    if len(paths) > SCENE_LIMIT:
        raise ValueError('Too many scene receipts for bounded observation')
    records, errors = [], 0
    for path in paths:
        try:
            value, digest = read_json(path)
            scene = value.get('metrics', {}).get('scene')
            if value.get('status') == 'failed' and scene is None:
                scene = int(path.parent.name.removeprefix('scene-'))
            if type(scene) is not int or not 0 <= scene < 74:
                raise ValueError('Unrecognized cohort scene receipt')
            records.append((path, value, digest, path.stat().st_mtime_ns, scene))
        except (OSError, ValueError, TypeError):
            errors += 1
    ids = [r[4] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate scene receipts in selected cohort')
    return records, errors


def status_at(path):
    if not path.is_file():
        return 'pending'
    try:
        value, _ = read_json(path)
        status = value.get('status', 'recorded')
        return status if status in {'passed', 'failed', 'completed', 'qualified', 'qualified_build', 'blocked'} else 'recorded'
    except (OSError, ValueError):
        return 'unreadable'


def status_fields(path, fields, schema=None):
    """Project named public scalars only, never journal arguments or results."""
    try:
        if path.is_symlink():
            raise ValueError('Observation receipt must not be a symlink')
        value, digest = read_json(path)
        if schema and value.get('schema', value.get('format')) != schema:
            raise ValueError('Unrecognized observation receipt')
        if not any(field in value for field in fields):
            raise ValueError('No observed status fields')
        result = {'observation': 'recorded', 'sha256': digest}
        for field in fields:
            token = value.get(field, 'not_recorded')
            if field in value and field in ('m3_service_switched', 'm3_complete', 'ordinary_admissions_open'):
                if type(token) is not bool:
                    raise ValueError('Invalid observation status boolean')
            elif not (
                    type(token) is str and 0 < len(token) <= 160
                    and all(c.isascii() and (c.isalnum() or c in '_-/.') for c in token)):
                raise ValueError('Invalid observation status scalar')
            result[field] = token
        return result
    except FileNotFoundError:
        return {'observation': 'absent'}
    except (OSError, ValueError):
        return {'observation': 'unreadable'}


M3_STEPS = ('m3-preflight-01', 'm3-preflight', 'm3-quiesce', 'm3-fresh-bookend',
            'm3-schema-upgrade', 'm3-domain-prepare', 'm3-domain-publish',
            'm3-domain-reconcile', 'm3-verify-current', 'm3-app-launch')
SMOKE_PHASES = ('open-project', 'create-model', 'publish-model', 'train-graph',
                'create-study', 'prepare-trial', 'approve-trial', 'open-admission',
                'admit-training', 'close-study', 'restore-graph', 'approve-restore',
                'admit-restore', 'issue-read-agent')


def status_text(value):
    if value['observation'] != 'recorded':
        return value['observation']
    return ', '.join(f'{key}={str(token).lower() if type(token) is bool else token}' for key, token in value.items()
                     if key not in ('observation', 'sha256'))


def status_summary(migration):
    paths = sorted(itertools.islice((migration / 'qualification').glob('torch-lifecycle-*/qualification.json'), 65))
    if len(paths) > 64:
        raise ValueError('Too many lifecycle receipts')
    latest = max(paths, key=lambda p: p.stat().st_mtime_ns, default=None)
    ops = migration / 'ops'
    steps = {name: status_fields(ops / (name + '.json'), ('status',),
                'rheon-fresh-stopped-original-bookend-v1' if name == 'm3-fresh-bookend'
                else 'rheon-m3-orchestrator-v1')
             for name in M3_STEPS}
    journals = {}
    for index, phase in enumerate(SMOKE_PHASES, 1):
        name = f'{index:02d}-{phase}'
        value = status_fields(migration / 'typed-pressure-live-01/mutations' / (name + '.json'), ('status',))
        if value['observation'] != 'absent':
            journals[name] = value
    return {'gpu_lifecycle': status_at(latest) if latest else 'pending',
            'catalog_rehearsal': status_fields(ops / 'rehearsal-status.json',
                ('current_stage', 'prepare04', 'publication', 'cold_read', 'completed_roundtrip', 'browser'),
                'rheon-migration-rehearsal-progress-v1'),
            'm3_steps': {name: value for name, value in steps.items() if value['observation'] != 'absent'},
            'm3_cutover': status_fields(ops / 'm3-cutover.json',
                ('m3_service_switched', 'm3_complete', 'ordinary_admissions_open'),
                'rheon-tuldok-migration-cutover-v1'),
            'm3_completion': status_fields(ops / 'm3-complete.json',
                ('status', 'm3_service_switched', 'm3_complete', 'ordinary_admissions_open'),
                'rheon-tuldok-migration-completion-v1'),
            'smoke_launcher': status_fields(ops / 'typed-pressure-smoke-launcher.json', ('status',),
                'rheon_typed_pressure_smoke_launcher_v1'),
            'smoke_journal': journals,
            'typed_pressure_live': status_fields(migration / 'typed-pressure-live-01/receipt.json',
                ('status', 'scientific_acceptance'), 'rheon_typed_pressure_qualification_v1'),
            'typed_pressure_failure': status_fields(migration / 'typed-pressure-live-01/failure.json', ('status',))}


def finite_positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid captured ' + name)
    return value


SCENE_KEYS = ('seed', 'center', 'acceleration', 'radius', 'swirl', 'fluctuation',
              'frequency', 'counts', 'cells_per_metre', 'steps', 'warmup_steps', 'sgs')


def current_forcing_metadata(scene, scene_path, scene_digest):
    """Project declared current Scene parameters; never evaluate its force kernel."""
    if not all(key in scene for key in SCENE_KEYS):
        return {}
    if (type(scene['seed']) is not int or not 0 <= scene['seed'] < 2**53
            or type(scene['sgs']) is not bool
            or type(scene['steps']) is not int or not 1 <= scene['steps'] <= 512
            or type(scene['warmup_steps']) is not int or not 0 <= scene['warmup_steps'] < scene['steps']
            or type(scene['cells_per_metre']) is not int or not 1 <= scene['cells_per_metre'] <= 1000):
        raise ValueError('Invalid current scene metadata')
    center = scene['center']
    if (not isinstance(center, list) or len(center) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in center)):
        raise ValueError('Invalid declared force center')
    for key, low, high in (('acceleration', 0, 4), ('radius', 0.1, 0.5),
                           ('swirl', -1, 1), ('fluctuation', 0, 0.8), ('frequency', 0, 4)):
        value = scene[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid declared force parameter')
    return {'forcing_kind': 'rheon-diagnostic-scene-v1',
            'forcing_parameters': {key: scene[key] for key in SCENE_KEYS},
            'forcing_scene_path': scene_path, 'forcing_scene_sha256': scene_digest,
            'forcing_units': {'center': 'normalized coordinates', 'radius': 'fraction of minimum domain extent',
                              'acceleration': 'm/s^2', 'frequency': 'Hz',
                              'swirl': 'dimensionless', 'fluctuation': 'dimensionless'},
            'forcing_scope': 'Nominal/base forcing parameters; actual force field not captured'}


def preview_from_receipt(root, receipt_path, receipt, receipt_digest):
    """Load one latest hash-bound actual frame; no procedural field synthesis."""
    if receipt.get('status') != 'passed':
        raise ValueError('Failed/incomplete case is not a qualified frame source')
    native, _ = read_json(receipt_path.parent / 'native/receipt.json')
    if native.get('native_process', {}).get('reaped') is not True or native.get('status') != 'completed':
        raise ValueError('Completed reaped native case required')
    scene_path = receipt_path.parent / 'native/scene.json'
    scene, scene_digest = read_json(scene_path)
    counts = scene['counts']
    if (not isinstance(counts, list) or len(counts) != 3
            or any(type(n) is not int or n <= 0 for n in counts)
            or math.prod(counts) > CELL_LIMIT):
        raise ValueError('Captured grid exceeds bounded preview')
    captures = receipt.get('captures', [])
    if not captures or len(captures) > 16:
        raise ValueError('Bounded sparse capture index required')
    capture = max(captures, key=lambda item: item['step'])
    step = capture['step']
    if type(step) is not int or not 0 < step <= scene['steps']:
        raise ValueError('Invalid captured generation')
    path = contained(root, capture['path'])
    if path.parent != receipt_path.parent.resolve():
        raise ValueError('Capture is not owned by this scene receipt')
    with path.open('rb') as stream:
        raw = stream.read(NPZ_LIMIT + 1)
    if len(raw) > NPZ_LIMIT or hashlib.sha256(raw).hexdigest() != capture['sha256']:
        raise ValueError('Sparse frame is oversized or hash changed')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) > 16 or len({m.filename for m in members}) != len(members):
            raise ValueError('Unsupported sparse frame archive')
        if sum(m.file_size for m in members) > EXPANDED_LIMIT:
            raise ValueError('Expanded sparse frame exceeds 8 MiB')
        if any(m.file_size > EXPANDED_LIMIT or '/' in m.filename for m in members):
            raise ValueError('Unsupported sparse frame member')
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported sparse array header')
                if (dtype.hasobject or dtype.kind != 'f' or dtype.itemsize not in (4, 8)
                        or fortran or math.prod(shape) > CELL_LIMIT
                        or stream.tell() + math.prod(shape) * dtype.itemsize != member.file_size):
                    raise ValueError('Sparse array shape/dtype is not bounded numeric storage')
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        pressure = arrays['prediction']
        expected_shape = (1, 1, counts[2], counts[1], counts[0])
        if pressure.shape != expected_shape or pressure.dtype != np.dtype('<f8') or not np.isfinite(pressure).all():
            raise ValueError('Actual pressure is not finite float64 x-fast ZYX')
        def scalar(key):
            array = arrays[key]
            if array.size != 1 or array.dtype.kind != 'f':
                raise ValueError('Invalid captured scalar')
            return finite_positive(array.reshape(-1)[0], key)
        h, dt, rho = scalar('cell_size_m'), scalar('dt'), scalar('rho')
        if not math.isclose(h, 1 / finite_positive(scene['cells_per_metre'], 'cells_per_metre'), rel_tol=1e-12):
            raise ValueError('Captured spacing disagrees with scene')
        preview = {'label': f"Current regression scene {receipt['metrics']['scene']}, step {step}",
                   'frame': step, 'time_s': step * dt, 'counts_xyz': counts,
                   'extents_m': [n * h for n in counts], 'cell_size_m': h, 'dt_s': dt,
                   'density_kg_m3': rho, 'pressure': pressure.reshape(-1).tolist(),
                   'pressure_label': 'Actual model full pressure (Pa); external observation; tracer not captured',
                   'source': {'path': str(path.relative_to(root)), 'sha256': capture['sha256'],
                              'receipt_path': str(receipt_path.relative_to(root)), 'receipt_sha256': receipt_digest,
                              'dataset': 'Already-consumed 74-scene regression cohort',
                              'scene': receipt['metrics']['scene'], 'seed': scene['seed'], 'field': 'prediction',
                              'authority': 'external_observation',
                              'qualification': 'Completed case preview; migration acceptance not asserted'}}
        preview['source'].update(current_forcing_metadata(
            scene, str(scene_path.relative_to(root)), scene_digest))
        if 'tracer' in arrays.files:
            tracer = arrays['tracer']
            if tracer.shape != expected_shape or tracer.dtype not in (np.dtype('<f4'), np.dtype('<f8')) or not np.isfinite(tracer).all():
                raise ValueError('Invalid captured tracer')
            preview['tracer'] = tracer.reshape(-1).tolist()
            preview['pressure_label'] = 'Actual model full pressure (Pa); external observation'
        if 'forcing_regime' in scene or 'amplitude_m_s2' in scene:
            regime = scene.get('forcing_regime')
            amplitude = scene.get('amplitude_m_s2')
            if (not isinstance(regime, str) or not 0 < len(regime) <= 160
                    or type(amplitude) not in (int, float) or not math.isfinite(amplitude) or amplitude < 0):
                raise ValueError('Incomplete declared forcing metadata')
            preview['forcing'] = {'regime': regime, 'amplitude_m_s2': amplitude}
        _, scene_after_digest = read_json(scene_path)
        if scene_after_digest != scene_digest:
            raise ValueError('Source scene changed during observation')
        if len(json.dumps(preview['source'], allow_nan=False).encode()) > 8192:
            raise ValueError('Scene provenance exceeds 8 KiB')
        if len(json.dumps(preview, allow_nan=False).encode()) > JSON_LIMIT - 4096:
            raise ValueError('Frame exceeds publication budget')
        return preview


class Publisher:
    def __init__(self, root, cohort, output):
        self.root = root.resolve()
        self.migration = self.root / 'runs/dependency-migration-v1'
        self.cohort = contained(self.root, cohort)
        if not self.cohort.is_relative_to(self.migration / 'numerical-qualification-v1'):
            raise ValueError('Cohort must be in numerical qualification')
        self.output = output.resolve()
        if self.output != self.migration / 'live-migration':
            raise ValueError('Only the dedicated migration observation root may be updated')
        self.observer = observer_type()(self.output)
        self.frame_key = None
        self.preview_error = None

    def poll(self):
        records, errors = scene_receipts(self.cohort)
        states = Counter(r[1].get('status', 'unknown') for r in records)
        passed = states['passed']
        latest = max((r for r in records if r[1].get('status') == 'passed'), key=lambda r: r[3], default=None)
        preview = None
        if latest:
            key = (str(latest[0]), latest[2])
            if key != self.frame_key:
                # Release the old in-memory frame before opening another capture.
                # Atomic frame.json remains available if this new read fails.
                self.observer.preview = None
                try:
                    preview = preview_from_receipt(self.root, *latest[:3])
                    self.preview_error = None
                    self.frame_key = key
                except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, EOFError) as error:
                    self.preview_error = type(error).__name__
                    # Never replace real fields with invented values on failure.
        statuses = status_summary(self.migration)
        # The current Project reader bounds activity messages to 1,000 characters.
        # Full per-receipt status remains in the returned projection and source receipts.
        steps = sum(value['observation'] == 'recorded' for value in statuses['m3_steps'].values())
        journals = sum(value['observation'] == 'recorded' for value in statuses['smoke_journal'].values())
        cutover = statuses['m3_cutover']
        completion = statuses['m3_completion']
        cutover_summary = status_text({'observation': cutover['observation'],
            'm3_complete': cutover.get('m3_complete', 'not_recorded')})
        completion_summary = status_text({'observation': completion['observation'],
            'status': completion.get('status', 'not_recorded'),
            'm3_complete': completion.get('m3_complete', 'not_recorded'),
            'ordinary_admissions_open': completion.get('ordinary_admissions_open', 'not_recorded')})
        machine = statuses['typed_pressure_live']
        machine_summary = ('status=' + machine['status'] if machine['observation'] == 'recorded'
                           else machine['observation'])
        message = (f'External observation: {passed}/74 current regression case receipts passed; '
                   f'{states["failed"]} failed, {errors} unreadable. GPU lifecycle: {statuses["gpu_lifecycle"]}; '
                   f'Current catalog rehearsal: {statuses["catalog_rehearsal"]["observation"]}. '
                   f'M3 receipts: {steps}/{len(M3_STEPS)} recorded; '
                   f'smoke journal: {journals}/{len(SMOKE_PHASES)} recorded. '
                   f'Cutover: {cutover_summary}. Final M3 completion: {completion_summary}. '
                   f'typed-pressure-live-01: {machine_summary}. '
                   'Full status details remain in the source receipts. '
                   'Migration acceptance is not asserted; this publisher creates no Tuldok Run.')
        if self.preview_error:
            message += ' New field preview unavailable; previous preview is retained.'
        if len(message) > 1000:
            raise ValueError('Observation summary exceeds current Project activity limit')
        self.observer.publish('Dependency migration observation', message, passed, 74, state='running', preview=preview)
        return {'passed_scene_receipts': passed, 'failed_scene_receipts': states['failed'],
                'unreadable_receipts': errors, 'preview_error': self.preview_error,
                'frame_source': self.observer.preview.get('source') if self.observer.preview else None, **statuses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, default=Path('runs/dependency-migration-v1/numerical-qualification-v1/new-cohort-01'))
    parser.add_argument('--live-root', type=Path, default=Path('runs/dependency-migration-v1/live-migration'))
    parser.add_argument('--interval', type=float, default=2.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1 or args.interval > 60:
        parser.error('--interval must be between 1 and 60 seconds')
    root = args.root.resolve()
    publisher = Publisher(root, args.cohort, contained(root, args.live_root))
    publisher.output.mkdir(parents=True, exist_ok=True)
    ownership = (publisher.output / '.publisher.lock').open('a')
    try:
        fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('A publisher already owns this live root')
    stop = threading.Event()
    def halt(_signal, _frame):
        stop.set()
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    while not stop.is_set():
        started = time.monotonic()
        try:
            result = publisher.poll()
        except (OSError, ValueError, KeyError, TypeError) as error:
            publisher.observer.publish('Observation unavailable',
                'External observation could not refresh receipts; no migration result is inferred (' + type(error).__name__ + ').',
                state='failed')
        else:
            if args.once:
                print(json.dumps({'authority': 'external_observation', **result,
                                  'poll_elapsed_s': time.monotonic() - started}), flush=True)
        if args.once:
            break
        stop.wait(args.interval)
    if not args.once:
        publisher.observer.publish('Observation stopped',
            'External observation publisher stopped; retained frame is a preview, not migration acceptance.', state='idle')


if __name__ == '__main__':
    main()
