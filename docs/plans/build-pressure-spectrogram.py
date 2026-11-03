"""Replay retained checkpoints on recorded teacher states; export temporal error PSD.

Run with ../../.venv/bin/python build-pressure-spectrogram.py, then
python3 refresh-cohort-progress.py. This performs CPU inference only, with no
training, simulator execution, or changes to the scientific acceptance records.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent.parent
VERSION = 1  # Increment when the numerical replay or spectrum method changes.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temporal_psd(values, dt, length, hop=1):
    """One-sided, periodic-Hann PSD, averaged over cells, no zero padding.

    Remove each cell's temporal window mean. Values have already had their
    spatial mean removed independently per frame to eliminate pressure gauge.
    """
    import numpy as np
    values = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(length) / length)
    starts = list(range(0, len(values) - length + 1, hop))
    power = []
    for start in starts:
        block = values[start:start + length]
        block = (block - block.mean(axis=0)) * window[:, None]
        spectrum = np.fft.rfft(block, axis=0)
        density = (np.abs(spectrum) ** 2).mean(axis=1) * dt / (window ** 2).sum()
        density[1:-1] *= 2  # All supported window lengths are even.
        power.append(density.tolist())
    return dict(window=length, hop=hop, starts=starts,
                frequencies_hz=np.fft.rfftfreq(length, dt).tolist(), power=power)


def replay(task):
    import numpy as np
    point, scenes, cpu, artifacts_source = task
    source_lab = (ROOT / point['replay_source_lab']).resolve() if point.get('replay_source_lab') else LAB
    if not (source_lab / 'experiments/water-residual-learning-v1/audit.py').exists():
        source_lab = LAB / 'runs/dependency-migration-v1/lab'
    sys.path.insert(0, str(source_lab / 'experiments/water-residual-learning-v1'))
    import audit
    receipt = (ROOT / point['timing_source']).resolve()
    managed = (ROOT / point['run_root'] / 'managed-full').resolve() if point.get('run_root') else receipt.parent.parent.parent
    project = (ROOT / point['replay_project']).resolve() if point.get('replay_project') else managed.parent / 'project'
    artifacts = (ROOT / artifacts_source).resolve()
    weights = artifacts / 'weights.npz'
    assert digest(weights) == point['weights_sha256']
    audit.initialize(project, artifacts, [cpu])
    torch = audit.TORCH
    records = []
    for scene in scenes:
        path = ROOT / scene['fields_source']
        with np.load(path, allow_pickle=False) as arrays:
            errors = []
            for step in scene['steps']:
                base = arrays[f'{step:03d}-base']
                residual = arrays[f'{step:03d}-residual']
                reference = arrays[f'{step:03d}-pressure']
                sample = {'base': torch.from_numpy(base.copy())[None, None],
                          'native_input_residual': torch.from_numpy(residual.copy())[None, None]}
                sample.update({key: torch.tensor([value], dtype=torch.float64)
                               for key, value in [('cell_size_m', scene['cell_size_m']),
                                                  ('rho', 1000.), ('dt', scene['dt_s'])]})
                with torch.no_grad():
                    prediction = audit.MODEL.predict_pressure(audit.NETWORK, sample, audit.NORM)
                error = prediction[0, 0].numpy() - reference
                error -= error.mean()  # Pressure gauge is not a physical error.
                assert np.isfinite(error).all()
                errors.append(error)
            spectra = [temporal_psd(errors, scene['dt_s'], n) for n in (16, 32, 48)]
            records.append(dict(scene_id=scene['id'], spectra=spectra))
    assert not torch.cuda.is_initialized()
    return dict(weights_sha256=point['weights_sha256'], model_id=point['id'],
                computed_by_source_sha256=digest(Path(__file__)),
                policy='Original predict_pressure call, no adaptive repetition or native refinement',
                weights_source=os.path.relpath(weights, ROOT),
                project_source=os.path.relpath(project / 'source-witness.json', ROOT),
                project_sha256=digest(project / 'source-witness.json'), scenes=records)


def worker(task):
    # Project-specific exception classes cannot be unpickled in the parent.
    try:
        return replay(task)
    except Exception:
        import traceback
        raise RuntimeError(f"Replay {task[0]['id']} failed:\n{traceback.format_exc()}") from None


def main():
    import numpy as np
    page = (ROOT / 'cohort-progress.html').read_text()
    data = json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', page, re.S)[1])
    eligible = {p['weights_sha256'] for p in data['frequency_probes']}
    points = [p for p in data['cohort_points'] if p['weights_sha256'] in eligible]
    scenes = []
    for folder in sorted((LAB / 'runs/water-targeted-feedback-v1/cpu-preparation-1').glob('parent-*')):
        receipt = json.loads((folder / 'receipt.json').read_text())
        config = receipt['scene']
        path = folder / 'fields.npz'
        assert receipt['status'] == 'qualified_cpu_teacher_parent'
        assert digest(path) == receipt['fields_sha256']
        frames = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
        steps = [s['step'] for s in receipt['samples']]
        times = [next(f['time_s'] for f in frames if f['step'] == step) for step in steps]
        dt = float(np.median(np.diff(times)))
        assert np.allclose(np.diff(times), dt, rtol=1e-10, atol=1e-12)
        assert len(steps) == 49 and steps == list(range(16, 65))
        with np.load(path, allow_pickle=False) as arrays:
            reference = np.stack([arrays[f'{step:03d}-pressure'] for step in steps])
        reference -= reference.mean(axis=(1, 2, 3), keepdims=True)
        assert tuple(reference.shape[1:]) == tuple(config['counts'][::-1])
        scenes.append(dict(id=folder.name, counts_xyz=config['counts'],
                           cell_size_m=1 / config['cells_per_metre'], dt_s=dt,
                           steps=steps, times_s=times, reference_mean_square=float((reference ** 2).mean()),
                           fields_source=os.path.relpath(path, ROOT), fields_sha256=digest(path),
                           receipt_source=os.path.relpath(folder / 'receipt.json', ROOT),
                           receipt_sha256=digest(folder / 'receipt.json'),
                           frames_source=os.path.relpath(folder / 'frames.jsonl', ROOT),
                           frames_sha256=digest(folder / 'frames.jsonl'),
                           forcing_frequency_hz=config['frequency']))
    cpus = sorted(os.sched_getaffinity(0))
    # Separate processes isolate each project's imported model implementation.
    import multiprocessing
    models = []
    tasks = []
    retained = {}
    old_output = ROOT / 'pressure-spectrogram-data.json'
    if old_output.exists():
        old = json.loads(old_output.read_text())
        if old.get('version') == VERSION and old.get('scenes') == scenes:
            for model in old['models']:
                if (digest(ROOT / model['weights_source']) == model['weights_sha256']
                        and digest(ROOT / model['project_source']) == model['project_sha256']):
                    model.setdefault('computed_by_source_sha256', old['source_sha256'])
                    retained[model['weights_sha256']] = model
    # Resolve and verify restore identities before starting expensive inference.
    for i, point in enumerate(points):
        if point.get('replay_artifacts'):
            project = (ROOT / point['replay_project']).resolve()
            artifacts = (ROOT / point['replay_artifacts']).resolve()
            receipt = json.loads((artifacts / 'receipt.json').read_text())
            if (receipt['mode'] != 'restore'
                    or digest(artifacts / 'weights.npz') != point['weights_sha256']
                    or receipt['closure']['source_witness_sha256'] != digest(project / 'source-witness.json')):
                raise ValueError(f"Current CPU restore mismatch: {point['id']}")
            tasks.append((point, scenes, cpus[i % min(4, len(cpus))], os.path.relpath(artifacts, ROOT)))
            continue
        managed = (ROOT / point['run_root'] / 'managed-full').resolve() if point.get('run_root') else (ROOT / point['timing_source']).resolve().parent.parent.parent
        project = managed.parent / 'project'
        witness = digest(project / 'source-witness.json')
        candidates = []
        for artifacts in sorted(managed.parent.glob('managed*/stage-01/artifacts')):
            if not (artifacts / 'weights.npz').exists():
                continue
            receipt = json.loads((artifacts / 'receipt.json').read_text())
            if (receipt.get('mode') == 'restore'
                    and digest(artifacts / 'weights.npz') == point['weights_sha256']
                    and receipt['closure']['source_witness_sha256'] == witness):
                candidates.append(artifacts)
        if not candidates:
            raise ValueError(f"No matching CPU restore: {point['id']}")
        tasks.append((point, scenes, cpus[i % min(4, len(cpus))], os.path.relpath(candidates[0], ROOT)))
    cache = Path('/tmp/rheon-pressure-spectrogram-cache')
    cache.mkdir(exist_ok=True)
    source_key = digest(Path(__file__))
    scene_key = hashlib.sha256(json.dumps(scenes, sort_keys=True).encode()).hexdigest()
    def cache_path(task):
        return cache / (hashlib.sha256((source_key + scene_key + task[0]['weights_sha256']).encode()).hexdigest() + '.json')
    pending = []
    for task in tasks:
        path = cache_path(task)
        if task[0]['weights_sha256'] in retained:
            model = retained[task[0]['weights_sha256']]
            model['model_id'] = task[0]['id']
            models.append(model)
        elif path.exists():
            models.append(json.loads(path.read_text()))
        else:
            pending.append(task)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(4, len(cpus)),
            mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        futures = {pool.submit(worker, task): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            model = future.result()
            models.append(model)
            cache_path(futures[future]).write_text(json.dumps(model, separators=(',', ':'), allow_nan=False))
            print(f"Computed {model['model_id']} ({len(models)}/{len(points)})", flush=True)
    models.sort(key=lambda m: int(m['model_id'][1:]))
    result = dict(schema='rheon-temporal-pressure-error-spectrogram-v1', version=VERSION,
                  source_sha256=digest(Path(__file__)), scenes=scenes, models=models,
                  scope='New CPU replay on 16 recorded teacher trajectories; 49 states each. '
                        'Development/preparation inputs, including targeted-feedback training exposure; '
                        'not independent validation or model-driven rollout. Scalar rollout errors are not transformed.',
                  method='One-sided temporal PSD in Pa²/Hz, averaged over all cells. Spatial gauge '
                         'removed per frame; temporal mean removed per cell and window. Periodic Hann, '
                         'hop 1 frame, no padding. Reference normalization divides by one full-record '
                         'reference spatial mean-square per scene, identical across models.')
    output = ROOT / 'pressure-spectrogram-data.json'
    output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(f'Wrote {output.name}', flush=True)


if __name__ == '__main__':
    main()
