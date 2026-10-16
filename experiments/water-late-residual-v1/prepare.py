"""Freeze train-only paired residual directions from a recorded CPU checkpoint."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-residual-learning-v1'))
import audit
sys.path.insert(0, str(LAB/'experiments/water-stability-learning-v1'))
from learned_iteration import correct, predict_proposal
sys.path.insert(0, str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import read, write, utc
from hashlib import sha256


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def limit_memory(growth_bytes):
    status = Path('/proc/self/status').read_text().splitlines()
    current = int(next(x for x in status if x.startswith('VmSize:')).split()[1])*1024
    resource.setrlimit(resource.RLIMIT_AS, (current+growth_bytes, current+growth_bytes))
    return current+growth_bytes


def initialize(source, cpus, growth_bytes=None):
    audit.initialize(LAB/source['project'], LAB/source['artifacts'], cpus)
    global ORACLE
    spec = spec_from_file_location('direction_oracle', LAB/'experiments/global-pressure-v1/verify_capture.py')
    ORACLE = module_from_spec(spec)
    spec.loader.exec_module(ORACLE)
    if growth_bytes is not None:
        limit_memory(growth_bytes)


def prepare(task):
    import numpy as np
    identity, arrays, amplitude, count, minimum = task
    torch = audit.TORCH
    sample = {k: torch.from_numpy(v.copy()) for k, v in arrays.items()}
    initial = sample['native_input_residual']
    with torch.no_grad():
        pressure = correct(audit.MODEL, audit.NETWORK, audit.NORM, sample, count, single_evaluation=True)
        late = initial-audit.MODEL.full_operator(pressure-sample['base'], sample['cell_size_m'], sample['rho'])
        ratio = audit.vector_norm(late)/max(audit.vector_norm(initial), 1e-30)
        record = {'identity': identity, 'late_over_initial': ratio,
            'cpu_affinity': sorted(os.sched_getaffinity(0)), 'cuda_initialized': torch.cuda.is_initialized()}
        if ratio <= minimum:
            return record | {'excluded': 'late residual at numerical-noise guard; both pair members excluded'}, None
        homogeneous = []
        proposal = predict_proposal(audit.MODEL, audit.NETWORK,
            {**sample, 'base': torch.zeros_like(pressure), 'native_input_residual': late}, audit.NORM, True)
        for gain in (.3, 2.):
            scaled = predict_proposal(audit.MODEL, audit.NETWORK,
                {**sample, 'base': torch.zeros_like(pressure), 'native_input_residual': late*gain}, audit.NORM, True)
            error = audit.vector_norm(scaled-gain*proposal)/max(audit.vector_norm(gain*proposal), 1e-30)
            assert error < 1e-5, ('homogeneity', identity, gain, error)
            homogeneous.append({'gain': gain, 'relative_error': error})
        stages, fields = {}, {}
        for name, residual in [('early', initial), ('late', late)]:
            compatible = residual-residual.mean()
            target = audit.spectral_correction(compatible, sample)
            rms = float(target.square().mean().sqrt())
            assert rms > 0 and torch.isfinite(target).all()
            gain = amplitude/rms
            rhs, target = compatible*gain, target*gain
            counts = list(target.shape[2:][::-1])
            action = ORACLE.full_action(target.flatten().tolist(), counts,
                [float(sample['cell_size_m'][0])]*3, float(sample['rho'][0]))
            action = torch.from_numpy(np.asarray(action, dtype='<f8')).reshape_as(target)
            defect = audit.vector_norm(action-rhs)/max(audit.vector_norm(rhs), 1e-30)
            assert defect < 1e-9 and float(target.flatten()[0]) == 0, (identity, name, defect)
            values = audit.MODEL.build_inputs(rhs, torch.zeros_like(target), sample['cell_size_m'], sample['rho'], sample['dt'], audit.NORM)
            assert torch.isfinite(values).all()
            stages[name] = {'gain': gain, 'unscaled_correction_rms_pa': rms,
                'scaled_correction_rms_pa': float(target.square().mean().sqrt()),
                'operator_relative_error': defect,
                'removed_constant_over_residual': audit.vector_norm(residual-compatible)/max(audit.vector_norm(residual), 1e-30)}
            fields[name] = {'rhs': rhs.numpy().copy(), 'reference': target.numpy().copy()}
        return record | {'stages': stages, 'homogeneity': homogeneous,
            'worker_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'worker_pid': os.getpid(),
            'counts_xyz': list(pressure.shape[2:][::-1]),
            **{k: float(sample[k][0]) for k in ('rho', 'dt', 'cell_size_m')}}, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = read(args.config)
    started = time.perf_counter()
    cpus = sorted(os.sched_getaffinity(0))
    initialize(config['source'], cpus)
    os.sched_setaffinity(0, cpus)
    require_available = config['minimum_available_bytes']
    available = int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    assert available >= require_available, 'insufficient memory reserve before preparation'
    parent_limit = limit_memory(config['parent_growth_bytes'])
    torch = audit.TORCH
    import analytical
    _, parents = audit.DATA.read_view(LAB/config['packet'])
    samples = [(p, i, audit.DATA.sample(p, i)) for p in parents if p['entry']['split'] == 'train'
               for i in range(len(p['frames']))]
    amplitude = analytical.training_correction_rms(s for _, _, s in samples)
    selected = [({'kind': 'real', 'group': 'real', 'trajectory_id': p['entry']['trajectory_id'],
        'frame': p['frames'][i]['index']}, s) for p, i, s in samples]
    source_config = read(LAB/config['source']['project']/'config.json')
    for bank_name, spec in [('primary', source_config['analytical_feedback']),
                            ('supplemental', source_config['supplemental_feedback'])]:
        for grid, detail in enumerate(spec['grids']):
            # First eight patterns are fixed in the canonical generator. Build
            # only one grid and its selected patterns, not all amplitude banks.
            template = {**samples[0][2], 'base': torch.zeros((1, 1, *detail['counts_xyz'][::-1]), dtype=torch.float64),
                'cell_size_m': torch.tensor([1/detail['cells_per_metre']], dtype=torch.float64)}
            values, _, _ = analytical.single_grid_bank(template,
                {**spec, 'patterns': config['patterns_per_grid']}, 1, amplitude=amplitude)
            for pattern in range(config['patterns_per_grid']):
                index = pattern*len(analytical.AMPLITUDES)+config['source_amplitude_slot']
                selected.append(({'kind': 'manufactured', 'group': f'{bank_name}-{grid:02d}',
                    'bank': bank_name, 'grid': detail, 'pattern': pattern,
                    'amplitude_slot': config['source_amplitude_slot']}, values[index]))
            del values
    if args.fixture:
        selected = [selected[0], next(x for x in selected if x[0]['kind'] == 'manufactured')]
    output = LAB/config['output']
    if args.fixture:
        output = output.with_name(output.name+'-fixture')
    if args.output:
        output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    tasks = [(identity, {k: s[k].numpy().copy() for k in ('base', 'native_input_residual', 'rho', 'dt', 'cell_size_m')},
        amplitude, config['corrections'], config['minimum_late_ratio']) for identity, s in selected]
    with ProcessPoolExecutor(max_workers=min(config['workers'], len(tasks), len(cpus)),
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(config['source'], cpus, config['worker_growth_bytes'])) as pool:
        results = list(pool.map(prepare, tasks, buffersize=config['workers']))
    arrays = {'early': {}, 'late': {}}
    records, excluded = [], []
    for record, fields in results:
        if fields is None:
            excluded.append(record)
            continue
        index = len(records)
        records.append(record)
        for stage in arrays:
            for name, value in fields[stage].items():
                arrays[stage][f'case_{index:03d}_{name}'] = value
    assert {r['identity']['group'] for r in records} == {r[0]['group'] for r in selected}
    import numpy as np
    stages = {}
    for stage, values in arrays.items():
        path = output/(stage+'.npz')
        with path.open('xb') as stream:
            np.savez(stream, **values)
        assert path.stat().st_size <= 32*1024**2
        stages[stage] = {'sha256': digest(path), 'bytes': path.stat().st_size, 'arrays': audit.DATA.inventory(values)}
    manifest = {'schema': 'rheon-pressure-direction-bank-v1', 'created_at_utc': utc(),
        'source': {**config['source'], 'weights_sha256': digest(LAB/config['source']['artifacts']/'weights.npz'),
            'normalization_sha256': digest(LAB/config['source']['artifacts']/'normalization.json'),
            'source_witness_sha256': digest(LAB/config['source']['project']/'source-witness.json')},
        'packet_splits_sha256': digest(LAB/config['packet']/'splits.json'),
        'train_trajectory_ids': [p['entry']['trajectory_id'] for p in parents if p['entry']['split'] == 'train'],
        'amplitude_rms_pa': amplitude, 'source_corrections': config['corrections'],
        'configuration': config, 'cases': records, 'excluded': excluded, 'stages': stages,
        'source_sha256': digest(__file__), 'elapsed_seconds': time.perf_counter()-started,
        'memory': {'parent_address_space_limit_bytes': parent_limit,
            'parent_growth_limit_bytes': config['parent_growth_bytes'],
            'worker_growth_limit_bytes': config['worker_growth_bytes'],
            'parent_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'available_bytes_at_start': available, 'workers': min(config['workers'], len(tasks), len(cpus)),
            'bounded_submission_buffer': config['workers']},
        'optimizer_updates': 0, 'scope': 'frozen train-only extra pressure directions; not new CFD parents or inference inputs'}
    write(output/'manifest.json', manifest)
    report = {k: v for k, v in manifest.items() if k != 'stages'}
    report.update(stages={k: {n: v for n, v in value.items() if n != 'arrays'} for k, value in stages.items()},
        manifest_sha256=digest(output/'manifest.json'), output=str(output.relative_to(LAB)))
    reports = LAB/'docs/plans/water-late-residual-v1/reports'
    report_name = (output.name if args.fixture else 'bank-preparation')+'.json'
    if (reports/report_name).exists():
        raise FileExistsError('report attempt already exists; preserve it and choose a fresh output')
    write(reports/report_name, report)
    print({'cases': len(records), 'excluded': len(excluded), 'stage_bytes': {k: v['bytes'] for k, v in stages.items()},
        'elapsed_seconds': manifest['elapsed_seconds']}, flush=True)


if __name__ == '__main__':
    main()
