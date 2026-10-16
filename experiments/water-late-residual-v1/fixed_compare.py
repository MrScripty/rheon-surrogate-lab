"""Compare checkpoints at equal CPU calls on already consumed common inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-pressure-accuracy-v1'))
import diagnose
from learned_iteration import equation_quality, predict_proposal


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def common_samples(study, all_captures):
    seen = set()
    for witness in study['witnesses']:
        for trajectory in (('retained', 'targeted') if all_captures else ('targeted',)):
            capture = diagnose.read(LAB/study['root']/(witness['name']+'-'+trajectory+'-original')/'capture.json')
            items = capture['samples'] if all_captures else (capture['samples'][0], capture['samples'][-1])
            for item in items:
                if all_captures and item['sha256'] in seen:
                    continue
                seen.add(item['sha256'])
                yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-captures', action='store_true', help='Include both trajectories and peak frames; deduplicate exact input bytes')
    args = parser.parse_args()
    assert not args.output.exists(), 'retain previous attempts'
    started = time.perf_counter()
    sequence, study = diagnose.read(args.sequence), diagnose.read(args.study)
    if args.trial in study['models']:
        source = study['models'][args.trial]
    else:
        assert args.trial in {t['name'] for t in sequence['trials']}
        directory = Path(sequence['root'])/args.trial
        source = {'project': str(directory/'project'),
                  'artifacts': str(directory/'managed-full/stage-01/artifacts')}
    audit = diagnose.audit
    audit.initialize(LAB/source['project'], LAB/source['artifacts'], sorted(os.sched_getaffinity(0)))
    current = int(next(x for x in Path('/proc/self/status').read_text().splitlines()
                       if x.startswith('VmSize:')).split()[1])*1024
    resource.setrlimit(resource.RLIMIT_AS, (current+512*1024**2, current+512*1024**2))
    import numpy as np
    torch = audit.TORCH
    cases = []
    with torch.no_grad():
        for item in common_samples(study, args.all_captures):
            path = LAB/item['path']
            assert digest(path) == item['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                sample = {k: torch.from_numpy(arrays[k].copy()) for k in arrays.files}
            initial = sample['native_input_residual']
            base = sample['base']
            delta = audit.spectral_correction(initial, sample)
            compatible = initial-initial.mean()
            defect = audit.vector_norm(diagnose.independent_action(delta, sample)-compatible)/max(audit.vector_norm(compatible), 1e-30)
            assert defect < 1e-9
            exact = base+delta
            rhs = initial+audit.MODEL.full_operator(base, sample['cell_size_m'], sample['rho'])
            pressure = base.clone()
            frontier = []
            inference_s = 0.
            for calls in range(1, 13):
                state = {**sample, 'base': pressure, 'native_input_residual': initial-audit.MODEL.full_operator(
                    pressure-base, sample['cell_size_m'], sample['rho'])}
                tick = time.perf_counter()
                pressure = predict_proposal(audit.MODEL, audit.NETWORK, state, audit.NORM, True)
                inference_s += time.perf_counter()-tick
                if calls in (4, 8, 12):
                    frontier.append({'calls': calls, 'cumulative_cpu_correction_seconds': inference_s,
                        **diagnose.error_metrics(pressure, exact), **diagnose.mode_metrics(pressure, exact),
                        'quality': equation_quality(audit.MODEL, pressure, sample, rhs, study['policies']['original'])})
            cases.append({'sample': item, 'counts_xyz': list(base.shape[2:][::-1]),
                'oracle_independent_relative_defect': defect, 'frontier': frontier})
    summary = {}
    for calls in (4, 8, 12):
        rows = [next(r for r in c['frontier'] if r['calls'] == calls) for c in cases]
        summary[str(calls)] = {'cases': len(rows), 'equation_passed': sum(r['quality']['equation_passed'] for r in rows),
            **{'worst_'+key: max(r[key] for r in rows) for key in ('pressure_relative_l2', 'mean_free_pressure_relative_l2', 'gradient_relative_l2')},
            'median_low_mode_error_energy_fraction': statistics.median(r['low_mode_error_energy_fraction'] for r in rows),
            'correction_seconds_sum': sum(r['cumulative_cpu_correction_seconds'] for r in rows)}
    result = {'schema': 'rheon-late-residual-fixed-call-comparison-v1', 'trial': args.trial,
        'source': source, 'weights_sha256': digest(LAB/source['artifacts']/'weights.npz'),
        'normalization_sha256': digest(LAB/source['artifacts']/'normalization.json'),
        'source_witness_sha256': digest(LAB/source['project']/'source-witness.json'),
        'source_sha256': digest(__file__), 'argv': [sys.executable, *sys.argv],
        'summary': summary, 'cases': cases, 'optimizer_updates': 0, 'simulations_advanced': 0,
        'cuda_initialized': torch.cuda.is_initialized(), 'cpu_threads': torch.get_num_threads(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'elapsed_seconds': time.perf_counter()-started,
        'scope': 'common consumed saved RHS; independently qualified exact pressure, not autonomous trajectory difference; CPU latency sampled during other work, not isolated benchmark'}
    assert not result['cuda_initialized']
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'trial': args.trial, 'summary': summary, 'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
