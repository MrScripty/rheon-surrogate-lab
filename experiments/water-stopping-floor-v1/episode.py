"""Run one frozen CPU policy, then diagnose sparse states after child reaping."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-pressure-accuracy-v1'))
import diagnose
rollout, audit = diagnose.rollout, diagnose.audit
read, write, digest = diagnose.read, diagnose.write, diagnose.digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scene', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--uninstrumented', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    scene = dict(config['scenes'][args.scene])
    if args.fixture:
        scene['steps'] = scene['warmup_steps']+1
    model = next((m for m in config['models'] if m['name'] == args.model), None)
    if args.model != 'native' and model is None:
        raise ValueError('declared model or native control required')
    args.output.mkdir(parents=True, exist_ok=False)
    captures = []
    original = rollout.correct
    step = scene['warmup_steps']
    wanted = {step+1, scene['steps'], *range(64, scene['steps']+1, 64)}

    def capture(*values, **options):
        nonlocal step
        step += 1
        pressure = original(*values, **options)
        if not args.uninstrumented and step in wanted:
            import numpy as np
            path = args.output/f'state-{step:04d}.npz'
            with path.open('xb') as stream:
                np.savez(stream, **{k: v.detach().cpu().numpy() for k, v in values[3].items()},
                         prediction=pressure.detach().cpu().numpy())
            captures.append({'step': step, 'path': str(path.relative_to(LAB)), 'sha256': digest(path)})
        return pressure

    started = time.perf_counter()
    rollout.correct = capture
    receipt = None
    try:
        receipt = rollout.run(LAB/config['binary'], scene, args.output/'native',
            LAB/model['project'] if model else None, LAB/model['artifacts'] if model else None,
            control=model is None, timeout=config['native_timeout_seconds'],
            proposals=32 if model else 1,
            stopping=read(LAB/model.get('stopping_profile',config['profile'])) if model else None,
            single_evaluation=model is not None, threads=1,
            minimum_network_evaluations=model['minimum_network_evaluations'] if model else 0)
    finally:
        rollout.correct = original
        write(args.output/'captures.json', {'samples': captures, 'scene': scene,
            'model': model, 'optimizer_updates': 0, 'native_elapsed_seconds': time.perf_counter()-started})
    assert receipt['native_process']['reaped']
    measurements = []
    if model and captures:
        import numpy as np
        torch = audit.TORCH
        current = int(next(x for x in Path('/proc/self/status').read_text().splitlines()
                           if x.startswith('VmSize:')).split()[1])*1024
        limit = current+config['oracle_address_space_growth_bytes']
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        with torch.no_grad():
            for item in captures:
                with np.load(LAB/item['path'], allow_pickle=False) as arrays:
                    sample = {k: torch.from_numpy(arrays[k].copy()) for k in arrays.files}
                pressure = sample.pop('prediction')
                delta = audit.spectral_correction(sample['native_input_residual'], sample)
                compatible = sample['native_input_residual']-sample['native_input_residual'].mean()
                defect = audit.vector_norm(diagnose.independent_action(delta, sample)-compatible)/max(audit.vector_norm(compatible), 1e-30)
                assert defect < 1e-9
                exact = sample['base']+delta
                measurement={'step': item['step'], 'sample_sha256': item['sha256'],
                    'oracle_independent_relative_defect': defect,
                    **diagnose.error_metrics(pressure, exact), **diagnose.mode_metrics(pressure, exact)}
                policy=read(LAB/model.get('stopping_profile',config['profile']))
                if policy['schema']=='rheon-learned-stopping-policy-v2':
                    rhs=sample['native_input_residual']+audit.MODEL.full_operator(sample['base'],sample['cell_size_m'],sample['rho'])
                    quality=diagnose.equation_quality(audit.MODEL,pressure,sample,rhs,policy)
                    bound=quality['mean_free_pressure_relative_error_bound']
                    covered=bound is not None and measurement['mean_free_pressure_relative_l2']<=bound*(1+1e-6)+1e-11
                    assert covered, 'independent same-RHS pressure error exceeds declared bound'
                    measurement.update(pressure_bound_relative_l2=bound,pressure_bound_covers_oracle=covered)
                measurements.append(measurement)
        assert not torch.cuda.is_initialized()
    write(args.output/'diagnostics.json', {'model': args.model, 'scene_index': args.scene,
        'samples': measurements, 'source_sha256': digest(__file__), 'config_sha256': digest(args.config),
        'weights_sha256': digest(LAB/model['artifacts']/'weights.npz') if model else None,
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'cpu_threads': 1,
        'optimizer_updates': 0, 'cuda_initialized': False,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'elapsed_seconds': time.perf_counter()-started,
        'timing_scope': 'uninstrumented' if args.uninstrumented else 'sparse capture overhead included; correctness only',
        'scope': 'same-RHS exact pressure and gradient at sparse declared steps; paired native trajectory differences are separate'})


if __name__ == '__main__':
    main()
