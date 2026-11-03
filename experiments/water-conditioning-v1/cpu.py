"""CPU fitting, conditioning and iterative-correction diagnostics; no optimization."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import time

from context import LAB, data, measured_model, model, policy
import numpy as np
import torch


def relative_residual(pressure, sample):
    error = model.full_operator(pressure, sample['cell_size_m'], sample['rho']) - sample['rhs']
    denominator = float(torch.linalg.vector_norm(sample['rhs']))
    return float(torch.linalg.vector_norm(error)) / max(denominator, 1e-30)


def controls(sample, normalization):
    scale = model.owner.scales(normalization)['pressure']
    correction = ((sample['reference'] - sample['base']) / scale).to(torch.float32)
    rounded = sample['base'] + model.frozen.gauge_correction(correction, scale)
    expected = sample['rhs'] - model.full_operator(sample['base'], sample['cell_size_m'], sample['rho'])
    captured = sample['native_input_residual']
    residual_error = float(torch.linalg.vector_norm(expected - captured)) / max(float(torch.linalg.vector_norm(captured)), 1e-30)
    variable = rounded.detach().requires_grad_(True)
    loss = model.pressure_loss(variable, sample, normalization).sum()
    gradient, = torch.autograd.grad(loss, variable)
    direction = torch.cos(torch.arange(variable.numel(), dtype=torch.float64)).reshape_as(variable)
    direction = direction / torch.linalg.vector_norm(direction)
    step = scale * 1e-4
    numeric = float((model.pressure_loss(variable.detach()+step*direction, sample, normalization)
                     - model.pressure_loss(variable.detach()-step*direction, sample, normalization)).sum())/(2*step)
    analytic = float((gradient*direction).sum())
    return {'reference_relative_residual': relative_residual(sample['reference'], sample),
            'float32_reference_correction_relative_residual': relative_residual(rounded, sample),
            'captured_residual_relative_difference': residual_error,
            'loss_directional_derivative': {'analytic': analytic, 'numeric': numeric,
                'absolute_difference': abs(analytic-numeric)},
            'rounded_reference_loss': float(model.pressure_loss(rounded, sample, normalization).sum())}


def initialize(artifacts, cpus):
    global NETWORK, NORMALIZATION
    index = multiprocessing.current_process()._identity[0] - 1
    os.sched_setaffinity(0, {cpus[index % len(cpus)]})
    NETWORK, NORMALIZATION, _receipt, _history = measured_model(Path(artifacts))
    torch.set_num_threads(1)


def amplitude_and_iteration(task):
    parent, slot = task
    sample = data.sample(parent, slot)
    base = sample['base']
    with torch.no_grad():
        prediction = model.predict_pressure(NETWORK, sample, NORMALIZATION)
        correction = prediction - base
        amplitudes = []
        base_rhs = model.full_operator(base, sample['cell_size_m'], sample['rho'])
        for amplitude in (-2., -1., .25, .5, 2., 4.):
            scaled = {**sample, 'native_input_residual': amplitude*sample['native_input_residual'],
                      'rhs': base_rhs + amplitude*sample['native_input_residual']}
            predicted = model.predict_pressure(NETWORK, scaled, NORMALIZATION)
            difference = predicted-base-amplitude*correction
            amplitudes.append({'amplitude': amplitude,
                'correction_linearity_error': float(torch.linalg.vector_norm(difference))/max(float(torch.linalg.vector_norm(amplitude*correction)), 1e-30),
                'relative_residual': relative_residual(predicted, scaled)})
        zero = {**sample, 'native_input_residual': torch.zeros_like(base)}
        zero_exact = torch.equal(model.predict_pressure(NETWORK, zero, NORMALIZATION), base)
        iterations = {}
        for method in ('raw', 'minimum_residual_step'):
            current = base.clone()
            trace = [relative_residual(current, sample)]
            steps = []
            for _ in range(8):
                residual = sample['rhs']-model.full_operator(current, sample['cell_size_m'], sample['rho'])
                state = {**sample, 'base': current, 'native_input_residual': residual}
                proposed = model.predict_pressure(NETWORK, state, NORMALIZATION)
                delta = proposed-current
                a_delta = model.full_operator(delta, sample['cell_size_m'], sample['rho'])
                denominator = float(a_delta.square().sum())
                step = 1. if method == 'raw' else float((residual*a_delta).sum())/denominator if denominator > 0 else 0.
                current = current + step*delta
                steps.append(step); trace.append(relative_residual(current, sample))
            iterations[method] = {'relative_residual': trace, 'step': steps,
                'monotonic': all(b <= a*(1+1e-12) for a,b in zip(trace,trace[1:]))}
    return {'trajectory_id': parent['entry']['trajectory_id'], 'native_frame': parent['frames'][slot]['index'],
            'counts_xyz': parent['counts_xyz'], 'cell_size_m': parent['cell_size_m'],
            'relative_residual': relative_residual(prediction, sample), 'zero_bypass_exact': zero_exact,
            'controls': controls(sample, NORMALIZATION), 'amplitudes': amplitudes, 'iterations': iterations,
            'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)),
            'threads': torch.get_num_threads(), 'cuda_initialized': torch.cuda.is_initialized()}


def fit_check(args):
    network, normalization, receipt, _history = measured_model(args.artifacts)
    _reservation, parents = data.read_view(args.packet)
    parent = next(p for p in parents if p['entry']['split'] == 'train')
    sample = data.sample(parent, 0)
    with torch.no_grad():
        value = relative_residual(model.predict_pressure(network, sample, normalization), sample)
    result = {'weights_sha256': receipt['outputs']['weights.npz']['sha256'],
              'relative_residual': value, 'target': 1e-6, 'equation_gate_passed': value <= 1e-6,
              'controls': controls(sample, normalization)}
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['equation_gate_passed'] else 2


def benchmark(args):
    network, normalization, _receipt, _history = measured_model(args.artifacts)
    _reservation, parents = data.read_view(args.packet)
    parent = next(p for p in parents if p['entry']['split'] == 'train')
    sample = data.sample(parent, 0)
    cpus = sorted(os.sched_getaffinity(0)); records = []; reference = None
    with torch.no_grad():
        for threads in sorted({1, 2, 4, 8, 12, len(cpus)} & set(range(1,len(cpus)+1))):
            torch.set_num_threads(threads)
            for _ in range(3): result = model.predict_pressure(network, sample, normalization)
            if reference is None: reference = result.clone()
            clocks = []
            for _ in range(9):
                start = time.perf_counter(); result = model.predict_pressure(network, sample, normalization)
                clocks.append(time.perf_counter()-start)
            records.append({'threads': threads, 'median_s': statistics.median(clocks), 'samples_s': clocks,
                            'pressure_max_abs_difference': float((result-reference).abs().max())})
    report = {'scope': 'isolated restored 16-cubed CPU inference; not end-to-end solver timing',
              'affinity': cpus, 'records': records, 'fastest_threads': min(records,key=lambda r:r['median_s'])['threads']}
    policy.write(args.output, report); print(json.dumps(report), flush=True)


def native_controls(args):
    import diagnostics
    _network, normalization, _receipt, _history = measured_model(args.artifacts)
    _reservation, parents = data.read_view(args.packet)
    parent = next(p for p in parents if p['entry']['split'] == 'train')
    sample = data.sample(parent, 0)
    root = Path(parent['capture_root'])
    manifest_raw = policy.read(root/'manifest.json', 4*policy.MIB)
    policy.require(policy.sha(manifest_raw) == parent['capture_manifest_sha256'], 'capture', 'original physical inputs')
    manifest = policy.decode(manifest_raw)
    trajectory = next(p for p in manifest['trajectories'] if p['trajectory_id'] == parent['scene']['trajectory_id'])
    frame = next(f for f in trajectory['frames'] if f['index'] == parent['frames'][0]['index'])
    velocity = [np.frombuffer(policy.read(root/frame['arrays']['pre_'+axis]['path'],policy.MIB),dtype='<f4').astype(float).tolist() for axis in 'uvw']
    scale = model.owner.scales(normalization)['pressure']
    rounded = sample['base'] + model.frozen.gauge_correction(((sample['reference']-sample['base'])/scale).float(),scale)
    diagnostics.admit(args.binary)
    oracle, validator = diagnostics.checks.historical_oracles()
    profile = diagnostics.targets.load(LAB/'experiments/float32-pressure-target-v1/configs/development.json')
    process = policy.load_module('conditioning_native_process', LAB/'experiments/local-pressure-v1/process.py')
    records = []
    for name, candidate in [('native_reference',sample['reference']),('rounded_reference_correction',rounded)]:
        request = {'schema':'rheon-variable-pressure-teacher-case-v1','counts_xyz':parent['counts_xyz'],
            'spacing_m':[parent['cell_size_m']]*3,'density_kg_m3':parent['density_kg_m3'],'dt_s':parent['entry']['dt_s'],
            'original_rhs':sample['rhs'].flatten().tolist(),'reference_pressure':sample['reference'].flatten().tolist(),
            'previous_pressure':sample['base'].flatten().tolist(),'pre_velocity':velocity,
            'candidate_pressure':candidate.flatten().tolist(),'candidate_error':None}
        response, lifecycle = process.native_call(args.binary,'--evaluate',request,time.monotonic()+63,60)
        independent = diagnostics.checks.check_response(request,response,oracle,validator)
        assessment = diagnostics.targets.assess(response['raw_learned'],profile)
        records.append({'name':name,'native_quality':response['raw_learned'],'float32_acceptance':assessment,
                        'independent_candidate_residual_l2':independent,'process':lifecycle})
    result = {'status':'passed' if all(r['float32_acceptance']['qualified'] for r in records) else 'failed',
              'scope':'original one-frame teacher-state physical controls; no optimizer', 'records':records}
    policy.write(args.output,result); print(json.dumps(result),flush=True)
    return 0 if result['status']=='passed' else 2


def fitting_curve(args):
    network, normalization, receipt, history = measured_model(args.artifacts)
    _reservation, parents = data.read_view(args.packet)
    parent = next(p for p in parents if p['entry']['split']=='train')
    sample = data.sample(parent,0); records = []
    with torch.no_grad():
        for name, spec in sorted(history['checkpoints'].items(),key=lambda item:item[1]['update']):
            policy.require(policy.sha(policy.read(args.artifacts/name))==spec['sha256'],'checkpoint',name)
            data.helpers().load_weights(network,args.artifacts/name,spec['inventory'])
            predicted = model.predict_pressure(network,sample,normalization)
            residual = (model.full_operator(predicted,sample['cell_size_m'],sample['rho'])-sample['rhs'])[0,0].numpy()
            coefficients = residual.copy(); wave_squared = np.zeros_like(coefficients)
            for axis,count in enumerate(residual.shape):
                modes = np.arange(count); cells = np.arange(count)+.5
                basis = np.cos(np.pi*modes[:,None]*cells[None,:]/count)*np.sqrt(2/count)
                basis[0] /= np.sqrt(2)
                coefficients = np.moveaxis(np.tensordot(basis,coefficients,axes=(1,axis)),0,axis)
                shape = [1]*3; shape[axis]=count
                wave_squared += (modes.reshape(shape)/count)**2
            energy = coefficients**2; energy[0,0,0]=0
            low = float(energy[wave_squared<=.25**2].sum())
            total = float(energy.sum())
            ratio = relative_residual(predicted,sample)
            records.append({'update':spec['update'],'weights_sha256':spec['sha256'],
                'relative_residual':ratio,'equation_gate_passed':ratio<=1e-6,
                'training_loss':float(model.pressure_loss(predicted,sample,normalization).sum()),
                'low_frequency_residual_energy_fraction':low/total if total else 0.})
    result = {'status':'complete','records':records,'final_weights_sha256':receipt['outputs']['weights.npz']['sha256'],
              'scope':'fitted single-frame CPU equation curve; cosine modes for sealed cubic-cell grid; full native gates separate'}
    policy.write(args.output,result); print(json.dumps(result),flush=True)


def audit(args):
    output = policy.fresh(args.output)
    _reservation, parents = data.read_view(args.packet)
    tasks = [(p,i) for p in parents for i in range(len(p['frames'])) if p['entry']['split'] == args.population]
    policy.require(tasks, 'population', 'nonempty declared population')
    cpus = sorted(os.sched_getaffinity(0))
    worker_cpus = cpus[1:] if len(cpus) > 1 else cpus
    workers = min(args.workers or len(worker_cpus), len(worker_cpus), len(tasks))
    policy.require(workers > 0, 'workers', 'positive worker budget')
    started = time.time(); records = []
    context = multiprocessing.get_context('spawn')
    # Workers own independent networks and fields. The parent drains every task;
    # a failure propagates and no successful report is published for partial work.
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialize,
                             initargs=(str(args.artifacts), worker_cpus)) as pool:
        for result in pool.map(amplitude_and_iteration, tasks):
            records.append(result)
            with (output/'cases.jsonl').open('a') as stream: stream.write(json.dumps(result,allow_nan=False)+'\n')
            status = {'completed': len(records), 'total': len(tasks), 'workers': workers,
                      'state': 'running', 'updated_at_unix': time.time()}
            temporary = output/'status.tmp'; temporary.write_text(json.dumps(status)); temporary.replace(output/'status.json')
            print('CPU', len(records), '/', len(tasks), flush=True)
    report = {'status': 'complete', 'population': args.population, 'cases': len(records),
              'workers': workers, 'available_cpus': cpus, 'worker_cpus': worker_cpus,
              'used_cpus': sorted({c for r in records for c in r['affinity']}),
              'elapsed_s': time.time()-started, 'started_at_unix': started,
              'completed_at_unix': time.time(), 'optimizer_updates': 0,
              'all_cpu_only': all(not r['cuda_initialized'] and r['threads']==1 for r in records),
              'zero_bypass_exact': all(r['zero_bypass_exact'] for r in records),
              'single_pass_median_residual': statistics.median(r['relative_residual'] for r in records),
              'amplitude_median_linearity_error': {str(a): statistics.median(next(x['correction_linearity_error'] for x in r['amplitudes'] if x['amplitude']==a) for r in records) for a in (-2.,-1.,.25,.5,2.,4.)},
              'iteration_final_median': {m: statistics.median(r['iterations'][m]['relative_residual'][-1] for r in records) for m in ('raw','minimum_residual_step')},
              'scope': 'algebraic amplitude/iteration diagnostics; not physical rollout qualification or trained candidate selection'}
    policy.write(output/'report.json', report)
    status.update(state='completed'); (output/'status.json').write_text(json.dumps(status))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fit-check','benchmark','audit','native-controls','curve'))
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--population', choices=('train','validation'), default='validation')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    if args.command != 'fit-check' and args.output is None: parser.error('--output is required')
    if args.workers is not None and args.workers < 1: parser.error('--workers must be positive')
    if args.command == 'fit-check': return fit_check(args)
    if args.command == 'native-controls':
        if args.binary is None: parser.error('--binary is required')
        return native_controls(args)
    if args.command == 'curve':
        fitting_curve(args)
        return 0
    if args.command == 'benchmark': benchmark(args)
    else: audit(args)
    return 0


if __name__ == '__main__': raise SystemExit(main())
