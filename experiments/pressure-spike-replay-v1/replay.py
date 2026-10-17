"""Capture unchanged CPU trajectories, then replay one extra learned correction.

Existing sealed rollout/adapter/iteration/oracle owners retain their arithmetic
and child lifecycle. This experiment owns selection, paired observations and
reports. Counterfactual pressures never advance the recorded trajectory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import sys
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def select_steps(row, scene, config):
    """Freeze two peak rankings, the worst step, and adjacent physical steps."""
    selected = {}
    def add(step, label):
        if scene['warmup_steps'] < step <= scene['steps']:
            selected.setdefault(step, set()).add(label)
    for key, count in [('excursion', config['peaks_by_excursion']),
                       ('absolute_rms_pa', config['peaks_by_absolute_error'])]:
        for peak in sorted(row['peaks'], key=lambda p: (-p[key], p['physical_step']))[:count]:
            add(peak['physical_step'], 'selected_peak')
    if config['include_worst_error_step']:
        # The caller supplies the physical step from the unchanged full trace.
        add(row['worst_physical_step'], 'worst_error_step')
    for step in list(selected):
        for offset in config['neighbor_offsets']:
            add(step+offset, 'neighbor')
    all_peaks = {p['physical_step'] for p in row['peaks']}
    return [{'step': step, 'roles': sorted(roles), 'detected_peak': step in all_peaks}
            for step, roles in sorted(selected.items())]


def prepare(root, config_path, output):
    config = read(config_path)
    if config['schema'] != 'rheon-pressure-spike-replay-v1':
        raise ValueError('Current spike replay configuration required')
    analysis_path = root/config['analysis']
    analysis = read(analysis_path)
    import re
    page = root/'docs/plans/cohort-progress.html'
    chart = json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', page.read_text(), re.S)[1])
    if digest(page) != analysis['sources']['docs/plans/cohort-progress.html']:
        raise ValueError('Spike analysis belongs to a different chart snapshot')
    cohorts = {c['weights_sha256']: c for c in chart['cohort_points']}
    jobs = []
    for identifier in config['cohorts']:
        row = dict(next(r for r in analysis['rows'] if r['id'] == identifier))
        trace = next(t for t in chart['drift_traces'] if t['frames_source'] == row['frames_source'])
        if trace['scene'] != analysis['scene'] or not trace['completed']:
            raise ValueError('Complete matching observed scene required')
        row['worst_physical_step'] = max(zip(trace['points'], trace['pressure_magnitudes']), key=lambda x: x[1][0])[0][0]
        frames = (root/'docs/plans'/row['frames_source']).resolve()
        receipt_path = frames.with_name('receipt.json')
        receipt = read(receipt_path)
        manifest = Path(receipt['adapter_manifest'])
        binary = Path(receipt['command'][0])
        if digest(manifest) != receipt['adapter_manifest_sha256'] or digest(binary) != receipt['binary_sha256']:
            raise ValueError('Original package/binary bytes differ')
        jobs.append(dict(id=identifier, weights_sha256=row['weights_sha256'],
            manifest=str(manifest.relative_to(root)), manifest_sha256=digest(manifest),
            binary=str(binary.relative_to(root)), binary_sha256=digest(binary),
            original_frames=str(frames.relative_to(root)), original_frames_sha256=digest(frames),
            original_callbacks=str(frames.with_name('callbacks.jsonl').relative_to(root)),
            original_callbacks_sha256=digest(frames.with_name('callbacks.jsonl')),
            selection=select_steps(row, trace['scene'], config)))
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(schema=config['schema'], root=str(root), config=config,
        config_sha256=digest(config_path), analysis_sha256=digest(analysis_path),
        scene=analysis['scene'], jobs=jobs, created_utc=datetime.now(timezone.utc).isoformat(),
        scope='One extra learned pass on identical input; independent same-RHS spectral reference. No training or deployment change.')
    write(output/'plan.json', plan)
    print(json.dumps({j['id']:j['selection'] for j in jobs}, indent=2))


def numerical_frames(frames):
    """Reuse the original benchmark's exact physical projection, ignoring clocks/UUIDs."""
    result = []
    for frame in frames:
        value = {k: v for k,v in frame.items() if k not in ('ai_seconds','reference_seconds')}
        if value.get('diagnostic_attempt'):
            value['diagnostic_attempt'] = {k:v for k,v in value['diagnostic_attempt'].items() if k != 'request_sha256'}
        result.append(value)
    return result


def worker(plan_path, identifier, cpu):
    plan = read(plan_path)
    root = Path(plan['root'])
    config = plan['config']
    job = next(j for j in plan['jobs'] if j['id'] == identifier)
    output = plan_path.parent/identifier
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    receipt = dict(status='running', id=identifier, command=sys.argv, cpu=cpu,
                   optimizer_updates=0, counterfactual_simulation_steps=0,
                   source_sha256=digest(__file__), plan_sha256=digest(plan_path))
    def cancelled(*_):
        raise InterruptedError('Spike replay worker cancelled or deadline reached')
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGALRM, cancelled)
    signal.alarm(config['worker_timeout_seconds'])
    os.sched_setaffinity(0, {cpu})
    os.environ.update(CUDA_VISIBLE_DEVICES='', TULDOK_TORCH_DEVICE='cpu',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    def progress(phase, completed, total):
        value = dict(id=identifier, phase=phase, completed=completed, total=total,
                     elapsed_s=time.perf_counter()-started)
        temporary = output/'progress.tmp'
        temporary.write_text(json.dumps(value))
        temporary.replace(output/'progress.json')
        print(json.dumps(value), flush=True)
    try:
        available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
        if available < config['minimum_available_memory_bytes']:
            raise RuntimeError('Insufficient available RAM for bounded CPU diagnostic')
        for key in ('manifest','binary','original_frames','original_callbacks'):
            if digest(root/job[key]) != job[key+'_sha256']:
                raise ValueError('Retained input changed: '+key)
        lab = root/config['source_lab']
        sys.path.insert(0, str(lab/'experiments/water-residual-learning-v1'))
        import rollout
        import audit
        adapter = module('spike_request_adapter', (root/job['manifest']).parent/'model_adapter.py')
        wanted = {s['step'] for s in job['selection']}
        captures = []
        state_holder = []
        progress('capture_original_trajectory', 0, plan['scene']['steps'])
        def observe(request, state, reply, trace):
            state_holder[:] = [state]
            step = request['context']['generation']
            if step % 50 == 0:
                progress('capture_original_trajectory', step, plan['scene']['steps'])
            if step not in wanted:
                return
            sample = adapter.sample_from_request(request, state)
            path = output/f'state-{step:04d}.npz'
            with path.open('xb') as stream:
                state.np.savez(stream, **{k:v.numpy() for k,v in sample.items()},
                               prediction=state.np.asarray(reply['pressure'], dtype='<f8').reshape(sample['base'].shape))
            captures.append(dict(step=step, path=path.name, sha256=digest(path),
                request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                original_network_passes=trace[-1]['proposals']))
        native = rollout.run(root/job['binary'], plan['scene'], output/'native',
            adapter_manifest=root/job['manifest'], timeout=config['native_timeout_seconds'], threads=1, observe=observe)
        if not native['native_process']['reaped'] or native['result']['hard_stop']:
            raise ValueError('Baseline native process did not complete and reap')
        original = [json.loads(l) for l in (root/job['original_frames']).read_text().splitlines()]
        current = [json.loads(l) for l in (output/'native/frames.jsonl').read_text().splitlines()]
        if numerical_frames(original) != numerical_frames(current):
            raise ValueError('Recaptured physical trajectory differs from the original')
        if [c['step'] for c in captures] != sorted(wanted):
            raise ValueError('Targeted capture inventory mismatch')
        receipt.update(native=native, original_trajectory_exact=True, captures=captures)
        state = state_holder[0]
        torch, np = state.torch, state.np
        if digest((root/job['manifest']).parent/'weights.npz') != job['weights_sha256']:
            raise ValueError('Observed cohort weight identity mismatch')
        oracle = module('spike_independent_oracle', lab/'experiments/global-pressure-v1/verify_capture.py')
        rows = []
        audit.TORCH = torch
        def norm(value):
            return float(torch.linalg.vector_norm(value))
        with torch.no_grad():
            for index, capture in enumerate(captures):
                with np.load(output/capture['path'], allow_pickle=False) as arrays:
                    sample = {k:torch.from_numpy(arrays[k].copy()) for k in arrays.files}
                stored = sample.pop('prediction')
                base, residual = sample['base'], sample['native_input_residual']
                delta = audit.spectral_correction(residual, sample)
                action = oracle.full_action(delta.flatten().tolist(), list(delta.shape[2:][::-1]),
                    [float(sample['cell_size_m'][0])]*3, float(sample['rho'][0]))
                independent = torch.from_numpy(np.asarray(action,dtype='<f8')).reshape_as(delta)
                defect = norm(independent-(residual-residual.mean()))/max(norm(residual-residual.mean()),1e-30)
                if defect >= 1e-9:
                    raise ValueError('Independent same-RHS spectral oracle failed')
                exact = base+delta
                rhs = residual+state.model.full_operator(base,sample['cell_size_m'],sample['rho'])
                def metrics(pressure):
                    error = pressure-exact
                    free = error-error.mean()
                    return dict(absolute_rms_pa=norm(error)/math.sqrt(error.numel()),
                        mean_free_rms_pa=norm(free)/math.sqrt(error.numel()),
                        mean_free_relative_l2=norm(free)/max(norm(exact-exact.mean()),1e-30),
                        gauge_offset_pa=float(error.mean()),
                        quality=state.iteration.equation_quality(state.model,pressure,sample,rhs,state.policy['stopping']))
                timings, calls, outputs = [], [], []
                for repetition in range(config['timing_repetitions']+1):
                    count = []
                    hook = state.network.register_forward_hook(lambda *_:count.append(1))
                    start = time.perf_counter()
                    try:
                        trace = []
                        baseline = state.iteration.correct(state.model,state.network,state.normalization,sample,
                            state.policy['max_network_calls'],state.policy['stopping'],trace,True)
                        baseline_s = time.perf_counter()-start
                        baseline_calls = len(count)
                        if not torch.equal(baseline,stored) or baseline_calls != capture['original_network_passes']:
                            raise ValueError('Replay baseline differs from captured candidate/call count')
                        start = time.perf_counter()
                        remaining = residual-state.model.full_operator(baseline-base,sample['cell_size_m'],sample['rho'])
                        extra = state.iteration.correct(state.model,state.network,state.normalization,
                            {**sample,'base':baseline,'native_input_residual':remaining},1,None,None,True)
                        state.iteration.equation_quality(state.model,extra,sample,rhs,state.policy['stopping'])
                        extra_s = time.perf_counter()-start
                        if len(count) != baseline_calls+1:
                            raise ValueError('Counterfactual did not execute exactly one extra network pass')
                    finally:
                        hook.remove()
                    if repetition:
                        timings.append(dict(baseline_s=baseline_s,extra_pass_s=extra_s,total_s=baseline_s+extra_s))
                        calls.append(baseline_calls)
                    outputs = [baseline,extra]
                before,after = map(metrics,outputs)
                selection = next(s for s in job['selection'] if s['step']==capture['step'])
                row = dict(**selection,baseline=before,extra=after,
                    absolute_error_ratio=after['absolute_rms_pa']/before['absolute_rms_pa'],
                    mean_free_error_ratio=after['mean_free_rms_pa']/before['mean_free_rms_pa'],
                    baseline_network_passes=calls[0],extra_network_passes=1,
                    timing_samples=timings,oracle_independent_relative_defect=defect)
                rows.append(row)
                progress('paired_replay', index+1, len(captures))
        receipt.update(status='passed',rows=rows,cuda_initialized=torch.cuda.is_initialized(),
            torch_threads=torch.get_num_threads(),numerical_sources={str(p.relative_to(root)):digest(p) for p in
                [Path(rollout.__file__),Path(audit.__file__),Path(oracle.__file__),Path(state.iteration.__file__)]})
    except BaseException as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        signal.alarm(0)
        receipt.update(elapsed_s=time.perf_counter()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
        write(output/'receipt.json',receipt)
    print(json.dumps({'id':identifier,'status':receipt['status'],'elapsed_s':receipt['elapsed_s']}),flush=True)


def summarize(plan_path):
    plan = read(plan_path)
    summary = []
    for job in plan['jobs']:
        result = read(plan_path.parent/job['id']/'receipt.json')
        if result['status'] != 'passed':
            raise ValueError('All paired replays must complete before summarizing')
        groups = {}
        for name, rows in [('all',result['rows']),('peaks',[r for r in result['rows'] if r['detected_peak']]),
                           ('neighbors',[r for r in result['rows'] if not r['detected_peak']])]:
            if not rows:
                continue
            groups[name] = dict(cases=len(rows),absolute_improved=sum(r['absolute_error_ratio']<1 for r in rows),
                median_absolute_error_ratio=statistics.median(r['absolute_error_ratio'] for r in rows),
                median_mean_free_error_ratio=statistics.median(r['mean_free_error_ratio'] for r in rows),
                worst_absolute_error_ratio=max(r['absolute_error_ratio'] for r in rows),
                baseline_median_rms_pa=statistics.median(r['baseline']['absolute_rms_pa'] for r in rows),
                extra_median_rms_pa=statistics.median(r['extra']['absolute_rms_pa'] for r in rows))
        samples = [s for r in result['rows'] for s in r['timing_samples']]
        summary.append(dict(id=job['id'],groups=groups,elapsed_s=result['elapsed_s'],
            baseline_median_s=statistics.median(s['baseline_s'] for s in samples),
            extra_pass_median_s=statistics.median(s['extra_pass_s'] for s in samples),
            peak_rss_bytes=result['peak_rss_bytes'],native_reaped=result['native']['native_process']['reaped']))
    write(plan_path.parent/'summary.json',dict(status='completed',scope=plan['scope'],models=summary,
        total_training_updates=0,baseline_simulation_steps=len(plan['jobs'])*plan['scene']['steps'],
        counterfactual_simulation_steps=0,plan_sha256=digest(plan_path)))
    print(json.dumps(summary,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('prepare')
    for name in ('root','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p = commands.add_parser('worker')
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--id',required=True)
    p.add_argument('--cpu',type=int,required=True)
    p = commands.add_parser('summarize')
    p.add_argument('--plan',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='prepare':
        prepare(args.root.resolve(),args.config.resolve(),args.output.resolve())
    elif args.command=='worker':
        worker(args.plan.resolve(),args.id,args.cpu)
    else:
        summarize(args.plan.resolve())


if __name__=='__main__':
    main()
