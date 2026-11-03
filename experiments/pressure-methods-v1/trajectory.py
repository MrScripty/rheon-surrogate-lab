"""One bounded independent trajectory for numerical/hybrid follow-up or periodic use."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha


def episode(args):
    material_path = ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json'
    material = read(material_path)
    sys.path[:0] = [str(ROOT/'experiments/pressure-pass-sweep-v1'),
        str(ROOT/material['config']['source_lab']/'experiments/water-residual-learning-v1')]
    import rollout
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[-1]})
    scene = dict(material['config']['scene'])
    scene['steps'] = args.steps
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = None
    if args.kind != 'native-control':
        shutil.copyfile(HERE/'trajectory_adapter.py', args.output/'model_adapter.py')
        manifest = args.output/'manifest.json'
        write(manifest, dict(schema='rheon-pressure-methods-trajectory-v1', root=str(ROOT),
            material=str(material_path), kind=args.kind, checkpoint=args.checkpoint,
            passes=0 if args.kind == 'full-numerical' else args.passes, interval=args.interval,
            warmup_steps=scene['warmup_steps'], numerics_sha256=sha(HERE/'numerics.py'),
            adapter_sha256=sha(args.output/'model_adapter.py')))
    def stop(*_):
        raise InterruptedError('Trajectory interrupted; native owner drains its child')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    tick = time.perf_counter()
    receipt = rollout.run(ROOT/material['config']['binary'], scene, args.output/'native',
        adapter_manifest=manifest, control=args.kind == 'native-control', timeout=600, threads=1)
    frames = [json.loads(line) for line in (args.output/'native/frames.jsonl').read_text().splitlines()]
    predicted = [f for f in frames if f.get('ai_quality')]
    if not receipt['native_process']['reaped'] or any(not f['reference_quality']['qualified'] for f in frames):
        raise ValueError('Native lifecycle/reference qualification failed')
    completed = bool(predicted) and predicted[-1]['step'] == scene['steps'] and not receipt['result']['hard_stop']
    callbacks_path = args.output/'native/callbacks.jsonl'
    callbacks = [json.loads(line) for line in callbacks_path.read_text().splitlines()] if callbacks_path.exists() else []
    points = [dict(step=f['step'], time_s=f['time_s'], pressure_trajectory_relative_l2=f['relative_pressure_difference'],
                   equation_residual=f['ai_quality']['relative_residual'],
                   divergence=f['ai_quality']['actual_divergence'], qualified=f['ai_quality']['qualified']) for f in predicted]
    result = dict(status='completed' if completed else 'pressure_rejection', kind=args.kind,
        checkpoint=args.checkpoint, passes=args.passes, interval=args.interval, scene=scene,
        points=points, callbacks=len(callbacks), network_calls=sum(c['network_passes'] for c in callbacks),
        numerical_corrections=sum(t.get('numerical_corrections', 0) for c in callbacks for t in c['trace']),
        first_pressure_1pct_crossing=next((p for p in points if p['pressure_trajectory_relative_l2'] > .01), None),
        pressure_1pct_scope='Diagnostic threshold only; no application tolerance was supplied.',
        drift_limitations='Existing diagnostic wire exposes pressure trajectory error, but no velocity/tracer drift. Those measurements remain required for campaign completion.',
        native_reaped=True, elapsed_s=time.perf_counter()-tick,
        native_elapsed_s=receipt['elapsed_s'], adapter_loading_s=receipt['adapter_loading_seconds'],
        summed_callback_s=sum(c['model_correction_seconds'] for c in callbacks),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        material_sha256=sha(material_path), source_sha256=sha(__file__),
        frames_sha256=sha(args.output/'native/frames.jsonl'))
    write(args.output/'results.json', result)
    print(json.dumps({k:result[k] for k in ['status', 'kind', 'network_calls', 'numerical_corrections', 'elapsed_s']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['native-control', 'full-numerical', 'neural', 'before', 'after', 'periodic'], required=True)
    parser.add_argument('--checkpoint', choices=['C53', 'C44', 'C37'], default='C53')
    parser.add_argument('--passes', type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument('--interval', type=int, choices=[4, 16, 64], default=16)
    parser.add_argument('--steps', type=int, choices=[16, 128, 512, 3600], default=512)
    parser.add_argument('--output', type=Path, required=True)
    episode(parser.parse_args())
