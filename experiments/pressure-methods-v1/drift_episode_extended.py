"""Measure pressure, velocity and tracer drift on independently advanced tracks."""
import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha


def episode(args):
    def stop(*_):
        raise InterruptedError('Drift episode interrupted; native owner reaps its child')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0):
            raise ValueError('Declared CPU is outside available affinity')
        os.sched_setaffinity(0,{args.cpu})
    binary = args.build/'rheon-drift-rollout'
    build = read(args.build/'build-receipt.json')
    if build['status'] != 'qualified_build' or sha(binary) != build['binary_sha256']:
        raise ValueError('Qualified isolated drift build required')
    material_path = ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json'
    material = read(material_path)
    sys.path[:0] = [str(ROOT/'experiments/pressure-pass-sweep-v1'),
        str(ROOT/material['config']['source_lab']/'experiments/water-residual-learning-v1')]
    import rollout
    scene = dict(material['config']['scene'],seed=args.seed,steps=args.steps)
    if args.counts is not None:
        scene['counts'] = args.counts
    if args.cells_per_metre is not None:
        scene['cells_per_metre'] = args.cells_per_metre
    if args.output.exists():
        raise ValueError('Fresh drift trajectory required')
    args.output.mkdir(parents=True)
    manifest = None
    if args.kind != 'native-control':
        adapter = 'fit_trajectory_extended_adapter.py' if args.fit_evaluation is not None else 'trajectory_extended_adapter.py'
        shutil.copyfile(HERE/adapter,args.output/'model_adapter.py')
        manifest = args.output/'manifest.json'
        contract = dict(schema='rheon-pressure-methods-trajectory-v1',root=str(ROOT),
            material=str(material_path),kind=args.kind,checkpoint=args.checkpoint,
            passes=0 if args.kind=='full-numerical' else args.passes,interval=args.interval,
            warmup_steps=scene['warmup_steps'],numerics_sha256=sha(HERE/'numerics.py'),
            adapter_sha256=sha(args.output/'model_adapter.py'))
        if args.fit_evaluation is not None:
            if args.kind != 'neural':
                raise ValueError('A trained fit owns its full inference method; select neural call budget')
            contract.update(schema='rheon-pressure-methods-fit-trajectory-v1',
                evaluation=str(args.fit_evaluation.resolve()),evaluation_sha256=sha(args.fit_evaluation))
        write(manifest,contract)
    tick = time.perf_counter()
    receipt = rollout.run(binary,scene,args.output/'native',adapter_manifest=manifest,
        control=args.kind=='native-control',timeout=600,threads=1)
    import json
    frames = [json.loads(line) for line in (args.output/'native/frames.jsonl').read_text().splitlines()]
    points = []
    keys = ['velocity_trajectory_relative_l2','tracer_trajectory_relative_l2',
            'velocity_trajectory_max_abs_m_s','tracer_trajectory_max_abs']
    for frame in frames:
        if not frame.get('reference_quality',{}).get('qualified'):
            raise ValueError('Unqualified native reference')
        if any(not isinstance(frame.get(k),(int,float)) or not math.isfinite(frame[k]) or frame[k]<0 for k in keys):
            raise ValueError('Complete finite host-observed drift required')
        if frame['step']<=scene['warmup_steps']:
            if any(frame[k] != 0 for k in keys):
                raise ValueError('Shared native warmup must have zero drift')
            continue
        if not frame.get('ai_quality',{}).get('qualified'):
            raise ValueError('Only accepted independently advanced frames belong in drift history')
        points.append(dict(step=frame['step'],time_s=frame['time_s'],
            pressure_trajectory_relative_l2=frame['relative_pressure_difference'],
            equation_residual=frame['ai_quality']['relative_residual'],**{k:frame[k] for k in keys}))
    thresholds = {k:.01 for k in ['pressure_trajectory_relative_l2','velocity_trajectory_relative_l2','tracer_trajectory_relative_l2']}
    crossings = {k:next((p for p in points if p[k]>limit),None) for k,limit in thresholds.items()}
    callbacks_path = args.output/'native/callbacks.jsonl'
    callbacks = [json.loads(line) for line in callbacks_path.read_text().splitlines()] if callbacks_path.exists() else []
    completed = bool(points) and points[-1]['step']==scene['steps'] and not receipt['result']['hard_stop']
    result = dict(status='completed' if completed else 'pressure_rejection',kind=args.kind,checkpoint=args.checkpoint,
        passes=args.passes,interval=args.interval,scene=scene,points=points,thresholds=thresholds,first_crossings=crossings,
        threshold_scope='Prespecified 1% relative L2 diagnostic limits; no application-specific tolerance supplied',
        norms='Velocity concatenates all staggered u/v/w faces; tracer uses cell concentration; relative to independently advanced native reference with norm floor 1e-30',
        velocity_absolute_units='m/s',tracer_absolute_units='dimensionless',
        completed_post_warmup_steps=len(points),callbacks=len(callbacks),network_calls=sum(c['network_passes'] for c in callbacks),
        numerical_corrections=sum(t.get('numerical_corrections',0) for c in callbacks for t in c['trace']),
        numerical_schedule='First post-warmup pressure request, then every interval requests' if args.kind=='periodic' else None,
        elapsed_s=time.perf_counter()-tick,native_elapsed_s=receipt['elapsed_s'],
        summed_callback_s=sum(c['model_correction_seconds'] for c in callbacks),
        summed_reference_track_s=sum(f['reference_seconds'] for f in frames),
        summed_candidate_track_s=sum(f['ai_seconds'] for f in frames),
        timing_scope='Instrumented dual-track diagnostic with host checks and IPC; not a single-track production benchmark',
        native_process=receipt['native_process'],source_sha256=sha(__file__),build_receipt_sha256=sha(args.build/'build-receipt.json'),
        frames_sha256=sha(args.output/'native/frames.jsonl'),material_sha256=sha(material_path),untouched_scenes_consumed=0)
    result['fit_evaluation'] = str(args.fit_evaluation) if args.fit_evaluation is not None else None
    if not receipt['native_process']['reaped']:
        raise ValueError('Native drift worker did not drain')
    write(args.output/'results.json',result)
    print(json.dumps({k:result[k] for k in ['status','kind','completed_post_warmup_steps','elapsed_s']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--kind',required=True,choices=['native-control','full-numerical','neural','before','after','periodic'])
    parser.add_argument('--checkpoint',choices=['C53','C44','C37'],default='C53')
    parser.add_argument('--passes',type=int,choices=range(1,17),default=8)
    parser.add_argument('--interval',type=int,choices=[4,16,64],default=16)
    parser.add_argument('--steps',type=int,choices=[16,128,512,3600],default=512)
    parser.add_argument('--seed',type=int,default=20261031)
    parser.add_argument('--counts',type=int,nargs=3)
    parser.add_argument('--cells-per-metre',type=int)
    parser.add_argument('--fit-evaluation',type=Path)
    parser.add_argument('--cpu',type=int)
    episode(parser.parse_args())
