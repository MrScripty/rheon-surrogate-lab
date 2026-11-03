"""Run a reserved stronger trajectory and retain complete multi-field drift curves."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def episode(args):
    if args.output.exists():raise ValueError('Fresh drift episode required')
    os.sched_setaffinity(0,{args.cpu})
    material_path=ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json'
    material=read(material_path)
    sys.path[:0]=[str(ROOT/'experiments/pressure-pass-sweep-v1'),
        str(ROOT/material['config']['source_lab']/'experiments/water-residual-learning-v1')]
    import rollout
    build=read(args.build/'build-receipt.json');binary=args.build/'rheon-drift-rollout'
    if build['status']!='qualified_build' or sha(binary)!=build['binary_sha256']:
        raise ValueError('Qualified longer-horizon diagnostic required')
    reservation_path=args.reservation.resolve()
    reservation=read(reservation_path)
    row=next(r for r in reservation['scenes'] if r['id']==args.scene)
    if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Reserved scene changed')
    scene=read(ROOT/row['path'])
    if args.steps:
        if args.steps>scene['steps']:raise ValueError('Cannot extend beyond reserved physical horizon')
        scene=dict(scene,steps=args.steps)
    args.output.mkdir(parents=True)
    manifest=None
    if args.kind!='native':
        shutil.copyfile(HERE/'cost_drift_adapter.py',args.output/'model_adapter.py')
        specification=dict(schema='rheon-pressure-methods-v2-cost-drift',root=str(ROOT),material=str(material_path),
            adapter_sha256=sha(args.output/'model_adapter.py'),kind=args.kind,passes=args.passes,interval=args.interval,
            warmup_steps=scene['warmup_steps'],evaluation=str(args.evaluation) if args.evaluation else None,
            evaluation_sha256=sha(args.evaluation) if args.evaluation else None)
        manifest=args.output/'manifest.json';write(manifest,specification)
    started=time.perf_counter()
    receipt=rollout.run(binary,scene,args.output/'native',adapter_manifest=manifest,
        control=args.kind=='native',timeout=args.timeout,threads=1)
    frames=[json.loads(line) for line in (args.output/'native/frames.jsonl').read_text().splitlines()]
    points=[]
    keys=['velocity_trajectory_relative_l2','tracer_trajectory_relative_l2',
          'velocity_trajectory_max_abs_m_s','tracer_trajectory_max_abs']
    for frame in frames:
        if not frame['reference_quality']['qualified']:raise ValueError('Native reference failed physical qualification')
        if frame['step']<=scene['warmup_steps']:continue
        if not frame['ai_quality']['qualified']:raise ValueError('Rejected frame advanced into drift history')
        point=dict(step=frame['step'],time_s=frame['time_s'],pressure_trajectory_relative_l2=frame['relative_pressure_difference'],
            equation_residual=frame['ai_quality']['relative_residual'],**{k:frame[k] for k in keys})
        if any(not math.isfinite(point[k]) for k in keys):raise ValueError('Nonfinite trajectory observations')
        points.append(point)
    callbacks_path=args.output/'native/callbacks.jsonl'
    callbacks=[json.loads(line) for line in callbacks_path.read_text().splitlines()] if callbacks_path.exists() else []
    crossings={}
    for field in ['pressure_trajectory_relative_l2','velocity_trajectory_relative_l2','tracer_trajectory_relative_l2']:
        crossings[field]={str(level):next((dict(step=p['step'],time_s=p['time_s'],value=p[field]) for p in points if p[field]>level),None)
            for level in reservation['relative_l2_levels']}
    complete=bool(points) and points[-1]['step']==scene['steps'] and not receipt['result']['hard_stop']
    result=dict(status='completed' if complete else 'pressure_rejection',scene=scene,kind=args.kind,
        passes=args.passes,interval=args.interval,points=points,first_crossings=crossings,
        completed_post_warmup_steps=len(points),network_calls=sum(c['network_passes'] for c in callbacks),
        numerical_corrections=sum(t.get('numerical_corrections',0) for c in callbacks for t in c['trace']),
        diagnostic_levels=reservation['relative_l2_levels'],absolute_limits=reservation['absolute_limits'],
        absolute_first_crossings={key:next((p for p in points if p[key]>1e-5),None)
            for key in ['velocity_trajectory_max_abs_m_s','tracer_trajectory_max_abs']},
        elapsed_s=time.perf_counter()-started,native_process=receipt['native_process'],
        timing_scope='Dual-track diagnostic, not single-track performance',source_sha256=sha(__file__),
        adapter_sha256=sha(HERE/'cost_drift_adapter.py'),reservation_sha256=sha(reservation_path),
        build_receipt_sha256=sha(args.build/'build-receipt.json'),frames_sha256=sha(args.output/'native/frames.jsonl'),
        evaluation=str(args.evaluation) if args.evaluation else None)
    if not result['native_process']['reaped']:raise ValueError('Native worker not drained')
    write(args.output/'results.json',result)
    print(json.dumps(dict(status=result['status'],kind=args.kind,steps=len(points))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene',type=int,required=True)
    parser.add_argument('--reservation',type=Path,default=ROOT/'docs/plans/pressure-methods-v2/drift-reservation.json')
    parser.add_argument('--kind',choices=['cost-hybrid'],required=True)
    parser.add_argument('--passes',type=int,choices=range(1,17),default=8)
    parser.add_argument('--interval',type=int,choices=[4,16,64],default=16)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--cpu',type=int,default=20)
    parser.add_argument('--timeout',type=int,default=1800)
    args=parser.parse_args()
    for name in ['build','output','evaluation']:
        if getattr(args,name):setattr(args,name,getattr(args,name).resolve())
    episode(args)
