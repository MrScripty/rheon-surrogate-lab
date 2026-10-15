"""Serial repeated whole-track CPU comparison; numerical failure is retained.

Uses actual forced-water trajectories and the unchanged native comparison engine.
The first AI step warms neural execution; subsequent steps supply timing samples.
This is not a pressure-kernel benchmark or a matched-tolerance solver comparison:
the existing native 1e-8 and learned 1e-6 profiles remain explicitly distinct.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
import rollout


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(output):
    receipt=json.loads((output/'receipt.json').read_text())
    frames=[json.loads(s) for s in (output/'frames.jsonl').read_text().splitlines()]
    frames=[f for f in frames if f.get('ai_quality')]
    callbacks=[json.loads(s) for s in (output/'callbacks.jsonl').read_text().splitlines()] if (output/'callbacks.jsonl').exists() else []
    # Exclude one explicit inference warm-up step from steady-state timing.
    measured=frames[1:];calls=callbacks[1:]
    return {'receipt_sha256':digest(output/'receipt.json'),'result':receipt['result'],
        'status':receipt['status'],'native_process':receipt['native_process'],
        'all_ai_steps_qualified':bool(frames) and all(f['ai_quality']['qualified'] for f in frames),
        'all_reference_steps_qualified':all(f['reference_quality']['qualified'] for f in frames),
        'timing_samples':len(measured),
        'native_seconds':[f['reference_seconds'] for f in measured],
        'ai_seconds':[f['ai_seconds'] for f in measured],
        'callback_seconds':[f['model_correction_seconds'] for f in calls],
        'network_passes':[f['network_passes'] for f in callbacks],
        'worst_relative_residual':receipt['worst_ai_relative_residual'],
        'final_pressure_difference':frames[-1]['relative_pressure_difference'] if frames else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','project','artifacts','binary','adaptive-profile','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cpu',type=int,required=True);args=p.parse_args()
    config=json.loads(args.config.read_text());stopping=rollout.stopping_policy(json.loads(args.adaptive_profile.read_text()))
    if config['schema']!='rheon-pressure-scaling-study-v1' or not 2<=config['repetitions']<=5:
        raise ValueError('bounded repeated scaling configuration required')
    if not 1<=config['reference_proposals']<=8 or not 0<=config['warmup_steps']<config['steps']<=32:
        raise ValueError('bounded trajectory and proposal budget required')
    if not 1<=config['fixed_cells_per_metre']<=128 or not all(type(n) is int and 5<=n<=40 for n in config['axis_counts']):
        raise ValueError('existing supported cubic grid budget required')
    if args.cpu not in os.sched_getaffinity(0):raise ValueError('CPU outside permitted affinity')
    if not 1<=config['timeout_seconds']<=180:raise ValueError('bounded native lifecycle deadline required')
    os.sched_setaffinity(0,{args.cpu});args.output.mkdir(parents=True,exist_ok=False)
    args.project=args.project.absolute();args.artifacts=args.artifacts.absolute();args.binary=args.binary.absolute()
    records=[];started=time.time();physical={}
    for sweep in ('fixed-volume','fixed-spacing'):
        for n in config['axis_counts']:
            density=n if sweep=='fixed-volume' else config['fixed_cells_per_metre']
            scene={k:config[k] for k in ('seed','steps','warmup_steps','center','acceleration','radius','swirl','fluctuation','frequency','sgs')}
            scene.update(counts=[n]*3,cells_per_metre=density)
            key=json.dumps(scene,sort_keys=True)
            if key in physical:
                records.append({'sweep':sweep,'n':n,'same_scene_as':physical[key]});continue
            label=f'{sweep}-{n}';physical[key]=label;root=args.output/label;root.mkdir()
            row={'label':label,'sweep':sweep,'n':n,'scene':scene,'extents_m':[n/density]*3,'cell_size_m':1/density,'runs':[]}
            try:
                control=rollout.run(args.binary,scene,root/'native-control',control=True,timeout=config['timeout_seconds'])
                row['native_control']=control
                if not control['result']['hard_stop'] and control['result']['qualified_ai_steps']==control['result']['ai_steps']:
                    for repetition in range(config['repetitions']):
                        # Alternate order to limit consistent temporal bias.
                        names=('fixed','adaptive') if repetition%2==0 else ('adaptive','fixed')
                        for name in names:
                            out=root/f'{name}-{repetition}'
                            try:
                                rollout.run(args.binary,scene,out,args.project,args.artifacts,timeout=config['timeout_seconds'],
                                    proposals=config['reference_proposals'],stopping=stopping if name=='adaptive' else None)
                            except (RuntimeError,TimeoutError,ValueError) as error:
                                row.setdefault('failures',[]).append({'run':out.name,'error':str(error)})
                            if (out/'receipt.json').exists():row['runs'].append({'policy':name,'repetition':repetition,**summarize(out)})
                else:row['not_run']='native control failed physical gates'
            except (RuntimeError,TimeoutError,ValueError) as error:row['control_failure']=str(error)
            records.append(row)
            for name in ('fixed','adaptive'):
                runs=[r for r in row['runs'] if r['policy']==name]
                native=[t for r in runs for t in r['native_seconds']];ai=[t for r in runs for t in r['ai_seconds']]
                if native and ai:
                    row[name]={'qualified':len(runs)==config['repetitions'] and all(r['all_ai_steps_qualified'] and r['all_reference_steps_qualified'] and r['status']=='completed' and not r['result']['hard_stop'] for r in runs),
                        'native_median_seconds':statistics.median(native),'ai_median_seconds':statistics.median(ai),
                        'ai_to_native_time_ratio':statistics.median(ai)/statistics.median(native),
                        'per_repetition_ai_median_seconds':[statistics.median(r['ai_seconds']) if r['ai_seconds'] else None for r in runs],
                        'mean_network_passes':statistics.mean(t for r in runs for t in r['network_passes']),
                        'timing_samples':len(ai)}
            report={'schema':'rheon-pressure-scaling-report-v1','source_sha256':digest(Path(__file__)),'config_sha256':digest(args.config),
                'binary_sha256':digest(args.binary),'weights_sha256':digest(args.artifacts/'weights.npz'),'configuration':config,
                'source_witness_sha256':digest(args.project/'source-witness.json'),'cpu':args.cpu,'torch_threads':1,
                'started_at_utc':datetime.fromtimestamp(started,timezone.utc).isoformat(),'elapsed_s':time.time()-started,
                'runtime':{'python':sys.executable,'cuda_initialized':False,
                    'checkpoint_runtime':json.loads((args.artifacts/'receipt.json').read_text())['runtime']},'records':records,
                'memory':{'python_process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    'reaped_native_child_peak_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
                    'scope':'process-lifetime high-water marks across all episodes so far; not per-policy allocation or simultaneous total'},
                'scope':'actual short trajectories; whole Engine.advance timing including prepare/projection/IPC; neural callback separately recorded; no isolated native pressure timing',
                'tolerances':{'native_relative_residual':1e-8,'ai_relative_residual':1e-6,'predicted_divergence':1e-7},
                'warmup':'15 native history steps plus one excluded AI timing step per run',
                'isolation':'serial CPU core; run only after managed training/validation and other task CPU workers finish',
                'larger_than_65536_cells':'unsupported; no extrapolated crossover or accuracy claim'}
            (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps({'scene':label,'fixed':row.get('fixed'),'adaptive':row.get('adaptive'),'control_failure':row.get('control_failure')}),flush=True)


if __name__=='__main__':main()
