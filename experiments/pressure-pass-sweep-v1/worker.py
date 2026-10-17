"""One CPU checkpoint: matched snapshot sweep, then independent fixed-pass tracks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import statistics
import sys
import time
from material import read,sha
import inference


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,allow_nan=False));temporary.replace(path)


def snapshots(state,material,output,fixture=False):
    root=Path(material['root']);inputs=root/material['config']['validation_inputs'];torch=state.torch
    for name,expected in material['validation_sources'].items():
        if sha(inputs/name)!=expected:raise ValueError('Common validation inputs changed')
    manifest=read(inputs/'validation_splits.json')
    _,parents=state.data.load(tuple((inputs/name).read_bytes() for name in ['validation_dataset.npz','validation_dataset_extra.npz']),manifest)
    cases=[(parent,slot,dict(trajectory_id=parent['entry']['trajectory_id'],native_frame=frame['index'],
        counts_xyz=parent['counts_xyz'],cell_size_m=parent['cell_size_m'])) for parent in parents for slot,frame in enumerate(parent['frames'])]
    expected=sorted((c['trajectory_id'],c.get('native_frame',c.get('frame')),tuple(c['counts_xyz']),c['cell_size_m'])for c in material['validation_identities'])
    actual=sorted((c['trajectory_id'],c['native_frame'],tuple(c['counts_xyz']),c['cell_size_m'])for _,_,c in cases)
    if expected!=actual or len(cases)!=128:raise ValueError('Common 128-case membership differs')
    passes=[1,2]if fixture else material['config']['passes'];repetitions=1 if fixture else material['config']['timing_repetitions']
    rows=[]
    with torch.no_grad():
        # Exclude initial module/kernel warm-up from the measured prefix costs.
        sample=state.data.sample(cases[0][0],cases[0][1]);inference.fixed(state,sample,2)
        if fixture:
            width=(state.config.get('correction')or{}).get('steps',1)
            native=state.model.predict_pressure(state.network,sample,state.normalization)
            repeated=inference.fixed(state,sample,width)
            if not torch.equal(native,repeated):raise ValueError('Single-pass recurrence changes captured original proposal')
        for index,(parent,slot,identity)in enumerate(cases[:1]if fixture else cases):
            sample=state.data.sample(parent,slot);measurements={n:[] for n in passes}
            if material['config'].get('warmup_each_case',False):inference.fixed(state,sample,1)
            for repetition in range(repetitions):
                current=sample['base'];elapsed=0.;calls=[];hook=state.network.register_forward_hook(lambda *_:calls.append(1))
                try:
                    for n in range(1,max(passes)+1):
                        start=time.perf_counter();current=inference.step(state,sample,current);elapsed+=time.perf_counter()-start
                        if not bool(torch.isfinite(current).all()):raise ValueError('Nonfinite fixed-pass snapshot')
                        if n in passes:
                            start=time.perf_counter()
                            residual=sample['rhs']-state.model.full_operator(current,sample['cell_size_m'],sample['rho'])
                            relative=float(torch.linalg.vector_norm(residual))/max(float(torch.linalg.vector_norm(sample['rhs'])),1e-30)
                            quality_s=time.perf_counter()-start
                            measurements[n].append(dict(relative_residual=relative,inference_s=elapsed+quality_s,network_calls=len(calls)))
                finally:hook.remove()
            for n,values in measurements.items():
                rows.append(dict(**identity,passes=n,relative_residual=values[-1]['relative_residual'],
                    inference_s=statistics.median(v['inference_s']for v in values),timing_samples_s=[v['inference_s']for v in values],
                    network_calls=values[-1]['network_calls']))
            if index%16==0:write(output/'progress.json',dict(phase='snapshots',completed=index+1,total=len(cases)))
    write(output/'snapshots.json',dict(status='passed',cases=1 if fixture else 128,rows=rows,
        precision='float32 neural network; original float64 physical reconstruction',cuda_initialized=torch.cuda.is_initialized()))


def trajectories(state,material,output):
    root=Path(material['root']);lab=root/material['config']['source_lab']
    sys.path.insert(0,str(lab/'experiments/water-residual-learning-v1'))
    import rollout
    source=Path(__file__).with_name('model_adapter.py')
    for n in material['config']['passes']:
        folder=output/f'passes-{n:02d}'
        if (folder/'trajectory.json').exists():
            retained=read(folder/'trajectory.json')
            if retained['id']!=state.identifier or retained['passes']!=n or not retained['native_reaped']:
                raise ValueError('Invalid retained trajectory on resume')
            continue
        if folder.exists():
            archive=output/'interrupted';archive.mkdir(exist_ok=True)
            folder.rename(archive/(folder.name+'-'+str(time.time_ns())))
        folder.mkdir()
        shutil.copyfile(source,folder/'model_adapter.py')
        manifest=folder/'manifest.json';write(manifest,dict(schema='rheon-pressure-pass-sweep-adapter-v1',material=str(output.parent/'material.json'),id=state.identifier,passes=n))
        write(output/'progress.json',dict(phase='trajectory',passes=n,total_pass_settings=len(material['config']['passes'])))
        receipt=rollout.run(root/material['config']['binary'],material['config']['scene'],folder/'native',
            adapter_manifest=manifest,threads=1,timeout=material['config']['native_timeout_seconds'])
        if not receipt['native_process']['reaped']:raise ValueError('Native process not reaped')
        frames=[json.loads(line)for line in(folder/'native/frames.jsonl').read_text().splitlines()]
        ai=[f for f in frames if f.get('ai_quality')]
        callbacks=[json.loads(line)for line in(folder/'native/callbacks.jsonl').read_text().splitlines()]
        attempts_path=folder/'native/rejected-attempts.jsonl'
        attempts=[json.loads(line)['attempt']for line in attempts_path.read_text().splitlines()]if attempts_path.exists()else[]
        if (not ai and not attempts)or not all(f['reference_quality']['qualified'] for f in frames):
            raise ValueError('Missing measured prediction or unqualified reference')
        # The existing bounded magnitude owner keeps same-request and trajectory errors distinct.
        sys.path.insert(0,str(root/'docs/plans'));from drift_pressure_metrics import pressure_magnitudes
        points=[]
        for f in ai:
            magnitudes,_=pressure_magnitudes(f,material['config']['scene']['counts'])
            points.append([f['step'],f['time_s'],f['relative_pressure_difference'],f['ai_quality']['qualified'],f['ai_quality']['relative_residual'],magnitudes])
        write(folder/'trajectory.json',dict(id=state.identifier,passes=n,points=points,
            completed=bool(ai)and ai[-1]['step']==material['config']['scene']['steps'] and not receipt['result']['hard_stop'],
            qualified_steps=sum(p[3]for p in points),recorded_ai_steps=len(points),
            termination='complete'if not receipt['result']['hard_stop']else'pressure_rejection',
            mean_actual_network_calls=statistics.mean(c['network_passes']for c in callbacks),
            median_ai_step_s=statistics.median(f['ai_seconds']for f in ai)if ai else None,
            median_reference_step_s=statistics.median(f['reference_seconds']for f in ai)if ai else None,
            rejected=[dict(step=a['generation'],time_s=a['interval_end_s'],equation_relative_residual=a['outcome'].get('relative_residual'),
                same_request_pressure_comparison=a.get('same_request_pressure_comparison'),history_transition=a['history_transition'])for a in attempts],
            native_reaped=True,elapsed_s=receipt['elapsed_s']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--material',type=Path,required=True)
    parser.add_argument('--id',required=True);parser.add_argument('--cpu',type=int,required=True);parser.add_argument('--fixture',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();material=read(args.material);output=args.material.parent/(args.id+'-fixture'if args.fixture else args.id);output.mkdir(exist_ok=args.resume)
    def stop(*_):raise InterruptedError('CPU pass sweep cancelled/deadline reached')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(material['config']['worker_timeout_seconds'])
    os.sched_setaffinity(0,{args.cpu});started=time.perf_counter();receipt=dict(status='running',id=args.id,cpu=args.cpu,fixture=args.fixture,command=sys.argv,source_sha256=sha(__file__))
    try:
        state=inference.load(material,args.id)
        if args.resume and (output/'snapshots.json').exists():
            previous=read(output/'snapshots.json')
            if previous['status']!='passed' or previous['cases']!=128 or len(previous['rows'])!=128*len(material['config']['passes']):
                raise ValueError('Incomplete retained snapshots on resume')
        else:
            snapshots(state,material,output,args.fixture)
        if not args.fixture:trajectories(state,material,output)
        receipt.update(status='passed',cuda_initialized=state.torch.cuda.is_initialized(),torch_threads=state.torch.get_num_threads())
    except BaseException as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        signal.alarm(0);receipt.update(elapsed_s=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024);write(output/'receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
