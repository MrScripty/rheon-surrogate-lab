"""Balanced CPU block timing, including encoding and every reconstructed future state."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import select
import signal
import sys
import time
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha,runtime


def aggregate(rows,steps):
    result=[]
    for regime in ('small','32cubed'):
        group=[r['seconds']for r in rows if ('small'if max(r['counts_xyz'])<=16 else'32cubed')==regime]
        ordered=sorted(group)
        result.append(dict(regime=regime,cases=len(group),median_block_ms=statistics.median(group)*1000,
            p95_block_ms=ordered[math.ceil(.95*len(ordered))-1]*1000,maximum_block_ms=max(group)*1000,
            median_ms_per_generated_step=statistics.median(group)*1000/steps,
            model_seconds_per_physical_second=statistics.median(group)*60/steps))
    return result


def worker(args):
    import torch
    import numpy as np
    from fixed_viscosity_evaluation_support import load_model
    from fast_preparation_model_support import wrap
    from fast_neural_state_codec import packed_state
    from fused_force_construction import auxiliary
    os.sched_setaffinity(0,{23});torch.set_num_threads(1);tick=time.perf_counter()
    identifier=args.worker.removesuffix('-fast');pair=BASE/(identifier+'-pair-01')
    with contextlib.redirect_stdout(sys.stderr):
        network,norm,identity=load_model(pair)
        project=ROOT/read(pair/'receipt.json')['source_project']
        network,norm,identity=wrap(network,norm,identity,project/'expert-assets')
    k=network.horizons
    cases=[];bank=read(args.bank)
    for row in bank['trajectories']:
        if row['split']!='development':raise ValueError('Only consumed development scenes for timing selection')
        path=ROOT/row['receipt']
        if sha(path)!=row['receipt_sha256']:raise ValueError('Native receipt changed')
        receipt=read(path);scene=receipt['scene']
        for step in (0,8,16,32,48,64,96,120):
            spec=receipt['states'][step];source=path.parent/spec['path']
            if sha(source)!=spec['sha256']:raise ValueError('Native current state changed')
            with np.load(source,allow_pickle=False)as arrays:
                fields={name:arrays[name].copy()for name in ('pressure','vx','vy','vz','tracer')}
            with torch.inference_mode():
                state=packed_state(fields,tuple(scene['counts']),norm['scales'])
                features=auxiliary(scene,step,k,norm['scales'])
            cases.append(dict(identifier=row['identifier'],step=step,scene=scene,fields=fields,state=state,features=features))
    def inventory():
        return {name:hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()for name,value in network.state_dict().items()}
    initial_inventory=inventory()
    def predict(case,variant):
        if variant=='tensor':return network(case['state'],case['features'])
        if variant=='full':return network.predict(case['fields'],case['scene'],case['step'],norm)
        if not variant.startswith('sequential')or k!=1:raise ValueError('Unknown temporal timing scope')
        count=int(variant.removeprefix('sequential'));current=case['fields'];outputs=[]
        for offset in range(count):
            generated=network.predict(current,case['scene'],case['step']+offset,norm)
            current=generated[-1];outputs.extend(generated)
        return outputs
    with torch.inference_mode():
        for case in cases:
            values=predict(case,'full')
            if len(values)!=k or any(not bool(torch.isfinite(v).all())for state in values for v in state.values()):
                raise ValueError('Full future output contract failed before timing')
    print(json.dumps(dict(kind='ready',model=identity,cases=len(cases),setup_s=time.perf_counter()-tick,
        full_scope='Current physical-state encoding, prescribed forcing, one joint network and physical reconstruction of every K output',
        tensor_scope='Pre-encoded tensors through joint network; excludes physical encoding and field reconstruction',
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)),flush=True)
    for line in sys.stdin:
        command=json.loads(line)
        if command['kind']=='stop':break
        variant=command['variant'];repeat=command['repeat'];rows=[]
        steps=int(variant.removeprefix('sequential'))if variant.startswith('sequential')else k
        ordering=list(range(len(cases)));random.Random(2026101950+repeat).shuffle(ordering)
        with torch.inference_mode():
            for case in cases:predict(case,variant)
            for index in ordering:
                case=cases[index];started=time.perf_counter();value=predict(case,variant);elapsed=time.perf_counter()-started
                rows.append(dict(identifier=case['identifier'],step=case['step'],counts_xyz=case['scene']['counts'],
                    cells_per_metre=case['scene']['cells_per_metre'],seconds=elapsed,generated_steps=steps))
        if inventory()!=initial_inventory or torch.cuda.is_initialized():raise ValueError('CPU timing changed weights or initialized CUDA')
        print(json.dumps(dict(kind='repeat',identifier=args.worker,horizons=steps,model_horizons=k,variant=variant,repeat=repeat,rows=rows,summaries=aggregate(rows,steps),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,threads=torch.get_num_threads(),
            affinity=sorted(os.sched_getaffinity(0)),cuda_initialized=False)),flush=True)


def controller(args):
    args.output=args.output.resolve();args.bank=args.bank.resolve()
    if args.output.exists():raise ValueError('Fresh temporal timing directory required')
    batch=read(BASE/'fixed-viscosity-training-batch-01/execution.json')
    if batch['status']!='completed'or not batch['owned_child_reaped']:
        raise ValueError('Complete fits/restores/evaluations and child reaping required')
    for k in (1,2,3,4):
        result=read(BASE/f'fixed-viscosity-execution-qualification-k{k}-01.json')
        if result['status']!='passed'or not result['all_physical_bytes_exact']or result['outputs']!=(121*k+240)*8:
            raise ValueError('Complete exact physical execution proof required for every model')
    registration_path=DOCS/'fixed-viscosity-timing-registration-01.json'
    registration=read(registration_path)
    if registration['source_sha256']!=sha(__file__) or registration['bank_sha256']!=sha(args.bank):
        raise ValueError('Registered timing source or population changed')
    for name,digest in registration['loader_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered model loader changed')
    for relative,digest in registration['execution_assets_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Registered execution asset changed')
    def stop(*_):raise InterruptedError('Timing interrupted; draining owned workers')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    args.output.mkdir(parents=True);children={};streams={};results=[];record=dict(status='running',active=None,completed=[])
    from common import environment
    selected=runtime()
    def receive(child):
        if not select.select([child.stdout],[],[],600)[0]:raise TimeoutError('Bounded worker response deadline; no automatic retry')
        line=child.stdout.readline()
        if not line:raise RuntimeError('Temporal timing worker exited; inspect retained stderr')
        return json.loads(line)
    try:
        identifiers=tuple(registration['models'])
        for k in identifiers:
            stream=(args.output/f'{k}-stderr.log').open('xb');streams[k]=stream
            child=subprocess.Popen([selected['python'],'-B',str(Path(__file__).resolve()),'--worker',str(k),'--bank',str(args.bank)],
                cwd=ROOT,env=environment(selected),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stream,text=True)
            children[k]=child;ready=receive(child);write(args.output/f'{k}-qualification.json',ready)
        variants=[(identifier,variant)for identifier in identifiers for variant in ('full','tensor')]
        variants.extend(('b1-fixed-viscosity-fast',f'sequential{k}')for k in (2,3,4))
        for repeat in range(5):
            offset=(repeat//2)*(len(variants)//2);order=variants[offset:]+variants[:offset]
            if repeat%2:order.reverse()
            for k,variant in order:
                record['active']=dict(identifier=k,variant=variant,repeat=repeat,pid=children[k].pid);write(args.output/'execution.json',record)
                children[k].stdin.write(json.dumps(dict(kind='time',variant=variant,repeat=repeat))+'\n');children[k].stdin.flush()
                result=receive(children[k]);results.append(result);path=args.output/f'{k}-{variant}-r{repeat}.json';write(path,result)
                record['completed'].append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)));record['active']=None
                write(args.output/'execution.json',record)
                print('Timed temporal',k,variant,'repeat',repeat,flush=True)
        summary=[]
        for k,variant in variants:
            group=[r for r in results if r['identifier']==k and r['variant']==variant]
            for regime in ('small','32cubed'):
                values=[next(s for s in r['summaries']if s['regime']==regime)for r in group]
                summary.append(dict(identifier=k,horizons=group[0]['horizons'],variant=variant,regime=regime,repeats=5,
                    median_block_ms=statistics.median(v['median_block_ms']for v in values),
                    median_ms_per_generated_step=statistics.median(v['median_ms_per_generated_step']for v in values),
                    min_repeat_median_ms=min(v['median_block_ms']for v in values),max_repeat_median_ms=max(v['median_block_ms']for v in values),
                    p95_block_ms=statistics.median(v['p95_block_ms']for v in values),maximum_block_ms=max(v['maximum_block_ms']for v in values)))
        write(args.output/'summary.json',dict(status='completed',rows=summary,source_sha256=sha(__file__),
            bank_sha256=sha(args.bank),cpu=23,threads=1,batch_size=1,balanced_serial=True,
            warmup='One full population before every variant and repeat',setup_reported_separately=True,
            application_timing_scope='Separate native integration and event episodes required; these are complete model blocks',
            registration_sha256=sha(registration_path),accuracy_targets_unchanged=True,accuracy_failure_does_not_prevent_cost_measurement=True,timed_requests=5*len(variants)*64))
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        for k,child in children.items():
            if child.poll()is None:
                try:child.stdin.write(json.dumps(dict(kind='stop'))+'\n');child.stdin.flush();child.wait(timeout=10)
                except BaseException:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
            streams[k].close()
        record.update(active=None,workers=[dict(identifier=k,pid=p.pid,returncode=p.returncode,reaped=p.poll()is not None)for k,p in children.items()])
        write(args.output/'execution.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',choices=[f'b{k}-fixed-viscosity-fast'for k in (1,2,3,4)])
    parser.add_argument('--bank',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args)
    else:controller(args)
