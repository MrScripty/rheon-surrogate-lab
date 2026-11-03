"""Five serial application repeats with deferred metrics and explicit controls.

Requires a concrete frozen registry of qualified checkpoint/precision choices.
No registry or model selection is inferred by this executable.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import resource
import select
import signal
import subprocess
import sys
import time
from study_support import ROOT,HERE,BASE,read,write,sha,runtime,load_checkpoint


def validate(args):
    protocol=read(args.protocol)
    if sha(args.registry)!=protocol['registry_sha256']:raise ValueError('Frozen application registry changed')
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Frozen application measurement source changed')
    for relative,digest in protocol['execution_assets_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Frozen native/compiled execution artifact changed')
    banks={mode:ROOT/protocol['banks'][mode]['path']for mode in ('steady','forcing')}
    for mode,path in banks.items():
        if sha(path)!=protocol['banks'][mode]['sha256']:raise ValueError('Frozen application reference population changed')
    registry=read(args.registry);methods=registry['methods']
    if len({m['identifier']for m in methods})!=len(methods)or len(methods)!=7:raise ValueError('Four neural primaries and three native controls required')
    forecasts=[m for m in methods if m['kind']=='forecast']
    if {m['model']['horizons']for m in forecasts}!={1,2,3,4}or len(forecasts)!=4:raise ValueError('All four actual trained horizons required')
    if {m['method']for m in methods if m['kind']=='control'}!={'native-pcg','spectral','C107'}:raise ValueError('All declared numerical/native-pressure controls required')
    return protocol,registry,banks


def worker(args):
    import numpy as np
    import torch
    import temporal_displacement_objective as teaching
    from application_model_support import load_model
    from application_model_evidence import require
    from forecast_application_episode import episode as forecast_episode
    from native_application_episode import episode as native_episode
    from application_episode_metrics import evaluate_forecast,evaluate_native
    from evaluate_coupled import references
    from native_client import FIELDS
    from temporal_train_displacement import inventory
    os.sched_setaffinity(0,{23});torch.set_num_threads(1)
    def stop(*_):raise InterruptedError('Application worker interrupted; reaping active native consumer')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    protocol,registry,banks=validate(args);method=next(m for m in registry['methods']if m['identifier']==args.worker)
    directory=args.output/args.worker;directory.mkdir(parents=True,exist_ok=False)
    setup_start=time.perf_counter();network=None;state=None;norm=read(ROOT/protocol['normalization']['path'])
    if sha(ROOT/protocol['normalization']['path'])!=protocol['normalization']['sha256']:raise ValueError('Original training-only physical normalization required')
    with contextlib.redirect_stdout(sys.stderr):
        if method['kind']=='forecast':
            qualification=require(method,protocol['banks']['steady']['sha256'],protocol['scene_identifiers'])
            network,actual_norm,identity=load_model(method['model'])
            if norm!=actual_norm:raise ValueError('Deployment normalization differs from original physical units')
            identity=dict(identity,qualified_deployment=qualification)
        elif method['method']=='C107':
            state=load_checkpoint('C107');identity=dict(identifier='C107',entry_sha256=sha(ROOT/'runs/pressure-methods-v4/entries/v4-student50-control.json'),role='Native pressure control; original gates unchanged')
        else:identity=dict(identifier=method['identifier'],role='Numerical native control')
    model_setup_s=time.perf_counter()-setup_start
    initial_inventory=inventory(network)if network is not None else inventory(state.network)if state is not None else None
    def forbidden(*args,**kwargs):raise ValueError('Training labels reached application inference')
    teaching.displacement_labels=forbidden
    cases=[]
    for scenario,path in banks.items():
        bank=read(path)
        if bank['status']!='completed':raise ValueError('Completed reference banks required')
        for row in bank['trajectories']:
            original=row.get('original_identifier',row['identifier'])
            if row['split']!='development'or original not in protocol['scene_identifiers']:raise ValueError('Only frozen consumed development cases required')
            receipt_path=ROOT/row['receipt']
            if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Application reference receipt changed')
            receipt=read(receipt_path);spec=receipt['states'][0];path0=receipt_path.parent/spec['path']
            if sha(path0)!=spec['sha256']:raise ValueError('Causal initial state changed')
            with np.load(path0,allow_pickle=False)as a:initial={n:a[n].copy()for n in FIELDS}
            cases.append(dict(identifier=original,scenario=scenario,row=row,scene=receipt['scene'],initial=initial,
                steps=240 if scenario=='steady'else 48))
    if len(cases)!=16:raise ValueError('Eight steady and eight interaction scenes required')
    def execute(case,target,warmup):
        tick=time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            if network is not None:
                result,outputs=forecast_episode(network,norm,case['scene'],case['initial'],target,steps=case['steps'],scenario=case['scenario'])
            else:
                predictor=(lambda sample:state.model.predict_pressure(state.network,sample,state.normalization))if state is not None else None
                result,outputs=native_episode(method['method'],predictor,case['scene'],case['initial'],target,steps=case['steps'],scenario=case['scenario'])
        # No future reference state is loaded until the measured episode closes.
        episode_wall_s=time.perf_counter()-tick
        if warmup:diagnostics=None
        else:
            scene,truth=references(case['row'])
            diagnostics=evaluate_forecast(result,outputs,truth,scene,norm)if network is not None else evaluate_native(result,outputs,truth,scene,norm)
        payload=dict(result,model=identity,identifier=case['identifier'],scenario=case['scenario'],warmup=warmup,
            diagnostics=diagnostics,measured_episode_call_wall_s=episode_wall_s,
            peak_worker_rss_including_deferred_validation_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            max_native_child_rss_so_far_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
            offline_references_excluded_from_episode=True,model_setup_seconds_separate=model_setup_s)
        path=target/'receipt.json';write(path,payload)
        return dict(identifier=case['identifier'],scenario=case['scenario'],counts_xyz=case['scene']['counts'],
            cells_per_metre=case['scene']['cells_per_metre'],requested_steps=case['steps'],physical_steps=result['physical_steps'],
            complete=result['completed_physical_interval'],application_seconds=result['application_seconds'],
            application_seconds_per_physical_second=result['application_seconds_per_physical_second'],
            native_setup_seconds=result['native_setup_seconds'],model_setup_seconds_separate=model_setup_s,
            event_discarded_states=result['event_discarded_states'],endpoint_discarded_states=result['endpoint_discarded_states'],
            accuracy_pass=None if diagnostics is None else diagnostics.get('accuracy_pass',diagnostics.get('diagnostic_accuracy_pass')),
            accuracy_scope='Every consumed forecast state'if network is not None else'Native diagnostic checkpoints',
            receipt=str(path.relative_to(ROOT)),receipt_sha256=sha(path))
    warmup_start=time.perf_counter();warmup=[]
    for case in cases:warmup.append(execute(case,directory/'warmup'/(case['identifier']+'-'+case['scenario']),True))
    def unchanged():
        module=network if network is not None else state.network if state is not None else None
        if(module is not None and inventory(module)!=initial_inventory)or torch.cuda.is_initialized():raise ValueError('Application measurement changed weights or initialized CUDA')
    unchanged()
    print(json.dumps(dict(kind='ready',identifier=args.worker,model=identity,cases=len(cases),model_setup_s=model_setup_s,
        warmup_s=time.perf_counter()-warmup_start,warmup=warmup,threads=1,affinity=[23],cuda_initialized=False,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)),flush=True)
    for raw in sys.stdin.buffer:
        command=json.loads(raw)
        if command['kind']=='stop':break
        if command['kind']!='repeat'or command['repeat']not in range(5):raise ValueError('Finite five-repeat application protocol required')
        repeat=command['repeat'];order=list(range(len(cases)));random.Random(2026102061+repeat).shuffle(order);rows=[]
        for index in order:
            case=cases[index];rows.append(execute(case,directory/f'r{repeat}'/(case['identifier']+'-'+case['scenario']),False))
        unchanged()
        print(json.dumps(dict(kind='repeat',identifier=args.worker,repeat=repeat,rows=rows,threads=1,affinity=[23],
            cuda_initialized=False,model=identity,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)),flush=True)


def controller(args):
    args.output=args.output.resolve();args.protocol=args.protocol.resolve();args.registry=args.registry.resolve()
    if args.output.exists():raise ValueError('Fresh application campaign directory required')
    protocol,registry,_=validate(args)
    if protocol['source_sha256']['benchmark_forecast_application.py']!=sha(__file__):raise ValueError('Registered application benchmark source required')
    args.output.mkdir(parents=True);record=dict(status='running',active=None,completed=[],automatic_retry=False)
    children={};streams={};buffers={};responses=[]
    write(args.output/'execution.json',record)
    def stop(*_):raise InterruptedError('Application campaign interrupted; draining owned workers')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def receive(child):
        deadline=time.monotonic()+600;incoming=buffers[child.pid]
        while True:
            end=incoming.find(b'\n')
            if end>=0:
                line=bytes(incoming[:end+1]);del incoming[:end+1];return json.loads(line)
            if len(incoming)>1024**2:raise ValueError('Application worker response exceeds bound')
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([child.stdout],[],[],remaining)[0]:raise TimeoutError('Same worker response deadline; no automatic retry')
            block=os.read(child.stdout.fileno(),65536)
            if not block:raise EOFError('Application worker exited before complete response')
            incoming.extend(block)
    from common import environment
    selected=runtime()
    try:
        for method in registry['methods']:
            identifier=method['identifier'];stream=(args.output/(identifier+'-stderr.log')).open('xb');streams[identifier]=stream
            command=[selected['python'],'-B',str(Path(__file__).resolve()),'--worker',identifier,'--registry',str(args.registry),
                '--protocol',str(args.protocol),'--output',str(args.output)]
            tick=time.perf_counter();child=subprocess.Popen(command,cwd=ROOT,env=environment(selected),
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stream,bufsize=0)
            children[identifier]=child;buffers[child.pid]=bytearray()
            record['active']=dict(identifier=identifier,phase='model_preparation_and_population_warmup',pid=child.pid);write(args.output/'execution.json',record)
            ready=receive(child)
            if ready['kind']!='ready'or ready['cases']!=16:raise ValueError('Complete application warmup population required')
            ready['worker_launch_to_ready_s']=time.perf_counter()-tick;write(args.output/(identifier+'-ready.json'),ready)
            record['active']=None;write(args.output/'execution.json',record)
            print('Prepared application worker and full population warmup:',identifier,flush=True)
        identifiers=[m['identifier']for m in registry['methods']]
        for repeat in range(5):
            order=identifiers[repeat:]+identifiers[:repeat]
            if repeat%2:order.reverse()
            for identifier in order:
                child=children[identifier];record['active']=dict(identifier=identifier,repeat=repeat,pid=child.pid);write(args.output/'execution.json',record)
                child.stdin.write((json.dumps(dict(kind='repeat',repeat=repeat))+'\n').encode());child.stdin.flush()
                value=receive(child)
                if value['kind']!='repeat'or value['identifier']!=identifier or value['repeat']!=repeat or len(value['rows'])!=16:
                    raise ValueError('Complete declared application repeat required')
                responses.append(value);path=args.output/f'{identifier}-r{repeat}.json';write(path,value)
                record['completed'].append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)));record['active']=None
                write(args.output/'execution.json',record);print('Measured application:',identifier,'repeat',repeat,flush=True)
        write(args.output/'summary.json',dict(status='completed',responses=len(responses),timed_episodes=560,
            full_population_warmups=112,repeats=5,balanced_serial=True,cpu=23,threads=1,batch_size=1,
            registry_sha256=sha(args.registry),protocol_sha256=sha(args.protocol),source_sha256=sha(__file__),
            application_physical_acceptance_claim=False,offline_metrics_excluded=True,reserved_scenes_consumed=False))
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        for identifier,child in children.items():
            if child.poll()is None:
                if record['status']=='completed':
                    try:child.stdin.write(b'{"kind":"stop"}\n');child.stdin.flush();child.wait(timeout=2)
                    except (BrokenPipeError,subprocess.TimeoutExpired):child.terminate()
                else:child.terminate()
                if child.poll()is None:
                    try:child.wait(timeout=8)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
            else:child.wait(timeout=2)
            streams[identifier].close()
        record.update(active=None,workers=[dict(identifier=k,pid=c.pid,returncode=c.returncode,reaped=c.poll()is not None)for k,c in children.items()])
        write(args.output/'execution.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker')
    for name in ('registry','protocol','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();worker(args)if args.worker else controller(args)
