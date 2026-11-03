"""Persistent single-checkpoint CPU workers and balanced serial timing."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha,runtime,load_checkpoint,historical_cases,equation_error

def worker(args):
    os.sched_setaffinity(0,{23})
    started=time.perf_counter()
    state=load_checkpoint(args.worker)
    torch=state.torch
    cases=historical_cases(state)
    inventory=state.data.helpers().state_inventory(state.network)
    from pressure_execution import CachedPrediction
    original=lambda sample:state.model.predict_pressure(state.network,sample,state.normalization)
    predictors={'original':original}
    if args.worker=='C107':
        predictors.update(cached=CachedPrediction(state),buffers=CachedPrediction(state,buffers=True),
                          channels_last=CachedPrediction(state,buffers=True,channels_last=True))
    reference=[]
    with torch.inference_mode():
        for identity,sample in cases:
            prediction=original(sample)
            reference.append(prediction.clone())
        expected=[json.loads(line) for line in (ROOT/'docs/plans/chart-diagnostics/pressure-methods-v4'/state.entry['id']/'cases.jsonl').read_text().splitlines()]
        reproduction=[]
        qualification={}
        for i,(identity,sample) in enumerate(cases):
            error=equation_error(state,reference[i],sample)
            old=expected[i]['raw_learned']['quality']['relative_residual']
            if abs(error-old)>max(1e-9,abs(old)*1e-6):
                raise ValueError(f'Historical checkpoint reproduction differs at {i}: observed={error:.17g}, expected={old:.17g}')
            reproduction.append(dict(**identity,equation_relative_residual=error,historical_error=old,
                absolute_difference=abs(error-old),reproduction_absolute_tolerance=1e-9,reproduction_relative_tolerance=1e-6))
        for name,predict in predictors.items():
            differences=[];errors=[];gauges=[]
            for i,(identity,sample) in enumerate(cases):
                predicted=predict(sample)
                differences.append(float(torch.linalg.vector_norm(predicted-reference[i]))/max(float(torch.linalg.vector_norm(reference[i])),1e-30))
                errors.append(equation_error(state,predicted,sample))
                gauges.append(float(predicted[:,:,:1,:1,:1].abs().max()))
                if not bool(torch.isfinite(predicted).all()):raise ValueError('Execution output nonfinite')
            qualification[name]=dict(cases=len(cases),worst_pressure_relative_difference=max(differences),
                worst_equation_relative_residual=max(errors),maximum_gauge=max(gauges),
                output_tolerance=1e-6,passed=max(differences)<=1e-6 and max(gauges)==0.)
            if not qualification[name]['passed']:
                raise ValueError('Execution output/gauge parity failed: '+name)
            if hasattr(predict,'cache'):
                sample=cases[0][1]
                saved=predict(sample).clone()
                own=predict(sample)
                predict(cases[1][1])
                if not torch.equal(own,saved):raise ValueError('Scratch storage aliases returned pressure')
                misses=predict.misses
                next(predict.network.parameters()).add_(0.)
                predict(sample)
                if predict.misses!=misses+1:raise ValueError('Weight-version cache invalidation failed')
                qualification[name].update(owned_output=True,weight_version_invalidation=True)
    setup=time.perf_counter()-started
    print(json.dumps(dict(kind='ready',checkpoint=args.worker,setup_seconds=setup,qualification=qualification,
                         reproduction=reproduction)),flush=True)
    for line in sys.stdin:
        command=json.loads(line)
        if command['kind']=='stop':break
        name=command['variant'];predict=predictors[name]
        ordering=list(range(len(cases)));random.Random(2026101920+command['repeat']).shuffle(ordering)
        rows=[]
        with torch.inference_mode():
            for _,sample in cases:
                predict(sample)
            for index in ordering:
                identity,sample=cases[index]
                tick=time.perf_counter();pressure=predict(sample);elapsed=time.perf_counter()-tick
                error=equation_error(state,pressure,sample)
                rows.append(dict(**identity,seconds=elapsed,equation_relative_residual=error))
        if inventory!=state.data.helpers().state_inventory(state.network):raise ValueError('Timing changed checkpoint weights')
        print(json.dumps(dict(kind='repeat',checkpoint=args.worker,variant=name,repeat=command['repeat'],
            rows=rows,median_ms=1000*statistics.median(r['seconds']for r in rows),
            worst_equation_relative_residual=max(r['equation_relative_residual']for r in rows),
            cuda_initialized=torch.cuda.is_initialized(),threads=torch.get_num_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)))),flush=True)

def controller(args):
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh timing output required')
    args.output.mkdir(parents=True)
    from common import environment
    selected=runtime();children={};streams={};ready={};results=[]
    record=dict(status='running',balanced_serial=True,concurrent_numerical_workers=False,active=None,completed=[])
    def receive(child):
        line=child.stdout.readline()
        if not line:raise RuntimeError('Timing worker exited unexpectedly; inspect retained stderr')
        return json.loads(line)
    try:
        for checkpoint in ('C106','C107'):
            stream=(args.output/(checkpoint+'-stderr.log')).open('xb');streams[checkpoint]=stream
            child=subprocess.Popen([selected['python'],'-B',str(Path(__file__).resolve()),'--worker',checkpoint],
                cwd=ROOT,env=environment(selected),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stream,text=True)
            children[checkpoint]=child;ready[checkpoint]=receive(child)
            write(args.output/(checkpoint+'-qualification.json'),ready[checkpoint])
            print('Qualified and warmed',checkpoint,flush=True)
        variants=[('C106','original'),('C107','original'),('C107','cached'),('C107','buffers'),('C107','channels_last')]
        for repeat in range(5):
            order=variants[repeat:]+variants[:repeat]
            if repeat%2:order=list(reversed(order))
            for checkpoint,variant in order:
                record['active']=dict(checkpoint=checkpoint,variant=variant,repeat=repeat,pid=children[checkpoint].pid)
                write(args.output/'execution.json',record)
                children[checkpoint].stdin.write(json.dumps(dict(kind='time',variant=variant,repeat=repeat))+'\n')
                children[checkpoint].stdin.flush()
                result=receive(children[checkpoint]);results.append(result)
                path=args.output/f'{checkpoint}-{variant}-r{repeat}.json';write(path,result)
                record['completed'].append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                record['active']=None;write(args.output/'execution.json',record)
                print(checkpoint,variant,'repeat',repeat,'median_ms',round(result['median_ms'],5),flush=True)
        summary=[]
        for checkpoint,variant in variants:
            group=[r for r in results if r['checkpoint']==checkpoint and r['variant']==variant]
            summary.append(dict(checkpoint=checkpoint,variant=variant,repeats=len(group),
                median_ms=statistics.median(r['median_ms']for r in group),
                min_repeat_median_ms=min(r['median_ms']for r in group),max_repeat_median_ms=max(r['median_ms']for r in group),
                worst_equation_relative_residual=max(r['worst_equation_relative_residual']for r in group)))
        write(args.output/'summary.json',dict(status='completed',rows=summary,setup_excluded=True,
            warmup='One complete population per variant before each repeat, plus output qualification',timed_requests=5*5*128,
            timing_scope='One complete predict_pressure invocation; no setup/sample assembly/error verification',
            cpu=23,threads=1,batch_size=1,source_sha256=sha(__file__),execution_source_sha256=sha(HERE/'pressure_execution.py')))
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        for checkpoint,child in children.items():
            if child.poll()is None:
                try:child.stdin.write(json.dumps(dict(kind='stop'))+'\n');child.stdin.flush();child.wait(timeout=10)
                except BaseException:child.terminate();child.wait(timeout=10)
            streams[checkpoint].close()
        record.update(active=None,owned_children=[dict(checkpoint=k,pid=p.pid,returncode=p.returncode,reaped=p.poll()is not None)for k,p in children.items()])
        write(args.output/'execution.json',record)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['C106','C107']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args)
    else:controller(args)
