"""Persistent single-checkpoint CPU workers and balanced serial timing."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import resource
import subprocess
import sys
import time
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha,runtime,load_checkpoint,historical_cases,equation_error

def worker(args):
    os.sched_setaffinity(0,{23})
    started=time.perf_counter()
    identifier,separator,update=args.worker.partition('@')
    state=load_checkpoint(identifier)
    if separator:
        restore=ROOT/state.entry['restore'];receipt=read(restore/'receipt.json');history=read(restore/'history.json')
        name=f'checkpoint-{int(update):03d}.npz';path=restore/name
        if sha(path)!=receipt['outputs'][name]['sha256']:raise ValueError('Restored snapshot changed')
        state.data.helpers().load_weights(state.network,path,history['checkpoints'][name]['inventory'])
    torch=state.torch
    from fresh_pressure_data import cases as fresh_cases
    cases=[(dict(identity,population='historical_small'),sample)for identity,sample in historical_cases(state)]
    cases.extend((dict(identity,population='fresh_'+identity['population']),sample)for identity,sample in fresh_cases(args.bank,state.torch,timing_only=True))
    inventory=state.data.helpers().state_inventory(state.network)
    from whole_pressure_trace import WholePrediction
    from tail_whole_pressure import TailWholePrediction
    from mixed_materialized_tail_pressure import MixedMaterializedTailPrediction
    original=lambda sample:state.model.predict_pressure(state.network,sample,state.normalization)
    predictors={'whole_graph':TailWholePrediction(state),'materialized_graph':MixedMaterializedTailPrediction(state)}
    reference=[]
    with torch.inference_mode():
        for identity,sample in cases:
            prediction=original(sample)
            reference.append(prediction.clone())
        reproduction=[]
        qualification={}
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
    setup=time.perf_counter()-started
    print(json.dumps(dict(kind='ready',checkpoint=args.worker,setup_seconds=setup,qualification=qualification,
                         reproduction=reproduction,compilation_preparation=predictors['whole_graph'].preparation,materialized_preparation=predictors['materialized_graph'].preparation,dense_cache_bytes=predictors['materialized_graph'].cache_bytes,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,actual_inventory=inventory,source_entry=state.entry,source_sha256=sha(__file__))),flush=True)
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
    for identifier in args.checkpoints:
        if identifier!='a-tail8':raise ValueError('Frozen learned-tail benchmark only')
        for directory,source in (('pressure-tail8-qualification-01','tail_whole_pressure.py'),
                                 ('mixed-materialized-pressure-qualification-01','mixed_materialized_tail_pressure.py')):
            qualification=read(BASE/directory/(identifier+'.json'))
            if qualification['status']!='passed'or qualification['execution_sha256']!=sha(HERE/source):
                raise ValueError('Every execution path must qualify before comparative timing')
        if qualification['materialization_sha256']!=sha(HERE/'materialized_fine_mixed.py')or qualification['precise_construction_sha256']!=sha(HERE/'materialized_fine_precise.py'):
            raise ValueError('Qualified learned matrix construction changed')
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
        for checkpoint in args.checkpoints:
            stream=(args.output/(checkpoint+'-stderr.log')).open('xb');streams[checkpoint]=stream
            child=subprocess.Popen([selected['python'],'-B',str(Path(__file__).resolve()),'--worker',checkpoint,'--bank',str(args.bank.resolve())],
                cwd=ROOT,env=environment(selected),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stream,text=True)
            children[checkpoint]=child;ready[checkpoint]=receive(child)
            write(args.output/(checkpoint+'-qualification.json'),ready[checkpoint])
            print('Qualified and warmed',checkpoint,flush=True)
        variants=[(checkpoint,variant)for checkpoint in args.checkpoints for variant in ('whole_graph','materialized_graph')]
        for repeat in range(5):
            offset=(repeat//2)*(len(variants)//2)
            order=variants[offset:]+variants[:offset]
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
            for population in('historical_small','fresh_small','fresh_32cubed_extension'):
                subsets=[[row for row in r['rows']if row['population']==population]for r in group]
                medians=[1000*statistics.median(row['seconds']for row in subset)for subset in subsets]
                summary.append(dict(checkpoint=checkpoint,variant=variant,population=population,repeats=len(group),
                    cases_per_repeat=len(subsets[0]),median_ms=statistics.median(medians),
                    min_repeat_median_ms=min(medians),max_repeat_median_ms=max(medians),
                    worst_equation_relative_residual=max(row['equation_relative_residual']for subset in subsets for row in subset)))
        write(args.output/'summary.json',dict(status='completed',rows=summary,setup_excluded=True,
            warmup='One complete192-request population before every repeat and variant',timed_requests=5*len(variants)*192,
            timing_scope='One complete predict_pressure invocation including runtime contracts; loading, trace setup, sample assembly and independent verification separate',
            CPU=23,threads=1,batch_size=1,source_sha256=sha(__file__),execution_source_sha256=sha(HERE/'mixed_materialized_tail_pressure.py'),reference_execution_source_sha256=sha(HERE/'tail_whole_pressure.py'),parent_execution_source_sha256=sha(HERE/'whole_pressure_trace.py'),materialization_sha256=sha(HERE/'materialized_fine_mixed.py'),precise_construction_sha256=sha(HERE/'materialized_fine_precise.py'),
            consumed_fresh_bank_sha256=sha(args.bank),new_independent_accuracy_claim=False,model_weights_unchanged=True))
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
    parser.add_argument('--worker');parser.add_argument('--checkpoints',nargs='+');parser.add_argument('--output',type=Path)
    parser.add_argument('--bank',type=Path,required=True)
    args=parser.parse_args()
    if args.worker:worker(args)
    else:controller(args)
