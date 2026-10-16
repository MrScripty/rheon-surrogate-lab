"""Fresh-process CPU trajectory qualification with explicit checkpoint populations.

Independent jobs may use distinct cores for throughput. Only workers=1 yields
isolated serial repetitions suitable for comparing elapsed timings. Each native
child retains the existing rollout owner's deadline and termination receipt.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED,ThreadPoolExecutor,wait
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-inference-budget-v1'))
from scaling import summarize
sys.path.insert(0,str(LAB/'experiments/live-project-v1'))
from observer import Observer


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_partitioned(tasks,workers,execute,consume):
    """Stream completed observations while one in-flight job owns each core group."""
    partitions=[iter(tasks[index::workers]) for index in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={}
        def submit(index):
            task=next(partitions[index],None)
            if task is not None:pending[pool.submit(execute,task)]=index
        for index in range(workers):submit(index)
        while pending:
            completed,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                index=pending.pop(future)
                consume(future.result())
                submit(index)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=23)
    parser.add_argument('--threads',type=int,choices=(1,4,8),default=1)
    parser.add_argument('--timing-study',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if set(config)!={'schema','binary','profile','models','scenes','repetitions'} or config['schema']!='rheon-multigrid-trajectory-study-v1':
        raise ValueError('closed study configuration required')
    if type(config['repetitions']) is not int or not 1<=config['repetitions']<=3:raise ValueError('one to three repetitions required')
    if not config['models'] or not config['scenes'] or len(config['models'])>4 or len(config['scenes'])>10:raise ValueError('bounded nonempty study required')
    allowed=sorted(os.sched_getaffinity(0));cpus=allowed[1:] or allowed
    partitions=[cpus[i:i+args.threads] for i in range(0,len(cpus),args.threads) if len(cpus[i:i+args.threads])==args.threads]
    if not 1<=args.workers<=len(partitions):raise ValueError('workers and threads must fit disjoint available core partitions')
    if args.timing_study and (args.workers!=1 or config['repetitions']<2):raise ValueError('timing requires one worker and repeated observations')
    args.output.mkdir(parents=True,exist_ok=False);started=time.time();commands=[];models=[]
    for model in config['models']:
        if set(model)!={'name','project','artifacts','single_evaluation'} or type(model['single_evaluation']) is not bool or not model['name'].replace('-','').isalnum():
            raise ValueError('named explicit model and proposal policy required')
        project=LAB/model['project'];artifacts=LAB/model['artifacts']
        width=1 if model['single_evaluation'] else json.loads((project/'config.json').read_text())['correction']['steps']
        models.append({**model,'proposals':32//width,'weights_sha256':digest(artifacts/'weights.npz'),
            'normalization_sha256':digest(artifacts/'normalization.json'),'source_witness_sha256':digest(project/'source-witness.json')})
    if len({m['name'] for m in models})!=len(models):raise ValueError('unique model names required')
    rollout=LAB/'experiments/water-residual-learning-v1/rollout.py'
    scenes=[]
    for index,scene in enumerate(config['scenes']):
        directory=args.output/f'scene-{index:02d}';directory.mkdir();path=directory/'scene.json'
        path.write_text(json.dumps(scene,indent=2)+'\n');scenes.append((index,path))
    def execute(item):
        index,path,name,model,repetition,cpu=item;output=path.parent/f'{name}-{repetition}'
        command=['taskset','-c',','.join(map(str,cpu)),sys.executable,str(rollout),'--binary',str(LAB/config['binary']),
            '--scene',str(path.absolute()),'--output',str(output.absolute()),'--threads',str(args.threads)]
        if model is None:command+=['--reference-control']
        else:
            command+=['--project',str(LAB/model['project']),'--artifacts',str(LAB/model['artifacts']),
                '--proposals',str(model['proposals']),'--adaptive-profile',str(LAB/config['profile'])]
            if model['single_evaluation']:command+=['--single-evaluation']
        # The controller alone writes the command ledger after futures finish;
        # each task owns its immutable command/log/output identities.
        (path.parent/f'{name}-{repetition}-command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (path.parent/f'{name}-{repetition}.log').open('xb') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait()
            except BaseException:
                child.send_signal(__import__('signal').SIGINT)
                try:child.wait(timeout=130)
                except subprocess.TimeoutExpired:
                    child.terminate();child.wait(timeout=5)
                raise
        result={'scene':index,'model':name,'repetition':repetition,'cpu':cpu,'command':command,'returncode':code,
            'controller_child':{'pid':child.pid,'reaped':child.poll() is not None}}
        if (output/'receipt.json').exists():result.update(summarize(output))
        else:result['failure']='no complete rollout receipt; inspect retained log'
        return result
    records=[];observer=Observer(args.output/'live')
    observer.publish('CPU trajectory study','Native controls before matched checkpoint episodes')
    def save():
        aggregate=[]
        for index,_ in scenes:
            for model in models:
                rows=[r for r in records if r['scene']==index and r['model']==model['name']]
                timings=[t for r in rows for t in r.get('ai_seconds',[])]
                passes=[n for r in rows for n in r.get('network_passes',[])]
                aggregate.append({'scene':index,'model':model['name'],'runs':len(rows),
                    'qualified':len(rows)==config['repetitions'] and all(r['returncode']==0 and r.get('status')=='completed'
                        and not r['result']['hard_stop'] and r['all_ai_steps_qualified'] and r['all_reference_steps_qualified'] for r in rows),
                    'mean_network_evaluations':statistics.mean(passes) if passes else None,
                    'maximum_network_evaluations':max(passes) if passes else None,
                    'worst_relative_residual':max((r.get('worst_relative_residual',0) for r in rows),default=None),
                    'ai_median_seconds':statistics.median(timings) if timings else None,
                    'native_median_seconds':statistics.median(t for r in rows for t in r.get('native_seconds',[])) if timings else None,
                    'per_repetition_ai_median_seconds':[statistics.median(r['ai_seconds']) if r.get('ai_seconds') else None for r in rows]})
        report={'schema':'rheon-multigrid-trajectory-report-v1','configuration':config,'models':models,'records':records,'aggregate':aggregate,
            'source_sha256':{str(p.relative_to(LAB)):digest(p) for p in (Path(__file__).resolve(),rollout,LAB/'experiments/water-stability-learning-v1/learned_iteration.py')},
            'config_sha256':digest(args.config),'binary_sha256':digest(LAB/config['binary']),
            'workers':args.workers,'threads_per_worker':args.threads,'available_cpus':allowed,'used_cpus':sorted({c for r in records for c in r['cpu']}),
            'python':sys.executable,'started_at_utc':datetime.fromtimestamp(started,timezone.utc).isoformat(),'elapsed_s':time.time()-started,
            'scope':'development whole-track timing; caller must exclude competing work; no native refinement' if args.timing_study else 'development correctness/throughput; no isolated latency claim'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (args.output/'commands.json').write_text(json.dumps([r['command'] for r in records],indent=2)+'\n')
    controls=[(i,p,'native',None,0,partitions[j%args.workers]) for j,(i,p) in enumerate(scenes)]
    def consume(row):
        records.append(row);save()
        observer.publish('CPU trajectory study',f"{row['model']}, scene {row['scene']}, repetition {row['repetition']+1}; qualified steps {(row.get('result') or {}).get('qualified_ai_steps','?')}",len(records))
        print(json.dumps({k:row.get(k) for k in ('scene','model','returncode','result')}),flush=True)
    try:execute_partitioned(controls,args.workers,execute,consume)
    except BaseException:
        observer.publish('CPU native controls failed','Inspect retained control outputs and lifecycle receipts',len(records),state='failed');raise
    valid={r['scene'] for r in records if r['returncode']==0 and r.get('result') and not r['result']['hard_stop'] and r['all_ai_steps_qualified'] and r['all_reference_steps_qualified']}
    tasks=[]
    for index,path in scenes:
        if index not in valid:continue
        for repetition in range(config['repetitions']):
            for model in (models if repetition%2==0 else models[::-1]):
                tasks.append((index,path,model['name'],model,repetition,partitions[len(tasks)%args.workers]))
    try:execute_partitioned(tasks,args.workers,execute,consume)
    except BaseException:
        observer.publish('CPU trajectory study failed','Inspect retained per-episode commands, logs and lifecycle receipts',len(records),state='failed');raise
    save()
    observer.publish('CPU trajectory study complete','All episode observations and physical outcomes retained',len(records),len(records),state='completed')


if __name__=='__main__':main()
