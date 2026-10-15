"""Fresh-process CPU trajectory qualification with explicit checkpoint populations.

Independent jobs may use distinct cores for throughput. Only workers=1 yields
isolated serial repetitions suitable for comparing elapsed timings. Each native
child retains the existing rollout owner's deadline and termination receipt.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
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


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=23)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if set(config)!={'schema','binary','profile','models','scenes','repetitions'} or config['schema']!='rheon-multigrid-trajectory-study-v1':
        raise ValueError('closed study configuration required')
    if type(config['repetitions']) is not int or not 1<=config['repetitions']<=3:raise ValueError('one to three repetitions required')
    if not config['models'] or not config['scenes'] or len(config['models'])>4 or len(config['scenes'])>10:raise ValueError('bounded nonempty study required')
    allowed=sorted(os.sched_getaffinity(0));cpus=allowed[1:] or allowed
    if not 1<=args.workers<=len(cpus):raise ValueError('workers must fit available distinct cores')
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
        command=['taskset','-c',str(cpu),sys.executable,str(rollout),'--binary',str(LAB/config['binary']),
            '--scene',str(path.absolute()),'--output',str(output.absolute())]
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
    records=[]
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
            'workers':args.workers,'available_cpus':allowed,'used_cpus':sorted({r['cpu'] for r in records}),
            'python':sys.executable,'started_at_utc':datetime.fromtimestamp(started,timezone.utc).isoformat(),'elapsed_s':time.time()-started,
            'scope':'development only, no native refinement; serial isolated timings' if args.workers==1 else 'development correctness/throughput; overlapping jobs, no latency claim'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (args.output/'commands.json').write_text(json.dumps([r['command'] for r in records],indent=2)+'\n')
    controls=[(i,p,'native',None,0,cpus[j%args.workers]) for j,(i,p) in enumerate(scenes)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(execute,controls):records.append(row);save();print(json.dumps({'control':row['scene'],'returncode':row['returncode'],'result':row.get('result')}),flush=True)
    valid={r['scene'] for r in records if r['returncode']==0 and r.get('result') and not r['result']['hard_stop'] and r['all_ai_steps_qualified'] and r['all_reference_steps_qualified']}
    tasks=[]
    for index,path in scenes:
        if index not in valid:continue
        for repetition in range(config['repetitions']):
            for model in (models if repetition%2==0 else models[::-1]):
                tasks.append((index,path,model['name'],model,repetition,cpus[len(tasks)%args.workers]))
    # Map fixed core partitions to persistent workers; uneven job duration cannot
    # cause two active subprocesses to contend on the same assigned core.
    def partition(items):return [execute(item) for item in items]
    partitions=[tasks[i::args.workers] for i in range(args.workers)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for group in pool.map(partition,partitions):
            for row in group:
                records.append(row);save();print(json.dumps({k:row.get(k) for k in ('scene','model','returncode','result')}),flush=True)
    save()


if __name__=='__main__':main()
