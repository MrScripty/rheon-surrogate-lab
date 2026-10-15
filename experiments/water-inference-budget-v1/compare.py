"""Compare two restored checkpoints at the same neural-work cap on one CPU core.

Each episode runs in a fresh Python process, preserving captured-module identity
when checkpoints have different configurations. Timings exclude initialization
and the first neural step, using the existing whole-track measurement owner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from scaling import summarize
from observer import Observer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-project','reference-artifacts','candidate-project','candidate-artifacts','binary','profile','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scenes',type=Path,nargs='+',required=True)
    p.add_argument('--cpu',type=int,required=True)
    p.add_argument('--repetitions',type=int,default=3)
    args=p.parse_args()
    if not 2<=args.repetitions<=5 or args.cpu not in os.sched_getaffinity(0):raise ValueError('bounded repetitions on a permitted CPU required')
    os.sched_setaffinity(0,{args.cpu});args.output.mkdir(parents=True,exist_ok=False)
    observer=Observer(args.output/'live');records=[];commands=[]
    sources={name:(getattr(args,name+'_project').absolute(),getattr(args,name+'_artifacts').absolute()) for name in ('reference','candidate')}
    try:
        for index,path in enumerate(args.scenes):
            scene=json.loads(path.read_text());scene['steps']=scene['warmup_steps']+8
            root=args.output/path.stem;root.mkdir();configuration=root/'scene.json'
            configuration.write_text(json.dumps(scene,indent=2)+'\n');row={'scene':scene,'name':path.stem,'runs':[]}
            control=root/'native-control'
            command=[sys.executable,str(Path(__file__).resolve().parents[1]/'water-residual-learning-v1/rollout.py'),
                '--binary',str(args.binary.absolute()),'--scene',str(configuration.absolute()),'--output',str(control.absolute()),'--reference-control']
            commands.append(command)
            with (root/'native-control.log').open('xb') as stream:subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
            row['native_control']=json.loads((control/'receipt.json').read_text())
            outcome=row['native_control']['result']
            if outcome['hard_stop'] or outcome['qualified_ai_steps']!=outcome['ai_steps']:raise ValueError('native control failed')
            for repetition in range(args.repetitions):
                for name in (('reference','candidate') if repetition%2==0 else ('candidate','reference')):
                    project,artifacts=sources[name];config=json.loads((project/'config.json').read_text());width=config['correction']['steps']
                    if type(width) is not int or not 1<=width<=4:raise ValueError('declared residual proposal width required')
                    output=root/f'{name}-{repetition}'
                    observer.publish('CPU checkpoint comparison · '+path.stem,f'{name}, repetition {repetition+1}/{args.repetitions}',index,len(args.scenes))
                    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'water-residual-learning-v1/rollout.py'),
                        '--binary',str(args.binary.absolute()),'--scene',str(configuration.absolute()),'--output',str(output.absolute()),
                        '--project',str(project),'--artifacts',str(artifacts),'--proposals',str(32//width),'--adaptive-profile',str(args.profile.absolute())]
                    commands.append(command);(args.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
                    with (root/(name+f'-{repetition}.log')).open('xb') as stream:subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
                    row['runs'].append({'policy':name,'repetition':repetition,**summarize(output)})
            for name in sources:
                runs=[r for r in row['runs'] if r['policy']==name];times=[t for r in runs for t in r['ai_seconds']]
                row[name]={'qualified':all(r['all_ai_steps_qualified'] and r['all_reference_steps_qualified'] and not r['result']['hard_stop'] for r in runs),
                    'ai_median_seconds':statistics.median(times),'native_median_seconds':statistics.median(t for r in runs for t in r['native_seconds']),
                    'mean_network_evaluations':statistics.mean(t for r in runs for t in r['network_passes']),
                    'per_repetition_ai_median_seconds':[statistics.median(r['ai_seconds']) for r in runs]}
            row['reference_to_candidate_time_ratio']=row['reference']['ai_median_seconds']/row['candidate']['ai_median_seconds'];records.append(row)
            result={'schema':'rheon-restored-checkpoint-latency-v1','records':records,'cpu':args.cpu,'repetitions':args.repetitions,
                'sources':{name:{'project':str(project),'artifacts':str(artifacts),'weights_sha256':hashlib.sha256((artifacts/'weights.npz').read_bytes()).hexdigest()} for name,(project,artifacts) in sources.items()},
                'scope':'same short scenes, fresh processes, one CPU core; whole-track timings after history/neural warm-up; zero additional training',
                'attribution':'checkpoint and stopping granularity both change; not an isolated learned-weight effect'}
            (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='runs'}),flush=True)
    except BaseException:
        observer.publish('CPU checkpoint comparison failed','Inspect retained episode receipts and logs',state='failed');raise
    else:observer.publish('CPU checkpoint comparison complete','Paired checkpoint timings and physical outcomes retained',len(args.scenes),len(args.scenes),state='completed')


if __name__=='__main__':main()
