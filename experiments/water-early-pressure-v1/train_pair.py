"""Serial managed CUDA fits with independent CPU diagnostics between fits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import command,read,write,utc
sys.path.insert(0,str(LAB/'experiments/live-project-v1'))
from observer import Observer, Watch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True);p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--torch-python',required=True)
    args=p.parse_args();sequence=read(args.sequence);root=LAB/sequence['root']
    assert read(root/'smoke/managed-smoke/receipt.json')['status']=='completed_and_drained'
    smoke=read(root/'smoke/managed-smoke/stage-00/artifacts/receipt.json')
    restore=read(root/'smoke/managed-smoke/stage-01/artifacts/receipt.json')
    assert smoke['mode']=='train' and smoke['optimizer_updates']==1
    assert restore['mode']=='restore' and restore['optimizer_updates']==0 and not restore['execution']['cuda_initialized']
    path=root/'pair-execution.json';assert not path.exists()
    result={'state':'running','started_at_utc':utc(),'commands':[],'optimizer_concurrency':1}
    started=time.perf_counter();write(path,result)
    observer=Observer(LAB/'runs/live-project')
    active=[None]
    def observe():
        if active[0] is not None:
            path=root/active[0]/'live/activity.json'
            if path.exists():
                value=read(path)
                observer.publish(value['phase'],value['message'],
                    value['progress']['completed'],value['progress']['total'])
    def diagnostic(name):
        return command(['taskset','-c','4',args.torch_python,LAB/'experiments/water-early-pressure-v1/amplitude.py',
            '--config',LAB/'experiments/water-early-pressure-v1/configs/amplitude.json','--model',name,
            '--output',LAB/f'docs/plans/water-early-pressure-v1/reports/amplitude-{name}.json'],root,'amplitude-'+name)
    try:
        with Watch(observe), ThreadPoolExecutor(max_workers=1) as pool:
            pending=[]
            for trial in sequence['trials']:
                name=trial['name'];active[0]=name
                observer.publish('Earlier pressure · CUDA '+name,'Matched5000-update fit through existing Tuldok catalog')
                result['commands'].append(command([sys.executable,args.controller,'--root',root,'--trial',name,'--phase','full'],root,'managed-'+name))
                pending.append(pool.submit(diagnostic,name))
                write(path,result)
            result['cpu_diagnostics']=[f.result() for f in pending]
        result['state']='completed'
    except BaseException as error:
        result.update(state='failed',error_type=type(error).__name__)
        raise
    finally:
        result.update(finished_at_utc=utc(),elapsed_seconds=time.perf_counter()-started)
        write(path,result)
        observer.publish('Earlier pressure · matched fits',result['state'],state='completed' if result['state']=='completed' else 'failed')


if __name__=='__main__':main()
