"""Observe admitted work, then run serial CPU phases through existing owners."""
from pathlib import Path
import sys
import time

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import read,write,command,utc


def main():
    root=LAB/'runs/water-pressure-recovery-v1'
    reports=LAB/'docs/plans/water-pressure-recovery-v1/reports'
    entry=LAB/'experiments/water-pressure-recovery-v1'
    python=read(entry/'configs/training-regression.json')['python']
    status=root/'cpu-continuation.json'
    assert not status.exists()
    record={'state':'observing_admitted_work','started_at_utc':utc(),'commands':[]}
    started=time.monotonic()
    write(status,record)
    def run(name,argv):
        record['state']=name;write(status,record)
        record['commands'].append(command(argv,root,name));write(status,record)
    try:
        deadline=time.monotonic()+7200
        while True:
            assert time.monotonic()<deadline,'Observation deadline; reconcile existing owners, never resubmit'
            policy=read(root/'policy-development-timing.json')
            if policy['status']!='running':
                assert policy['status']=='completed'
                break
            time.sleep(1)
        run('policy-analysis',[python,entry/'measure.py','analyze','--config',entry/'configs/policy-regression.json','--reports',reports/'policy'])
        while True:
            assert time.monotonic()<deadline,'Observation deadline; reconcile existing managed Runs'
            training=read(root/'training-execution.json')
            assert training['state'] in ('running','completed'),'Managed run failed; reconcile rather than replay'
            if training['state']=='completed':break
            time.sleep(1)
        config=entry/'configs/training-regression.json'
        run('training-admission',[python,entry/'measure.py','admit','--config',config,'--reports',reports/'training'])
        for phase in ('fixture','development'):
            run('training-'+phase,[python,LAB/'experiments/water-stopping-floor-v1/run.py','--config',config,'--phase',phase])
        run('training-analysis',[python,entry/'measure.py','analyze','--config',config,'--reports',reports/'training'])
        record['state']='consumed_comparisons_completed'
    except BaseException as error:
        record.update(state='failed',error_type=type(error).__name__,error=str(error)[:1000])
        raise
    finally:
        record.update(finished_at_utc=utc(),elapsed_seconds=time.monotonic()-started)
        write(status,record)


if __name__=='__main__':
    main()
