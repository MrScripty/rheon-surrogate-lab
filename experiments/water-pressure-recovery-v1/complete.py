"""Finish predefined diagnostics and selection after all consumed trajectories."""
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
    path=root/'completion-execution.json'
    assert not path.exists()
    record={'state':'observing_consumed_trajectories','started_at_utc':utc(),'commands':[]}
    started=time.monotonic()
    write(path,record)
    def run(name,argv):
        record['state']=name;write(path,record)
        record['commands'].append(command(argv,root,name));write(path,record)
    try:
        deadline=time.monotonic()+7200
        while True:
            assert time.monotonic()<deadline,'Observation deadline; reconcile existing CPU owner'
            state=read(root/'cpu-continuation.json')['state']
            assert state!='failed','CPU owner failed; retain and reconcile, never resubmit'
            if state=='consumed_comparisons_completed':break
            time.sleep(1)
        run('trained-frontier-gentle-025',[python,entry/'replay_all.py','--config',entry/'configs/trained-frontier.json',
                                        '--python',python,'--model','gentle-025'])
        run('amplitude-gentle-025',['taskset','-c','4',python,LAB/'experiments/water-early-pressure-v1/amplitude.py',
            '--config',entry/'configs/amplitude.json','--model','gentle-025','--output',reports/'amplitude-gentle-025.json'])
        run('frozen-selection',[python,entry/'select.py'])
        selection=read(reports/'selection.json')
        if selection['eligible']:
            record['state']='awaiting_seed_replication'
        elif selection['selected'] is not None:
            chosen=selection['selected'];phase=chosen['phase']
            assert read(reports/phase/'regression-summary.json')['frozen_challenger']==chosen['name']
            run('current-serial-timing',[python,LAB/'experiments/water-early-pressure-v1/time_selected.py',
                '--config',LAB/chosen['config'],'--reports',reports/phase])
            record['state']='measurements_completed_fresh_deferred'
        else:
            record['state']='no_qualified_timing_candidate'
    except BaseException as error:
        record.update(state='failed',error_type=type(error).__name__,error=str(error)[:1000])
        raise
    finally:
        record.update(finished_at_utc=utc(),elapsed_seconds=time.monotonic()-started)
        write(path,record)


if __name__=='__main__':
    main()
