"""Continue qualified matched fits through the existing bounded CPU owners."""
import argparse
from pathlib import Path
import sys
import time

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import read,write,command,utc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--reports',type=Path,required=True)
    args=p.parse_args();config=read(args.config);root=LAB/'runs/water-early-pressure-v1'
    status=root/'cpu-sequence-execution.json';assert not status.exists()
    record={'state':'waiting_for_matched_fits','started_at_utc':utc(),'commands':[]}
    write(status,record);started=time.perf_counter()
    def run(name,argv):
        record['state']=name;write(status,record)
        result=command(argv,root,name);record['commands'].append(result);write(status,record)
    try:
        deadline=time.monotonic()+3600
        while True:
            value=read(root/'pair-execution.json')
            if value['state']=='completed':break
            assert value['state']=='running','matched managed pair failed; reconcile retained Runs'
            assert time.monotonic()<deadline,'managed observation deadline; do not resubmit'
            time.sleep(1)
        record['waiting_seconds']=time.perf_counter()-started
        python=config['python'];entry=LAB/'experiments/water-early-pressure-v1'
        run('admission',[python,entry/'admit.py','--config',args.config,'--reports',args.reports])
        for phase in ('fixture','development'):
            run('regression-'+phase,[python,LAB/'experiments/water-stopping-floor-v1/run.py',
                '--config',args.config,'--phase',phase])
        run('regression-analysis',[python,entry/'compare.py','--config',args.config,'--reports',args.reports])
        run('amplitude-summary',[python,entry/'summarize_amplitude.py','--reports',args.reports,
            '--models','retained','late','control','first-pressure','--output',args.reports/'amplitude-summary.json'])
        run('current-serial-timing',[python,entry/'time_selected.py','--config',args.config,'--reports',args.reports])
        summary=read(args.reports/'regression-summary.json')
        record.update(state='consumed_measurements_completed',fresh_validation='pending eligible frozen challenger' if summary['all_consumed_gates_passed'] else 'deferred: consumed replacement gates failed')
        write(args.reports/'current.json',{'state':record['state'],'eligible':summary['eligible_for_fresh_validation'],
            'fresh_validation':record['fresh_validation'],'candidate_promoted':False})
    except BaseException as error:
        record.update(state='failed',error_type=type(error).__name__,error=str(error)[:1000])
        raise
    finally:
        record.update(finished_at_utc=utc(),elapsed_seconds=time.perf_counter()-started)
        write(status,record)


if __name__=='__main__':main()
