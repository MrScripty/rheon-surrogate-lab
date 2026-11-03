"""Admit only the freshly checked immutable plan, then own its finite batch."""
import argparse,subprocess,sys,time
from pathlib import Path
from v6_support import ROOT,read,write,sha
def run(a):
    d=read(a.readiness);age=time.time()-d['checked_unix']
    if d['status']!='ready_for_serial_timing'or not d['host_namespace_process_audit']or d['live_owned_workers']or not 0<=age<=120:raise ValueError('Fresh completed host readiness required')
    if d['plan_sha256']!=sha(a.plan):raise ValueError('Timing plan changed after readiness')
    write(a.admission,dict(started_unix=time.time(),readiness_sha256=sha(a.readiness),plan_sha256=sha(a.plan),source_sha256=sha(__file__)))
    subprocess.run([sys.executable,'-B',str(ROOT/'experiments/pressure-methods-v2/run_batch.py'),'--specification',str(a.plan),'--output',str(a.output)],cwd=ROOT,check=True,timeout=a.timeout)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['plan','readiness','admission','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--timeout',type=int,default=14400);run(p.parse_args())
