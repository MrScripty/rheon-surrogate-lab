"""Admit only a fresh, completed readiness audit and exact reserved timing plan."""
import os,sys,time
from v5_support import ROOT,BASE,DOCS,read,write,sha
def start():
    ready=read(BASE/'quiet-ready-03.json');now=time.time()
    if ready['status']!='ready_for_serial_timing'or not 0<=now-ready['checked_unix']<=120:raise ValueError('Fresh completed readiness required')
    if ready['plan_sha256']!=sha(BASE/'all-timing-plan-03.json')or ready['selection_sha256']!=sha(DOCS/'selection.json'):raise ValueError('Readiness identities changed')
    output=BASE/'all-timing-batch-03'
    if output.exists():raise ValueError('Fresh timing output required')
    write(BASE/'timing-admission-03.json',dict(status='admitted_after_completed_readiness',readiness_sha256=sha(BASE/'quiet-ready-03.json'),
        plan_sha256=sha(BASE/'all-timing-plan-03.json'),selection_sha256=sha(DOCS/'selection.json'),started_unix=now,source_sha256=sha(__file__)))
    os.execv(sys.executable,[sys.executable,'-B',str(ROOT/'experiments/pressure-methods-v2/run_batch.py'),'--specification',str(BASE/'all-timing-plan-03.json'),'--output',str(output)])
if __name__=='__main__':start()
