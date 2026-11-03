"""Admit the serial timing controller only after a completed fresh readiness audit."""
import os
import sys
import time
from v4_support import ROOT,BASE,read,write,sha

def start():
    ready=BASE/'quiet-start-01.json';record=read(ready)
    if record['status']!='ready_for_serial_timing'or record['live_owned_workers']:
        raise ValueError('A completed quiet-start audit is required')
    if not 0<=time.time()-record['checked_unix']<120:
        raise ValueError('Quiet-start observation is stale; reobserve before timing')
    plan=BASE/'all-timing-plan-01.json';output=BASE/'all-timing-batch-02'
    if output.exists():raise ValueError('Fresh timing controller required')
    write(BASE/'timing-admission-02.json',dict(status='admitted_after_completed_audit',readiness_sha256=sha(ready),plan_sha256=sha(plan),
        source_sha256=sha(__file__),started_unix=time.time(),prior_unqualified_timings_excluded=True))
    os.execv(sys.executable,[sys.executable,'-B',str(ROOT/'experiments/pressure-methods-v4/run_batch.py'),
        '--plan',str(plan),'--output',str(output)])

if __name__=='__main__':start()
