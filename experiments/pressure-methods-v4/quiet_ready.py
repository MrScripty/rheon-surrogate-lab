"""Check fresh public Run custody and actual host processes before timing."""
import os
from pathlib import Path
import sys
import time
from v4_support import ROOT,BASE,read,write,sha

def ready():
    output=BASE/'quiet-start-01.json'
    if output.exists():raise ValueError('Fresh quiet-start audit required')
    sys.path[:0]=[str(ROOT/'runs/dependency-migration-v1/Tuldok'),str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
    import agent_api
    from qualify_typed_pressure import OwnerHTTP
    owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
    managed=[];run_ids=[]
    arms=['v4-student75-distill','v4-student75-control','v4-student50-distill','v4-student50-control','v4-replay','v4-gcr','v4-diverse','v4-onpolicy']
    for arm in arms:
        generation=2 if arm=='v4-student75-distill'else 1
        receipt=read(BASE/(arm+f'-study-{generation:02d}')/'receipt.json')
        if receipt['status']!='completed'or not all(receipt['artifact_parity'].values()):raise ValueError('All fits and fresh CPU restores must complete before timing')
        for phase,key in [('fit','training_run_id'),('restore','restore_run_id')]:
            run=owner.call('runs.get',dict(run_id=receipt[key]));run_ids.append(run['id'])
            if run['state']!='completed'or run['custody_phase']!='drained':raise ValueError('Managed process custody is not complete')
            managed.append(dict(arm=arm,phase=phase,run_id=run['id'],state=run['state'],custody_phase=run['custody_phase']))
    workers=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit()or int(process.name)==os.getpid():continue
        try:
            arguments=(process/'cmdline').read_bytes().split(b'\0');exe=Path(arguments[0].decode()).name
            if not(exe.startswith('python')or exe=='rheon-large-single-track'):continue
            campaign=any(b'pressure-methods-v4'in a for a in arguments)or any(any(r.encode()in a for r in run_ids)for a in arguments)
            if campaign:workers.append(dict(pid=int(process.name),executable=exe))
        except(OSError,UnicodeError,IndexError):pass
    if workers:raise ValueError('Live campaign workers remain: '+str(workers))
    write(output,dict(status='ready_for_serial_timing',managed_runs=managed,live_owned_workers=workers,
        checked_unix=time.time(),host_namespace_process_audit=True,cpu=23,threads=1,source_sha256=sha(__file__),
        scope='No live campaign worker observed; unrelated desktop activity is not claimed absent. Final benchmark batches must run serially.'))
    print('Fresh Run custody and host process checks passed; ready for serial timing',flush=True)

if __name__=='__main__':ready()
