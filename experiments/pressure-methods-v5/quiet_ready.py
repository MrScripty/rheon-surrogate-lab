"""Fresh public Run custody and actual host-process admission for serial timing."""
import os,sys,time
from pathlib import Path
from v5_support import ROOT,BASE,DOCS,read,write,sha
def ready():
    output=BASE/'quiet-ready-03.json'
    if output.exists():raise ValueError('Fresh readiness audit required')
    for name in ['v5-raw-training-batch-01','conditioned-training-batch-01','controls-evaluation-batch-02','trained-evaluation-batch-02',
        'native-development-batch-01','rollout-controls-batch-02','rollout-model-batch-02','composition-batch-03','drift-batch-03',
        'blind-capture-batch-03','blind-evaluation-batch-03-0','blind-evaluation-batch-03-1','blind-rollout-batch-03','blind-drift-batch-03','transfer-batch-04']:
        d=read(BASE/name/'execution.json')
        if d['status']!='completed'or not d['owned_child_reaped']:raise ValueError('Unfinished dependency: '+name)
    sys.path[:0]=[str(ROOT/'runs/dependency-migration-v1/Tuldok'),str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
    import agent_api
    from qualify_typed_pressure import OwnerHTTP
    owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
    records=[]
    for arm in ['v5-raw','v5-multilevel','v5-spectrum']:
        receipt=read(BASE/(arm+'-study-01/receipt.json'))
        if receipt['status']!='completed'or not all(receipt['artifact_parity'].values()):raise ValueError('Exact completed fit/restore required')
        for phase,key in [('fit','training_run_id'),('restore','restore_run_id')]:
            run=owner.call('runs.get',dict(run_id=receipt[key]))
            if run['state']!='completed'or run['custody_phase']!='drained':raise ValueError('Managed custody incomplete')
            records.append(dict(arm=arm,phase=phase,run_id=run['id'],state=run['state'],custody_phase=run['custody_phase']))
    workers=[];ids=[r['run_id'].encode()for r in records]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit()or int(p.name)==os.getpid():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0');exe=Path(args[0].decode()).name
            if(exe.startswith('python')or exe=='rheon-large-single-track')and(any(b'pressure-methods-v5'in a for a in args)or any(any(id in a for id in ids)for a in args)):
                workers.append(dict(pid=int(p.name),executable=exe))
        except(OSError,UnicodeError,IndexError):pass
    if workers:raise ValueError('Live campaign workers remain: '+str(workers))
    write(output,dict(status='ready_for_serial_timing',managed_runs=records,live_owned_workers=workers,checked_unix=time.time(),host_namespace_process_audit=True,
        selection_sha256=sha(DOCS/'selection.json'),plan_sha256=sha(BASE/'all-timing-plan-03.json'),source_sha256=sha(__file__),cpu=23,threads=1,
        scope='Fresh six-Run custody and actual host process scan. Unrelated desktop activity is not claimed absent.'))
    print('All development/blind work drained; fresh six-Run and host readiness passed',flush=True)
if __name__=='__main__':ready()

