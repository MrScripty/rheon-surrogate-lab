"""Fresh eight-Run and actual host-worker admission for one exact timing plan."""
import argparse,os,sys,time
from pathlib import Path
from v6_support import ROOT,BASE,DOCS,read,write,sha,fit_record
def run(a):
    if a.output.exists():raise ValueError('Fresh timing-readiness record required')
    dependencies={}
    for name in a.dependencies:
        p=BASE/name/'execution.json';d=read(p)
        if d['status']!='completed'or not d['owned_child_reaped']or d.get('active'):raise ValueError('Dependency has not drained: '+name)
        dependencies[str(p.relative_to(ROOT))]=sha(p)
    sys.path[:0]=[str(ROOT/'runs/dependency-migration-v1/Tuldok'),str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
    import agent_api
    from qualify_typed_pressure import OwnerHTTP
    owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
    records=[]
    for arm in read(DOCS/'campaign.json')['fits']:
        r=read(ROOT/fit_record(arm)['study']/'receipt.json')
        if r['status']!='completed'or not all(r['artifact_parity'].values()):raise ValueError('Four completed fits/restores required')
        for phase,key in [('fit','training_run_id'),('restore','restore_run_id')]:
            d=owner.call('runs.get',dict(run_id=r[key]))
            if d['state']!='completed'or d['custody_phase']!='drained':raise ValueError('Managed Run custody has not drained')
            records.append(dict(arm=arm,phase=phase,run_id=d['id'],state=d['state'],custody_phase=d['custody_phase']))
    ids=[r['run_id'].encode()for r in records];workers=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit()or int(p.name)==os.getpid():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0');exe=Path(args[0].decode()).name
            owned=(any(b'pressure-methods-v6'in x for x in args)or any(any(run in x for run in ids)for x in args))
            if (exe.startswith('python')and owned)or exe=='rheon-large-single-track':workers.append(dict(pid=int(p.name),executable=exe))
        except(OSError,UnicodeError,IndexError):pass
    if workers:raise ValueError('Owned workers still live: '+str(workers))
    write(a.output,dict(status='ready_for_serial_timing',checked_unix=time.time(),managed_runs=records,dependencies=dependencies,
        live_owned_workers=workers,host_namespace_process_audit=True,plan_sha256=sha(a.plan),source_sha256=sha(__file__),cpu=23,threads=1,
        selection_sha256=sha(DOCS/'selection.json')if(DOCS/'selection.json').exists()else None,
        scope='Fresh managed custody and actual host-process check. Unrelated desktop activity is not claimed absent.'))
    print('Fresh eight-Run custody and host-process readiness passed',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dependencies',nargs='+',required=True);run(p.parse_args())
