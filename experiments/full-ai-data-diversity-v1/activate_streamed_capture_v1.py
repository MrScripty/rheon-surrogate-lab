"""Activate qualified local capture code only after a quiescent-state snapshot.

Preserve the current service command/environment and authorized owner session.
No alternative credentials, simultaneous catalog supervisors or training retries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from inspect_live_capture_service_v1 import current_service,snapshot,CATALOG,ROOT


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix('.pending');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.chmod(0o600);temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();reg=read(args.protocol)
    if args.output.exists():raise ValueError('Fresh activation receipt required')
    for name,expected in reg['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Qualified activation source changed: '+name)
    if read(ROOT/reg['compatibility_results'])['status']!='passed':raise ValueError('Actual production compatibility tests required')
    if read(ROOT/reg['actual_trainer_qualification'])['status']!='passed_actual_captured_trainer_full148scene_objective':
        raise ValueError('Actual captured trainer qualification required')
    before=read(ROOT/reg['preflight']);command,environment,ticks=current_service(before['pid'])
    if ticks!=before['start_ticks']:raise ValueError('Current service PID identity changed')
    saved=snapshot(environment)
    if saved!=before['run_snapshot'] or any(r['state'] not in ('completed','failed','cancelled','interrupted') for r in saved.values()):
        raise ValueError('Run state changed or work is active; do not restart')
    source=ROOT/reg['candidate'];manifest=read(source/'deployment-manifest.json')
    for item in manifest['files']:
        if sha(source/item['path'])!=item['sha256']:raise ValueError('Deployment file changed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    receipt=dict(status='stop_intent',catalog_id=CATALOG,old_pid=before['pid'],old_start_ticks=ticks,
        candidate=reg['candidate'],candidate_manifest_sha256=sha(source/'deployment-manifest.json'),
        existing_runs=len(saved),active_runs=0,run_snapshot_before=saved,automatic_retry=False,
        original_source_preserved=True,owner_session_file='runs/tuldok-performance-v1/private/owner-session.txt',
        new_parameter_fits=0,cpu_model_forward=False)
    write(args.output,receipt);started=time.monotonic()
    os.kill(before['pid'],signal.SIGINT);proc=Path('/proc')/str(before['pid'])
    for _ in range(120):
        if not proc.exists():break
        fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':break
        time.sleep(.5)
    else:raise RuntimeError('Old service still live; replacement not started')
    receipt['status']='launch_intent';write(args.output,receipt)
    old=ROOT/'runs/dependency-migration-v1/Tuldok-deployed/current-07/app.py'
    command[command.index(str(old))]=str(source/'app.py')
    log_path=args.output.with_suffix('.private.log')
    with os.fdopen(os.open(log_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as log:
        process=subprocess.Popen(command,env=environment,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(status='health_pending',pid=process.pid);write(args.output,receipt)
    for _ in range(180):
        if process.poll() is not None:raise RuntimeError('Replacement exited; inspect private activation log')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8091/',timeout=2) as response:
                if response.status==200:break
        except OSError:time.sleep(.5)
    else:raise RuntimeError('Replacement health observation expired; do not launch a second service')
    if snapshot(environment)!=saved:raise ValueError('Existing Run documents changed')
    sys.path[:0]=[str(source),str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
    import agent_api
    from qualify_typed_pressure import OwnerHTTP
    owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091',CATALOG,ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
    if owner.identity['kind']!='owner':raise ValueError('Saved owner session not preserved')
    receipt.update(status='passed',existing_run_documents_preserved=True,original_owner_session_preserved=True,
        elapsed_seconds=time.monotonic()-started,start_ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
        source_sha256=reg['source_sha256'])
    write(args.output,receipt)
    print(json.dumps(dict(status='passed',pid=process.pid,existing_runs=len(saved),existing_run_documents_preserved=True)),flush=True)


if __name__=='__main__':main()
