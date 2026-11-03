"""Read-only launch/state preflight using the existing service configuration.

Never searches credentials or prints environment values. Owner authentication
continues to use the explicitly authorized saved session through the HTTP API.
The existing service's DB configuration is reused only for a read-only snapshot
without reopening every historical Run's source/input files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import psycopg
from psycopg import sql

ROOT=Path(__file__).resolve().parents[2]
CATALOG='a7da4a8de53541a2a0e2aad08005bb71'


def current_service(pid):
    proc=Path('/proc')/str(pid)
    fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
    command=proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
    old=ROOT/'runs/dependency-migration-v1/Tuldok-deployed/current-07/app.py'
    if str(old) not in command or '--port' not in command or command[command.index('--port')+1]!='8091':
        raise ValueError('Selected PID is not the expected current07 service')
    environment=dict(item.split('=',1) for item in proc.joinpath('environ').read_bytes().decode().strip('\0').split('\0') if '=' in item)
    if 'TULDOK_DATABASE_URL' not in environment:raise ValueError('Existing service database configuration unavailable')
    return command,environment,fields[19]


def snapshot(environment):
    with psycopg.connect(environment['TULDOK_DATABASE_URL'],connect_timeout=5) as db:
        db.execute('SET TRANSACTION READ ONLY')
        db.execute("SET LOCAL statement_timeout='10s'")
        rows=db.execute(sql.SQL("SELECT id,document::jsonb->>'state',encode(sha256(convert_to(document,'UTF8')),'hex') FROM {}.workspace_runs").format(sql.Identifier('catalog_'+CATALOG))).fetchall()
    return {key:dict(state=state,document_sha256=digest) for key,state,digest in rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh read-only inspection receipt required')
    command,environment,ticks=current_service(args.pid)
    rows=snapshot(environment);active={key:r for key,r in rows.items() if r['state'] not in ('completed','failed','cancelled','interrupted')}
    output=dict(status='quiescent' if not active else 'active_runs_prevent_cutover',created_unix=time.time(),pid=args.pid,start_ticks=ticks,
        catalog_id=CATALOG,old_source='runs/dependency-migration-v1/Tuldok-deployed/current-07',
        expected_deployment_manifest_sha256=hashlib.sha256((ROOT/'runs/dependency-migration-v1/Tuldok-deployed/current-07/deployment-manifest.json').read_bytes()).hexdigest(),
        existing_runs=len(rows),active_runs=active,run_snapshot=rows,read_only=True,service_restarted=False,model_fits=0,
        owner_authentication='Only the authorized saved owner session is used for application API requests.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(output,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(status=output['status'],existing_runs=len(rows),active_runs=len(active),pid=args.pid)),flush=True)


if __name__=='__main__':main()
