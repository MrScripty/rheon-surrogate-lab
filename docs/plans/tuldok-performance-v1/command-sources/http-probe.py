"""Bounded read-only reproduction of slow completed-Run and tiny-artifact reads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path.cwd()
sys.path[:0]=[str(ROOT/'runs/dependency-migration-v1/Tuldok'),str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
import agent_api
from qualify_typed_pressure import OwnerHTTP

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=float,default=2);a=p.parse_args()
run=json.loads((ROOT/'runs/pressure-shape-decay-v1/shape-later-decay-study-01/terminal-e17624f982b54e858698207227010410.json').read_text())
artifact=next(r for r in run['artifacts']if r['name']=='training-profile.json')
t=time.perf_counter();owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
rows=[dict(operation='owner_initialization',elapsed_s=time.perf_counter()-t)]
for operation,arguments in [('runs.get',{'run_id':run['id']}),('runs.artifact_chunk',dict(run_id=run['id'],name=artifact['name'],expected_sha256=artifact['sha256'],offset=0,max_bytes=4096))]*2:
 t=time.perf_counter();result=owner.call(operation,arguments);elapsed=time.perf_counter()-t
 if operation=='runs.get':assert result['state']=='completed'and result['custody_phase']=='drained'
 rows.append(dict(operation=operation,elapsed_s=elapsed));print(operation,round(elapsed,4),'s',flush=True)
record=dict(status='passed'if all(r['elapsed_s']<=a.limit for r in rows if r['operation']!='owner_initialization')else'latency_reproduced',
 scope='Two current typed reads of a known completed Run and its 189-byte artifact; no admission or training.',request_target_s=a.limit,artifact_bytes=artifact['bytes'],run_id=run['id'],rows=rows,
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
a.output.write_text(json.dumps(record,indent=2)+'\n');print(record['status'],flush=True)
raise SystemExit(0 if record['status']=='passed'else 1)
