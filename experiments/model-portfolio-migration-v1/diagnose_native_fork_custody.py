"""Trace actual fork preservation decisions without global profiling or writes.

The real mark boundary retains its SHA validator. This captures only its finite
artifact decisions, not unrelated interpreter calls or historical source code.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
from unittest.mock import patch


ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/model-portfolio-migration-v1/sources/Tuldok'


def diagnose(project,report_path,require_complete,source):
    if report_path.exists():raise ValueError('Fresh custody diagnosis report required')
    if not source.is_relative_to(ROOT/'runs/model-portfolio-migration-v1/sources') or not (source/'project_fork_composition.py').is_file():
        raise ValueError('Task-owned native source required')
    sys.path.insert(0,str(source))
    import project_fork_composition as owner
    from project_material import decode_records as material_records
    start=time.monotonic();raw=(project/'.tuldok/project.json').read_bytes();manifest=json.loads(raw)
    records=[]
    for entry in manifest['inventory']:
        path=Path(entry['path'])
        if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid canonical record path')
        value=(project/'.tuldok'/path).read_bytes()
        if len(value)!=entry['bytes'] or hashlib.sha256(value).hexdigest()!=entry['sha256']:
            raise ValueError('Canonical record identity changed')
        records.append(dict(family=entry['family'],id=entry['id'],document=json.loads(value)))
    missing={r['id'] for r in records if r['family']=='project_artifact'
             and not (project/'artifacts/sha256'/r['id']).is_file()}
    obligations=[];original_sha=owner._sha
    def observe(value):
        original_sha(value)
        caller=sys._getframe(1)
        if value in missing and caller.f_code.co_name=='mark' and caller.f_code.co_filename==owner.__file__:
            local=caller.f_locals
            decision={key:local[key] for key in ('sha','role','name','size','policy')}
            decision['record']={key:local['record'][key] for key in ('family','id')}
            obligations.append(decision)
    # Restrict observation to an existing boundary; no global tracing overhead.
    with patch.object(owner,'_sha',observe):plan=owner.artifact_plan(records)
    missing_unique=[entry for entry in plan if entry['declaration']['sha256'] in missing and entry['preservation']=='unique']
    required={entry['declaration']['sha256'] for entry in missing_unique}
    marked_unique={row['sha'] for row in obligations if row['policy']=='unique'}
    marked_regenerable={row['sha'] for row in obligations if row['policy']=='regenerable'}
    materials=material_records(records)
    runtime_details=[]
    for (family,identity),doc in materials.items():
        if family!='runtime_material':continue
        value=doc['value']
        runtime_details.append(dict(id=identity,keys=list(value),profile=value.get('descriptor',{}).get('profile'),
            artifact_edges=len(doc['artifacts']),edge_roles=dict(Counter(edge['role'] for edge in doc['artifacts']))))
    if (project/'.tuldok/project.json').read_bytes()!=raw:raise ValueError('Source checkpoint changed during diagnosis')
    result=dict(schema='rheon.native-fork-custody-diagnosis.v1',created_unix=time.time(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        source_worktree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()),
        owner_source_sha256=hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),
        history_owner_source_sha256=hashlib.sha256((source/'project_history.py').read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        project_id=manifest['project_id'],checkpoint=manifest['checkpoint'],records=len(records),families=dict(Counter(r['family'] for r in records)),
        missing_unique=len(required),missing_unique_bytes=sum(entry['declaration']['bytes'] for entry in missing_unique),
        unique_owner_obligations=len(required&marked_unique),regenerable_owner_obligations=len(required&marked_regenerable),
        unclaimed_missing_declarations=sorted(required-marked_unique-marked_regenerable),
        decisions=dict(Counter(row['role']+' / '+row['policy'] for row in obligations)),obligations=obligations,
        missing_unique_declarations=missing_unique,runtime_details=runtime_details,
        canonical_source_unchanged=True,new_neural_forwards=0,elapsed_seconds=time.monotonic()-start)
    report_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('missing_unique','missing_unique_bytes','unique_owner_obligations',
        'regenerable_owner_obligations','decisions','runtime_details','elapsed_seconds')}),flush=True)
    if require_complete and required:raise SystemExit('Incomplete native fork custody: actual missing unique bytes remain')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true');parser.add_argument('--source',type=Path,default=SOURCE)
    args=parser.parse_args()
    diagnose(args.project.resolve(),args.report.resolve(),args.require_complete,args.source.resolve())
