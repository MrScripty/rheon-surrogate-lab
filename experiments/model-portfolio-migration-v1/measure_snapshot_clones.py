"""Compare pure JSON snapshot cloning on existing pinned metadata, without I/O owners.

The actual private Store value-projection method uses no instance state; calling
that pure seam does not construct a fake Store or qualify a native lifecycle.
The JSON alternative is a diagnostic prototype, never installed into the app.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/model-portfolio-migration-v1'


def digest(raw):return hashlib.sha256(raw).hexdigest()


def run(source,project,manifest_sha,report):
    if (not source.is_relative_to(BASE/'sources') or not project.is_relative_to(ROOT)
            or not report.is_relative_to(ROOT/'docs/plans/model-portfolio-migration-v1/reports')
            or report.exists()):
        raise ValueError('Explicit task source, pinned existing metadata and fresh report required')
    resource.setrlimit(resource.RLIMIT_AS,(18*1024**3,18*1024**3))
    header=project/'.tuldok/project.json';raw_header=header.read_bytes()
    if digest(raw_header)!=manifest_sha:raise ValueError('Selected metadata checkpoint differs')
    manifest=json.loads(raw_header);records=[]
    code=source/'project_store.py';original_code=code.read_bytes()
    result=dict(schema='rheon.snapshot-clone-mechanism.v1',status='reading_fixed_metadata',
        source=str(source.relative_to(ROOT)),source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        source_sha256=digest(original_code),runner_sha256=digest(Path(__file__).read_bytes()),
        project=str(project.relative_to(ROOT)),checkpoint=manifest['checkpoint'],manifest_sha256=manifest_sha,
        records=len(manifest['inventory']),canonical_record_bytes=sum(r['bytes'] for r in manifest['inventory']),
        phases=[],model_forwards=0,SQL_opened=False,Store_or_Session_constructed=False,
        payload_copies=0,project_written=False,full_native_Open_qualified=False,
        scope=__doc__.strip(),memory_limit_bytes=18*1024**3)

    def save():report.write_text(json.dumps(result,indent=2)+'\n')
    def expired(*_):raise TimeoutError('Snapshot-clone diagnostic budget expired')
    save();signal.signal(signal.SIGALRM,expired);signal.alarm(240)
    active=None;started={}
    def gc_event(phase,info):
        generation=info['generation']
        if phase=='start':started[generation]=time.process_time()
        elif active is not None and generation in started:
            active['gc_cpu_seconds']+=time.process_time()-started.pop(generation)
            active['gc_collections'][str(generation)]+=1
    try:
        for entry in manifest['inventory']:
            relative=Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unconfined canonical path')
            raw=(project/'.tuldok'/relative).read_bytes()
            if len(raw)!=entry['bytes'] or digest(raw)!=entry['sha256']:raise ValueError('Canonical metadata changed')
            records.append(dict(family=entry['family'],id=entry['id'],document=json.loads(raw)))
        sys.path.insert(0,str(source));from project_store import ProjectStore
        gc.callbacks.append(gc_event)
        for name in ('actual_native_snapshot_value_projection','prototype_compact_JSON_clone'):
            gc.collect();started.clear()
            active=dict(name=name,gc_cpu_seconds=0.,gc_collections={'0':0,'1':0,'2':0})
            result.update(status='snapshot_clone_running');save()
            cpu,wall=time.process_time(),time.monotonic()
            if name=='actual_native_snapshot_value_projection':
                value=ProjectStore._snapshot(None,manifest,records)
            else:
                value=dict(project_id=manifest['project_id'],name=manifest['name'],checkpoint=manifest['checkpoint'],
                    records=json.loads(json.dumps(records,ensure_ascii=False,allow_nan=False,separators=(',',':'))))
            active.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
            # Full-value oracle and isolation check are outside the measured call.
            active['values_sha256']=digest(json.dumps(value,sort_keys=True,ensure_ascii=False,
                allow_nan=False,separators=(',',':')).encode())
            before=records[0]['document']['format'];value['records'][0]['document']['format']='diagnostic_changed'
            active['caller_mutation_isolated']=records[0]['document']['format']==before
            result['phases'].append(active);active=None;save();del value
        if (len({p['values_sha256'] for p in result['phases']})!=1
                or not all(p['caller_mutation_isolated'] for p in result['phases'])):
            raise ValueError('Clone prototype changes values or isolation')
        result['status']='snapshot_clone_component_comparison_completed'
    except BaseException as error:
        result.update(status='snapshot_clone_diagnostic_incomplete',error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0)
        if gc_event in gc.callbacks:gc.callbacks.remove(gc_event)
        result.update(source_unchanged=code.read_bytes()==original_code,manifest_unchanged=header.read_bytes()==raw_header,
            neural_runtime_imported='torch' in sys.modules,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save()
    print(json.dumps(dict(status=result['status'],phases=result['phases'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','project','report'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True);args=parser.parse_args()
    run(args.source.resolve(),args.project.resolve(),args.manifest_sha256,args.report.resolve())
