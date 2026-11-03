"""Measure Unicode validation in two progression reads on pinned project metadata.

Derived from benchmark_history_facts.py with agent_api.py explicitly source-bound.

The frozen Session substitute reproduces independent snapshot copying and counts
freshness-guard calls. Actual filesystem guard cost, SQL, authorization and whole
UI/Open latency are excluded. All selected canonical bytes are hashed initially.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/model-portfolio-migration-v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()


def run(source,project,manifest_sha,output):
    if (not source.is_relative_to(BASE/'sources') or not project.is_relative_to(ROOT)
            or not output.is_relative_to(ROOT/'docs/plans/model-portfolio-migration-v1/reports')
            or output.exists()):
        raise ValueError('Explicit task-owned sources, pinned metadata and fresh report required')
    resource.setrlimit(resource.RLIMIT_AS,(18*1024**3,18*1024**3))
    original=(project/'.tuldok/project.json').read_bytes()
    if digest(original)!=manifest_sha:
        raise ValueError('Pinned project header differs')
    header=json.loads(original)
    files=('project_history.py','project_material.py','project_session.py','project_store.py','agent_api.py')
    identities={name:digest((source/name).read_bytes()) for name in files}
    result=dict(schema='rheon.history-unicode-repeat-read-component.v1',status='loading_fixed_metadata',
        created_unix=time.time(),source=str(source.relative_to(ROOT)),
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        source_sha256=identities,producer_sha256=digest(Path(__file__).read_bytes()),
        project=str(project.relative_to(ROOT)),checkpoint=header['checkpoint'],manifest_sha256=manifest_sha,
        canonical_records=len(header['inventory']),canonical_record_bytes=sum(e['bytes'] for e in header['inventory']),
        scope=__doc__.strip(),phases=[],neural_forwards=0,payload_copies=0,SQL_opened=False,
        live_Store_or_Session_constructed=False,whole_native_UI_or_Open_speedup_qualified=False,
        memory_limit_bytes=18*1024**3,deadline_seconds=600)

    def save():
        output.write_text(json.dumps(result,indent=2)+'\n')

    def expired(*_):
        raise TimeoutError('Bounded history repeat-read component deadline')

    save();signal.signal(signal.SIGALRM,expired);signal.alarm(600)
    try:
        records=[]
        for entry in header['inventory']:
            path=Path(entry['path'])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unconfined canonical metadata path')
            raw=(project/'.tuldok'/path).read_bytes()
            if len(raw)!=entry['bytes'] or digest(raw)!=entry['sha256']:
                raise ValueError('Canonical body differs from pinned header')
            records.append(dict(family=entry['family'],id=entry['id'],document=json.loads(raw)))
        origins=sorted({record['document']['origin_catalog_id'] for record in records if record['family']=='trial_history'})
        if len(origins)!=1:
            raise ValueError('This admitted component workload requires its one original Trial namespace')
        origin=origins[0]
        raw_trials=sorted((record['document'] for record in records if record['family']=='trial_history'),
                          key=lambda document:document['trial']['id'])
        studies={(record['document']['origin_catalog_id'],record['document']['study']['id']):record['document']['study']
                 for record in records if record['family']=='study_history'}
        sys.path.insert(0,str(source))
        from project_history import ProjectHistory
        calls=dict(snapshot_copies=0,frozen_freshness_guard=0)

        class FrozenSession:
            artifacts=None
            access=SimpleNamespace(require=lambda *args,**kwargs:None)

            @property
            def summary(self):
                return {key:header[key] for key in ('project_id','name','checkpoint')}

            @property
            def snapshot(self):
                calls['snapshot_copies']+=1
                return dict(self.summary,records=copy.deepcopy(records))

            def check_current(self):
                calls['frozen_freshness_guard']+=1

        history=ProjectHistory(FrozenSession())
        for repetition in (1,2):
            gc.collect()
            before=dict(calls)
            result['status']='measuring_progression_read_'+str(repetition)
            save()
            cpu,wall=time.process_time(),time.monotonic()
            value=history.progression(origin,offset=0,limit=20)
            phase=dict(repetition=repetition,cpu_seconds=time.process_time()-cpu,
                       wall_seconds=time.monotonic()-wall,
                       snapshot_copies=calls['snapshot_copies']-before['snapshot_copies'],
                       frozen_freshness_guard_calls=calls['frozen_freshness_guard']-before['frozen_freshness_guard'])
            if value['total']!=94 or len(value['nodes'])!=20:
                raise ValueError('Actual progression population differs')
            for node,original_trial in zip(value['nodes'],raw_trials[:20],strict=True):
                trial=original_trial['trial'];study=studies[(origin,trial['study_id'])]
                if (node['trial']!=trial or node['lineage']!=original_trial['lineage']
                        or node['study']!={key:study[key] for key in ('id','name')}):
                    raise ValueError('Progression output differs from literal original Trial/Study facts')
            phase['complete_public_result_sha256']=digest(encode(value))
            result['phases'].append(phase)
            # A retained private cache must not let a prior public result become
            # authoritative. The next repetition checks the original facts.
            value['nodes'][0]['study']['name']='Caller mutation outside measured interval'
            value['nodes'][0]['trial']['replicate']=99
            del value
            save()
            print(json.dumps(phase),flush=True)
        if len({phase['complete_public_result_sha256'] for phase in result['phases']})!=1:
            raise ValueError('Repeated public result was changed by a caller mutation')
        if (project/'.tuldok/project.json').read_bytes()!=original:
            raise ValueError('Project changed during observation')
        if identities!={name:digest((source/name).read_bytes()) for name in files}:
            raise ValueError('Observed source changed')
        if 'torch' in sys.modules:
            raise ValueError('Metadata-only probe imported a neural runtime')
        result.update(status='history_repeat_read_component_completed',all_canonical_bodies_initially_hashed=True,
            exact_original_trial_and_study_values=True,caller_mutation_isolated=True,
            total_snapshot_copies=calls['snapshot_copies'],frozen_freshness_guard_calls=calls['frozen_freshness_guard'])
    except BaseException as error:
        result.update(status='history_repeat_read_component_incomplete',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        signal.alarm(0)
        result.update(source_unchanged=identities=={name:digest((source/name).read_bytes()) for name in files},
            header_unchanged=(project/'.tuldok/project.json').read_bytes()==original,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            neural_runtime_imported='torch' in sys.modules)
        save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--manifest-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.source.resolve(),args.project.resolve(),args.manifest_sha,args.output.resolve())
