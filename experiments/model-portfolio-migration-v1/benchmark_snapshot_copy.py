"""Measure native Store/Session copying on one fixed real canonical checkpoint.

Excludes file loading, live freshness checks, SQL, UI latency and inference.
This is a copy-path comparison, not proof of complete Open/Save performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from qualify_native_workspace import ROOT


def value_proof(value, forbidden=()):
    """Fingerprint JSON types/float bits and enumerate every mutable container.

    This oracle is independent of JSON serialization and runs outside timing.
    It checks all descendants, including shared mutable source identities that
    would escape a first-record-only mutation assertion.
    """
    digest = hashlib.sha256()
    containers = set()

    def visit(item):
        kind = type(item)
        if kind in (list, dict):
            identity = id(item)
            if identity in forbidden:
                raise ValueError('Snapshot shares a mutable source container')
            containers.add(identity)
            digest.update(b'L' if kind is list else b'D')
            digest.update(struct.pack('>Q', len(item)))
            if kind is dict:
                for key, member in item.items():
                    visit(key)
                    visit(member)
            else:
                for member in item:
                    visit(member)
        elif item is None:
            digest.update(b'N')
        elif kind is bool:
            digest.update(b'T' if item else b'F')
        elif kind is float:
            digest.update(b'R' + struct.pack('>d', item))
        elif kind is int:
            digest.update(b'I' + struct.pack('>q', item))
        elif kind is str:
            raw = item.encode('utf-8')
            digest.update(b'S' + struct.pack('>Q', len(raw)))
            digest.update(raw)
        else:
            raise ValueError('Unexpected non-JSON scalar in copy oracle')

    visit(value)
    return digest.hexdigest(), containers


def run(source,report,project,manifest_sha):
    base=ROOT/'runs/model-portfolio-migration-v1'
    if (report.exists() or not source.is_relative_to(base/'sources')
            or not project.is_relative_to(ROOT/'projects')
            or not report.is_relative_to(ROOT/'docs/plans/model-portfolio-migration-v1/reports')):
        raise ValueError('Explicit owned source and fresh metadata report required')
    resource.setrlimit(resource.RLIMIT_AS,(18*1024**3,18*1024**3))
    def expired(*_):
        raise TimeoutError('Bounded copy-seam comparison expired')
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(600)
    original=(project/'.tuldok/project.json').read_bytes();manifest=json.loads(original)
    if hashlib.sha256(original).hexdigest()!=manifest_sha:
        raise ValueError('Explicit pinned project header differs')
    records=[]
    for entry in manifest['inventory']:
        relative=Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unconfined canonical metadata member')
        raw=(project/'.tuldok'/entry['path']).read_bytes()
        if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('Canonical source record differs')
        records.append(dict(family=entry['family'],id=entry['id'],document=json.loads(raw)))
    sys.path.insert(0,str(source))
    from project_store import ProjectStore,_json_value
    from project_session import ProjectSession
    _json_value(records)
    files=('project_store.py','project_session.py','project_projection.py')
    before={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}
    expected=dict(project_id=manifest['project_id'],name=manifest['name'],checkpoint=manifest['checkpoint'],records=records)
    expected_sha,source_containers=value_proof(expected);measurements=[]
    for name,invoke in (
            ('Store._snapshot',lambda:ProjectStore._snapshot(None,manifest,records)),
            ('Session.snapshot',lambda:ProjectSession.snapshot.fget(SimpleNamespace(lock=threading.RLock(),_snapshot=expected)))):
        cpu=time.process_time();wall=time.monotonic();value=invoke()
        elapsed_cpu=time.process_time()-cpu;elapsed_wall=time.monotonic()-wall
        actual_sha,owned_containers=value_proof(value,source_containers)
        if actual_sha!=expected_sha:raise ValueError('Snapshot copy changed JSON values/types/float bits')
        value['records'].clear()
        if len(records)!=len(manifest['inventory']):raise ValueError('Snapshot copy shares mutable source state')
        measurements.append(dict(operation=name,CPU_seconds=elapsed_cpu,wall_seconds=elapsed_wall,
            exact_values_types_and_float_bits=True,independent_mutable_tree=True,
            source_mutable_containers=len(source_containers),owned_mutable_containers=len(owned_containers)))
        del value,owned_containers
    if original!=(project/'.tuldok/project.json').read_bytes() or 'torch' in sys.modules:
        raise ValueError('Source checkpoint changed or model runtime imported')
    if before!={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}:
        raise ValueError('Copy implementation changed during comparison')
    if value_proof(expected)[0]!=expected_sha:
        raise ValueError('A source scalar or nested value changed after caller mutation')
    result=dict(schema='rheon.native-snapshot-copy-benchmark.v1',scope=__doc__.strip(),
                source=str(source.relative_to(ROOT)),source_commit=subprocess.check_output(
                    ['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),source_sha256=before,
                project_checkpoint=manifest['checkpoint'],project_manifest_sha256=hashlib.sha256(original).hexdigest(),
                project=str(project.relative_to(ROOT)),all_canonical_bodies_freshly_hashed=True,
                input_records=len(records),canonical_record_bytes=sum(r['bytes'] for r in manifest['inventory']),
                exact_value_sha256=expected_sha,measurements=measurements,
                process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                memory_limit_bytes=18*1024**3,wall_budget_seconds=600,
                Store_or_Session_constructed=False,full_native_Open_qualified=False,
                models_executed=0,project_written=False,SQL_opened=False)
    signal.alarm(0)
    report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(measurements))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args();run(args.source.resolve(),args.report.resolve(),args.project.resolve(),args.manifest_sha256)
