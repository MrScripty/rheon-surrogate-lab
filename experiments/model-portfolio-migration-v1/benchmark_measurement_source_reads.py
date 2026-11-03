"""Compare fixed-checkpoint measurement document lookup work in place.

The canonical bytes are hashed before measurement. A fixture session's adopted
records stand in for the already-open native session. Freshness I/O, SQL,
artifact bodies and whole Open are excluded; this is an allocation/codec
component diagnostic, never a native endpoint latency qualification.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(source, project, report):
    if (not source.is_relative_to(ROOT/'runs/model-portfolio-migration-v1/sources')
            or not project.is_relative_to(ROOT) or report.exists()
            or not report.is_relative_to(ROOT/'docs/plans/model-portfolio-migration-v1/reports')):
        raise ValueError('Bound task source, existing project and fresh task report required')
    resource.setrlimit(resource.RLIMIT_AS, (22*1024**3, 22*1024**3))
    budget = 600
    def expired(_signal, _frame):
        raise TimeoutError('Read-only lookup diagnostic exceeded its wall budget')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(budget)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    header_path = project/'.tuldok/project.json'
    raw_header = header_path.read_bytes()
    header = json.loads(raw_header)
    tracked = subprocess.check_output(['git','-C',str(source),'ls-files','-z']).decode().split('\0')
    tracked = sorted(name for name in tracked if name)
    tracked += sorted(subprocess.check_output(['git','-C',str(source),'ls-files','--others','--exclude-standard','-z']).decode().rstrip('\0').split('\0'))
    tracked = sorted(set(name for name in tracked if name))
    def source_fingerprint():
        rows = [(name, sha((source/name).read_bytes())) for name in tracked]
        return sha(json.dumps(rows, separators=(',',':')).encode())
    source_before = source_fingerprint()
    result = dict(status='reading_fixed_canonical_metadata', created_unix=time.time(), pid=os.getpid(),
        source=str(source.relative_to(ROOT)), project=str(project.relative_to(ROOT)),
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        source_tracked_and_untracked_files=len(tracked), source_sha256=source_before,
        producer_sha256=sha(Path(__file__).read_bytes()), canonical_header_sha256=sha(raw_header),
        checkpoint=header['checkpoint'], records=len(header['inventory']),
        canonical_record_bytes=sum(row['bytes'] for row in header['inventory']),
        memory_limit_bytes=22*1024**3, wall_budget_seconds=budget, CPU_affinity=sorted(os.sched_getaffinity(0)),
        python=platform.python_version(), samples=[], scope=__doc__.strip(),
        freshness_IO_in_timing=False, SQL_opened=False, native_Session_constructed=False,
        native_Store_constructed=False, artifact_bodies_read=0, model_forwards=0,
        project_dataset_or_weight_copies=0, whole_Open_qualified=False, native_endpoint_latency_qualified=False)
    def save():
        temporary = report.with_suffix('.pending')
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
        temporary.replace(report)
    save()
    try:
        records = []
        for entry in header['inventory']:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unconfined canonical path')
            path = project/'.tuldok'/relative
            for part in (path, *path.parents):
                if part.is_symlink():
                    raise ValueError('Symlink in canonical path')
                if part == project:
                    break
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError('Canonical record must be an ordinary file')
            raw = path.read_bytes()
            if len(raw) != entry['bytes'] or sha(raw) != entry['sha256']:
                raise ValueError('Canonical bytes differ from the fixed checkpoint')
            records.append(dict(family=entry['family'], id=entry['id'], document=json.loads(raw)))
        if header_path.read_bytes() != raw_header:
            raise ValueError('Checkpoint changed during loading')
        sys.path.insert(0, str(source))
        import project_measurements
        import project_session
        session = object.__new__(project_session.ProjectSession)
        session._snapshot = dict(project_id=header['project_id'], name=header['name'],
                                 checkpoint=header['checkpoint'], records=records)
        session.available, session.lock = True, threading.RLock()
        session.store = SimpleNamespace(summary=lambda: {key: header[key] for key in ('project_id','name','checkpoint')})
        owner = object.__new__(project_measurements.ProjectMeasurements)
        owner.session = session
        expected = {row['id']: row['document'] for row in records if row['family'] == project_measurements.FAMILY}
        expected_hash = sha(json.dumps(expected, sort_keys=True, separators=(',',':'), allow_nan=False).encode())
        original_copy = project_session.copy_validated_json
        original_material = project_measurements.material_records
        result.update(status='fixed_record_lookup_running', measurement_documents=len(expected),
                      expected_documents_sha256=expected_hash)
        save()
        for index in range(2):
            counters = dict(full_graph_copies=0, records_in_full_graph_copies=0,
                            full_graph_copy_cpu_seconds=0.0, material_decodes=0, material_decode_cpu_seconds=0.0)
            def counted_copy(value):
                top = isinstance(value, dict) and 'records' in value
                if top:
                    counters['full_graph_copies'] += 1
                    counters['records_in_full_graph_copies'] += len(value['records'])
                started = time.process_time()
                try:
                    return original_copy(value)
                finally:
                    if top:
                        counters['full_graph_copy_cpu_seconds'] += time.process_time()-started
            def counted_material(*args, **kwargs):
                counters['material_decodes'] += 1
                started = time.process_time()
                try:
                    return original_material(*args, **kwargs)
                finally:
                    counters['material_decode_cpu_seconds'] += time.process_time()-started
            gc.collect()
            with patch.object(project_session, 'copy_validated_json', counted_copy), \
                    patch.object(project_measurements, 'material_records', counted_material):
                cpu, wall = time.process_time(), time.monotonic()
                observed = owner._documents()
                sample = dict(index=index, cpu_seconds=time.process_time()-cpu,
                              wall_seconds=time.monotonic()-wall, **counters)
            actual_hash = sha(json.dumps(observed, sort_keys=True, separators=(',',':'), allow_nan=False).encode())
            if actual_hash != expected_hash or observed != expected:
                raise ValueError('Lookup documents differ from the literal fixed canonical records')
            sample['documents_sha256'] = actual_hash
            result['samples'].append(sample)
            del observed
            save()
        if source_fingerprint() != source_before or header_path.read_bytes() != raw_header:
            raise ValueError('Source or canonical checkpoint changed during lookup measurement')
        if 'torch' in sys.modules:
            raise ValueError('Neural runtime unexpectedly imported')
        result.update(status='fixed_measurement_source_lookup_completed', source_unchanged=True,
                      canonical_header_unchanged=True, exact_literal_documents=True,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException as error:
        result.update(status='fixed_measurement_source_lookup_failed', error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0)
        result['finished_unix'] = time.time()
        save()
    print(json.dumps({key: result[key] for key in ('status','measurement_documents','samples','peak_RSS_KiB')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.project.resolve(), args.report.resolve())
