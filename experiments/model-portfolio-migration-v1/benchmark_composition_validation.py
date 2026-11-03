"""Measure selected project codec work on pinned canonical metadata in place.

This is a read-only component diagnostic, not native Open qualification. It
never opens a ProjectStore, SQL, payload, model runtime or project fixture copy.
Two sequential calls expose within-call hashing work and call-to-call variation.
"""
import argparse
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
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(source, project, expected_manifest, report, component='composition'):
    if (not source.is_relative_to(BASE / 'sources') or not project.is_relative_to(ROOT)
            or not report.is_relative_to(ROOT / 'docs/plans/model-portfolio-migration-v1/reports')
            or report.exists() or component not in ('composition','operations')):
        raise ValueError('Explicit task source, existing project and fresh task report required')
    resource.setrlimit(resource.RLIMIT_AS, (18 * 1024**3, 18 * 1024**3))
    header = project / '.tuldok/project.json'
    original = header.read_bytes()
    if sha(original) != expected_manifest:
        raise ValueError('Canonical checkpoint differs from the selected workload')
    manifest = json.loads(original)
    files = ('project_composition.py', 'project_fork.py', 'project_history.py', 'project_store.py', 'project_session.py')
    fingerprints = {name: sha((source / name).read_bytes()) for name in files}
    result = dict(schema='rheon.project-codec-work.v1', status='reading_fixed_canonical_metadata',
                  component=component,runner_sha256=sha(Path(__file__).read_bytes()),
                  created_unix=time.time(), source=str(source.relative_to(ROOT)),
                  source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                  source_worktree_dirty=bool(subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()),
                  source_sha256=fingerprints, project=str(project.relative_to(ROOT)),
                  checkpoint=manifest['checkpoint'], manifest_sha256=expected_manifest,
                  records=len(manifest['inventory']), canonical_record_bytes=sum(e['bytes'] for e in manifest['inventory']),
                  python=platform.python_version(), CPU_affinity=sorted(os.sched_getaffinity(0)),
                  sample_policy='Two sequential same-input component calls; no end-to-end or uncontended UI latency claim.',
                  environment_note='Original cold Open is active on another CPU; native SQL qualification may also be active.',
                  samples=[], scope=__doc__.strip(), memory_limit_bytes=18 * 1024**3,
                  project_written=False, payload_copies=0, model_forwards=0, SQL_opened=False,
                  full_native_Open_qualified=False)

    def save():
        report.write_text(json.dumps(result, indent=2) + '\n')

    def expired(_signal, _frame):
        raise TimeoutError('Composition diagnostic budget expired')

    save()
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    try:
        records = []
        for entry in manifest['inventory']:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Canonical record path is not confined')
            path = project / '.tuldok' / relative
            for part in (path, *path.parents):
                if part.is_symlink():
                    raise ValueError('Canonical record path contains a symlink')
                if part == project:
                    break
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError('Canonical record is not an ordinary file')
            raw = path.read_bytes()
            if len(raw) != entry['bytes'] or sha(raw) != entry['sha256']:
                raise ValueError('Canonical record bytes differ from the selected checkpoint')
            records.append(dict(family=entry['family'], id=entry['id'], document=json.loads(raw)))
        if header.read_bytes() != original:
            raise ValueError('Canonical checkpoint changed during the diagnostic read')
        sys.path.insert(0, str(source))
        import project_composition as native
        original_hash, original_encode = native._record_sha, native._encode
        counts = {}

        def counted_hash(record):
            counts['record_hash_calls'] += 1
            return original_hash(record)

        def counted_encode(value):
            encoded = original_encode(value)
            counts['encode_calls'] += 1
            counts['encoded_bytes'] += len(encoded)
            return encoded

        if component=='composition':
            native._record_sha, native._encode = counted_hash, counted_encode
            invoke=lambda:native.decode_records(records,project_id=manifest['project_id'])
        else:
            import project_fork
            from project_session import operation_records
            deepcopy=project_fork.copy.deepcopy
            def counted_copy(value):
                if type(value) is dict and {'family','id','document'}<=set(value):
                    counts['active_envelope_copies']+=1
                    family=value['family'];families=counts['active_copied_families']
                    families[family]=families.get(family,0)+1
                return deepcopy(value)
            # Replace this module's reference only; recursive deepcopy and
            # the receipt owner's independent copies remain uninstrumented.
            project_fork.copy=SimpleNamespace(deepcopy=counted_copy)
            invoke=lambda:operation_records(records,project_id=manifest['project_id'])
        result['status'] = 'project_codec_measurement_running'
        save()
        for number in (1, 2):
            counts = (dict(record_hash_calls=0,encode_calls=0,encoded_bytes=0) if component=='composition'
                      else dict(active_envelope_copies=0,active_copied_families={}))
            cpu, wall = time.process_time(), time.monotonic()
            decoded = invoke()
            cpu, wall = time.process_time() - cpu, time.monotonic() - wall
            # Independent compact representation, outside the measured calls.
            values = sha(json.dumps(decoded, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode())
            result['samples'].append(dict(number=number, cpu_seconds=cpu, wall_seconds=wall,
                                          work=dict(counts), decoded_values_sha256=values))
            save()
        if len({s['decoded_values_sha256'] for s in result['samples']}) != 1:
            raise ValueError('Repeated codec calls returned different values')
        result['status'] = 'completed_read_only_project_codec_measurement'
    except BaseException as error:
        result.update(status='failed_or_incomplete_project_codec_measurement',
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        result.update(manifest_unchanged=header.read_bytes() == original,
                      source_unchanged=fingerprints == {name: sha((source / name).read_bytes()) for name in files},
                      neural_runtime_imported='torch' in sys.modules,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_unix=time.time())
        save()
    if not result['manifest_unchanged'] or not result['source_unchanged'] or result['neural_runtime_imported']:
        raise ValueError('Diagnostic source or checkpoint boundary changed')
    print(json.dumps(dict(status=result['status'], samples=result['samples'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--component',choices=('composition','operations'),default='composition')
    args = parser.parse_args()
    run(args.source.resolve(), args.project.resolve(), args.manifest_sha256, args.report.resolve(),args.component)
