"""Managed experiment with measured API costs and verified artifact-byte reuse.

Current owner authorization and terminal Run inventory are always read normally.
Only already downloaded exact bytes are reused; no private Run directory is read.
"""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write
from manage import execute


def cached_collector(original, receipts):
    known = {}
    def collect(reader, run, row, target):
        key = row['sha256'], row['bytes']
        destination = target/row['name']
        previous = known.get(key)
        started = time.perf_counter()
        if previous is not None and not destination.exists() and not destination.is_symlink():
            source, source_run_id = previous
            if source.is_symlink() or not source.is_file() or source.stat().st_size != row['bytes'] or sha(source) != row['sha256']:
                raise ValueError('Previously verified local artifact changed')
            os.link(source, destination, follow_symlinks=False)
            result = dict(run_id=run['id'], name=row['name'], output=str(destination), sha256=row['sha256'],
                          bytes=row['bytes'], verification='current_published_hash_matches_verified_local_bytes',
                          source_run_id=source_run_id, source=str(source))
        else:
            result = original(reader, run, row, target)
        if destination.is_symlink() or destination.stat().st_size != row['bytes'] or sha(destination) != row['sha256']:
            raise ValueError('Collected artifact differs from current published inventory')
        known[key] = destination, run['id']
        receipts.append(dict(run_id=run['id'], name=row['name'], bytes=row['bytes'],
                             elapsed_s=time.perf_counter()-started, verification=result['verification']))
        return result
    return collect


def managed(args):
    config = read(args.config)
    source = ROOT/config['source_lab']/'experiments'
    sys.path[:0] = [str(source/'pressure-throughput-feedback-v1'), str(source/'dependency-migration-v1')]
    import run_study
    original_owner = run_study.OwnerHTTP
    original_collector = run_study.collect_artifact
    calls, downloads = [], []
    class MeasuredOwner(original_owner):
        def call(self, operation, arguments):
            started = time.perf_counter()
            status = 'failed'
            try:
                result = super().call(operation, arguments)
                status = 'completed'
                return result
            finally:
                calls.append(dict(operation=operation, status=status, elapsed_s=time.perf_counter()-started))
                print('API', operation, status, round(calls[-1]['elapsed_s'], 3), flush=True)
    run_study.OwnerHTTP = MeasuredOwner
    run_study.collect_artifact = cached_collector(original_collector, downloads)
    started = time.perf_counter()
    try:
        execute(args)
    finally:
        run_study.OwnerHTTP = original_owner
        run_study.collect_artifact = original_collector
        if args.output.is_dir():
            write(args.output/'readback-profile.json', dict(elapsed_s=time.perf_counter()-started,
                calls=calls, artifacts=downloads, source_sha256=sha(__file__),
                authority='Unchanged current owner API, typed output and terminal Run inventory checks; no retries',
                reused_artifacts=sum(r['verification']=='current_published_hash_matches_verified_local_bytes' for r in downloads)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'assembly', 'qualification', 'tuldok', 'session-file', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['endpoint', 'catalog-id', 'project-alias']:
        parser.add_argument('--'+name, required=True)
    managed(parser.parse_args())
