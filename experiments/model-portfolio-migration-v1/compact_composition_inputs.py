"""Retire only duplicate artifacts in the two closed composition inputs.

Canonical metadata stays in place. A durable map precedes every deletion;
interrupted cleanup resumes from it. Reconstruct only when an old input is
actually needed. The published keeper is never modified or hard-linked.
"""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

from compact_retired_catalog import digest, inactive, safe, signature, write
from verify_native_composition import INPUTS
from verify_native_portfolio_fork import inventory

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'runs/model-portfolio-migration-v1'
KEEPER = ROOT/'projects/rheon-model-portfolio-unified-04'
SOURCES = dict(primary=INPUTS[0], current=INPUTS[1])
PLAN = BASE/'discovery/composition-input-payload-compaction-01.json'
REPORT = ROOT/'docs/plans/model-portfolio-migration-v1/reports/composition-input-payload-compaction-01.json'
CUSTODY = ROOT/'docs/plans/model-portfolio-migration-v1/reports/published-composition-custody-01.json'
PUBLICATION = ROOT/'docs/plans/model-portfolio-migration-v1/reports/committed-composition-recovery-01.json'
KEEPER_MANIFEST = '7f1591bd065577c95d75866071756c0128d350edbd4c4a6d2fe412a4558d2e46'
SCHEMA = 'rheon.composition-input-payload-compaction.v1'
MARKER = 'COMPACTED-PAYLOADS.json'


@contextmanager
def closed_roots():
    roots = [KEEPER] + [row[0] for row in SOURCES.values()]
    # Native ProjectStore uses an exclusive flock on the physical root itself.
    # Check readers first, then exclude cooperating opens for the entire pass.
    for root in roots:
        safe(ROOT, str(root.relative_to(ROOT))); inactive(root)
    with ExitStack() as stack:
        for root in sorted(roots):
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            info = os.fstat(fd)
            if info.st_uid != os.getuid():
                raise ValueError('Owned closed project required')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def manifest(root, expected):
    path = safe(root, '.tuldok/project.json')
    if digest(path) != expected:
        raise ValueError('Project manifest changed')


def file_row(path):
    before = path.lstat()
    sha = digest(path)
    after = path.lstat()
    if signature(before) != signature(after) or after.st_uid != os.getuid():
        raise ValueError('Payload identity or ownership changed')
    return dict(sha256=sha, bytes=after.st_size, mode=stat.S_IMODE(after.st_mode),
                allocated_bytes=after.st_blocks*512, links=after.st_nlink)


def matches(path, row):
    actual = file_row(path)
    if any(actual[k] != row[k] for k in ('sha256', 'bytes', 'mode')):
        raise ValueError('Payload bytes or mode changed')


def verify_keeper(plan, *, allow_later_checkpoint=False):
    if allow_later_checkpoint:
        descriptor = json.loads(safe(KEEPER, '.tuldok/project.json').read_bytes())
        if descriptor['project_id'] != plan['keeper_project_id']:
            raise ValueError('Reconstruction keeper project differs')
    else:
        manifest(KEEPER, plan['keeper_manifest_sha256'])
    folder = safe(KEEPER, 'artifacts/sha256')
    present = {p.name for p in folder.iterdir()}
    required = set(plan['keeper_payloads'])
    if not required <= present or (not allow_later_checkpoint and present != required):
        raise ValueError('Keeper payload population changed')
    for sha, row in plan['keeper_payloads'].items():
        matches(safe(folder, sha), dict(row, sha256=sha))


def make_plan():
    if PLAN.exists() or REPORT.exists() or any((s[0]/MARKER).exists() for s in SOURCES.values()):
        raise ValueError('Existing cleanup requires explicit resume, not a new scan')
    publication = json.loads(PUBLICATION.read_bytes())
    custody = json.loads(CUSTODY.read_bytes())
    if (publication['status'] != 'native_committed_project_published_without_payload_copy'
            or custody['status'] != 'all_original_archive_scientific_facts_memberships_and_copied_payloads_exact'
            or custody['composition_manifest_sha256'] != KEEPER_MANIFEST
            or publication['result']['project_id'] != custody['project_id']
            or publication['result']['checkpoint'] != custody['checkpoint']
            or custody['project'] != str(KEEPER.relative_to(ROOT))):
        raise ValueError('Completed native publication and custody required')
    manifest(KEEPER, KEEPER_MANIFEST)
    descriptor, retained = inventory(KEEPER)
    payloads = {}
    for path in sorted(safe(KEEPER, 'artifacts/sha256').iterdir()):
        if not re.fullmatch('[a-f0-9]{64}', path.name):
            raise ValueError('Unexpected artifact path')
        row = file_row(path)
        if row.pop('sha256') != path.name:
            raise ValueError('Content-addressed keeper differs')
        payloads[path.name] = row
    groups = {}
    for name, (source, checkpoint, sha) in SOURCES.items():
        manifest(source, sha)
        old, original = inventory(source)
        if old['checkpoint'] != checkpoint:
            raise ValueError('Source checkpoint changed')
        for key, (_, path) in original.items():
            if key not in retained or path.read_bytes() != retained[key][1].read_bytes():
                raise ValueError('Source metadata absent from published keeper')
        files = {}
        for path in sorted(safe(source, 'artifacts/sha256').iterdir()):
            row = file_row(path)
            if row['sha256'] != path.name or path.name not in payloads:
                raise ValueError('Unique or changed source payload; no deletion allowed')
            matches(safe(KEEPER, 'artifacts/sha256/'+path.name), row)
            if ('project_artifact', path.name) not in original:
                raise ValueError('Undeclared source artifact')
            files[path.name] = row
        groups[name] = dict(source=str(source.relative_to(ROOT)), checkpoint=checkpoint,
                            manifest_sha256=sha, canonical_records=len(original), files=files)
    plan = dict(schema=SCHEMA, created_unix=time.time(), keeper=str(KEEPER.relative_to(ROOT)),
                keeper_manifest_sha256=KEEPER_MANIFEST, keeper_project_id=descriptor['project_id'],
                keeper_payloads=payloads, sources=groups, publication_sha256=digest(PUBLICATION),
                custody_sha256=digest(CUSTODY), runner_sha256=digest(Path(__file__)))
    write(PLAN, plan, True)
    return plan


def load_plan():
    plan = json.loads(PLAN.read_bytes())
    if (plan['schema'] != SCHEMA or plan['keeper'] != str(KEEPER.relative_to(ROOT))
            or plan['keeper_manifest_sha256'] != KEEPER_MANIFEST or set(plan['sources']) != set(SOURCES)):
        raise ValueError('Fixed cleanup scope differs')
    for name, group in plan['sources'].items():
        source, checkpoint, sha = SOURCES[name]
        if (group['source'], group['checkpoint'], group['manifest_sha256']) != (str(source.relative_to(ROOT)), checkpoint, sha):
            raise ValueError('Source identity differs')
        manifest(source, sha)
        for key, row in group['files'].items():
            if (not re.fullmatch('[a-f0-9]{64}', key) or row['sha256'] != key
                    or any(row[k] != plan['keeper_payloads'][key][k] for k in ('bytes', 'mode'))):
                raise ValueError('Invalid reconstruction binding')
    return plan


def marker_value(status):
    return dict(schema=SCHEMA, status=status, manifest=str(PLAN.relative_to(ROOT)),
                manifest_sha256=digest(PLAN), keeper=str(KEEPER.relative_to(ROOT)),
                reconstruct_before_native_reuse=True)


def compact(resume=False):
    with closed_roots():
        plan = load_plan() if resume else make_plan()
        verify_keeper(plan)
        for name, group in plan['sources'].items():
            source = SOURCES[name][0]
            folder = safe(source, 'artifacts/sha256')
            if not {p.name for p in folder.iterdir()} <= set(group['files']):
                raise ValueError('Unexpected source payload')
            marker = source/MARKER
            if marker.exists():
                value = json.loads(marker.read_bytes())
                if value['manifest_sha256'] != digest(PLAN) or value['schema'] != SCHEMA:
                    raise ValueError('Compaction marker differs')
            elif resume and {p.name for p in folder.iterdir()} != set(group['files']):
                raise ValueError('Missing unmarked source payload')
            for sha, row in group['files'].items():
                path = safe(folder, sha)
                if path.exists(): matches(path, row)
            write(marker, marker_value('compacting'), not marker.exists())
        result = dict(schema=SCHEMA, status='removing_verified_duplicates', created_unix=time.time(),
                      manifest=str(PLAN.relative_to(ROOT)), manifest_sha256=digest(PLAN),
                      free_bytes_before=shutil.disk_usage(ROOT).free, unique_payloads_deleted=0,
                      payload_copies=0, cold_lifecycle_verified=False)
        write(REPORT, result, not REPORT.exists())
        for name, group in plan['sources'].items():
            source = SOURCES[name][0]
            for sha, row in group['files'].items():
                path = safe(source, 'artifacts/sha256/'+sha)
                if not path.exists(): continue
                matches(path, row); matches(safe(KEEPER, 'artifacts/sha256/'+sha), row)
                path.unlink()
            fd = os.open(source/'artifacts/sha256', os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            manifest(source, group['manifest_sha256'])
            write(source/MARKER, marker_value('compacted_requires_reconstruction'))
        verify_keeper(plan)
        result.update(status='completed_verified_duplicate_payload_retirement', completed_unix=time.time(),
                      files_removed=sum(len(g['files']) for g in plan['sources'].values()),
                      logical_bytes_removed=sum(r['bytes'] for g in plan['sources'].values() for r in g['files'].values()),
                      allocated_bytes_of_removed_files=sum(r['allocated_bytes'] for g in plan['sources'].values() for r in g['files'].values() if r['links']==1),
                      keeper_complete_and_rehashed=True, source_canonical_metadata_unchanged=True,
                      free_bytes_after=shutil.disk_usage(ROOT).free)
        write(REPORT, result)
        print(json.dumps(result), flush=True)


def reconstruct(name):
    with closed_roots():
        plan = load_plan(); verify_keeper(plan, allow_later_checkpoint=True)
        group = plan['sources'][name]; source = SOURCES[name][0]
        marker = source/MARKER
        if json.loads(marker.read_bytes())['manifest_sha256'] != digest(PLAN):
            raise ValueError('Reconstruction map changed')
        if shutil.disk_usage(ROOT).free < sum(r['bytes'] for r in group['files'].values()) + 1024**3:
            raise ValueError('Insufficient reconstruction space')
        for sha, row in group['files'].items():
            path = safe(source, 'artifacts/sha256/'+sha)
            if not path.exists():
                # A fresh ordinary file; an interrupted partial copy must be
                # explicitly diagnosed, never overwritten on a guessed retry.
                with safe(KEEPER, 'artifacts/sha256/'+sha).open('rb') as src, path.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 4*1024*1024); dst.flush(); os.fsync(dst.fileno())
                path.chmod(row['mode'])
            matches(path, row)
        manifest(source, group['manifest_sha256'])
        write(marker, marker_value('reconstructed_payloads_verified'))
        print(json.dumps(dict(status='reconstructed_independent_payloads', source=name)))


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('compact', 'resume', 'reconstruct'))
    parser.add_argument('--source', choices=tuple(SOURCES))
    args = parser.parse_args()
    if args.command == 'reconstruct':
        if args.source is None: parser.error('Reconstruction requires an explicit source')
        reconstruct(args.source)
    else:
        if args.source is not None: parser.error('Compaction always uses both fixed closed sources')
        compact(args.command == 'resume')
