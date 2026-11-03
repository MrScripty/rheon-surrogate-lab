"""Independently check original canonical bytes and payloads after native Save As.

Reads portable files only. Does not invoke native decoders, SQL, model runtimes,
or credentials. The native operation must already have produced the destination.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import stat
import time


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(project):
    descriptor = json.loads((project / '.tuldok/project.json').read_bytes())
    indexed = {}
    for entry in descriptor['inventory']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Canonical record path escapes its portable project')
        path = project / '.tuldok' / relative
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('Canonical record is not an independent regular file')
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Canonical record identity differs from its inventory')
        key = entry['family'], entry['id']
        if key in indexed:
            raise ValueError('Duplicate canonical identity')
        indexed[key] = entry, path
    return descriptor, indexed


def verify(source, destination, expected_checkpoint, report):
    if report.exists():
        raise ValueError('A fresh verification receipt is required')
    original_manifest = (source / '.tuldok/project.json').read_bytes()
    destination_manifest = (destination / '.tuldok/project.json').read_bytes()
    old, originals = inventory(source)
    new, forked = inventory(destination)
    if old['checkpoint'] != expected_checkpoint or old['project_id'] == new['project_id']:
        raise ValueError('Source checkpoint or independent destination identity differs')
    present = missing = 0
    for key, (entry, path) in originals.items():
        retained = forked.get(key)
        if retained is None or retained[1].read_bytes() != path.read_bytes():
            raise ValueError('Original canonical fact changed or disappeared: ' + str(key))
        if path.stat().st_ino == retained[1].stat().st_ino and path.stat().st_dev == retained[1].stat().st_dev:
            raise ValueError('Fork shares an original canonical file inode')
        if stat.S_IMODE(path.stat().st_mode) != stat.S_IMODE(retained[1].stat().st_mode):
            raise ValueError('Original canonical record mode changed')
        if key[0] != 'project_artifact':
            continue
        declaration = json.loads(path.read_bytes())
        before = source / 'artifacts/sha256' / key[1]
        after = destination / 'artifacts/sha256' / key[1]
        if not before.exists():
            if after.exists():
                raise ValueError('Previously missing dependency appeared without retained source bytes')
            missing += 1
            continue
        for payload in (before, after):
            if (not stat.S_ISREG(payload.lstat().st_mode)
                    or payload.stat().st_size != declaration['bytes'] or digest(payload) != key[1]):
                raise ValueError('Original artifact bytes are missing or changed')
        if before.stat().st_ino == after.stat().st_ino and before.stat().st_dev == after.stat().st_dev:
            raise ValueError('Fork shares an original artifact inode')
        if stat.S_IMODE(before.stat().st_mode) != stat.S_IMODE(after.stat().st_mode):
            raise ValueError('Original artifact mode changed')
        present += 1
    if (source / '.tuldok/project.json').read_bytes() != original_manifest:
        raise ValueError('Source checkpoint changed during verification')
    if (destination / '.tuldok/project.json').read_bytes() != destination_manifest:
        raise ValueError('Destination checkpoint changed during verification')
    result = dict(schema='rheon.native-portfolio-fork-preservation.v1',
        status='all_original_canonical_facts_and_present_artifacts_exact', created_unix=time.time(),
        source=str(source.relative_to(ROOT)), destination=str(destination.relative_to(ROOT)),
        source_project_id=old['project_id'], destination_project_id=new['project_id'],
        source_checkpoint=old['checkpoint'], destination_checkpoint=new['checkpoint'],
        original_records_exact=len(originals), destination_records=len(forked),
        original_families=dict(Counter(key[0] for key in originals)),
        original_present_payloads_exact=present, explicit_original_missing_dependencies=missing,
        independent_inodes=True, original_modes_exact=True, source_manifest_unchanged=True,
        destination_manifest_unchanged=True,
        source_manifest_sha256=hashlib.sha256(original_manifest).hexdigest(),
        runner_sha256=digest(Path(__file__)), new_neural_forwards=0,
        limitations='Byte preservation only; actual native Save/Close/reopen and portfolio ingestion require separate receipts.')
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'original_records_exact',
        'destination_records', 'original_present_payloads_exact', 'explicit_original_missing_dependencies')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--expected-checkpoint', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    verify(args.source.resolve(), args.destination.resolve(), args.expected_checkpoint, args.report.resolve())
