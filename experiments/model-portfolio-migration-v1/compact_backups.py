"""Remove verified duplicate payloads from two retired migration backups.

Keep the complete current backup and each older backup's unique metadata and
payloads. The manifest records every removed file and its retained source. An
older backup must be materialized before using the original restore tools.
Only the two explicit retired roots are eligible for compaction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/dependency-migration-v1'
KEEPER = BASE / 'completed-current-backup-04'
RETIRED = ('original-schema8-backup-01', 'original-schema8-backup-02')
SCHEMA = 'retired-backup-compaction-v1'


def digest(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError('Expected regular file: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    after = path.lstat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('File changed during verification: ' + str(path))
    return h.hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def payload(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid backup relative path')
    path = root / 'catalog' / relative
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Symlink in backup path: ' + str(parent))
    return path


def verify_backup(root):
    receipt_path = root / 'backup-receipt.json'
    receipt_hash = digest(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if digest(root / 'metadata.json') != receipt['metadata_sha256']:
        raise ValueError('Backup metadata hash mismatch: ' + str(root))
    files = {}
    for name, record in receipt['files'].items():
        path = payload(root, name)
        if record['kind'] == 'directory':
            if not path.is_dir():
                raise ValueError('Missing backup directory: ' + str(path))
        elif record['kind'] == 'file':
            if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
                raise ValueError('Backup payload hash mismatch: ' + str(path))
            files[name] = record
        else:
            raise ValueError('Unsupported backup entry kind')
    return receipt_hash, files


def scan(manifest):
    if manifest.exists():
        raise ValueError('Use a fresh manifest path')
    keeper_hash, keeper_files = verify_backup(KEEPER)
    groups = []
    for name in RETIRED:
        root = BASE / name
        if (root / 'COMPACTED-BACKUP.json').exists():
            raise ValueError('Backup already compacted; use its existing manifest')
        receipt_hash, files = verify_backup(root)
        duplicates = []
        for relative, record in files.items():
            kept = keeper_files.get(relative)
            if kept is not None and (record['bytes'], record['sha256']) == (
                    kept['bytes'], kept['sha256']):
                duplicates.append(dict(path=relative, bytes=record['bytes'],
                                       sha256=record['sha256']))
        groups.append(dict(name=name, receipt_sha256=receipt_hash,
                           duplicates=duplicates,
                           preserved_unique_payload_files=len(files) - len(duplicates)))
    value = dict(schema=SCHEMA, created_unix=time.time(),
                 authorization='User: we dont need the identical backup files',
                 keeper=str(KEEPER.relative_to(ROOT)),
                 keeper_receipt_sha256=keeper_hash, backups=groups,
                 removed_file_count=sum(len(g['duplicates']) for g in groups),
                 removed_logical_bytes=sum(f['bytes'] for g in groups for f in g['duplicates']),
                 scope='Only duplicate catalog payloads; metadata and original receipts retained')
    write_new(manifest, value)
    print(json.dumps({k: v for k, v in value.items() if k != 'backups'}))


def load(manifest):
    value = json.loads(manifest.read_text())
    if (value['schema'] != SCHEMA or value['keeper'] != str(KEEPER.relative_to(ROOT))
            or tuple(g['name'] for g in value['backups']) != RETIRED):
        raise ValueError('Manifest does not match fixed compaction scope')
    if digest(KEEPER / 'backup-receipt.json') != value['keeper_receipt_sha256']:
        raise ValueError('Retained backup receipt changed')
    keeper_files = json.loads((KEEPER / 'backup-receipt.json').read_text())['files']
    for group in value['backups']:
        root = BASE / group['name']
        if digest(root / 'backup-receipt.json') != group['receipt_sha256']:
            raise ValueError('Original backup receipt changed')
        original_files = json.loads((root / 'backup-receipt.json').read_text())['files']
        names = set()
        for item in group['duplicates']:
            name = item['path']
            if name in names:
                raise ValueError('Duplicate manifest entry')
            names.add(name)
            for expected in (keeper_files.get(name), original_files.get(name)):
                if expected is None or expected.get('kind') != 'file' or (
                        expected['bytes'], expected['sha256']) != (item['bytes'], item['sha256']):
                    raise ValueError('Compaction entry differs from original backup receipts')
            payload(root, name)
    if (sum(len(g['duplicates']) for g in value['backups']) != value['removed_file_count']
            or sum(f['bytes'] for g in value['backups'] for f in g['duplicates'])
            != value['removed_logical_bytes']):
        raise ValueError('Manifest totals differ from entries')
    return value


def apply(manifest, report):
    manifest = manifest.resolve()
    if report.exists():
        raise ValueError('Use a fresh completion report path')
    value = load(manifest)
    # Verify the complete keeper, not merely the copies selected for removal.
    verify_backup(KEEPER)
    for group in value['backups']:
        for item in group['duplicates']:
            original = payload(BASE / group['name'], item['path'])
            if original.exists() and digest(original) != item['sha256']:
                raise ValueError('Original copy changed; refusing removal')
            if digest(payload(KEEPER, item['path'])) != item['sha256']:
                raise ValueError('Retained copy changed; refusing removal')
    for group in value['backups']:
        root = BASE / group['name']
        marker = root / 'COMPACTED-BACKUP.json'
        identity = dict(schema=SCHEMA, manifest=str(manifest.resolve()),
                        manifest_sha256=digest(manifest), keeper=str(KEEPER),
                        status='Compacted: materialize before legacy restore',
                        unique_metadata_and_original_receipt_retained=True)
        if marker.exists():
            if json.loads(marker.read_text()) != identity:
                raise ValueError('Existing compaction marker differs')
        else:
            write_new(marker, identity)
        for item in group['duplicates']:
            path = payload(root, item['path'])
            path.unlink(missing_ok=True)
    # Hash again after deletion, including all unique files in compacted roots.
    verify_backup(KEEPER)
    for group in value['backups']:
        removed = {f['path'] for f in group['duplicates']}
        root = BASE / group['name']
        receipt = json.loads((root / 'backup-receipt.json').read_text())
        if digest(root / 'metadata.json') != receipt['metadata_sha256']:
            raise ValueError('Unique metadata changed')
        for name, record in receipt['files'].items():
            if record['kind'] == 'file' and name not in removed:
                if digest(payload(root, name)) != record['sha256']:
                    raise ValueError('Unique payload verification failed')
    result = dict(schema=SCHEMA, completed_unix=time.time(),
                  manifest=str(manifest.relative_to(ROOT)), manifest_sha256=digest(manifest),
                  status='complete', files_removed=value['removed_file_count'],
                  logical_bytes_removed=value['removed_logical_bytes'],
                  complete_keeper_verified_after_removal=True,
                  unique_metadata_and_payloads_verified=True)
    write_new(report, result)
    print(json.dumps(result))


def materialize(manifest, name, destination):
    value = load(manifest)
    if destination.exists():
        raise ValueError('Destination must not exist')
    group = next(g for g in value['backups'] if g['name'] == name)
    root = BASE / name
    shutil.copytree(root, destination, ignore=shutil.ignore_patterns('COMPACTED-BACKUP.json'))
    for item in group['duplicates']:
        source = payload(KEEPER, item['path'])
        if digest(source) != item['sha256']:
            raise ValueError('Retained source hash mismatch')
        target = payload(destination, item['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    verify_backup(destination)
    print(json.dumps(dict(status='materialized-and-verified', destination=str(destination))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for action in ('scan', 'apply', 'materialize'):
        p = sub.add_parser(action)
        p.add_argument('--manifest', required=True, type=Path)
        if action == 'apply':
            p.add_argument('--report', required=True, type=Path)
        if action == 'materialize':
            p.add_argument('--backup-name', choices=RETIRED, required=True)
            p.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'scan':
        scan(args.manifest)
    elif args.command == 'apply':
        apply(args.manifest, args.report)
    else:
        materialize(args.manifest, args.backup_name, args.destination)


if __name__ == '__main__':
    main()
