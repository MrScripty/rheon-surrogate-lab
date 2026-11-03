"""Retire only duplicate payloads from one fixed, verified rehearsal backup.

Requires the completed out-of-place compaction and actual native restore proof.
Original metadata/receipt and a relocation marker remain at the retired path.
The complete native compact backup stays usable directly by restore_catalog.
No live catalog, unique weights or other backup root is an eligible target.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import traceback

from qualify_native_workspace import ROOT, SOURCE


RUN = ROOT / 'runs/model-portfolio-migration-v1'
REPORTS = ROOT / 'docs/plans/model-portfolio-migration-v1/reports'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def retire(attempt, report):
    require(attempt in (3, 4), 'Only fixed retired rehearsal backups are eligible')
    source = RUN / 'isolated-history-metadata-restore-01/private' / (
        f'extraction-exchange-native-history-extraction-rehearsal-{attempt:02}') / 'stopped-copy'
    keeper = RUN / f'native-backup-compaction-{attempt:02}/private/backup'
    proof_path = REPORTS / f'native-backup-compaction-{attempt:02}.json'
    require(not report.exists() and report.is_relative_to(REPORTS), 'Fresh task report required')
    proof = json.loads(proof_path.read_text())
    require(proof.get('format') == 'rheon_native_backup_compaction_restore_v1'
            and proof.get('attempt') == attempt
            and proof['status'] == 'qualified_compact_backup_and_native_restore'
            and all(proof.get(key) is True for key in ('compact_copy_verified','native_restore_verified',
                                                      'postgres_stopped','temporary_restore_removed')),
            'Completed compact copy, native restore, shutdown and temporary cleanup required')
    require(proof['source'] == str(source.relative_to(ROOT))
            and proof['compact_backup'] == str(keeper.relative_to(ROOT)), 'Proof paths differ from fixed scope')
    marker = source/'COMPACTED-BACKUP.json'
    require(not marker.exists(), 'A prior retirement must be reconciled explicitly')
    for path in (source, keeper, source/'catalog', keeper/'catalog'):
        info = path.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(), 'Ordinary owned backup roots required')
    sys.path.insert(0, str(SOURCE))
    from backup_metadata import tree_manifest, tree_modes, _remove_partial_contents, write_json
    require(all(sha(SOURCE/name) == expected for name, expected in proof['source_sha256'].items()),
            'Qualified native backup source changed')
    receipt_hash = sha(source/'backup-receipt.json')
    require(receipt_hash == proof['source_receipt_sha256'] == sha(keeper/'backup-receipt.json'),
            'Original and complete keeper receipts differ')
    receipt = json.loads((source/'backup-receipt.json').read_text())
    metadata_hash = sha(source/'metadata.json')
    require(metadata_hash == receipt['metadata_sha256'] == sha(keeper/'metadata.json'),
            'Exact original SQL metadata must remain in both backups')
    require(tree_manifest(source/'catalog') == receipt['files']
            and tree_manifest(keeper/'catalog') == receipt['files'],
            'Every original and retained payload must match the original receipt')
    require(tree_modes(source/'catalog', receipt['files']) == tree_modes(keeper/'catalog', receipt['files']),
            'Original and keeper permissions differ')
    expected_files = sum(entry['kind'] == 'file' for entry in receipt['files'].values())
    expected_bytes = sum(entry.get('bytes', 0) for entry in receipt['files'].values())
    journal = dict(format='rheon_retired_native_backup_payloads_v1', status='retiring_payloads',
        authorization='User: we dont need the identical backup files',
        original_backup_receipt_sha256=receipt_hash, original_metadata_sha256=metadata_hash,
        complete_keeper=str(keeper.relative_to(ROOT)),
        native_restore_proof=str(proof_path.relative_to(ROOT)), native_restore_proof_sha256=sha(proof_path),
        source_catalog_files=expected_files, source_catalog_bytes=expected_bytes,
        original_metadata_and_receipt_retained=True, unique_weight_retirement=False)
    value = dict(format='rheon_rehearsal_backup_retirement_v1', status='verified_pending_retirement',
        created_unix=time.time(), attempt=attempt, source=str(source.relative_to(ROOT)),
        keeper=str(keeper.relative_to(ROOT)), original_live_catalog_modified=False,
        metadata_deleted=False, unique_weights_deleted=False,
        journal=journal, completed=False, runner_sha256=sha(Path(__file__)))

    def save():
        pending = report.with_suffix('.pending')
        pending.write_text(json.dumps(value, indent=2)+'\n')
        pending.replace(report)

    os.umask(0o077)
    save()
    parent_fd = os.open(source, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    catalog_fd = None
    try:
        before = os.stat('catalog', dir_fd=parent_fd, follow_symlinks=False)
        catalog_fd = os.open('catalog', os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=parent_fd)
        held = os.fstat(catalog_fd)
        require((before.st_dev,before.st_ino,before.st_uid) == (held.st_dev,held.st_ino,os.getuid()),
                'Retired catalog binding changed')
        # Durable relocation proof precedes removal. It points to a complete,
        # byte/mode-equal backup whose actual native restore already passed.
        write_json(marker, journal)
        value['status'] = 'retiring_duplicate_payloads'
        save()
        _remove_partial_contents(catalog_fd)
        named = os.stat('catalog', dir_fd=parent_fd, follow_symlinks=False)
        require((named.st_dev,named.st_ino) == (held.st_dev,held.st_ino), 'Retired catalog identity changed')
        os.rmdir('catalog', dir_fd=parent_fd)
        os.fsync(parent_fd)
        require(sha(source/'metadata.json') == metadata_hash
                and sha(source/'backup-receipt.json') == receipt_hash,
                'Retained original metadata or receipt changed')
        require(tree_manifest(keeper/'catalog') == receipt['files'], 'Complete keeper changed during retirement')
        journal['status'] = 'duplicate_payloads_retired_use_complete_keeper'
        write_json(marker, journal)
        value.update(status='completed_verified_duplicate_backup_payload_retirement',
                     completed=True, removed_logical_bytes=expected_bytes,
                     removed_files=expected_files, metadata_and_receipt_retained=True,
                     complete_keeper_still_verified=True,
                     available_bytes_after=os.statvfs(RUN).f_bavail*os.statvfs(RUN).f_frsize)
        save()
    except BaseException as error:
        value.update(status='incomplete_retirement_keep_complete_keeper', error_type=type(error).__name__,
            error_frames=[dict(file=Path(f.filename).name,function=f.name,line=f.lineno)
                          for f in traceback.extract_tb(error.__traceback__)])
        save()
        raise
    finally:
        if catalog_fd is not None:
            os.close(catalog_fd)
        os.close(parent_fd)
    print(json.dumps({key:value[key] for key in ('status','removed_logical_bytes','removed_files',
                                                'metadata_and_receipt_retained','available_bytes_after')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=int,choices=(3,4),required=True)
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    try:
        retire(args.attempt,args.report.resolve())
    except BaseException as error:
        print('Retired-backup payload removal incomplete:',type(error).__name__,file=sys.stderr)
        raise SystemExit(1)
