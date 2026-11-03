"""Complete the owned rehearsal's current-catalog publication filesystem.

Copy missing publication files only. Refuse to replace a differing retained
file, verify original artifact declarations, bytes, modes and independent inodes.
This is preparation for native owner extraction, not a full original backup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time

from qualify_native_workspace import ROOT, SOURCE
from catalog_metadata_snapshot import CATALOG, CATALOG_ID

BASE = ROOT / 'runs/model-portfolio-migration-v1'
TARGET = BASE / 'isolated-history-metadata-restore-01/private/catalog'


def save(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def prepare(report):
    if report.exists() or not TARGET.is_dir():
        raise ValueError('Fresh report and existing owned rehearsal catalog required')
    for root in (CATALOG, TARGET):
        if json.loads((root / '.tuldok.json').read_bytes())['dataset_id'] != CATALOG_ID:
            raise ValueError('Original or rehearsal source identity differs')
    source = BASE / 'discovery/run-reconciliation-02.json'
    raw = source.read_bytes()
    entries = [row for row in json.loads(raw)['records']
               if row['epoch'] == 'current_catalog' and row['run']['state'] == 'completed']
    if len(entries) != 291 or len({row['run']['id'] for row in entries}) != 291:
        raise ValueError('Expected all291 current completed publications')
    sys.path.insert(0, str(SOURCE))
    from backup_metadata import tree_manifest, tree_modes, canonical
    inventories = []
    for row in entries:
        run = row['run']
        relative = Path('workspace-runs') / run['id']
        original = CATALOG / relative
        target = TARGET / relative
        inventory = tree_manifest(original)
        modes = tree_modes(original, inventory)
        for artifact in run['artifacts']:
            actual = inventory.get(artifact['name'])
            if actual != dict(kind='file', bytes=artifact['bytes'], sha256=artifact['sha256']):
                raise ValueError('Original Run artifact declaration differs from its publication')
        current = tree_manifest(target) if target.exists() else {}
        if set(current) - set(inventory) or any(current[name] != inventory[name] for name in current):
            raise ValueError('Existing task-owned publication has differing or extra files')
        inventories.append((run['id'], relative, inventory, modes, current))
    needed = sum(entry['bytes'] for _, _, inventory, _, current in inventories
                 for name, entry in inventory.items() if name not in current and entry['kind'] == 'file')
    free = shutil.disk_usage(TARGET).free
    if free < needed + 10 * 1024 ** 3:
        raise ValueError('Insufficient reserve for additional publication copies')
    result = dict(schema='rheon.current-history-publication-preparation.v1', status='copying_missing_publication_files',
                  created_unix=time.time(), source_reconciliation_sha256=hashlib.sha256(raw).hexdigest(),
                  current_completed_Runs=291, expected_new_file_bytes=needed, free_bytes_before=free,
                  copied_files=0, copied_bytes=0, rows=[], original_catalog_changed=False,
                  original_database_opened=False, native_import_performed=False,
                  complete_original_catalog_backup=False, new_neural_forwards=0,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(report, result)
    for run_id, relative, inventory, modes, current in inventories:
        original, target = CATALOG / relative, TARGET / relative
        target.mkdir(parents=True, exist_ok=True)
        for name, entry in inventory.items():
            path = target / name
            if entry['kind'] == 'directory':
                path.mkdir(parents=True, exist_ok=True)
                continue
            if name not in current:
                path.parent.mkdir(parents=True, exist_ok=True)
                with (original / name).open('rb') as src, path.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                    dst.flush()
                    os.fsync(dst.fileno())
                result['copied_files'] += 1
                result['copied_bytes'] += entry['bytes']
        for name, mode in sorted(modes.items(), key=lambda pair: pair[0].count('/'), reverse=True):
            os.chmod(target / name, mode)
        if (tree_manifest(target) != inventory or tree_modes(target, inventory) != modes
                or tree_manifest(original) != inventory or tree_modes(original, inventory) != modes):
            raise ValueError('Original or copied publication bytes/modes changed')
        for name, entry in inventory.items():
            if entry['kind'] != 'file':
                continue
            before, after = (original / name).stat(), (target / name).stat()
            if (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino):
                raise ValueError('Copied publication shares an original inode')
        result['rows'].append(dict(run_id=run_id, files=sum(entry['kind'] == 'file' for entry in inventory.values()),
                                  bytes=sum(entry.get('bytes', 0) for entry in inventory.values()),
                                  original_manifest_sha256=hashlib.sha256(canonical(inventory)).hexdigest(),
                                  original_modes_sha256=hashlib.sha256(canonical(modes)).hexdigest(),
                                  independent_inodes=True))
        save(report, result)
    result.update(status='all291_current_publications_exact_in_owned_rehearsal',
                  free_bytes_after=shutil.disk_usage(TARGET).free)
    save(report, result)
    print(json.dumps({key: result[key] for key in ('status', 'current_completed_Runs', 'copied_files', 'copied_bytes')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.report.resolve())
