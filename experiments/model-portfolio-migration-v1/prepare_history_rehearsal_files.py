"""Populate only the filesystem closure needed by the isolated history rehearsal.

Native startup validates all retained model/recipe materials. Native Stop checks
all original custody receipts. Only C106/C107 publication folders are copied.
This is explicitly a partial copy of the original catalog, never its full backup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

from qualify_native_workspace import ROOT, SOURCE
from published_artifact_inventory import CATALOG, CATALOG_ID


SELECTED_RUNS = ('74240450dfb642599b9aea0d83517a0f', '2a3555fc5d9c44928316c6f1c03064f5')


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def run(snapshot, rehearsal, report):
    scope = (ROOT / 'runs/model-portfolio-migration-v1').resolve()
    if (not snapshot.is_relative_to(scope) or not rehearsal.is_relative_to(scope)
            or not rehearsal.is_dir() or report.exists()):
        raise ValueError('Owned snapshot, existing restored copy and fresh report required')
    catalog = rehearsal / 'private/catalog'
    if (stat.S_IMODE((rehearsal / 'private').stat().st_mode) != 0o700
            or json.loads((catalog / '.tuldok.json').read_bytes())['dataset_id'] != CATALOG_ID
            or json.loads((CATALOG / '.tuldok.json').read_bytes())['dataset_id'] != CATALOG_ID):
        raise ValueError('Exact private source-copy identities required')
    summary = json.loads((snapshot / 'summary.json').read_bytes())
    if sha(snapshot / 'metadata.json') != summary['metadata_sha256']:
        raise ValueError('Snapshot identity changed')
    data = json.loads((snapshot / 'metadata.json').read_bytes())
    table = data['tables']['workspace_runs']
    position = table['columns'].index('document')
    runs = [json.loads(row[position]) for row in table['rows']]
    if any(run['state'] not in ('completed', 'failed', 'interrupted', 'cancelled') for run in runs):
        raise ValueError('Rehearsal source has nonterminal work')
    sys.path.insert(0, str(SOURCE))
    from backup_metadata import tree_manifest, copy_tree, tree_modes

    os.umask(0o077)
    value = dict(schema='rheon.isolated-history-filesystem-preparation.v1',
                 status='inventorying', complete_original_catalog_backup=False,
                 source_snapshot_sha256=summary['metadata_sha256'],
                 copied_trees=[], custody_receipts=[], live_service_changed=False,
                 original_database_opened=False, models_executed=0, controller_started=False)

    def save():
        pending = report.with_suffix('.pending')
        pending.write_text(json.dumps(value, indent=2) + '\n')
        pending.replace(report)

    save()
    trees = []
    for relative in ('model-materials', 'workspace-recipe-materials',
                     *(f'workspace-runs/{run_id}' for run_id in SELECTED_RUNS)):
        original = CATALOG / relative
        target = catalog / relative
        if target.exists():
            raise ValueError('Rehearsal tree already exists; reconcile rather than overwrite')
        inventory = tree_manifest(original)
        trees.append((relative, inventory))
    byte_count = sum(item['bytes'] for _, inventory in trees for item in inventory.values()
                     if item['kind'] == 'file')
    free = shutil.disk_usage(catalog).free
    # Retain room for the copy owner's native stopped backup and later extraction.
    if free < 2 * byte_count + 10 * 1024 ** 3:
        raise ValueError('Insufficient room for this isolated copy and its native backup')
    value.update(status='copying_material_and_selected_publication_closure',
                 tree_bytes=byte_count, free_bytes_before=free)
    save()
    for relative, inventory in trees:
        original = CATALOG / relative
        target = catalog / relative
        copy_tree(original, target, inventory)
        if tree_manifest(target) != inventory or tree_modes(target, inventory) != tree_modes(original, inventory):
            raise ValueError('Native filesystem copy differs')
        value['copied_trees'].append(dict(path=relative, files=sum(
            item['kind'] == 'file' for item in inventory.values()),
            bytes=sum(item['bytes'] for item in inventory.values() if item['kind'] == 'file'),
            inventory_sha256=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()))
        save()
        print('Copied and verified', relative, flush=True)
    for run in runs:
        if run.get('custody_identity') is None or run['id'] in SELECTED_RUNS:
            continue
        relative = (f"workspace-runs/{run['id']}/custody-receipt.json" if run['state'] == 'completed'
                    else f"workspace-runs/.external-{run['id']}/custody/receipt.json")
        original = CATALOG / relative
        target = catalog / relative
        info = original.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 16384
                or target.exists()):
            raise ValueError('Exact bounded original custody receipt required')
        expected = sha(original)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with original.open('rb') as reader, target.open('xb') as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fchmod(writer.fileno(), stat.S_IMODE(info.st_mode))
            os.fsync(writer.fileno())
        if sha(target) != expected or sha(original) != expected:
            raise ValueError('Original custody receipt changed during copy')
        value['custody_receipts'].append(dict(run_id=run['id'], path=relative, sha256=expected))
    value.update(status='verified_partial_catalog_files_for_rehearsal',
                 original_publications_copied=len(SELECTED_RUNS),
                 custody_receipts_copied=len(value['custody_receipts']),
                 metadata_unchanged_by_this_file_copy=True)
    save()
    print(json.dumps({k: v for k, v in value.items() if k not in ('custody_receipts', 'copied_trees')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--rehearsal', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.snapshot.resolve(), args.rehearsal.resolve(), args.report.resolve())
    except BaseException as error:
        print('History rehearsal filesystem preparation failed:', type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
