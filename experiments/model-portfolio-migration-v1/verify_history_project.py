"""Compare extracted historical facts with the authorized private SQL snapshot.

This is a literal preservation check, independent of Tuldok's decoders. It
opens no service, credentials or model runtime and prints no private SQL rows.
Run after publication, then again after the independent cold Open and Close.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[2]
ORIGIN = 'a7da4a8de53541a2a0e2aad08005bb71'
SELECTED = {
    'study_history': ('model_studies', 'study', {
        '18f9d041a5e647628660e8cb8f634b6d', '8905e808a42f400292d21deb82c94ddd'}),
    'trial_history': ('model_trials', 'trial', {
        '9d9e45f10fe6455dae8503a051f65b9f', 'cc783784295a4a18ad8218e3bfd21578'}),
    'run_history': ('workspace_runs', 'run', {
        '74240450dfb642599b9aea0d83517a0f', '2a3555fc5d9c44928316c6f1c03064f5'}),
}
# The external-project contract excludes these seven local continuation fields.
# Everything else in each original Run must survive literally, including its
# metrics, errors, timestamps, artifacts and captured input/source identities.
LOCAL_CONTINUATION = {
    'worker', 'supervisor', 'python', 'worker_command', 'process_phase',
    'release_path', 'publication_continuation',
}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def files(directory):
    result = {}
    for path in directory.rglob('*'):
        relative = path.relative_to(directory)
        if '.tuldok-local' in relative.parts:
            continue
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), 'Portable tree contains a symlink')
        if stat.S_ISREG(mode):
            result[relative.as_posix()] = {
                'sha256': digest(path), 'bytes': path.stat().st_size,
                'mode': stat.S_IMODE(mode),
            }
        else:
            require(stat.S_ISDIR(mode), 'Portable tree contains a special file')
    return result


def display(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def original_fact_checks(snapshot, records, *, all_current=False, all_archive=False):
    """Compare original scientific observations without native codecs or IO."""
    checks = []
    scope = SELECTED
    if all_current or all_archive:
        # Independent literal projections: do not call Tuldok's decoders or use
        # the extraction selection/report as the original membership oracle.
        scope = {
            'study_history': ('model_studies', 'study', None),
            'trial_history': ('model_trials', 'trial', None),
            'run_history': ('workspace_runs', 'run', None),
            'model_definition': ('model_definition_drafts', 'current', None),
            'pipeline_graph': ('workspace_graphs', 'current', None),
            'model_material': ('model_materials', 'value', None),
            'recipe_material': ('workspace_recipe_materials', 'value', None),
        }
    original_counts = {}
    for family, (table_name, field, selected) in scope.items():
        if table_name not in snapshot['tables']:
            require(all_archive, 'Original current scientific table is missing')
            rows=[]
        else:
            table = snapshot['tables'][table_name]
            rows = [dict(zip(table['columns'], row)) for row in table['rows']]
        column = 'sha256' if family in ('model_material', 'recipe_material') else 'id'
        if selected is None:
            selected = {row[column] for row in rows}
        expected = {row[column]: json.loads(row['document']) for row in rows if row[column] in selected}
        require(set(expected) == selected, 'Original selected history is incomplete')
        actual = {identity: doc for (kind, identity), doc in records.items()
                  if kind == family and doc.get('origin_catalog_id') == ORIGIN}
        require(set(actual) == {ORIGIN + '.' + identity for identity in selected},
                'Extracted original history membership differs')
        original_counts[family] = len(selected)
        for identity in sorted(selected):
            doc = records[(family, ORIGIN + '.' + identity)]
            require(doc['origin_catalog_id'] == ORIGIN, 'Historical origin differs')
            original = expected[identity]
            if family == 'run_history':
                original = {key: value for key, value in original.items()
                            if key not in LOCAL_CONTINUATION}
            require(doc[field] == original, 'Historical scientific observation changed')
            checks.append(dict(family=family, id=identity, exact=True))
    snapshot_checks=[]
    if all_archive:
        table=snapshot['tables']['workspace_projects']
        registrations={r['id']:json.loads(r['document']) for raw in table['rows']
                       if (r:=dict(zip(table['columns'],raw)))}
        for (family,identity),doc in records.items():
            if family!='project_snapshot' or doc.get('origin_catalog_id')!=ORIGIN:continue
            value=doc['value']
            require(value['id'] in registrations and value==registrations[value['id']],
                    'Archived registered project snapshot changed')
            require(identity==ORIGIN+'.'+value['manifest_sha256'],'Archived snapshot identity differs')
            snapshot_checks.append(value['id'])
    return checks, original_counts, snapshot_checks


def verify(project, snapshot_path, report, relocated=None, *, all_current=False, all_archive=False):
    require(not report.exists(), 'A fresh verification report is required')
    require(not (all_current and all_archive), 'Choose one original catalog observation')
    if all_current:
        summary = json.loads((snapshot_path.parent / 'summary.json').read_bytes())
        require(digest(snapshot_path) == summary['metadata_sha256'],
                'Retained original private snapshot bytes changed')
    if all_archive:
        observation=json.loads((ROOT/'docs/plans/model-portfolio-migration-v1/reports/catalog-archive-summary-01.json').read_bytes())
        require(snapshot_path==ROOT/observation['source_metadata']['path'], 'Use the original archived metadata')
        require(digest(snapshot_path)==observation['source_metadata']['sha256']
                and digest(snapshot_path.parent/'backup-receipt.json')==observation['backup_receipt']['sha256'],
                'Original archive metadata or receipt bytes changed')
    snapshot = json.loads(snapshot_path.read_text())
    require(snapshot['dataset_id'] == ORIGIN, 'Private snapshot origin differs')
    descriptor = json.loads((project / '.tuldok/project.json').read_text())
    records = {}
    for entry in descriptor['inventory']:
        relative = Path('.tuldok') / entry['path']
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Canonical inventory path escapes the project')
        path = project / relative
        require(path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'],
                'Canonical record bytes differ from their inventory')
        identity = entry['family'], entry['id']
        require(identity not in records, 'Canonical record identity repeats')
        records[identity] = json.loads(path.read_text())
    checks, original_counts, snapshot_checks = original_fact_checks(
        snapshot, records, all_current=all_current, all_archive=all_archive)
    extraction = records[('project_extraction', descriptor['project_id'])]
    present = set(extraction['artifacts']['copied'])
    missing = set(extraction['artifacts']['missing_regenerable'])
    require(not present & missing, 'Artifact custody states overlap')
    for sha in sorted(present):
        declaration = records[('project_artifact', sha)]
        path = project / 'artifacts/sha256' / sha
        require(path.stat().st_size == declaration['bytes'] and digest(path) == sha,
                'Copied historical payload changed')
    baseline = files(project)
    if relocated is not None:
        require(files(relocated) == baseline, 'Relocated portable file bytes or modes differ')
        require(all((project / name).stat().st_ino != (relocated / name).stat().st_ino
                    or (project / name).stat().st_dev != (relocated / name).stat().st_dev
                    for name in baseline), 'Relocated tree shares source file inodes')
    result = dict(format='rheon_extracted_history_preservation_v1',
        status=('exact_all_archived_original_facts_and_payloads' if all_archive else
                'exact_all_current_original_facts_and_payloads' if all_current else 'exact_selected_history_and_payloads'),
        project=display(project), project_id=descriptor['project_id'],
        checkpoint=descriptor['checkpoint'], private_snapshot_sha256=digest(snapshot_path),
        runner_sha256=digest(Path(__file__)),
        original_fact_checks=checks, family_counts=dict(Counter(e['family'] for e in descriptor['inventory'])),
        original_fact_scope=('all_archived_seven_scientific_families_and_retained_registrations' if all_archive else
                             'all_current_seven_scientific_families' if all_current else 'C106_C107_selected_history'),
        exact_original_family_counts=original_counts,
        exact_retained_archived_registration_checks=snapshot_checks,
        verified_payloads=len(present), visible_missing_regenerable=len(missing),
        portable_files=len(baseline), portable_bytes=sum(v['bytes'] for v in baseline.values()),
        portable_file_inventory_sha256=hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest(),
        relocated=display(relocated) if relocated else None,
        relocated_independent_inodes=relocated is not None,
        native_decoders_used=False, service_opened=False, models_executed=0,
        limitations='Literal selected-fact/byte preservation only; fresh native Open and usability require separate receipts.')
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in
                      ('status', 'project_id', 'verified_payloads', 'portable_files')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--relocated', type=Path)
    scope=parser.add_mutually_exclusive_group()
    scope.add_argument('--all-current', action='store_true')
    scope.add_argument('--all-archive', action='store_true')
    args = parser.parse_args()
    verify(args.project.resolve(), args.snapshot.resolve(), args.report.resolve(),
           args.relocated.resolve() if args.relocated else None, all_current=args.all_current,
           all_archive=args.all_archive)
