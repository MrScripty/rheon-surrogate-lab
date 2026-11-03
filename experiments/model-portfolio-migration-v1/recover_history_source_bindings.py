"""Recover exact registered manifests and locate original large Run inputs.

Manifest recovery uses retained values and their original SHA256, with the
retained writers' field order. A matching digest proves exact original bytes;
no producer or model is executed. Large inputs are located only, and require
native descriptor hashing before their custody can be claimed.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import stat

from qualify_native_workspace import ROOT
from catalog_metadata_snapshot import CATALOG

BASE = ROOT / 'runs/model-portfolio-migration-v1'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def recover(output, preview_path=None, prior_bindings=None, project_ids=()):
    if output.exists() or not output.is_relative_to(BASE):
        raise ValueError('Fresh task-owned recovery directory required')
    preview_path = preview_path or BASE / 'full-current-history-preview-02/native-preview.json'
    if not preview_path.is_relative_to(BASE):
        raise ValueError('Task-owned native preview required')
    if preview_path.exists():
        preview_raw = preview_path.read_bytes()
    else:
        # Lossless archived reports retain the same logical bytes and digest.
        preview_raw = gzip.decompress(preview_path.with_suffix('.json.gz').read_bytes())
    summary_path = ROOT / 'docs/plans/model-portfolio-migration-v1/reports' / (preview_path.parent.name + '.json')
    expected_preview = json.loads(summary_path.read_bytes())['native_preview_sha256']
    if hashlib.sha256(preview_raw).hexdigest() != expected_preview:
        raise ValueError('Retained native preview bytes changed')
    preview = json.loads(preview_raw)
    manifest_hashes = {row['id'].split('.')[-1] for row in preview['blockers'] if row['family'] == 'project_snapshot'}
    missing_inputs = {row['sha256']: row for row in preview['artifacts']
                      if row['preservation'] == 'unique' and row['state'] == 'missing'}
    snapshot = BASE / 'private-history-rehearsal-snapshot-01'
    summary = json.loads((snapshot / 'summary.json').read_bytes())
    with (snapshot / 'metadata.json').open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != summary['metadata_sha256']:
            raise ValueError('Original snapshot identity changed')
    metadata = json.loads((snapshot / 'metadata.json').read_bytes())
    output.mkdir(mode=0o700)
    manifest_rows = []
    bindings = []
    table = metadata['tables']['workspace_projects']
    column = table['columns'].index('document')
    registrations = [json.loads(row[column]) for row in table['rows']]
    for registration in registrations:
        if registration['id'] in project_ids:
            manifest_hashes.add(registration['manifest_sha256'])
    if set(project_ids) - {row['id'] for row in registrations}:
        raise ValueError('Explicit original project registration is absent')
    for registration in registrations:
        expected = registration['manifest_sha256']
        if expected not in manifest_hashes:
            continue
        value = registration['manifest']
        order = ('schema_version', 'id', 'name', 'data', 'recipes', 'evidence', 'requirements', 'extensions')
        ordered = {key: value[key] for key in order if key in value}
        if set(ordered) != set(value):
            raise ValueError('Unknown original manifest fields')
        matches = set()
        for document in (value, ordered):
            for sort_keys in (False, True):
                for ascii_only in (False, True):
                    for indent in (None, 2, 3, 4):
                        for ending in ('', '\n', '\n\n'):
                            raw = (json.dumps(document, indent=indent, sort_keys=sort_keys,
                                              ensure_ascii=ascii_only, allow_nan=False) + ending).encode()
                            if hashlib.sha256(raw).hexdigest() == expected:
                                matches.add(raw)
        if len(matches) != 1:
            raise ValueError('Exact original registered manifest bytes not recovered: ' + expected)
        raw = matches.pop()
        path = output / (expected + '.json')
        path.write_bytes(raw)
        path.chmod(0o600)
        manifest_rows.append(dict(registration=registration, sha256=expected, bytes=len(raw),
                                  recovered_path=str(path.relative_to(ROOT)),
                                  state='exact_original_registered_manifest_bytes'))
        bindings.append(dict(role='registered_manifest', sha256=expected, bytes=len(raw),
                             root='current-lab', relative=str(path.relative_to(ROOT))))
    if len(manifest_rows) != len(manifest_hashes):
        raise ValueError('Missing original registration rows')
    table = metadata['tables']['workspace_runs']
    column = table['columns'].index('document')
    input_rows = {sha: dict(sha256=sha, bytes=decl['bytes'], observations=[], candidates=[])
                  for sha, decl in missing_inputs.items()}
    for row in table['rows']:
        run = json.loads(row[column])
        for name, blob in run.get('material', {}).get('captures', {}).items():
            if blob['sha256'] not in input_rows:
                continue
            target = input_rows[blob['sha256']]
            if blob['bytes'] != target['bytes']:
                raise ValueError('Original input byte counts disagree')
            target['observations'].append(dict(run_id=run['id'], captured_name=name))
            relative = Path('workspace-runs') / ('.external-' + run['id']) / name
            path = CATALOG / relative
            if path.exists():
                info = path.lstat()
                if stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == blob['bytes']:
                    target['candidates'].append(str(relative))
    for sha, row in input_rows.items():
        if not row['candidates']:
            raise ValueError('Retained original large input not located: ' + sha)
        row['state'] = 'original_Run_bound_location_found_native_hash_verification_pending'
        bindings.append(dict(role='large_original_Run_input', sha256=sha, bytes=row['bytes'],
                             root='catalog-source', relative=sorted(row['candidates'])[0]))
    prior_digest = None
    if prior_bindings is not None:
        if not prior_bindings.is_relative_to(BASE):
            raise ValueError('Task-owned prior source bindings required')
        raw = prior_bindings.read_bytes()
        prior_digest = hashlib.sha256(raw).hexdigest()
        previous = json.loads(raw)
        if previous['schema'] != 'rheon.explicit-history-payload-bindings.v1':
            raise ValueError('Unknown prior source binding representation')
        existing = {row['sha256']: row for row in bindings}
        for row in previous['bindings']:
            if row['sha256'] in existing:
                if row['bytes'] != existing[row['sha256']]['bytes']:
                    raise ValueError('Original payload binding sizes conflict')
            else:
                existing[row['sha256']] = row
        bindings = list(existing.values())
    write(output / 'bindings.json', dict(schema='rheon.explicit-history-payload-bindings.v1', bindings=bindings))
    write(output / 'recovery.json', dict(schema='rheon.history-source-recovery.v1',
          original_metadata_sha256=summary['metadata_sha256'], manifest_rows=manifest_rows,
          prior_bindings_sha256=prior_digest,
          large_input_rows=list(input_rows.values()), source_files_changed=False,
          source_or_model_code_executed=False, new_neural_forwards=0,
          runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(exact_registered_manifests=len(manifest_rows),
                         large_inputs_located=len(input_rows),
                         large_input_bytes=sum(row['bytes'] for row in input_rows.values()),
                         native_large_input_hashing_still_required=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', type=Path)
    parser.add_argument('--prior-bindings', type=Path)
    parser.add_argument('--project-ids', nargs='*', default=[])
    args = parser.parse_args()
    recover(args.output.resolve(), args.preview.resolve() if args.preview else None,
            args.prior_bindings.resolve() if args.prior_bindings else None, args.project_ids)
