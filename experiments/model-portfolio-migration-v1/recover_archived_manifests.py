"""Recover archived manifest bytes only when the original SHA256 proves equality."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from preview_portfolio_history import ROOTS, sha, save
from qualify_native_workspace import ROOT


def recover(output, report):
    base = ROOT / 'runs/model-portfolio-migration-v1'
    docs = ROOT / 'docs/plans/model-portfolio-migration-v1/reports'
    qualification = json.loads((docs / 'archived-history-metadata-restore-01.json').read_bytes())
    metadata = ROOT / 'runs/dependency-migration-v1/original-schema8-backup-02/metadata.json'
    if (output.exists() or report.exists() or not output.is_relative_to(base)
            or not report.is_relative_to(docs) or sha(metadata) != qualification['source_snapshot_sha256']):
        raise ValueError('Original qualified archive metadata and fresh task-owned outputs required')
    original = json.loads(metadata.read_bytes())
    table = original['tables']['workspace_projects']
    registrations = [json.loads(row[table['columns'].index('document')]) for row in table['rows']]
    output.mkdir(mode=0o700)
    rows, bindings, missing = [], [], []
    for saved in registrations:
        expected, value = saved['manifest_sha256'], saved['manifest']
        matches = set()
        for directory in ROOTS.values():
            path = directory / saved['manifest_path']
            if path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536:
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() == expected:
                    matches.add(raw)
        keys = [k for k in ('schema_version','id','name','data','recipes','evidence','requirements','extensions') if k in value]
        orders = [list(value)]
        if 'activity_path' in value:
            orders += [keys[:i]+['activity_path']+keys[i:] for i in range(len(keys)+1)]
        else:
            orders.append(keys)
        for order in orders:
            if set(order) != set(value):
                continue
            document = {key: value[key] for key in order}
            for sort in (False, True):
                for ascii_only in (False, True):
                    for indent in (None, 2, 3, 4):
                        for ending in ('', '\n', '\n\n'):
                            raw = (json.dumps(document, sort_keys=sort, ensure_ascii=ascii_only,
                                              indent=indent, allow_nan=False)+ending).encode()
                            if hashlib.sha256(raw).hexdigest() == expected:
                                matches.add(raw)
        if len(matches) != 1:
            missing.append(dict(project_id=saved['id'], sha256=expected,
                                state='exact_original_manifest_bytes_not_recovered'))
            continue
        raw = matches.pop()
        path = output / (expected+'.json')
        if not path.exists():
            path.write_bytes(raw);path.chmod(0o600)
        relative = str(path.relative_to(ROOT))
        rows.append(dict(project_id=saved['id'], sha256=expected, bytes=len(raw), recovered_path=relative,
                         state='exact_original_registered_manifest_bytes'))
        bindings.append(dict(role='registered_manifest', sha256=expected, bytes=len(raw),
                             root='current-lab', relative=relative))
    save(output / 'bindings.json', dict(schema='rheon.archived-manifest-bindings.v1', bindings=bindings))
    result = dict(schema='rheon.archived-manifest-recovery.v1', created_unix=time.time(),
                  archive_metadata_sha256=qualification['source_snapshot_sha256'],
                  runner_sha256=sha(Path(__file__)), registrations=len(registrations),
                  recovered=len(rows), rows=rows, missing=missing,
                  binding_index=str((output/'bindings.json').relative_to(ROOT)),
                  binding_index_sha256=sha(output/'bindings.json'), original_files_changed=False,
                  database_opened=False, models_executed=0)
    save(report, result)
    print(json.dumps(dict(registrations=len(registrations), recovered=len(rows), missing=len(missing))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    recover(args.output.resolve(), args.report.resolve())
