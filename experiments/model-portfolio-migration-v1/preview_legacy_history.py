"""Prepare a native read-only extraction report for genuine C106/C107 history.

Execution needs separate authorization for the live connection setting and
native preview scope. The earlier public Run/graph snapshot permission does
not authorize this script. No stopped-source lease, backup or import is made.
The native catalog fingerprint reads all metadata tables internally; only the
non-secret domain report is retained. Selected artifact bytes may be read.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import traceback
import urllib.request

from catalog_metadata_snapshot import ROOT, SOURCE, CATALOG, CATALOG_ID, PID, OLD_SERVICE, configuration

RUN_IDS = ('74240450dfb642599b9aea0d83517a0f', '2a3555fc5d9c44928316c6f1c03064f5')
HOST_FILES = frozenset(('agent_api.py', 'extension_bindings.py', 'extension_contract.py',
    'extension_evidence.py', 'extension_schemas/contribution-v1.json',
    'extension_schemas/extension-v1.json', 'extension_schemas/parameter-v1.json',
    'field_evidence.py', 'pipeline.py', 'projects.py', 'recipe_contract.py',
    'recipe_process.py', 'scientific_fields.py', 'scientific_release.py', 'torch_runtimes.py'))
HOST_ROOTS = {'migration-host': OLD_SERVICE,
              'migration-original-host': ROOT / 'runs/dependency-migration-v1/Tuldok'}
PREVIEW_SOURCE_FILES = ('project_transfer.py', 'project_extraction.py', 'project_history.py',
    'project_projection.py', 'project_material.py', 'project_data.py', 'project_store.py', 'metadata.py')


def project_roots():
    arguments = [value.decode() for value in (Path('/proc') / str(PID) / 'cmdline').read_bytes().split(b'\0') if value]
    if str(OLD_SERVICE / 'app.py') not in arguments or str(CATALOG) not in arguments:
        raise ValueError('Existing service identity differs')
    result = {}
    for index, argument in enumerate(arguments):
        if argument == '--project-root':
            alias, separator, path = arguments[index + 1].partition('=')
            if not separator or alias in result:
                raise ValueError('Ambiguous current service root')
            result[alias] = Path(path).resolve(strict=True)
    return result


def owner():
    path = ROOT / 'runs/tuldok-performance-v1/private/owner-session.txt'
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4096 or info.st_mode & 0o077:
            raise ValueError('Existing private owner session differs')
        token = stream.read(4096).decode('ascii').strip()
    request = urllib.request.Request('http://127.0.0.1:8091/api/access/session',
                                    headers={'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(request, timeout=10) as response:
        value = json.loads(response.read())
    if value.get('kind') != 'owner' or value.get('catalog_id') != CATALOG_ID:
        raise ValueError('Current owner of the exact source catalog required')
    return value


def selection():
    manifest = ROOT / 'runs/pressure-methods-v4/v4-student50-control-project-01/project.json'
    document = json.loads(manifest.read_text())
    assert document['id'] == '2b6b6e48ba8943f0bff2daae6769f4dc'
    return dict(project_snapshot=dict(id=document['id'],
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),
        roots=dict(studies=['18f9d041a5e647628660e8cb8f634b6d',
                            '8905e808a42f400292d21deb82c94ddd']), decisions=[])


def run(output, authorized):
    if not authorized:
        print(json.dumps(dict(status='prepared_not_executed', selection=selection(),
            requires='Expanded user authorization for live DSN read and native read-only preview.')))
        return
    if output.exists():
        raise ValueError('Fresh read-only preview output required')
    identity = owner()
    source_sha256 = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                     for name in PREVIEW_SOURCE_FILES}
    sys.path.insert(0, str(SOURCE))
    from project_transfer import SourceView, ReadonlyCatalog, ProjectTransfer
    from metadata import connection_info
    db = ReadonlyCatalog(connection_info(configuration()), 'catalog_' + CATALOG_ID, CATALOG_ID, CATALOG)
    root_fd = None
    roots = {}
    try:
        assert db.execute("SELECT current_setting('transaction_read_only')").fetchone()[0] == 'on'
        assert db.execute("SELECT current_setting('transaction_isolation')").fetchone()[0] == 'repeatable read'
        root_fd = os.open(CATALOG, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        bindings = project_roots()
        if set(HOST_ROOTS) & bindings.keys():
            raise ValueError('Host-source capability alias conflict')
        bindings.update(HOST_ROOTS)
        for alias, path in bindings.items():
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(descriptor)
            roots[alias] = (descriptor, [info.st_dev, info.st_ino])
        with SourceView(db, root_fd, CATALOG, roots, actor_id=identity['actor_id']) as source:
            root_fd = None
            roots = {}
            source._authorize = owner
            host_matches, host_gaps, seen = [], [], set()
            for run_id in RUN_IDS:
                hosts = db.execute("SELECT document::jsonb#>'{material,host_files}' FROM workspace_runs WHERE id=%s", (run_id,)).fetchone()[0]
                if not isinstance(hosts, dict):
                    raise ValueError('Historical host-source map is unavailable')
                for name in sorted(set(hosts) & HOST_FILES):
                    expected = hosts[name]
                    matches = []
                    for alias, directory in HOST_ROOTS.items():
                        path = directory / name
                        if path.is_file() and not path.is_symlink():
                            raw = path.read_bytes()
                            if hashlib.sha256(raw).hexdigest() == expected:
                                matches.append((alias, len(raw)))
                    if not matches:
                        host_gaps.append(dict(run_id=run_id, name=name, expected_sha256=expected))
                        continue
                    alias, size = matches[0]
                    if expected not in seen:
                        source.bind_payload(expected, size, alias, name)
                        seen.add(expected)
                    host_matches.append(dict(run_id=run_id, name=name, sha256=expected, bytes=size, recovered_from=alias))
            report = ProjectTransfer(source).preview(selection())
        if source_sha256 != {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                            for name in PREVIEW_SOURCE_FILES}:
            raise ValueError('Preview code changed during this observation')
        output.mkdir(parents=True, mode=0o700)
        path = output / 'native-preview.json'
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        summary = dict(format='rheon_native_history_preview_v1',
            report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source=report['source'], selection=report['selection'],
            included=len(report['included']), required=len(report['required']),
            unassigned=len(report['unassigned']), can_extract=report['can_extract'],
            blockers=dict(Counter(item['code'] for item in report['blockers'])),
            included_families=dict(Counter(item['family'] for item in report['included'])),
            root_aliases=sorted(bindings), host_matches=host_matches, host_gaps=host_gaps,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            upstream_source_sha256=source_sha256, upstream_code_stable=True,
            transaction_read_only=True, database_writes=0, models_executed=0,
            stopped_source_capability_issued=False, migration_performed=False)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(dict(output=str(output.relative_to(ROOT)), included=summary['included'],
            can_extract=summary['can_extract'], blockers=summary['blockers'],
            host_matches=len(host_matches), host_gaps=len(host_gaps))), flush=True)
    finally:
        if root_fd is not None:
            os.close(root_fd)
        for descriptor, _identity in roots.values():
            os.close(descriptor)
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--authorized-native-preview', action='store_true')
    args = parser.parse_args()
    try:
        run(args.output.resolve(), args.authorized_native_preview)
    except Exception as error:
        # Database exceptions may carry conninfo. Never print their messages.
        print('Native read-only preview failed:', type(error).__name__,
              [(Path(frame.filename).name, frame.lineno, frame.name) for frame in traceback.extract_tb(error.__traceback__)],
              file=sys.stderr)
        raise SystemExit(1)
