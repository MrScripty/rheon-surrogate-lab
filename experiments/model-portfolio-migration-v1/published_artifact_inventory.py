"""Inspect public artifact manifests and hash checkpoint candidates in place.

No credentials, PostgreSQL connection or private process documents are read.
Directory membership is not a complete Run-table inventory. Artifact declarations
are separate from measured file identities and current custody authorization.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import stat
import time


ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path('/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok/data-postgres')
CATALOG_ID = 'a7da4a8de53541a2a0e2aad08005bb71'


def identity(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError('Independent regular file required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Artifact changed during measurement')
    return dict(sha256=h.hexdigest(), bytes=after.st_size)


def collect(output, summary_path):
    if output.exists() or summary_path.exists():
        raise ValueError('Fresh artifact inventory outputs required')
    if CATALOG.is_symlink() or json.loads((CATALOG / '.tuldok.json').read_text())['dataset_id'] != CATALOG_ID:
        raise ValueError('Selected catalog differs')
    manifests, missing, failures, groups = [], [], [], {}
    folder = CATALOG / 'workspace-runs'
    for directory in sorted(folder.iterdir()):
        if not re.fullmatch('[a-f0-9]{32}', directory.name):
            continue
        if directory.is_symlink() or not directory.is_dir():
            failures.append(dict(run_id=directory.name, reason='unexpected_directory_type'))
            continue
        path = directory / 'artifact-manifest.json'
        if not path.exists():
            missing.append(dict(run_id=directory.name, reason='no_publication_manifest'))
            continue
        try:
            if path.stat().st_size > 1024 * 1024:
                raise ValueError('Oversize manifest')
            info = identity(path)
            manifest = json.loads(path.read_text())
            if manifest['run_id'] != directory.name:
                raise ValueError('Manifest Run identity differs')
            measured = []
            for artifact in manifest['artifacts']:
                name = artifact['name']
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', name):
                    raise ValueError('Artifact name must be flat')
                if not re.search('weight|checkpoint|parameter', name + ' ' + artifact.get('role', ''), re.I):
                    continue
                actual = identity(directory / name)
                if actual != {k: artifact[k] for k in ('sha256', 'bytes')}:
                    raise ValueError('Declared checkpoint differs from measured bytes')
                measured.append(dict(name=name, role=artifact.get('role'), **actual))
                group = groups.setdefault(actual['sha256'], dict(**actual, paths=[]))
                group['paths'].append(str(directory / name))
            manifests.append(dict(run_id=directory.name, path=str(path), **info,
                                  declared_artifacts=manifest['artifacts'],
                                  material_sha256=manifest['material_sha256'],
                                  checkpoint_candidates=measured))
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(dict(run_id=directory.name, error_type=type(error).__name__))
    result = dict(schema='rheon.published-artifact-file-inventory.v1', created_unix=time.time(),
                  catalog_id=CATALOG_ID, status='captured' if not failures else 'partial_with_failures',
                  manifests=manifests, unpublished_directories=missing, failures=failures,
                  checkpoint_groups=[groups[k] for k in sorted(groups)],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  credentials_read=False, database_opened=False, inference_calls=0, files_deleted=0,
                  limitations=['A public manifest is not a transitive-custody or scientific-validity certificate.',
                               'Only selected checkpoint candidate payloads are hashed; other artifacts are declarations.',
                               'Run directories may omit queued or otherwise unpublished records; full Run accounting remains separate.'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    summary = dict(schema=result['schema'], status=result['status'], output=str(output),
                   output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                   publication_manifests=len(manifests), unpublished_directories=missing,
                   checkpoint_candidate_paths=sum(len(g['paths']) for g in groups.values()),
                   checkpoint_byte_identities=len(groups),
                   checkpoint_role_counts=dict(Counter(a['role'] for m in manifests for a in m['checkpoint_candidates'])),
                   failures=failures, credentials_read=False, database_opened=False, inference_calls=0)
    with summary_path.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    collect(args.output, args.summary)
