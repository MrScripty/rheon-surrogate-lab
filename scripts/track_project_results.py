"""Write a Tuldok project's .gitignore so Git keeps its irreplaceable stored files.

Tuldok ignores a project's whole artifacts/ folder by default. This lab keeps
two kinds of stored files in Git, because they cannot be cheaply regenerated:

- benchmark and measurement results (blobs referenced by measurement_source records)
- captured model source and policy (model_material artifacts with those roles)

Training inputs, checkpoints and run payloads stay ignored. Run this after a
project gains new benchmark results or models, then commit the .gitignore and
the newly allowed files.

Usage: python scripts/track_project_results.py projects/<project-directory> [--check]
"""
import argparse
import json
import sys
from pathlib import Path

# GitHub rejects files over 100 MB; stay clearly below it.
MAX_TRACKED_BYTES = 95 * 1024 * 1024
MODEL_ROLES = {'model_source', 'model_policy'}
BEGIN = '# BEGIN tracked stored files (scripts/track_project_results.py)'
END = '# END tracked stored files'
# Tuldok's default project .gitignore, without its blanket /artifacts/ rule.
LOCAL_ONLY = ['/.tuldok-local/', '/.venv/', '/__pycache__/', '/PGDATA/', '/output/']


def referenced_blobs(value, found):
    if isinstance(value, dict):
        if isinstance(value.get('sha256'), str) and isinstance(value.get('bytes'), int):
            found[value['sha256']] = value['bytes']
        for item in value.values():
            referenced_blobs(item, found)
    elif isinstance(value, list):
        for item in value:
            referenced_blobs(item, found)


def tracked_blobs(project):
    manifest = json.loads((project / '.tuldok' / 'project.json').read_bytes())
    kept = {}
    for entry in manifest['inventory']:
        family = entry['family']
        if family not in ('measurement_source', 'model_material'):
            continue
        document = json.loads((project / '.tuldok' / entry['path']).read_bytes())
        if family == 'measurement_source':
            referenced_blobs(document, kept)
        else:
            for artifact in document.get('artifacts', []):
                if artifact.get('role') in MODEL_ROLES:
                    kept[artifact['blob']['sha256']] = artifact['blob']['bytes']
    return kept


def render(kept):
    lines = [*LOCAL_ONLY, '', BEGIN, '/artifacts/*', '!/artifacts/sha256/', '/artifacts/sha256/*']
    lines += [f'!/artifacts/sha256/{sha}' for sha in sorted(kept)]
    return '\n'.join([*lines, END, ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('project', type=Path)
    parser.add_argument('--check', action='store_true', help='fail if the .gitignore is out of date')
    arguments = parser.parse_args()
    project = arguments.project
    kept = tracked_blobs(project)
    store = project / 'artifacts' / 'sha256'
    missing = [sha for sha in kept if not (store / sha).is_file()]
    too_large = [sha for sha, size in kept.items() if size > MAX_TRACKED_BYTES]
    wrong_size = [sha for sha, size in kept.items()
                  if sha not in missing and (store / sha).stat().st_size != size]
    for label, shas in (('missing', missing), ('over the Git size limit', too_large), ('size differs from its record', wrong_size)):
        for sha in shas:
            print(f'{label}: artifacts/sha256/{sha}', file=sys.stderr)
    if missing or too_large or wrong_size:
        return 1
    text = render(kept)
    target = project / '.gitignore'
    if arguments.check:
        if not target.is_file() or target.read_text() != text:
            print(f'{target} is out of date; run without --check.', file=sys.stderr)
            return 1
        return 0
    target.write_text(text)
    print(f'{target}: {len(kept)} stored files tracked, {sum(kept.values()) / 1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
