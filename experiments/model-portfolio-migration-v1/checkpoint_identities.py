"""Hash checkpoint candidates without deserializing models or running inference.

A byte identity groups copies, not architectures or execution variants. Keep
every alias; no files are removed. Reserved/holdout/private candidates stay
unopened. Candidate naming is a discovery heuristic, not a complete model count.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import time

from discover import CHECKPOINT_NAME, CHECKPOINT_SUFFIXES, protected


ROOT = Path(__file__).resolve().parents[2]


def identity(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('Expected regular candidate file')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Candidate changed while hashing')
    return dict(sha256=h.hexdigest(), bytes=after.st_size)


def collect(local_path, output):
    if output.exists():
        raise ValueError('Fresh report path required')
    local = json.loads(local_path.read_text())
    names = {r['path'] for r in local['checkpoint_filename_candidates']}
    names.update(n for n in local['tracked_payload_extension_candidates']
                 if Path(n).suffix in CHECKPOINT_SUFFIXES)
    command = ['rg', '--files', '--hidden', '--no-ignore', '--null']
    for suffix in sorted(CHECKPOINT_SUFFIXES):
        command.extend(['-g', '*' + suffix])
    command.append('experiments')
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError('Additional experiment candidate discovery failed')
    names.update(p.decode() for p in result.stdout.split(b'\0')
                 if p and CHECKPOINT_NAME.search(Path(p.decode()).name))
    groups, unavailable, excluded = {}, [], []
    start = time.monotonic()
    for index, name in enumerate(sorted(names), 1):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Candidate must be repository relative')
        if protected(relative):
            excluded.append(dict(path=name, reason='reserved_holdout_or_private'))
            continue
        path = ROOT / relative
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise ValueError('Symlink or escaped candidate requires separate classification')
            actual = identity(path)
        except (ValueError, OSError) as error:
            unavailable.append(dict(path=name, error_type=type(error).__name__))
            continue
        group = groups.setdefault(actual['sha256'], dict(**actual, paths=[]))
        if group['bytes'] != actual['bytes']:
            raise ValueError('Hash identity has inconsistent byte sizes')
        group['paths'].append(name)
        if index % 1000 == 0:
            print('Hashed', index, 'candidates;', len(groups), 'byte identities', flush=True)
    value = dict(schema='rheon.checkpoint-byte-identities.v1', created_unix=time.time(),
                 status='candidate_bytes_verified_model_reconciliation_pending',
                 input=dict(path=str(local_path), **identity(local_path)),
                 source_sha256=identity(Path(__file__))['sha256'],
                 groups=[groups[k] for k in sorted(groups)], unavailable=unavailable,
                 excluded=excluded, candidate_count=len(names),
                 paths_hashed=sum(len(g['paths']) for g in groups.values()),
                 distinct_byte_identities=len(groups),
                 logical_bytes=sum(g['bytes'] * len(g['paths']) for g in groups.values()),
                 distinct_identity_bytes=sum(g['bytes'] for g in groups.values()),
                 payloads_deserialized=False, inference_calls=0, files_deleted=0,
                 elapsed_seconds=time.monotonic() - start,
                 limitations=['Content identity is not model identity: source and execution contracts remain required.',
                              'Different container bytes can encode equal parameters; no numerical equivalence asserted.',
                              'Extensionless artifacts, remote artifacts and source-only prototypes require other discovery sources.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in value.items() if k not in ('groups', 'excluded', 'unavailable')}), flush=True)
    print('Unavailable:', dict(Counter(r['error_type'] for r in unavailable)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-inventory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    collect(args.local_inventory, args.output)
