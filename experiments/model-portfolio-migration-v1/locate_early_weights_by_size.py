"""Find missing original C01/C02 byte states, including extensionless payloads.

Only ordinary model-compatible files with the exact original artifact size are
hashed. Private/reserved inputs, symlinks, environment caches and database storage
are excluded. No model deserialization, credential read or mutation occurs.
"""
import hashlib
import json
from pathlib import Path
import stat
import subprocess

from discover import protected
from checkpoint_identities import identity


ROOT = Path(__file__).resolve().parents[2]
ROOTS = [ROOT, Path('/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok')]
TARGETS = dict(C01='f46aec5e0de59bca9a2a7a701e85c4904782c11b7ac6c113cfd3f7625c8391fc',
               C02='4adf55ca940c2013b6352edc4a6716492e0a697484331c65cdf80495312db147')
SIZE = 1500186


def run():
    destination = ROOT / 'runs/model-portfolio-migration-v1/discovery/early-weight-size-search-01.json'
    if destination.exists():
        raise ValueError('Fresh search report required')
    matches, candidates, errors = [], [], []
    files_seen = 0
    for root in ROOTS:
        command = ['rg', '--files', '--hidden', '--no-ignore', '--null']
        for directory in ['.git', '.venv*', 'node_modules', 'site-packages', '__pycache__',
                          'private', 'postgres-data', 'PGDATA', 'profile', '.cache']:
            command.extend(['-g', '!**/' + directory + '/**'])
        value = subprocess.run(command, cwd=root, capture_output=True)
        if value.returncode not in (0, 1):
            raise ValueError('Bounded file discovery failed')
        for raw in value.stdout.split(b'\0'):
            if not raw:
                continue
            files_seen += 1
            relative = Path(raw.decode())
            if (relative.name.startswith('.') or protected(relative)
                    or relative.suffix.lower() not in ('', '.npz', '.pt', '.pth', '.bin', '.safetensors')):
                continue
            path = root / relative
            try:
                info = path.lstat()
                if info.st_size != SIZE or not stat.S_ISREG(info.st_mode):
                    continue
                if not path.resolve().is_relative_to(root):
                    continue
                result = identity(path)
                entry = dict(root=str(root), path=str(relative), **result)
                candidates.append(entry)
                for alias, target in TARGETS.items():
                    if result['sha256'] == target:
                        matches.append(dict(alias=alias, **entry))
            except OSError as error:
                errors.append(dict(root=str(root), path=str(relative), error_type=type(error).__name__))
    result = dict(format='rheon_exact_early_weight_size_search_v1', roots=list(map(str, ROOTS)),
                  expected_bytes=SIZE, expected_sha256=TARGETS, files_seen=files_seen,
                  candidates_hashed=candidates, matches=matches, errors=errors,
                  models_executed=False, unique_weights_deleted=False,
                  limitation='Bounded known local roots only; remote or unmounted archives are not searched.')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    summary = dict(format=result['format'], expected_bytes=SIZE, expected_sha256=TARGETS,
        roots=result['roots'], files_seen=files_seen, candidates_hashed=len(candidates),
        extensionless_candidates=sum(Path(p['path']).suffix == '' for p in candidates),
        matches=matches, errors=errors,
        report_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        models_executed=False, limitation=result['limitation'])
    (ROOT / 'docs/plans/model-portfolio-migration-v1/reports/early-weight-size-search-01.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    run()
