"""Make an independent portable fixture for a fresh-installation cold Open.

The source project must be idle at its expected checkpoint. Local continuation
files are excluded by the existing portable inventory reader. This fixture is
test evidence preparation, not an original catalog backup or native Save As.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from verify_history_project import files

ROOT = Path(__file__).resolve().parents[2]


def prepare(destination, report, checkpoint):
    source = ROOT / 'projects/rheon-model-portfolio-unified-03'
    if (destination.exists() or report.exists()
            or not destination.is_relative_to(ROOT / 'runs/model-portfolio-migration-v1')):
        raise ValueError('Fresh task-owned fixture and report required')
    manifest = (source / '.tuldok/project.json').read_bytes()
    descriptor = json.loads(manifest)
    if descriptor['checkpoint'] != checkpoint:
        raise ValueError('Current source differs from expected acknowledged checkpoint')
    before = files(source)
    destination.mkdir(parents=True)
    for relative in before:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target, follow_symlinks=False)
        old, new = (source / relative).stat(), target.stat()
        if (old.st_dev, old.st_ino) == (new.st_dev, new.st_ino):
            raise ValueError('Portable fixture shares an original inode')
    if before != files(destination) or before != files(source):
        raise ValueError('Portable fixture bytes/modes or source changed during copying')
    if (source / '.tuldok/project.json').read_bytes() != manifest:
        raise ValueError('Source checkpoint changed during fixture preparation')
    result = dict(schema='rheon.native-portfolio-cold-preparation.v1', status='independent_portable_copy_exact',
        created_unix=time.time(), source=str(source.relative_to(ROOT)), destination=str(destination.relative_to(ROOT)),
        project_id=descriptor['project_id'], checkpoint=checkpoint, portable_files=len(before),
        portable_bytes=sum(item['bytes'] for item in before.values()),
        portable_file_inventory_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        independent_inodes=True, native_cold_Open_qualified=False,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    prepare(args.destination.resolve(), args.report.resolve(), args.checkpoint)
