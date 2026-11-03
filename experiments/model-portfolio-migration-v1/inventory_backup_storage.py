"""Read-only content/mode inventory for full-catalog backup capacity planning.

This is a live filesystem observation, not a quiesced backup or stop proof.
Private per-file identities stay in mode0600 storage; stdout contains totals.
Equal sizes alone never count as duplicate bytes. No originals are linked,
deleted, chmodded or otherwise changed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import stat
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok/data-postgres')


def identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


def entries(root):
    result = {}
    for base, directories, names in os.walk(root, followlinks=False):
        for name in directories + names:
            path = Path(base) / name
            info = path.lstat()
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError('Source contains a symlink or special file')
            result[path.relative_to(root).as_posix()] = identity(info)
    return result


def write_private(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(output):
    if (output.exists() or not output.is_relative_to(ROOT / 'runs/model-portfolio-migration-v1')
            or SOURCE.is_symlink() or not SOURCE.is_dir()):
        raise ValueError('Existing ordinary source and fresh private task output required')
    output.mkdir(mode=0o700)
    before = entries(SOURCE)
    manifest = {}
    groups = defaultdict(list)
    inodes = {}
    started = time.time()
    for name, expected in sorted(before.items()):
        if not stat.S_ISREG(expected[2]):
            continue
        descriptor = os.open(SOURCE / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if identity(info) != expected:
                raise ValueError('Source changed before file inventory')
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
            if identity(os.fstat(stream.fileno())) != expected:
                raise ValueError('Source changed during file inventory')
        if identity((SOURCE / name).lstat()) != expected:
            raise ValueError('Source file binding changed during inventory')
        sha = digest.hexdigest()
        mode = stat.S_IMODE(info.st_mode)
        manifest[name] = dict(sha256=sha, bytes=info.st_size, mode=mode)
        groups[(sha, info.st_size, mode)].append(name)
        inodes[(info.st_dev, info.st_ino)] = info.st_blocks * 512
        if len(manifest) % 10_000 == 0:
            print(json.dumps(dict(phase='hashing_original_files_read_only', files=len(manifest))), flush=True)
    if entries(SOURCE) != before:
        raise ValueError('Live source tree changed; observation cannot support a stable capacity estimate')
    logical = sum(value['bytes'] for value in manifest.values())
    unique = sum(key[1] for key in groups)
    duplicates = [dict(sha256=key[0], bytes=key[1], mode=key[2], paths=names)
                  for key, names in sorted(groups.items()) if len(names) > 1]
    write_private(output / 'file-inventory.json', manifest)
    write_private(output / 'duplicate-groups.json', duplicates)
    filesystem = os.statvfs(SOURCE)
    value = dict(format='rheon_readonly_backup_content_capacity_v1',
        status='stable_live_filesystem_observation', observed_unix=time.time(),
        elapsed_seconds=time.time() - started, source_modified=False, source_stopped=False,
        original_SQL_opened=False, regular_files=len(manifest), logical_file_bytes=logical,
        original_distinct_inode_allocated_bytes=sum(inodes.values()),
        distinct_content_and_mode_groups=len(groups),
        unique_content_and_mode_logical_bytes=unique,
        duplicate_groups=len(duplicates), duplicate_file_members=sum(len(g['paths']) for g in duplicates),
        theoretical_avoided_copy_bytes=logical-unique,
        available_bytes=filesystem.f_bavail*filesystem.f_frsize,
        private_file_inventory_sha256=hashlib.sha256((output/'file-inventory.json').read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        existing_native_full_copy_bytes=logical,
        deduplicated_native_backup_implemented=False,
        limits='Content/mode totals exclude metadata.json, receipt, directories, allocation rounding and staging overhead. This live observation is not a complete backup, stopped-source proof or authorization to delete source files.')
    write_private(output / 'summary.json', value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
