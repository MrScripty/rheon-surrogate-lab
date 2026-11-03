"""Reuse verified backup compaction for four fixed inactive archive copies.

Distinct SQL snapshots, receipts and payloads remain at their original paths.
The complete current backup owns the retained duplicate bytes. Materialize a
compacted copy at a fresh external destination before using legacy restore.
"""
import os
import stat
import sys

import compact_backups as backups
import compact_retired_catalog as catalogs


STAGES = ('domain-original-stage-01', 'domain-stage-02',
          'domain-stage-03', 'domain-stage-04')
RETIRED = tuple(name + '/original-archive' for name in STAGES)


def require_inactive():
    for name in STAGES:
        catalogs.OLD = backups.BASE / name
        catalogs.inactive()


def require_duplicate_modes(manifest):
    value = backups.load(manifest)
    for group in value['backups']:
        for item in group['duplicates']:
            old = backups.payload(backups.BASE / group['name'], item['path'])
            kept = backups.payload(backups.KEEPER, item['path'])
            if stat.S_IMODE(old.lstat().st_mode) != stat.S_IMODE(kept.lstat().st_mode):
                raise ValueError('Duplicate file modes differ; preserve the original')


if __name__ == '__main__':
    from pathlib import Path
    os.umask(0o077)
    backups.RETIRED = RETIRED
    # The existing owner still controls receipt validation, byte comparison,
    # removal, markers, post-verification and independent reconstruction.
    if len(sys.argv) > 1 and sys.argv[1] in ('scan', 'apply'):
        require_inactive()
    if len(sys.argv) > 1 and sys.argv[1] == 'apply':
        index = sys.argv.index('--manifest')
        require_duplicate_modes(Path(sys.argv[index + 1]))
    if len(sys.argv) > 1 and sys.argv[1] == 'materialize':
        index = sys.argv.index('--destination')
        destination = Path(sys.argv[index + 1]).resolve()
        if destination.is_relative_to(backups.ROOT):
            raise ValueError('Reconstruct at an external destination')
    backups.main()
