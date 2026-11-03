"""Retire exact duplicate payloads in four fixed inactive external stages.

The existing compaction owner verifies the complete keeper, every duplicate's
bytes and mode, inactivity, retained files, and independent reconstruction.
No active native project, SQL snapshot, or unique payload is selected.
"""
import argparse
import os
from pathlib import Path

import compact_retired_catalog as owner


STAGES = ('domain-original-stage-01', 'domain-stage-02',
          'domain-stage-03', 'domain-stage-04')


def select(stage):
    if stage not in STAGES:
        raise ValueError('Only the four fixed inactive external stages qualify')
    owner.OLD = owner.ROOT / 'runs/dependency-migration-v1' / stage / 'external'
    name = stage + '-external-compaction-01.json'
    owner.PLAN = owner.BASE / 'discovery' / name
    owner.REPORT = owner.ROOT / 'docs/plans/model-portfolio-migration-v1/reports' / name
    owner.MARKER = owner.OLD / 'COMPACTED-SOURCE.json'


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('compact', 'materialize'))
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    select(args.stage)
    if args.command == 'compact':
        if args.destination is not None:
            parser.error('Compaction selects only the fixed inactive stage')
        owner.compact()
    else:
        if args.destination is None or args.destination.resolve().is_relative_to(owner.ROOT):
            parser.error('Reconstruction requires an absent external destination')
        owner.materialize(args.destination)
