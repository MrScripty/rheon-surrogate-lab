"""Compact the one fixed inactive cold catalog using the existing owner.

Actual SHA/length/mode matches are required against the complete backup04.
Distinct/root metadata remains. Reconstruct before any legacy catalog reuse.
"""
import argparse
from pathlib import Path

import compact_retired_catalog as owner


def select():
    owner.OLD = owner.ROOT/'runs/dependency-migration-v1/catalog-current-cold-04'
    owner.PLAN = owner.BASE/'discovery/inactive-cold-catalog-compaction-01.json'
    owner.REPORT = owner.ROOT/'docs/plans/model-portfolio-migration-v1/reports/inactive-cold-catalog-compaction-01.json'
    owner.MARKER = owner.OLD/'COMPACTED-SOURCE.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('compact','materialize'))
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    select()
    if args.command == 'compact':
        if args.destination is not None:
            parser.error('Compaction uses only the fixed inactive catalog')
        owner.compact()
    else:
        if args.destination is None or args.destination.resolve().is_relative_to(owner.ROOT):
            parser.error('Reconstruction requires an absent external destination')
        owner.materialize(args.destination)
