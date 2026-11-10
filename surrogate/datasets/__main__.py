"""List, size, link and generate reference datasets.

    python -m surrogate.datasets list
    python -m surrogate.datasets plan NAME [--split SPLIT ...] [--root DIRECTORY] [--json]
    python -m surrogate.datasets link NAME --bank INDEX [--bank INDEX ...] [--root DIRECTORY]
    python -m surrogate.datasets generate NAME --split SPLIT ... [--scene ID ...] [--root DIRECTORY]

`plan` only reads the definition and what is already on disk. `link` reuses
recordings that already exist on this machine (the earlier studies' index files
under runs/) instead of simulating them again. `generate` runs the solver, one
scene after another on one CPU thread, and skips scenes already complete.
"""
import argparse
import json
import shutil
import sys

from . import SPLITS, available, load
from .generate import generate, plan, select
from .link import link
from .store import data_root


def totals(summary):
    """The plan summed over its splits, as the flat object `plan --json` prints."""
    entries = summary.values()
    complete = sum(entry['complete'] for entry in entries)
    items = sum(entry['scenes'] for entry in entries)
    return {'disk_bytes': sum(entry['estimated_bytes'] for entry in entries),
            'items': items,
            'complete': complete,
            'to_run': items - complete,
            'partial': sum(entry['partial'] for entry in entries),
            'states': sum(entry['states'] for entry in entries),
            'cell_states': sum(entry['cell_states'] for entry in entries)}


def main(arguments=None):
    parser = argparse.ArgumentParser(prog='python -m surrogate.datasets', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='list the dataset definitions')
    for name in ('plan', 'generate'):
        command = commands.add_parser(name)
        command.add_argument('dataset')
        command.add_argument('--split', action='append', choices=SPLITS, dest='splits')
        command.add_argument('--scene', action='append', dest='scenes')
        command.add_argument('--root', help='data root (default: $SURROGATE_DATA or data/reference in the repository)')
        if name == 'plan':
            command.add_argument('--json', action='store_true',
                                 help='print the totals as one JSON object instead of the text summary')
        if name == 'generate':
            command.add_argument('--include-reserved', action='store_true',
                                 help='allow generating reserved scenes, which are for final checks only')
    linking = commands.add_parser('link')
    linking.add_argument('dataset')
    linking.add_argument('--bank', action='append', required=True, dest='banks',
                         help='an earlier index file listing recordings, relative to the repository')
    linking.add_argument('--root', help='data root (default: $SURROGATE_DATA or data/reference in the repository)')
    options = parser.parse_args(arguments)

    if options.command == 'list':
        for name in available():
            definition = load(name)
            counts = {split: sum(row['split'] == split for row in definition['scenes']) for split in SPLITS}
            print(f"{name}\n    {definition['label']}\n    scenes: "
                  + ', '.join(f'{count} {split}' for split, count in counts.items()))
        return 0
    if options.command == 'link':
        print(link(options.dataset, options.banks, options.root))
        return 0
    definition = load(options.dataset)
    root = data_root(options.root)
    if options.command == 'plan':
        rows = select(definition, options.splits, options.scenes)
        summary = plan(definition, rows, options.root)
        if options.json:
            print(json.dumps(totals(summary)))
            return 0
        total = 0
        print(f"{definition['id']} -> {root / definition['id']}")
        for split, entry in summary.items():
            total += entry['estimated_bytes']
            to_run = entry['scenes'] - entry['complete']
            print(f"  {split:12s} {entry['scenes']:4d} scenes: {entry['complete']} complete on disk, {to_run} to "
                  f"run ({entry['partial']} of them have some steps) -> {entry['states']} states, "
                  f"{entry['cell_states'] / 1e9:.2f} billion cell-states, about {entry['estimated_bytes'] / 1e9:.1f} GB")
        existing = next(folder for folder in (root, *root.parents) if folder.exists())
        print(f'  total about {total / 1e9:.1f} GB to write; '
              f'{shutil.disk_usage(existing).free / 1e9:.0f} GB free at the data root')
        return 0
    if not options.splits and not options.scenes:
        parser.error('Name what to generate with --split or --scene.')
    print(generate(options.dataset, options.root, options.splits, options.scenes, options.include_reserved))
    return 0


if __name__ == '__main__':
    sys.exit(main())
