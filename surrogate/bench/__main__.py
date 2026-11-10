"""Run the benchmark and prepare its results for Tuldok.

    python -m surrogate.bench suites
    python -m surrogate.bench run --suite NAME --subject SPEC [--subject SPEC ...] --output result.json
    python -m surrogate.bench packets result.json --output-directory DIRECTORY

Subject specifications:

    persistence                 the state never changes; the error any model must beat
    checkpoint:PATH             a trained stepper saved by surrogate.train
    rheon-native                Rheon's own solver, one CPU thread
    spectral-composition[:K]    the earlier GPU solver composition, K outputs per pass (1 to 4)
"""
import argparse
import sys
from pathlib import Path

from . import packet, results, suites
from .runner import run
from .subjects import LearnedStepper, Persistence


def subject_from(specification, device):
    kind, _, argument = specification.partition(':')
    if kind == 'persistence' and not argument:
        return Persistence()
    if kind == 'checkpoint' and argument:
        return LearnedStepper(argument, device)
    if kind == 'rheon-native' and not argument:
        from .native import NativeRheon
        return NativeRheon()
    if kind == 'spectral-composition':
        from .composition import SpectralComposition
        return SpectralComposition(int(argument or 1))
    raise ValueError(f'Unknown subject {specification!r}; see python -m surrogate.bench --help.')


def main(arguments=None):
    parser = argparse.ArgumentParser(prog='python -m surrogate.bench', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('suites', help='list the available suites')
    running = commands.add_parser('run', help='score subjects on a suite')
    running.add_argument('--suite', required=True)
    running.add_argument('--subject', action='append', required=True, dest='subjects')
    running.add_argument('--output', type=Path, required=True, help='result document to write; must not exist')
    running.add_argument('--device', default='cuda', help='device for learned checkpoints (default: cuda)')
    running.add_argument('--allow-uncommitted', action='store_true',
                         help='run with uncommitted changes; the result records that its commit does not pin the code')
    packing = commands.add_parser('packets', help='write the Tuldok packets of a result document')
    packing.add_argument('result', type=Path)
    packing.add_argument('--output-directory', type=Path, required=True)
    options = parser.parse_args(arguments)

    if options.command == 'suites':
        for name, specification in sorted(suites.SPECIFICATIONS.items()):
            print(f"{name}\n    {specification['label']} ({specification['population']})\n"
                  f"    {specification['description']}")
        return 0
    if options.command == 'run':
        if options.output.exists():
            parser.error(f'{options.output} already exists; results are never overwritten.')
        if results.git_identity()['git_dirty'] and not options.allow_uncommitted:
            parser.error('There are uncommitted changes, so the recorded commit would not identify this code. '
                         'Commit first, or pass --allow-uncommitted for a development run.')
        suite = suites.build(options.suite)
        subjects = [subject_from(specification, options.device) for specification in options.subjects]
        print(results.write(run(subjects, suite), options.output))
        return 0
    document = results.read(options.result)
    options.output_directory.mkdir(parents=True, exist_ok=True)
    built = packet.build(document)
    for index, raw in enumerate(built, start=1):
        name = f"{document['suite']['id']}" + (f'.part-{index}' if len(built) > 1 else '') + '.measurements.json'
        target = options.output_directory / name
        if target.exists():
            parser.error(f'{target} already exists; packets are never overwritten.')
        target.write_bytes(raw)
        print(target)
    return 0


if __name__ == '__main__':
    sys.exit(main())
