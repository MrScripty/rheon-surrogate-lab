#!/usr/bin/env python3
"""Build and qualify variable domains with one shared physical cell size."""
import argparse
import json
from pathlib import Path
import sys

import grid
from contract import ContractError, decode, load_module, read, require
import build_native
import cubic_capture


def model_preflight(config):
    specification = grid.plan(config)
    unsupported = [g['counts_xyz'] for g in specification['grids'] if not g['model_shape_compatible']]
    require(not unsupported, 'model_shape_unsupported',
            f'Unchanged GroupNorm v1 accepts only 16³ or 32³; got {unsupported}. '
            'Keep cubic cells; rectangular model support requires a separate model change.')
    return specification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('describe', 'build', 'generate', 'verify', 'model-preflight'))
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/development.json')
    parser.add_argument('--prepared-rheon', type=Path)
    parser.add_argument('--cargo', type=Path, default=Path.home()/'.cargo/bin/cargo')
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--live-root', type=Path)
    args = parser.parse_args()
    if args.operation == 'verify':
        require(args.capture is not None, 'argument', '--capture required')
        return cubic_capture.verify(args.capture.absolute(), args.candidate)
    config = grid.config(decode(read(args.config.absolute())))
    if args.operation == 'describe':
        return grid.plan(config)
    if args.operation == 'model-preflight':
        return model_preflight(config)
    require(args.output is not None, 'argument', '--output required')
    if args.operation == 'build':
        require(args.prepared_rheon is not None, 'argument', '--prepared-rheon required')
        binary = build_native.build('capture', config, args.prepared_rheon.absolute(), args.cargo, args.output)
        return {'status': 'qualified_build', 'binary': str(binary)}
    require(args.binary is not None, 'argument', '--binary required')
    observer = None
    if args.live_root:
        module = load_module('cubic_live_observer', grid.LAB/'experiments/live-project-v1/observer.py')
        observer = module.Observer(args.live_root)
    try:
        fixture = args.output.absolute().with_name(args.output.name+'-fixture')
        fixture_report = cubic_capture.generate(config, args.binary.absolute(), fixture, fixture=True, observer=observer)
        if observer:
            observer.publish('Cubic-cell fixture qualified', 'Independent one-frame physics gates passed', 1, 1)
        report = cubic_capture.generate(config, args.binary.absolute(), args.output.absolute(), observer=observer, qualified_fixture=fixture)
        if observer:
            module.atomic(args.live_root/'capture-manifest.json', decode(read(args.output.absolute()/'manifest.json')))
            preview = {**observer.preview, 'pressure_label': 'Native reference pressure (Pa); independent physics verification passed'}
            observer.publish('Cubic-cell capture verified',
                             f'{report["frames"]} frames; cubic cells at {config["cells_per_metre"]} cells/metre. Model training not run.',
                             report['frames'], report['frames'], state='completed', preview=preview)
        return {'fixture': fixture_report, 'development': report}
    except BaseException:
        if observer:
            observer.publish('Cubic-cell capture failed', 'Capture did not qualify; retained evidence is diagnostic', state='failed')
        raise


if __name__ == '__main__':
    try:
        print(json.dumps(main(), allow_nan=False))
    except ContractError as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
