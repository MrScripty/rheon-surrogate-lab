#!/usr/bin/env python3
"""Qualify fresh cubic-cell captures and heterogeneous pressure training inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from policy import HERE, LAB, canonical, decode, experiment, fresh, load_module, read, write
import build_native
import data


def observer(root):
    module = load_module('variable_pressure_live_observer', LAB/'experiments/live-project-v1/observer.py')
    value = module.Observer(root)
    if (root/'frame.json').exists():
        value.preview = decode(read(root/'frame.json', 2*1024**2))
    return value


def material(config, prepared, cargo, output, live, captures=None):
    output = fresh(output); write(output/'config.json', config)
    selected = [] if captures is None else list(captures)
    for index, capture in enumerate(config['captures'] if captures is None else []):
        directory = fresh(output/f'density-{capture["cells_per_metre"]}-{index}')
        write(directory/'config.json', capture)
        executable = LAB/'experiments/cubic-pressure-v1/pipeline.py'
        for name, arguments in [('build', ['--prepared-rheon', str(prepared), '--cargo', str(cargo), '--output', str(directory/'build')]),
                                ('generate', ['--binary', str(directory/'build/rheon-cubic-pressure-capture'), '--output', str(directory/'capture'), '--live-root', str(live)])]:
            with (directory/(name+'.log')).open('xb') as log:
                subprocess.run([sys.executable, str(executable), name, '--config', str(directory/'config.json'), *arguments], stdout=log, stderr=log, check=True, timeout=600)
        selected.append(directory/'capture')
    native = build_native.build('evaluation', config, prepared, cargo, output/'native-build')
    observation = observer(live)
    data.pack(config, selected, native, output/'packet', observation)
    frames = sum(e['frames'] for c in config['captures'] for e in c['experiment']['trajectories'])
    observation.publish('Variable-grid inputs qualified', 'Multiple densities and odd/rectangular grids ready for managed training', frames, frames)
    result = {'status': 'qualified_material', 'packet': str(output/'packet'), 'native_binary': str(native),
              'captures': list(map(str, selected)), 'reused_captures': captures is not None}
    write(output/'material.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'configs/development.json')
    parser.add_argument('--prepared-rheon', type=Path, required=True)
    parser.add_argument('--cargo', type=Path, default=Path.home()/'.cargo/bin/cargo')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live-root', type=Path, default=LAB/'runs/live-project')
    parser.add_argument('--captures', type=Path, nargs='+', help='Reverify existing whole-parent captures instead of generating new histories')
    args = parser.parse_args(); config = experiment(decode(read(args.config)))
    try:
        print(json.dumps(material(config, args.prepared_rheon.absolute(), args.cargo, args.output.absolute(), args.live_root.absolute(),
                                  None if args.captures is None else [p.absolute() for p in args.captures])))
    except BaseException:
        observer(args.live_root.absolute()).publish('Variable-grid material failed', 'Retained capture/input evidence is diagnostic; no training admitted', state='failed')
        raise


if __name__ == '__main__':
    main()
