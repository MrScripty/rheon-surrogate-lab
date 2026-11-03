"""Select the captured sharing constructor, then use the frozen joint boundary.

The original worker's timing, packing, scoring and CUDA guards are unchanged.
The process-local import alias changes only model construction, before timing.
Each invocation owns one model and exits; no captured source is modified.
"""
import argparse
import importlib
from pathlib import Path
import sys
from types import ModuleType

from joint_generator_gpu_benchmark import run as measure
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]


def run(protocol_path, identifier, phase, output):
    protocol = read(protocol_path)
    prepared = read(ROOT/protocol['cases']['path'])
    selected = [m for m in prepared['models'] if m['identifier'] == identifier]
    require(len(selected) == 1, 'One reviewed sharing model required')
    selected = selected[0]
    config_path = ROOT/selected['configuration']['path']
    require(binding(config_path) == selected['configuration'], 'Original sharing configuration changed')
    config = read(config_path)
    require(config['schema'] == 'rheon-temporal-fit-sharing-v1'
            and config['architecture'] == selected['architecture'] in ('endpoint','reuse')
            and config['horizons'] == selected['K'] in (2,3,4) and config['width'] == 12,
            'Reviewed sharing architecture required')
    require('bundle_model_v5' not in sys.modules and 'bundle_model_sharing' not in sys.modules,
            'Fresh process with no previously loaded model required')
    source = next(s for s in selected['sources'] if Path(s['path']).name == 'bundle_model_sharing.py')

    def construct(k, width):
        require(k == config['horizons'] and width == config['width'], 'Captured constructor dimensions differ')
        module = importlib.import_module('bundle_model_sharing')
        require(Path(module.__file__).resolve() == (ROOT/source['path']).resolve()
                and binding(ROOT/source['path']) == source, 'Captured sharing implementation required')
        return module.BundleNet(k,width,config['architecture'])

    alias = ModuleType('bundle_model_v5')
    alias.BundleNet = construct
    sys.modules['bundle_model_v5'] = alias
    try:
        measure(protocol_path,identifier,phase,output)
    finally:
        del sys.modules['bundle_model_v5']
    measured = read(output)
    require(measured['status'] == 'all64_native_cases_measured', 'Complete original worker result required')
    measured.update(architecture=config['architecture'],constructor_source=source,
        constructor_arguments=[config['horizons'],config['width'],config['architecture']],
        constructor_adapter=binding(Path(__file__)),
        measurement_boundary='Unchanged frozen joint_generator_gpu_benchmark.run; only captured BundleNet constructor differs.')
    output.write_bytes(canonical(measured)+b'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--identifier',required=True)
    parser.add_argument('--phase',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(),args.identifier,args.phase,args.output.resolve())
