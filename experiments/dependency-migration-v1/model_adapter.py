"""CPU float32 model behind Rheon's current generic pressure adapter protocol.

Only current adapter/numerical contracts are accepted. Historical deployment,
recipe, history and Run records belong to the offline packager, not this process.
"""
import argparse
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

LIMIT = 8 * 1024**2
STAGE = 'sealed-mac-pressure-context-v1'
STOPPING = {'schema': 'rheon-learned-stopping-policy-v2', 'relative_residual': 1e-6,
            'absolute_residual': 1e-12, 'predicted_divergence': 1e-7,
            'mean_free_pressure_relative_error': 1e-6}


class ProtocolError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, message, code='invalid'):
    if not condition:
        raise ProtocolError(code, message)


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result

    def constant(value):
        raise ProtocolError('invalid', 'Nonfinite JSON')

    def floating(value):
        result = float(value)
        require(math.isfinite(result), 'Nonfinite JSON')
        return result

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)


def closed(value, keys, message):
    require(type(value) is dict and set(value) == set(keys.split()), message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value, minimum=0, maximum=2**53-1):
    return type(value) is int and minimum <= value <= maximum


def measure(path):
    h = hashlib.sha256()
    total = 0
    with Path(path).open('rb') as stream:
        while raw := stream.read(1024**2):
            total += len(raw)
            h.update(raw)
    return {'sha256': h.hexdigest(), 'bytes': total}


def read(path, cap=1024**2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= cap,
            'Invalid measured file')
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    require(len(raw) <= cap, 'Measured file exceeds bound')
    return raw


@dataclass
class State:
    np: object
    torch: object
    model: object
    network: object
    normalization: dict
    policy: dict
    manifest: dict
    iteration: object = None


def load(path, emit=None):
    """Verify current material and every installed asset before numerical imports.

    A process owns one package. Use separate processes to compare packages, since
    their independently measured numerical modules have intentionally flat names.
    """
    path = Path(path).absolute()
    root = path.parent.resolve()
    raw = read(path)
    manifest = strict(raw)
    closed(manifest, 'schema id version stage python entrypoint files support context inputs output',
           'Closed current adapter manifest required')
    require(manifest['schema'] == 'rheon-pressure-adapter-v1' and manifest['stage'] == STAGE,
            'Unsupported adapter contract', 'unsupported')
    require(manifest['entrypoint'] == 'model_adapter.py', 'Measured current entrypoint required')
    require(manifest['context'] == {'minimum_native_steps': 15, 'prior_pressure': 'required'}
            and manifest['inputs'] == {'base': 'previous_pressure', 'residual': 'initial_residual',
                                       'rho': 'density', 'dt': 'dt'}
            and manifest['output'] == {'role': 'full_pressure', 'dtype': 'f64',
                                       'layout': 'x_fast', 'gauge': 'cell_zero'},
            'Frozen context, named inputs and full-pressure output required')
    files = manifest['files']
    require(type(files) is dict and 1 <= len(files) <= 64, 'Package inventory bound')
    required = {'model_adapter.py', 'model_adapter_numerics.py', 'numerical.json',
                'policy.py', 'portable_model.py', 'frozen_gn.py', 'original_model.py',
                'normalization.json', 'weights.npz', 'installed-runtime.json', 'torch_runtime.py'}
    require(required <= set(files), 'Incomplete current package')
    require(files['weights.npz']['bytes'] <= 8*1024**2, 'Weight archive bound')
    total = 0
    for relative, expected in files.items():
        require(type(relative) is str and re.fullmatch(r'[A-Za-z0-9_.\-/]{1,240}', relative)
                and all(part not in ('', '.', '..') for part in relative.split('/')),
                'Invalid package path')
        asset = root / relative
        require(asset.resolve() == asset and asset.is_relative_to(root), 'Package path escapes root')
        closed(expected, 'sha256 bytes', 'Closed measured identity required')
        require(integer(expected['bytes'], 0, 256*1024**2), 'Package file bound')
        total += expected['bytes']
        require(total <= 512*1024**2 and measure(asset) == expected, 'Package inventory drift')
    # The current runtime owner validates topology, aliases and hashes. Capture
    # that owner, rather than duplicating or weakening its manifest contract.
    owner_spec = importlib.util.spec_from_file_location('rheon_adapter_torch_runtime', root / 'torch_runtime.py')
    runtime_owner = importlib.util.module_from_spec(owner_spec)
    owner_spec.loader.exec_module(runtime_owner)
    runtime = runtime_owner.load(root / 'installed-runtime.json')
    closed(manifest['python'], 'executable sha256 bytes', 'Closed interpreter identity required')
    executable = Path(manifest['python']['executable'])
    require(executable.is_absolute() and executable.resolve() == Path(sys.executable).resolve()
            and executable.resolve() == Path(runtime['python']).resolve()
            and measure(executable) == {k: manifest['python'][k] for k in ('sha256', 'bytes')},
            'Interpreter identity mismatch')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and os.environ.get('TULDOK_TORCH_DEVICE') == 'cpu', 'Explicit CPU environment required')
    if emit:
        emit({'kind': 'loading', 'phase': 'Checking current managed CPU runtime assets'})
    runtime_owner.verify(runtime)
    spec = strict(read(root / 'numerical.json'))
    closed(spec, 'schema kind correction weights inference spatial_coarse', 'Closed current numerical mapping required')
    require(spec['schema'] == 'rheon-pressure-numerics-v2', 'Current numerical mapping required', 'unsupported')
    kind = spec['kind']
    require(kind in ('seven_channel', 'physical_residual'), 'Unsupported numerical mapping', 'unsupported')
    require(type(spec['spatial_coarse']) is bool and (not spec['spatial_coarse'] or
            kind == 'physical_residual' and 'coarse.py' in files),
            'Explicit captured residual coarse architecture required')
    policy = spec['inference']
    if kind == 'physical_residual':
        require(spec['correction'] == {'activation': 'linear', 'steps': 2}
                and policy == {'max_network_calls': 32, 'single_evaluation': True,
                               'minimum_network_evaluations': 0, 'stopping': STOPPING},
                'Frozen control correction/stopping policy required')
        require({'physical_residual.py', 'learned_iteration.py'} <= set(files), 'Missing residual numerical owners')
    else:
        require(spec['correction'] is None and policy == {'max_network_calls': 1},
                'Frozen seven-channel one-call policy required')
    sys.path[:0] = [str(root), runtime['site_packages']]
    np = importlib.import_module('numpy')
    torch = importlib.import_module('torch')
    require(np.__version__ == runtime['packages']['numpy']
            and torch.__version__ == runtime['packages']['torch'], 'Numerical package identity mismatch')
    model = importlib.import_module('model_adapter_numerics')
    model.select(spec)
    model.configure('cpu')
    torch.set_num_threads(1)
    network = model.PressureCorrectionNet().eval()
    expected = network.state_dict()
    require(type(spec['weights']) is dict and set(expected) == set(spec['weights']), 'Architecture tensor names')
    tensors = {}
    with np.load(root / 'weights.npz', allow_pickle=False) as arrays:
        require(set(arrays.files) == set(expected), 'Weight archive tensor names')
        for name, tensor in expected.items():
            value = arrays[name]
            identity = {'shape': list(value.shape), 'dtype': value.dtype.str,
                        'bytes': value.nbytes, 'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
            require(identity == spec['weights'][name] and value.dtype.str == '<f4'
                    and list(value.shape) == list(tensor.shape) and bool(np.isfinite(value).all()),
                    'Weight tensor identity mismatch')
            tensors[name] = torch.from_numpy(value.copy())
    network.load_state_dict(tensors, strict=True)
    require(not torch.cuda.is_initialized() and all(p.device.type == 'cpu' and p.dtype == torch.float32
            for p in network.parameters()), 'CPU float32 model required')
    normalization = strict(read(root / 'normalization.json'))
    iteration = importlib.import_module('learned_iteration') if kind == 'physical_residual' else None
    state = State(np, torch, model, network, normalization, policy, manifest, iteration)
    if emit:
        emit({'kind': 'adapter_ready', 'id': manifest['id'], 'version': manifest['version'],
              'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'device': 'cpu', 'optimizer_updates': 0})
    return state


def validate_request(value, manifest):
    """Reject unsupported physical/history context before accessing the model."""
    closed(value, 'kind schema id stage context inputs', 'Closed current prediction required')
    require(value['kind'] == 'predict' and value['schema'] == 'rheon-pressure-context-request-v1'
            and value['stage'] == STAGE, 'Unsupported prediction contract', 'unsupported')
    require(integer(value['id']), 'Bounded integer request id required')
    context = value['context']
    closed(context, 'run_id generation mode interval_start_s interval_end_s dt_s grid prior', 'Closed context required')
    require(type(context['run_id']) is str and 0 < len(context['run_id']) <= 4096
            and not any(ord(c) < 32 or ord(c) == 127 for c in context['run_id']), 'Invalid context run identity')
    require(integer(context['generation'], 1) and context['mode'] in ('accepted_candidate', 'diagnostic_candidate'),
            'Invalid context generation/mode')
    require(context['generation'] > manifest['context']['minimum_native_steps'],
            'Testing setup lacks required native warm-up', 'unsupported_context')
    start, end, dt = (context[k] for k in ('interval_start_s', 'interval_end_s', 'dt_s'))
    require(all(number(x) for x in (start, end, dt)) and start > 0 and dt == 1/60
            and start+dt > start and end > start
            and abs(struct.unpack('<Q', struct.pack('<d', end))[0]
                    - struct.unpack('<Q', struct.pack('<d', start+dt))[0]) <= 1,
            'Unsupported pressure interval', 'unsupported')
    grid = context['grid']
    closed(grid, 'counts_xyz spacing_xyz_m origin_xyz_m', 'Closed cubic grid required')
    counts, spacing = grid['counts_xyz'], grid['spacing_xyz_m']
    support = manifest['support']
    require(type(counts) is list and len(counts) == 3
            and all(integer(n, support['counts_min'][i], support['counts_max'][i]) for i, n in enumerate(counts))
            and math.prod(counts) <= support['max_cells'], 'Unsupported grid counts', 'unsupported')
    require(type(spacing) is list and len(spacing) == 3 and all(number(x) for x in spacing)
            and spacing[0] == spacing[1] == spacing[2]
            and support['spacing_min_m'] <= spacing[0] <= support['spacing_max_m']
            and type(grid['origin_xyz_m']) is list and len(grid['origin_xyz_m']) == 3
            and all(number(x) and x == 0 for x in grid['origin_xyz_m']),
            'Unsupported cubic grid geometry', 'unsupported')
    prior = context['prior']
    require(prior is not None, 'Solved prior pressure is required', 'unsupported_context')
    closed(prior, 'generation interval_end_s status', 'Closed prior lineage required')
    require(integer(prior['generation'], 1) and prior['generation']+1 == context['generation']
            and number(prior['interval_end_s']) and prior['interval_end_s'] == start, 'Stale prior pressure lineage', 'unsupported_context')
    status = prior['status']
    require(type(status) is dict and (status == {'kind': 'accepted'}
            or set(status) == {'kind', 'qualified'} and status['kind'] == 'diagnostic'
            and status['qualified'] is True),
            'Solved prior pressure is not qualified', 'unsupported_context')
    require(context['mode'] != 'accepted_candidate' or status == {'kind': 'accepted'},
            'Diagnostic prior cannot become accepted history', 'unsupported_context')
    require(type(value['inputs']) is dict and set(value['inputs']) == set(manifest['inputs']), 'Named input inventory mismatch')
    fields = {}
    units = {'previous_pressure': 'Pa', 'initial_residual': 'm3/s2', 'density': 'kg/m3', 'dt': 's'}
    for name, semantic in manifest['inputs'].items():
        field = value['inputs'][name]
        closed(field, 'semantic units association dimensions scalar layout values', 'Closed field metadata required')
        scalar = semantic in ('density', 'dt')
        require(field['semantic'] == semantic and field['units'] == units[semantic]
                and field['association'] == ('global' if scalar else 'cell')
                and type(field['dimensions']) is list and len(field['dimensions']) == 3
                and all(integer(n, 1, 128) for n in field['dimensions'])
                and field['dimensions'] == ([1, 1, 1] if scalar else counts)
                and field['scalar'] == 'f64' and field['layout'] == 'x_fast', 'Field semantic/unit/shape mismatch')
        values = field['values']
        require(type(values) is list and len(values) == (1 if scalar else math.prod(counts))
                and all(number(x) for x in values), 'Finite full-grid values required')
        fields[semantic] = values
    require(fields['density'] == [1000.] and fields['dt'] == [dt], 'Unsupported water physical scalars', 'unsupported')
    require(fields['previous_pressure'][0] == 0., 'Cell-zero pressure gauge required')
    return counts, spacing[0], fields


def sample_from_request(value, state):
    counts, h, fields = validate_request(value, state.manifest)
    np, torch = state.np, state.torch
    sample = {target: torch.from_numpy(np.asarray(fields[source], dtype='<f8').copy()).reshape(1, 1, *counts[::-1])
              for target, source in [('base', 'previous_pressure'), ('native_input_residual', 'initial_residual')]}
    sample.update(cell_size_m=torch.tensor([h], dtype=torch.float64),
                  rho=torch.tensor(fields['density'], dtype=torch.float64),
                  dt=torch.tensor(fields['dt'], dtype=torch.float64))
    return sample


def predict(value, state, trace=None):
    sample = sample_from_request(value, state)
    with state.torch.no_grad():
        if state.iteration is None:
            pressure = state.model.predict_pressure(state.network, sample, state.normalization)
        else:
            policy = state.policy
            pressure = state.iteration.correct(state.model, state.network, state.normalization, sample,
                policy['max_network_calls'], policy['stopping'], trace, policy['single_evaluation'],
                minimum_network_evaluations=policy['minimum_network_evaluations'])
    array = pressure.reshape(-1).numpy()
    require(array.dtype.str == '<f8' and bool(state.np.isfinite(array).all()) and array[0] == 0
            and not state.torch.cuda.is_initialized(), 'Invalid CPU full pressure output')
    return {'kind': 'candidate', 'id': value['id'], 'role': 'full_pressure',
            'pressure': array.tolist(), 'device': 'cpu', 'optimizer_updates': 0}


def failure(error, request_id=None):
    return {'kind': 'error', 'code': getattr(error, 'code', 'unavailable' if isinstance(error, (OSError, ImportError)) else 'invalid'),
            'id': request_id, 'message': str(error)[:2000]}


def send(value):
    raw = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    require(len(raw) <= LIMIT, 'Reply exceeds protocol bound')
    sys.stdout.buffer.write(raw+b'\n')
    sys.stdout.buffer.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    args = parser.parse_args()
    state = load(args.deployment, emit=send)
    while raw := sys.stdin.buffer.readline(LIMIT+1):
        request_id = None
        try:
            require(len(raw) <= LIMIT and raw.endswith(b'\n'), 'Incomplete or oversized protocol record')
            value = strict(raw)
            if type(value) is dict and integer(value.get('id')):
                request_id = value['id']
            send(predict(value, state))
        except (ValueError, OSError, ImportError) as error:
            send(failure(error, request_id))
            return 2
    return 0


if __name__ == '__main__':
    try:
        code = main()
    except (ValueError, OSError, ImportError) as error:
        send(failure(error))
        code = 2
    sys.exit(code)
