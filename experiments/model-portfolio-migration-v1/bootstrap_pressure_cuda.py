"""Two captured GroupNorm fits: CUDA CNN with original CPU physical boundary.

The only in-memory forward-source changes are the CPU device guard and its
diagnostic string. All neural operation ASTs, modules and learned tensors remain
original. A plain callable bridges original CPU feature/reconstruction functions
to the guarded CUDA module. No captured file is edited.
"""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import textwrap

from import_common_generator_measurements import ROOT, binding, read, require


def forward_variant(original):
    source = textwrap.dedent(inspect.getsource(original.PressureCorrectionNet.forward))
    before = ast.parse(source)
    after = ast.parse(source)
    function = after.body[0]
    require(len(function.body)>4 and all(isinstance(n,ast.Expr) for n in function.body[:4]),
        'Original four input guards required')
    guard = function.body[1]
    constants = [n for n in ast.walk(guard) if isinstance(n,ast.Constant)]
    require([n.value for n in constants] == ['eager f32 CPU inputs required','cpu'],
        'Exact original device guard literals required')
    for node in constants:
        node.value = {'cpu':'cuda','eager f32 CPU inputs required':'eager f32 CUDA inputs required'}[node.value]
    restored = ast.parse(ast.unparse(after))
    for node in ast.walk(restored.body[0].body[1]):
        if isinstance(node,ast.Constant) and node.value in ('cuda','eager f32 CUDA inputs required'):
            node.value = {'cuda':'cpu','eager f32 CUDA inputs required':'eager f32 CPU inputs required'}[node.value]
    require(ast.dump(restored,include_attributes=False) == ast.dump(before,include_attributes=False),
        'Device boundary inverse must restore entire original forward AST')
    namespace = dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(after),original.__file__,'exec'),namespace)
    return namespace['forward'],dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        adapted_forward_source=ast.unparse(after),only_two_device_guard_literals_changed=True,
        all_operation_and_other_guard_AST_exact=True,original_file_unchanged=True)


@contextmanager
def restore(entry):
    import numpy as np
    import torch
    from cuda_forward_observer import CUDAForwardObserver
    for spec in [entry['original_receipt'],entry['original_configuration'],entry['source_witness'],
            *entry['sources'],*entry['outputs'].values()]:
        require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Original bootstrap material changed')
    path = ROOT/next(s['path'] for s in entry['sources'] if s['original_name']=='gn_model.py')
    require(not any(n.startswith('rheon_groupnorm_frozen_original') for n in sys.modules),
        'Fresh original private package required for each fit')
    spec = importlib.util.spec_from_file_location('captured_bootstrap_groupnorm',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    forward,proof = forward_variant(module.original)
    observer = CUDAForwardObserver(torch)
    observer.enable('construction')
    try:
        with torch.device('cuda'):
            network = module.PressureCorrectionNet().eval().requires_grad_(False)
        inventory = entry['selected_inventory']
        require(len(inventory) == len(network.state_dict()) == 38 and entry['serialized_tensor_elements'] == 372377,
            'Complete original GroupNorm architecture/inventory required')
        with np.load(ROOT/entry['outputs']['weights.npz']['path'],allow_pickle=False) as archive:
            require(set(archive.files) == set(inventory) == set(network.state_dict()), 'All selected tensors required')
            values = {}
            for name in archive.files:
                value = archive[name]
                expected = inventory[name]
                require(value.dtype.str == expected['dtype'] == '<f4' and list(value.shape) == expected['shape']
                    and hashlib.sha256(value.tobytes()).hexdigest() == expected['sha256'], 'Selected learned tensor differs')
                values[name] = torch.from_numpy(value).to('cuda')
            network.load_state_dict(values,strict=True)
        network.forward = MethodType(forward,network)
        normalization = read(ROOT/entry['outputs']['normalization.json']['path'])
        def verify_state():
            for name,value in network.state_dict().items():
                require(value.device.type == 'cuda' and value.dtype == torch.float32
                    and hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() == inventory[name]['sha256'],
                    'Original CUDA parameters changed')
        verify_state()
        def bridge(features):
            require(features.device.type == 'cpu' and features.dtype == torch.float32,
                'Original fixed CPU preprocessing must emit F32 features')
            return network(features.to('cuda')).to('cpu',copy=True).contiguous()
        state = SimpleNamespace(module=module,network=network,normalization=normalization,
            bridge=bridge,proof=proof,observer=observer,verify_state=verify_state)
        observer.set_phase('prediction')
        yield state
        verify_state()
        require(not observer.forbidden_attempts and not observer.calls.get(('construction','PressureCorrectionNet')),
            'No CPU or constructor neural query allowed')
    finally:
        observer.close()
