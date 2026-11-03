"""CUDA CNN execution for the two original CPU-interface variable-grid models.

Keep their captured preprocessing, reconstruction, modules and parameters. The
CNN body below must exactly match both original forward bodies after their four
CPU input guards (with only the bound object name normalized). Transfers bracket
the CNN. The original CPU neural forward is never called.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def cnn(network, values):
    skips = []
    for encoder in network.encoders:
        values = encoder(values)
        skips.append(values)
        kernel = tuple(min(2, n) for n in values.shape[2:])
        values = F.avg_pool3d(values, kernel, stride=kernel, ceil_mode=True, count_include_pad=False)
    values = network.bottleneck(values)
    values = values + network.global_pool(values).expand_as(values)
    for decoder, skip in zip(network.decoders, reversed(skips)):
        values = F.interpolate(values, size=skip.shape[2:], mode='nearest')
        values = decoder(torch.cat((values, skip), dim=1))
    return network.head(values)


def load(material, identifier):
    import json
    global torch, F
    if identifier not in ('C03', 'C04'):
        raise ValueError('Only the two source-reviewed legacy interfaces are allowed')
    entry = next(e for e in material['models'] if e['id'] == identifier)
    for name, expected in {**entry['identities'], **entry['source_bindings']}.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Legacy captured source/checkpoint changed')
    project, restore = ROOT / entry['project'], ROOT / entry['restore']
    sys.path.insert(0, str(project))
    import torch
    from torch.nn import functional as F
    import model
    import data
    model.configure_cpu()
    config = json.loads((project / 'config.json').read_text())
    receipt = json.loads((restore / 'receipt.json').read_text())
    history = json.loads((restore / 'history.json').read_text())
    if (receipt['mode'] != 'restore' or receipt['optimizer_updates'] != 0
            or receipt['architecture'] != model.ARCHITECTURE
            or receipt['closure']['source_witness_sha256'] != entry['identities'][str((project / 'source-witness.json').relative_to(ROOT))]
            or receipt['closure']['experiment_sha256'] != entry['identities'][str((project / 'config.json').relative_to(ROOT))]):
        raise ValueError('Legacy source/configuration/restore contract differs')
    for name in ('weights.npz', 'normalization.json', 'history.json'):
        if hashlib.sha256((restore / name).read_bytes()).hexdigest() != receipt['outputs'][name]['sha256']:
            raise ValueError('Legacy restore output changed')
    network = model.PressureCorrectionNet().eval()
    inventory = history['checkpoints'][f"checkpoint-{config['training']['updates']:03d}.npz"]['inventory']
    data.helpers().load_weights(network, restore / 'weights.npz', inventory)
    if torch.cuda.is_initialized() or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in network.parameters()):
        raise ValueError('CPU parameter restoration without a neural forward required')
    original = ast.parse(inspect.getsource(model.PressureCorrectionNet))
    forward = next(n for n in original.body[0].body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    if len(forward.body) < 5 or any(not isinstance(n, ast.Expr) for n in forward.body[:4]):
        raise ValueError('Legacy forward boundary changed')
    class BoundName(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id='network', ctx=node.ctx), node) if node.id == 'self' else node
    original_body = BoundName().visit(ast.Module(body=forward.body[4:], type_ignores=[]))
    port_body = ast.Module(body=ast.parse(inspect.getsource(cnn)).body[0].body, type_ignores=[])
    if ast.dump(original_body, include_attributes=False) != ast.dump(port_body, include_attributes=False):
        raise ValueError('CUDA CNN computation differs from captured model operations')
    return SimpleNamespace(identifier=identifier, entry=entry, torch=torch, np=data.np,
                           model=model, data=data, network=network, config=config,
                           normalization=json.loads((restore / 'normalization.json').read_text()),
                           CPU_forward_executed=False, CNN_operation_AST_preserved=True)


def predict(state, sample):
    def network(values):
        if values.device.type != 'cpu' or values.dtype != torch.float32:
            raise ValueError('Captured CPU feature preparation required')
        return cnn(state.network, values.to('cuda')).cpu()
    inputs = dict(sample)
    if state.identifier == 'C03':
        inputs['spacing'] = sample['cell_size_m'][:, None].expand(-1, 3)
    return state.model.predict_pressure(network, inputs, state.normalization)


def operator(state, pressure, sample):
    spacing = (sample['cell_size_m'][:, None].expand(-1, 3)
               if state.identifier == 'C03' else sample['cell_size_m'])
    return state.model.full_operator(pressure, spacing, sample['rho'])
