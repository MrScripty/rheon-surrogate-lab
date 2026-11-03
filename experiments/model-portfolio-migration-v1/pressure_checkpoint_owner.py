"""Load a captured deployment or declared training checkpoint without a CPU forward.

The caller owns one numerical process and installs its device guard before this
owner imports captured model code. A training artifact is never described as a
CPU restore or a new admission. It retains its original receipt and source.
"""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from pressure_gpu_timing import ROOT, read, sha


def load(specification):
    if specification['kind'] == 'captured_deployment':
        reference_path = ROOT / specification['reference']['path']
        if sha(reference_path) != specification['reference']['sha256']:
            raise ValueError('Captured deployment reference changed')
        entry = read(reference_path)['source_entry']
        sys.path.insert(0, str(ROOT / 'experiments/full-ai-next-priorities-v1'))
        import study_support
        import inference
        state = inference.load(dict(root=str(ROOT), models=[entry]), entry['id'])
        state.custody = dict(kind='captured_deployment', source_entry=entry,
                             reference=specification['reference'])
        return state
    if specification['kind'] != 'training_checkpoint':
        raise ValueError('Unsupported checkpoint owner kind')
    project = (ROOT / specification['project']).resolve()
    artifacts = (ROOT / specification['training_output']).resolve()
    if not project.is_relative_to(ROOT) or not artifacts.is_relative_to(ROOT):
        raise ValueError('Captured task-local material required')
    receipt = read(artifacts / 'receipt.json')
    if (sha(artifacts / 'receipt.json') != specification['receipt_sha256']
            or receipt['mode'] != 'train' or receipt['status'] != 'complete'):
        raise ValueError('Original completed training receipt required')
    witness = read(project / 'source-witness.json')
    if (sha(project / 'source-witness.json') != receipt['closure']['source_witness_sha256']
            or sha(project / 'config.json') != receipt['closure']['experiment_sha256']):
        raise ValueError('Captured executable/configuration witness changed')
    for name, expected in witness['sources'].items():
        if Path(name).name != name or sha(project / name) != expected:
            raise ValueError('Captured numerical source changed: ' + name)
    for name in ('weights.npz', 'normalization.json', 'history.json'):
        if sha(artifacts / name) != receipt['outputs'][name]['sha256']:
            raise ValueError('Original declared training output changed: ' + name)
    if receipt['outputs']['weights.npz']['sha256'] != specification['weights_sha256']:
        raise ValueError('Another training checkpoint was selected')
    sys.path.insert(0, str(project))
    import torch
    model = importlib.import_module('model')
    data = importlib.import_module('data')
    config = read(project / 'config.json')
    model.configure('cpu')
    model.select(config)
    if model.ARCHITECTURE != receipt['architecture']:
        raise ValueError('Constructed captured architecture differs')
    network = model.PressureCorrectionNet().eval()
    history = read(artifacts / 'history.json')
    inventory = history['checkpoints'][f"checkpoint-{receipt['optimizer_updates']:03d}.npz"]['inventory']
    data.helpers().load_weights(network, artifacts / 'weights.npz', inventory)
    if data.helpers().state_inventory(network) != inventory:
        raise ValueError('Loaded checkpoint inventory differs from the original training record')
    return SimpleNamespace(torch=torch, np=data.np, model=model, data=data, config=config,
        network=network, normalization=read(artifacts / 'normalization.json'),
        custody=dict(kind='training_checkpoint', project=specification['project'],
            training_output=specification['training_output'], receipt_sha256=specification['receipt_sha256'],
            source_witness_sha256=receipt['closure']['source_witness_sha256'],
            weights_sha256=specification['weights_sha256'], optimizer_updates_in_loaded_execution=0))
