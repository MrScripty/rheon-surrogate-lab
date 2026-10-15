"""Reuse the historical water pipeline without changing its source or results."""
from pathlib import Path
import sys

LAB = Path(__file__).resolve().parents[2]
SEQUENCE = LAB / 'experiments/water-learning-sequence-v1'
sys.path.insert(0, str(SEQUENCE))

import data
import model
import policy


def measured_model(artifacts):
    import torch
    model.configure('cpu')
    receipt = policy.decode(policy.read(artifacts / 'receipt.json'))
    history = policy.decode(policy.read(artifacts / 'history.json'))
    normalization = policy.decode(policy.read(artifacts / 'normalization.json'))
    weights = policy.read(artifacts / 'weights.npz')
    policy.require(receipt['mode'] == 'restore' and receipt['optimizer_updates'] == 0
                   and receipt['execution']['cuda_initialized'] is False,
                   'checkpoint', 'completed CPU restore artifacts required')
    for name in ('weights.npz', 'normalization.json', 'history.json'):
        policy.require(policy.sha(policy.read(artifacts / name)) == receipt['outputs'][name]['sha256'],
                       'artifact_hash', name)
    final_update = max(spec['update'] for spec in history['checkpoints'].values())
    final = history['checkpoints'][f'checkpoint-{final_update:03d}.npz']
    policy.require(final['sha256'] == policy.sha(weights), 'checkpoint', 'declared final state')
    network = model.PressureCorrectionNet()
    data.helpers().load_weights(network, artifacts / 'weights.npz', final['inventory'])
    network.eval()
    policy.require(not torch.cuda.is_initialized(), 'cpu_only', 'diagnostics must not initialize CUDA')
    return network, normalization, receipt, history
