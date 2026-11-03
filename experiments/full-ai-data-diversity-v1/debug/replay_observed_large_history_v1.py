"""DEBUG: assert actual large-grid own-history accuracy against a frozen endpoint."""
import argparse
import hashlib
import importlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT, read, write, digest
from cuda_native_transfer_v1 import upload_state, download_outputs
from diagnose_causal_composition import inventory, measure

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')


def run(args):
    if args.output.exists():
        raise ValueError('Fresh actual failure evidence required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Learned inference must be CUDA')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen dependency changed: ' + path)
    os.sched_setaffinity(0, {4})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    case = reg['case']
    scene = dict(read(ROOT / case['receipt'])['scene'],
                 reference_substeps=case['reference_substeps'], forcing_control_substeps=1)
    specs = {s['step']: s for s in case['states']}
    for step in (0, reg['endpoint_step']):
        if digest(ROOT / specs[step]['path']) != specs[step]['sha256']:
            raise ValueError('Native endpoint fixture changed')
    with np.load(ROOT / specs[0]['path'], allow_pickle=False) as archive:
        initial = {name: torch.from_numpy(archive[name].copy()) for name in FIELDS}
    with np.load(ROOT / specs[reg['endpoint_step']]['path'], allow_pickle=False) as archive:
        truth = {name: archive[name].copy() for name in (*FIELDS, 'rhs')}
    observed = read(ROOT / reg['observation'])
    expected = next(r['output_sha256'] for r in observed['rows']
                    if r['K'] == 1 and r['step'] == reg['endpoint_step'])
    reference = next(r['actual_field_sha256'] for r in observed['reference_rows']
                     if r['step'] == reg['endpoint_step'])
    if any(hashlib.sha256(v.tobytes()).hexdigest() != reference[n] for n, v in truth.items()):
        raise ValueError('Endpoint does not match actual uninterrupted reference')
    cls = getattr(importlib.import_module(reg['candidate_module']), reg['candidate_class'])
    model = cls(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model)
    args.output.mkdir(parents=True)
    began = time.monotonic()
    try:
        with torch.inference_mode():
            current = upload_state(initial)
            model.prepare_geometry(tuple(scene['counts']))
            model.prepare_block(current, scene, 0, 1, case['reference_substeps'])
            for start in range(reg['endpoint_step']):
                current = model.predict(current, scene, start, 1, case['reference_substeps'])[0]
            emitted = download_outputs([current])[0]
            hashes = {n: hashlib.sha256(v.numpy().tobytes()).hexdigest() for n, v in emitted.items()}
            equal = hashes == expected
            if reg['require_observed_bytes'] and not equal:
                raise ValueError('Replay differs from the actual observed failing endpoint')
            metrics = measure(emitted, truth, scene, read(ROOT / reg['normalization']))
        if inventory(model) != before:
            raise ValueError('Diagnostic changed restored parameters')
        passed = all(v <= reg['maximum_error'] for v in metrics['relative'].values())
        write(args.output / 'results.json', dict(
            status='passed_target' if passed else 'reproduced_actual_equation_failure',
            endpoint_step=reg['endpoint_step'], metrics=metrics,
            original_failure_endpoint_bitwise_equal=equal, actual_endpoint_hashes=hashes,
            elapsed_diagnostic_seconds=time.monotonic() - began,
            native_oracle_in_candidate=False, cpu_model_forward=False,
            optimizer_updates=0, latency_claim=False,
            protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256']))
        print('Actual history endpoint', reg['endpoint_step'], 'equation',
              metrics['relative']['equation'], 'target', reg['maximum_error'],
              'observed bytes equal', equal, flush=True)
        if not passed:
            raise AssertionError('Actual large-grid history exceeds original error limit')
    finally:
        model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
