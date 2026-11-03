"""Qualify the shared request decoder against a completed independent reader.

This constructs CPU input tensors and hashes them; it executes no model and
does not initialize CUDA. The earlier C107 replay used its captured data reader.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import pressure_cases


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(report):
    if report.exists():
        raise ValueError('Fresh decoder qualification report required')
    torch.set_num_threads(1)
    baseline_path = ROOT / 'runs/model-portfolio-migration-v1/pressure-GPU-C107-accuracy-01/results.json'
    baseline = json.loads(baseline_path.read_text())
    cases = pressure_cases.load(torch, np)
    if len(cases) != len(baseline['rows']) or len(cases) != 128:
        raise ValueError('Exact prior case denominator required')
    for (identity, sample), row in zip(cases, baseline['rows'], strict=True):
        if any(identity[k] != row[k] for k in identity):
            raise ValueError('Request identity/order differs from the independently decoded baseline')
        for name in ('base', 'native_input_residual', 'cell_size_m', 'rho', 'dt'):
            if hashlib.sha256(sample[name].numpy().tobytes()).hexdigest() != row['input_sha256'][name]:
                raise ValueError('Input tensor differs from the original reader: ' + name)
        for field, key in (('reference', 'reference_pressure_sha256'), ('rhs', 'reference_rhs_sha256')):
            if hashlib.sha256(sample[field].numpy().tobytes()).hexdigest() != row[key]:
                raise ValueError('Reference field differs from the original reader: ' + field)
    if torch.cuda.is_initialized():
        raise ValueError('No inference or CUDA initialization belongs in decoder qualification')
    value = dict(schema='rheon.pressure-request-decoder-qualification.v1', created_unix=time.time(),
                 status='all128_inputs_and_references_exactly_match_captured_C107_reader',
                 cases=128, source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                     (Path(__file__), ROOT / 'experiments/model-portfolio-migration-v1/pressure_cases.py')},
                 baseline=dict(path=str(baseline_path.relative_to(ROOT)), sha256=sha(baseline_path)),
                 CPU_neural_forward=False, GPU_neural_forward=False, CUDA_initialized=False,
                 new_parameter_fits=0, latency_claim=False)
    report.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    run(args.report.resolve())
