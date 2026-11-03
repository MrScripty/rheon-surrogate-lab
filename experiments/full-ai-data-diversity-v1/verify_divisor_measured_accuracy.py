"""Verify accuracy of the exact complete outputs recorded by a CPU benchmark."""
import argparse
import os
from pathlib import Path
from support import ROOT, read, write, digest
from divisor_generator import load_model
from diagnose_causal_composition import FIELDS, inventory, measure
from evaluate_recurrent_sampler import identity
import numpy as np
import torch


def run(args):
    if args.output.exists():
        raise ValueError('Fresh measured-output accuracy evidence required')
    reg, measured = read(args.protocol), read(args.measurement)
    if reg['mode'] != 'benchmark' or measured['protocol_sha256'] != digest(args.protocol):
        raise ValueError('Actual benchmark output must bind this registered protocol')
    if len(measured['rows']) != reg['expected_measurement_rows'] or measured['cores'] != reg['cores']:
        raise ValueError('Complete measured population and exact CPU policy required')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Measured model/data source changed: '+path)
    expected = {}
    for row in measured['rows']:
        key = (row['identifier'], row['start'], row['horizons'])
        outputs = row['measured_complete_output_identities']
        if len(outputs) != row['horizons'] or key in expected and outputs != expected[key]:
            raise ValueError('Measured repetitions changed complete output bytes')
        expected[key] = outputs
    for task in Path('/proc/self/task').iterdir():
        os.sched_setaffinity(int(task.name), set(reg['cores']))
    torch.set_num_threads(len(reg['cores']))
    torch.set_num_interop_threads(1)
    norm = read(ROOT/reg['normalization'])
    model = load_model(reg['pair'], reg['library_path'])
    before = inventory(model)
    rows = []

    def state(path, receipt, step):
        spec = receipt['states'][step]
        source = path.parent/spec['path']
        if spec['step'] != step or digest(source) != spec['sha256']:
            raise ValueError('Current/reference state bytes or clock changed')
        with np.load(source, allow_pickle=False) as archive:
            return {k: archive[k].copy() for k in (*FIELDS, 'rhs')}

    try:
        with torch.inference_mode():
            for trajectory in read(ROOT/reg['bank'])['trajectories']:
                path = ROOT/trajectory['receipt']
                if digest(path) != trajectory['receipt_sha256']:
                    raise ValueError('Measured scene receipt changed')
                receipt = read(path)
                for start in reg['starts']:
                    initial = state(path, receipt, start)
                    fields = {k: initial[k] for k in FIELDS}
                    for k in reg['horizons']:
                        outputs = model.predict(fields, receipt['scene'], start, k)
                        if len(outputs) != k or [identity(v) for v in outputs] != expected[(trajectory['identifier'], start, k)]:
                            raise ValueError('Accuracy replay differs from actual timed output bytes')
                        for lead, output in enumerate(outputs, 1):
                            truth = state(path, receipt, start+lead)
                            if set(output) != set(FIELDS):
                                raise ValueError('Complete physical outputs required')
                            for name, value in output.items():
                                dtype = torch.float64 if name == 'pressure' else torch.float32
                                if value.dtype != dtype or tuple(value.shape) != truth[name].shape or not bool(torch.isfinite(value).all()):
                                    raise ValueError('Typed finite complete physical output required')
                            metrics = measure(output, truth, receipt['scene'], norm)
                            passed = all(v <= reg['maximum_error'] for v in metrics['relative'].values())
                            rows.append(dict(identifier=trajectory['identifier'], start=start,
                                horizons=k, lead=lead, counts_xyz=receipt['scene']['counts'],
                                cells_per_metre=receipt['scene']['cells_per_metre'],
                                metrics=metrics, all_accuracy_pass=passed,
                                bitwise_equal_actual_measured_output=True))
                print('Verified exact measured outputs', trajectory['identifier'], flush=True)
            if before != inventory(model) or torch.cuda.is_initialized():
                raise ValueError('Measured-output replay changed parameters or device')
    finally:
        model.close()
    if len(rows) != len(read(ROOT/reg['bank'])['trajectories'])*len(reg['starts'])*sum(reg['horizons']):
        raise ValueError('Every complete intermediate measured state must be scored')
    for path, expected_hash in reg['source_sha256'].items():
        if digest(ROOT/path) != expected_hash:
            raise ValueError('Frozen source changed during measured-output qualification: '+path)
    write(args.output, dict(status='completed_exact_measured_output_accuracy_verification',
        rows=rows, cores=reg['cores'], all_accuracy_pass=all(r['all_accuracy_pass'] for r in rows),
        measured_output_identity_parity=True, measurement=str(args.measurement.relative_to(ROOT)),
        measurement_sha256=digest(args.measurement), protocol_sha256=digest(args.protocol),
        source_sha256=reg['source_sha256'], optimizer_updates=0, latency_claim=False,
        full_generator_target_pass=False, short_blocks_only=True,
        full_trajectory_accuracy_evidence='runs/full-ai-data-diversity-v1/learned-divisor-independent-evaluation-02/results.json',
        qualifying_multicore_latency_requires_full_trajectory_accuracy_at_matching_policy=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.measurement = args.measurement.resolve()
    args.output = args.output.resolve()
    run(args)
