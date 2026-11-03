"""Check unchanged fine GPU baseline against a new finite-F32 reference epoch.

This is a compatibility check, not a reduced-pass successor or speed claim.
Future reference fields and RHS enter the offline metric only.
"""
import argparse
import os
from pathlib import Path
import time

import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index, LongTemporalDataset, FIELDS
from temporal_frozen_history_graph_v1 import TemporalFrozenGraph
from diagnose_causal_composition import inventory, measure


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only reference compatibility check required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen compatibility source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), inference_device='cuda',
        cpu_model_forward=False, new_parameter_fits=0, optimizer_updates=0,
        reserved52_references_opened=0, rows=[], latency_claim=False,
        reference_training_adopted=False, full_generator_target_pass=False)
    model = None
    try:
        dataset = LongTemporalDataset(build_index([ROOT / p for p in reg['audits']]))
        model = TemporalFrozenGraph(reg['transport_pair'], reg['shared_pair'],
            reg['build'], reg['control_build'])
        before = inventory(model)
        normalization = read(ROOT / reg['normalization'])
        examples = [e for e in dataset.index['examples'] if e['start_fine_step'] in reg['starts']]
        with torch.inference_mode():
            for example in examples:
                identifier, start, rate, k = (example[n] for n in
                    ('identifier', 'start_fine_step', 'output_steps_per_s', 'K'))
                scene = dataset.receipts[identifier][1]['scene']
                current = dataset.inputs(example)['current']
                fields = {n: torch.from_numpy(current[n]).to('cuda') for n in FIELDS}
                outputs = model.predict(fields, scene, start, rate, k, (1,) * (240 // rate))
                if len(outputs) != k:
                    raise AssertionError('Every physical output required')
                for lead, (output, step) in enumerate(zip(outputs, example['target_fine_steps']), 1):
                    actual = {n: v.cpu() for n, v in output.items()}
                    if any(not torch.isfinite(v).all() for v in actual.values()):
                        raise AssertionError('Nonfinite unchanged baseline')
                    truth = dataset._load(identifier, step)
                    values = measure(actual, truth, scene, normalization)
                    result['rows'].append(dict(identifier=identifier,
                        start_fine_step=start, output_steps_per_s=rate, K=k,
                        lead=lead, counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                        metrics=values, all_accuracy_pass=all(v <= reg['maximum_error']
                            for v in values['relative'].values())))
                if (time.monotonic() - started > reg['maximum_wall_seconds']
                        or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                    raise ValueError('Registered compatibility resource budget exceeded')
        if len(result['rows']) != reg['expected_complete_outputs'] or inventory(model) != before:
            raise AssertionError('Incomplete population or modified baseline')
        result['failures'] = sum(not r['all_accuracy_pass'] for r in result['rows'])
        result['worst_relative'] = {n: max(r['metrics']['relative'][n] for r in result['rows'])
            for n in ('equation', 'pressure', 'velocity', 'tracer')}
        result['status'] = 'passed_finite_reference_unchanged_GPU_model_compatibility' if not result['failures'] else 'failed_finite_reference_GPU_model_compatibility'
        if result['failures']:
            raise AssertionError('Finite-F32 reference differs beyond unchanged baseline accuracy gates')
    except BaseException as error:
        if result['status'] == 'running':
            result['status'] = 'failed'
        result['error'] = repr(error)
        raise
    finally:
        if model is not None:
            model.close()
        result.update(elapsed_seconds=time.monotonic() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
