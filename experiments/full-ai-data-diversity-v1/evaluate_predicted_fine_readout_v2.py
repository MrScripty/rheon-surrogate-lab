"""Predicted pre-output fine-readout ablation with complete GPU timing."""
import argparse
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v3 import LongTemporalDataset
from shared_temporal_control_guides_v2 import ControlGuideGraph
from predicted_fine_readout_v2 import FineReadoutGraph
from shared_temporal_graph_v3 import SharedTemporalGraph
from shared_temporal_native_transfer_v1 import transaction
from captured_parameter_inventory_v1 import inventory
from evaluate_trained_shared_pair_v1 import hashes, load_network
from diagnose_causal_composition import measure


def construct(reg, mode):
    model = FineReadoutGraph(reg['transport_pair'], reg['shared_pair'], reg['build'],
        reg['control_build'], hidden=4, force_sum_build=reg['force_sum_build'], fine_mode=mode)
    load_network(model, ROOT / reg['pair'], 4, 'selected')
    return model


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only ablation required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen guide experiment source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    dataset = LongTemporalDataset(read(ROOT / reg['qualified_index']))
    examples = [e for e in dataset.index['examples'] if e['split'] == 'train'
        and (reg['starts'] is None or e['start_fine_step'] in reg['starts'])]
    examples.sort(key=lambda e: (e['identifier'], e['output_steps_per_s'], e['K'], e['start_fine_step']))
    norm = read(ROOT / reg['normalization'])
    result = dict(status='running', rows=[], preparations=[],
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        inference_device='cuda', display_label='GPU (CUDA)',
        new_parameter_fits=0, reserved52_references_opened=0, cpu_model_forward=False,
        model_promotion=False, full_generator_target_pass=False,
        reference_labels_changed=False, pressure_dtype='float64', carrier_dtype='float32',
        neural_dtype='float32', baseline_wrapper_parity_blocks=0,
        scope='Complete CPU currentfields to K ownedCPU outputstates, controls/events/H2D, wholecoarse sharedforecast, knownprefix/preoutputguides, K final1/240s learnedadvances, optionalK knowncontrolmoment guided transports and alloutputcopies/D2H/sync. Fine-readout adds K actuallearnedadvances, not extra physicaloutputs. Divide onlybyK. Setup/prewarmups/warmups separate; training-design ablation.')
    started = time.monotonic()
    model = reference = None
    try:
        for mode in reg['modes']:
            model = construct(reg, mode)
            before = inventory(model)
            if mode == 'prefix_guide':
                reference = ControlGuideGraph(reg['transport_pair'], reg['shared_pair'], reg['build'],
                    reg['control_build'], hidden=4, force_sum_build=reg['force_sum_build'],
                    guide_mode='last_force_pressure_prefix_impulse')
                load_network(reference, ROOT / reg['pair'], 4, 'selected')
            for example in examples:
                identifier, start, rate, k = (example[n] for n in ('identifier', 'start_fine_step', 'output_steps_per_s', 'K'))
                scene = dataset.receipts[identifier][1]['scene']
                fields = {n: torch.from_numpy(a) for n, a in dataset.inputs(example)['current'].items()}
                original = hashes(fields)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                outputs = transaction(model, fields, scene, start, rate, k)
                prewarmup_ms = 1000 * (time.perf_counter() - tick)
                expected = [hashes(o) for o in outputs]
                if reference is not None:
                    if [hashes(o) for o in transaction(reference, fields, scene, start, rate, k)] != expected:
                        raise ValueError('Fine readout baseline wrapper changed the qualified prefix-guide bytes')
                    result['baseline_wrapper_parity_blocks'] += 1
                targets = [dataset._load(identifier, step) for step in example['target_fine_steps']]
                metrics = [measure(output, truth, scene, norm) for output, truth in zip(outputs, targets)]

                def check(actual):
                    if len(actual) != k or hashes(fields) != original or [hashes(o) for o in actual] != expected:
                        raise ValueError('Guide replay changed caller fields or repeated outputs')

                for unused in range(reg['warmups']):
                    check(transaction(model, fields, scene, start, rate, k))
                samples = []
                for unused in range(reg['repetitions']):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    actual = transaction(model, fields, scene, start, rate, k)
                    torch.cuda.synchronize()
                    samples.append(1000 * (time.perf_counter() - tick))
                    check(actual)
                budget = reg['strict_ms_per_physical_output'] * max(1, math.prod(scene['counts']) / 32768)
                result['rows'].append(dict(identifier=identifier, split='train', mode=mode,
                    counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                    force_profile=scene['forcing_program']['profile'], force_motion=scene['forcing_program']['motion'],
                    start_fine_step=start, output_steps_per_s=rate, K=k, block_duration_s=k / rate,
                    native_metrics_per_output=metrics,
                    all_accuracy_pass=all(all(v <= reg['maximum_error'] for v in m['relative'].values()) for m in metrics),
                    block_wall_ms=samples, median_ms_per_physical_output=float(np.median(samples)) / k,
                    maximum_ms_per_physical_output=max(samples) / k,
                    strict_ms_per_physical_output_limit=budget, all_latency_samples_pass=max(samples) / k < budget,
                    all_outputs_owned_and_caller_unchanged=True))
                result['preparations'].append(dict(identifier=identifier, start_fine_step=start,
                    output_steps_per_s=rate, K=k, mode=mode, first_complete_prewarmup_ms=prewarmup_ms,
                    warmups=reg['warmups']))
                if (time.monotonic() - started > reg['maximum_wall_seconds']
                        or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                    raise ValueError('Finite guide ablation budget exceeded')
                if len(result['rows']) % 12 == 0:
                    write(args.output / 'results.json', result)
            if inventory(model) != before:
                raise ValueError('Zero-fit diagnostic changed parameters')
            if reference is not None:
                reference.close()
                reference = None
            model.close()
            model = None
        if len(result['rows']) != reg['expected_blocks'] or result['baseline_wrapper_parity_blocks'] != len(examples):
            raise ValueError('Frozen guide ablation population is incomplete')
        result.update(status='completed_predicted_preoutput_fine_readout_ablation',
            independent_training_masters=len({e['identifier'] for e in examples}),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), gpu_name=torch.cuda.get_device_name())
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if reference is not None:
            reference.close()
        if model is not None:
            model.close()
        result['elapsed_seconds'] = time.monotonic() - started
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
