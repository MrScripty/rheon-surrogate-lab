"""Pairwise complete-output parity and GPU cost for zero-head folding."""
import argparse
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v3 import LongTemporalDataset
from predicted_fine_readout_v2 import FineReadoutGraph
from folded_fine_readout_v1 import FoldedFineReadoutGraph
from shared_temporal_native_transfer_v1 import transaction
from evaluate_trained_shared_pair_v1 import hashes, load_network
from captured_parameter_inventory_v1 import inventory


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only folding qualification required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen folding source differs: ' + name)
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
    result = dict(status='running', rows=[], preparations=[], source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), inference_device='GPU (CUDA)',
        new_parameter_fits=0, cpu_model_forward=False, reserved52_references_opened=0,
        native_future_fields_read=0, full_generator_target_pass=False, model_promotion=False,
        scope='Pairwise exactCPUoutputbytes for allKfields. Complete CPUcurrentfields to ownedCPUoutputs, controls/events/H2D/allmodelwork/packing/D2H/sync. Balanced alternating original/folded order. Divide onlyK. Setup/prewarmups/warmups separate. No nativefuture labels read by this qualification.')
    models, before = {}, {}
    started = time.monotonic()
    try:
        for mode, cls in [('original', FineReadoutGraph), ('folded', FoldedFineReadoutGraph)]:
            model = cls(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'],
                hidden=4, force_sum_build=reg['force_sum_build'],
                fine_mode='preoutput_fine_fullstate_moment_tracer')
            models[mode] = model
            load_network(model, ROOT / reg['pair'], 4, 'selected')
            if mode == 'folded':
                result['zero_head_certificate'] = model.certify_zero_heads()
            before[mode] = inventory(model)
        if before['original'] != before['folded']:
            raise ValueError('Folding qualification requires identical complete parameter inventories')
        for index, example in enumerate(examples):
            identifier, start, rate, k = (example[n] for n in ('identifier', 'start_fine_step', 'output_steps_per_s', 'K'))
            scene = dataset.receipts[identifier][1]['scene']
            fields = {n: torch.from_numpy(v) for n, v in dataset.inputs(example)['current'].items()}
            original_fields = hashes(fields)
            expected, preparations = None, {}
            for mode in ('original', 'folded'):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                outputs = transaction(models[mode], fields, scene, start, rate, k)
                preparations[mode] = 1000 * (time.perf_counter() - tick)
                actual = [hashes(o) for o in outputs]
                if expected is None:
                    expected = actual
                if actual != expected:
                    result['mismatch'] = dict(identifier=identifier, start_fine_step=start,
                        output_steps_per_s=rate, K=k, mode=mode, expected_sha256=expected, actual_sha256=actual)
                    raise ValueError('Folded forecast changes complete output bytes')
            samples = {mode: [] for mode in models}
            for unused in range(reg['warmups']):
                for mode in ('original', 'folded'):
                    if [hashes(o) for o in transaction(models[mode], fields, scene, start, rate, k)] != expected:
                        raise ValueError('Warmup parity changed')
            for repetition in range(reg['repetitions']):
                order = ('original', 'folded') if (index + repetition) % 2 == 0 else ('folded', 'original')
                for mode in order:
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    outputs = transaction(models[mode], fields, scene, start, rate, k)
                    torch.cuda.synchronize()
                    samples[mode].append(1000 * (time.perf_counter() - tick))
                    if hashes(fields) != original_fields or [hashes(o) for o in outputs] != expected:
                        raise ValueError('Repeated parity or caller ownership changed')
            limit = reg['strict_ms_per_physical_output'] * max(1, math.prod(scene['counts']) / 32768)
            result['rows'].append(dict(identifier=identifier, start_fine_step=start, output_steps_per_s=rate, K=k,
                counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                force_profile=scene['forcing_program']['profile'], force_motion=scene['forcing_program']['motion'],
                all_complete_output_bytes_equal=True, output_sha256=expected, block_wall_ms=samples,
                median_GPU_ms_per_physical_output={mode: float(np.median(v)) / k for mode, v in samples.items()},
                maximum_GPU_ms_per_physical_output={mode: max(v) / k for mode, v in samples.items()},
                strict_ms_per_physical_output_limit=limit,
                all_latency_samples_pass={mode: max(v) / k < limit for mode, v in samples.items()}))
            result['preparations'].append(dict(identifier=identifier, start_fine_step=start,
                output_steps_per_s=rate, K=k, first_complete_prewarmup_ms=preparations, warmups=reg['warmups']))
            if (time.monotonic() - started > reg['maximum_wall_seconds']
                    or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                raise ValueError('Finite folding qualification budget exceeded')
            if len(result['rows']) % 12 == 0:
                write(args.output / 'results.json', result)
        if len(result['rows']) != reg['expected_blocks'] or sum(r['K'] for r in result['rows']) != reg['expected_physical_outputs']:
            raise ValueError('Incomplete folding population')
        for mode, model in models.items():
            if inventory(model) != before[mode]:
                raise ValueError('Folding qualification changed parameters')
        result.update(status='passed_frozen_zero_head_complete_output_byte_parity',
            independent_training_masters=len({e['identifier'] for e in examples}),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), gpu_name=torch.cuda.get_device_name())
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        for model in models.values():
            model.close()
        result['elapsed_seconds'] = time.monotonic() - started
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
