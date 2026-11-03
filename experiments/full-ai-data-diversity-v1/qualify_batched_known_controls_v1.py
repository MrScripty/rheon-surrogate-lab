"""Bitwise comparison against original per-fine CUDA force construction."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index, LongTemporalDataset
from temporal_control_packets_v1 import control_packet
from dynamic_cuda_controls_v1 import CudaKnownControls
from batched_known_controls_v1 import CudaKnownControlSum


def hashes(values):
    return [hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for v in values]


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only prescribed-control qualification required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen prescribed-control source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    result = dict(status='running', rows=[], source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), inference_device='cuda', display_label='GPU (CUDA)',
        GPU=torch.cuda.get_device_name(), new_parameter_fits=0, learned_forwards=0,
        cpu_model_forward=False, reserved52_references_opened=0, latency_claim=False,
        full_generator_target_pass=False, future_physical_fields_loaded=False,
        scope='Known prescribed controls only; no fluid fields or labels are loaded. F64 byte hashes include signed zero. Compare original fine-force generation plus ordered Torch additions against two batched CUDA kernels.')
    original = candidate = None
    started = time.monotonic()
    try:
        dataset = LongTemporalDataset(build_index([ROOT / p for p in reg['audits']]))
        original = CudaKnownControls(reg['original_control_build'])
        candidate = CudaKnownControlSum(reg['candidate_build'])
        for identifier, (_, receipt, _) in dataset.receipts.items():
            scene = receipt['scene']
            counts = tuple(scene['counts'])
            for start in reg['starts']:
                for width in reg['widths']:
                    packet = torch.from_numpy(control_packet(scene, start, width)).to('cuda')
                    packet_hash = hashes([packet])
                    fine = original.generate_from_packet(counts, packet)
                    expected = fine[0]
                    for row in fine[1:]:
                        expected = [a + b for a, b in zip(expected, row)]
                    actual = candidate.sum_from_packet(counts, packet)
                    if hashes(actual) != hashes(expected) or hashes([packet]) != packet_hash:
                        raise ValueError('Batched force sum changed F64 output or input packet bytes')
                    result['rows'].append(dict(identifier=identifier, split=receipt['split'],
                        counts_xyz=list(counts), cells_per_metre=scene['cells_per_metre'],
                        start_fine_step=start, width=width, all_force_fields_bitwise_equal=True,
                        input_packet_unchanged=True, output_sha256=hashes(actual)))
                    del fine, expected, actual, packet
                    if (time.monotonic() - started > reg['maximum_wall_seconds']
                            or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                        raise ValueError('Registered force qualification resource budget exceeded')
            write(args.output / 'results.json', result)
            print('Batched force sum exact:', identifier, flush=True)
        if len(result['rows']) != reg['expected_cases']:
            raise ValueError('Incomplete prescribed-control qualification population')
        result.update(status='passed_bitwise_batched_known_force_sum',
            all_force_sum_bytes_exact=True, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if candidate is not None:
            candidate.close()
        if original is not None:
            original.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
