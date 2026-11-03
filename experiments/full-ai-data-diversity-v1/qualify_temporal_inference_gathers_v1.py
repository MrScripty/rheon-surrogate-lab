"""Exact nonzero correction equivalence for the inference-only learned gathers."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import LongTemporalDataset, FIELDS
from temporal_group_projector_v2 import TemporalGroupProjector
from inference_restored_linear_v1 import InferenceRestoredLinear
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory


def hashes(fields):
    return {n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n, v in fields.items()}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only gather qualification required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Registered source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    result = dict(status='running', rows=[], inference_device='cuda', display_label='GPU (CUDA)',
        cpu_model_forward=False, new_parameter_fits=0, reserved52_references_opened=0,
        latency_claim=False, full_generator_target_pass=False,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    model = None
    started = time.monotonic()
    try:
        dataset = LongTemporalDataset(read(ROOT / reg['index']))
        model = TemporalGroupProjector(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
        original = model.linear
        accelerated = InferenceRestoredLinear(model.model)
        before = inventory(model)
        rng = torch.Generator(device='cuda').manual_seed(reg['seed'])
        for identifier, (_, receipt, _) in dataset.receipts.items():
            scene = receipt['scene']
            sizes = dimensions(tuple(scene['counts']))
            for start in reg['dense_starts']:
                initial = dataset._load(identifier, start)
                fields = {n: torch.from_numpy(initial[n]).to('cuda') for n in FIELDS}
                current_hash = hashes(fields)
                for width in reg['widths']:
                    model.linear = original
                    phase = model.phase(fields, scene, start, width)
                    for scale in reg['dimensionless_delta_scales']:
                        with torch.no_grad():
                            delta_v = [torch.randn(s, generator=rng, dtype=torch.float64, device='cuda')
                                * (scale * phase['h'] / phase['dt']) for s in sizes[:3]]
                            delta_t = torch.randn(phase['shape'], generator=rng, dtype=torch.float64, device='cuda') * scale
                            model.linear = original
                            expected = model.emit(phase, delta_v, delta_t)
                            model.linear = accelerated
                            actual = model.emit(phase, delta_v, delta_t)
                            if hashes(actual) != hashes(expected):
                                raise ValueError('Fused learned gather changed nonzero correction output bytes')
                        result['rows'].append(dict(identifier=identifier, start_fine_step=start,
                            width=width, dimensionless_delta_scale=scale,
                            all_complete_fields_bitwise_equal=True))
                    if hashes(fields) != current_hash:
                        raise ValueError('Correction qualification changed current fields')
                    if (time.monotonic() - started > reg['maximum_wall_seconds']
                            or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                        raise ValueError('Qualification resource bound exceeded')
                write(args.output / 'results.json', result)
            print('Qualified fused nonzero correction gathers:', identifier, flush=True)
        # The differentiable route must retain the original implementation.
        model.linear = accelerated
        value = torch.randn((3, 3, 3), generator=rng, dtype=torch.float64, device='cuda', requires_grad=True)
        expected = original([value], (3, 3, 3), 'momentum', 0)
        actual = accelerated([value], (3, 3, 3), 'momentum', 0)
        expected_gradient = torch.autograd.grad(expected.square().sum(), value, retain_graph=True)[0]
        actual_gradient = torch.autograd.grad(actual.square().sum(), value)[0]
        if not torch.equal(actual, expected) or not torch.equal(actual_gradient, expected_gradient):
            raise ValueError('Autograd route differs from qualified original')
        if len(result['rows']) != reg['expected_rows'] or inventory(model) != before:
            raise ValueError('Full registered population and fixed parameters required')
        result.update(status='completed_bitwise_temporal_correction_inference_gather_qualification',
            all_nonzero_and_zero_complete_outputs_bitwise_equal=True,
            autograd_original_forward_and_gradient_bitwise_equal=True,
            learned_parameters_unchanged=True, elapsed_operational_seconds=time.monotonic() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if model is not None:
            model.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
