"""Restored coefficients, signs, dtypes and channel-selection equivalence controls."""
import argparse
import os
from pathlib import Path
import time

import torch

from support import ROOT, read, write, digest
from internal_residual_fused_graph_v1 import InternalResidualFusedGraphGenerator
from coupled_macrostep_v1 import RestoredLinear
from recurrent_sampler_native import dimensions
from capture_memory_telemetry_v1 import process_memory
from diagnose_causal_composition import inventory


def run(args):
    reg = read(args.protocol)
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only primitive controls required')
    for p, expected in reg['source_sha256'].items():
        if digest(ROOT / p) != expected:
            raise ValueError('Frozen precise linear source changed: ' + p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1730)
    model = InternalResidualFusedGraphGenerator(reg['residual_pair'], reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    original = RestoredLinear(model.backbone.model)
    initial = inventory(model)
    args.output.mkdir(parents=True)
    rows = []
    started = time.monotonic()
    result = dict(status='running', optimizer_updates=0, new_parameter_fits=0, cpu_model_forward=False,
        expanded_reserved52_references_opened=0, full_generator_target_pass=False,
        inference_device='cuda', display_label='GPU (CUDA)')
    try:
        for counts in reg['counts_xyz']:
            shape = tuple(reversed(counts))
            sizes = dimensions(counts)[:3]
            for dtype in (torch.float32, torch.float64):
                for pattern in ('random', 'zero'):
                    fields = [(torch.randn(s, device='cuda', dtype=dtype) if pattern == 'random' else
                        torch.zeros(s, device='cuda', dtype=dtype)) for s in sizes]
                    pressure = torch.randn(shape, device='cuda', dtype=dtype) if pattern == 'random' else torch.zeros(shape, device='cuda', dtype=dtype)
                    cases = [('flux', fields, shape, 0, channel, sign) for channel in (None, 0, 1, 2) for sign in (-1, 1)]
                    cases += [('momentum', [pressure], size, head, None, sign) for head, size in enumerate(sizes) for sign in (-1, 1)]
                    for name, inputs, target, head, channel, sign in cases:
                        expected = original(inputs, target, name, head, channel, sign)
                        actual = model.backbone.linear(inputs, target, name, head, channel, sign)
                        if not torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)):
                            raise ValueError('Precise fused learned-linear bytes differ')
                        rows.append(dict(counts_xyz=counts, dtype=str(dtype), pattern=pattern,
                            name=name, head=head, channel=channel, sign=sign, elements=actual.numel(), byte_exact=True))
            if (time.monotonic()-started > reg['maximum_wall_seconds'] or
                    process_memory()['peak_since_exec_bytes'] > reg['maximum_peak_rss_bytes'] or
                    torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                raise ValueError('Finite primitive resource budget exceeded')
        if inventory(model) != initial:
            raise ValueError('Primitive controls changed learned weights')
        result.update(status='passed_precise_fused_learned_linear_controls', rows=rows,
            all_arrays_byte_exact=True, learned_parameters_unchanged=True,
            memory_counters=process_memory(), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    except BaseException as error:
        result.update(status='failed', error=repr(error), rows=rows)
        raise
    finally:
        model.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
