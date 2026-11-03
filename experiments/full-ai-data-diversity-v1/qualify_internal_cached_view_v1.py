"""Full148scene CUDA cache capacity/objective/gradient check without optimization."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from internal_residual_dataset_v1 import TrainingView, METRICS
from internal_residual_cached_view_v1 import GPUCachedTrainingView
from internal_residual_network_v2 import InternalGroupResidual
from internal_residual_loss_v1 import metrics
from qualify_internal_objective_v2 import inventory
from managed_input_stream_v1 import streamed_identity
from capture_memory_telemetry_v1 import process_memory

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only training-cache qualification required')
    reg = read(args.protocol)
    for p, expected in reg['source_sha256'].items():
        if streamed_identity(ROOT / p)['sha256'] != expected:
            raise ValueError('Frozen cache dependency changed: ' + p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    mp = ROOT / reg['manifest']
    manifest = read(mp)
    archive = mp.parent / manifest['archive']['path']
    if streamed_identity(archive) != {k: manifest['archive'][k] for k in ('bytes', 'sha256')}:
        raise ValueError('Original training archive differs')
    qualified = read(ROOT / reg['initial_qualification'])
    optimization = read(mp.parent / 'optimization-normalization.json')
    model = InternalGroupResidual(normalization=read(mp.parent / 'network-normalization.json'))
    initial = inventory(model)
    if initial != qualified['parameter_inventory']:
        raise ValueError('Qualified initial student differs')
    source = TrainingView(mp, archive)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    cache = None
    result = dict(status='running', optimizer_updates=0, new_parameter_fits=0,
        cpu_model_forward=False, expanded_reserved52_references_opened=0,
        full_generator_target_pass=False, fitting_allowed=False,
        inference_device='cuda', display_label='GPU (CUDA)')
    def bounded():
        if (time.monotonic()-started > reg['maximum_wall_seconds'] or
                process_memory()['peak_since_exec_bytes'] > reg['maximum_peak_rss_bytes'] or
                torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes'] or
                torch.cuda.max_memory_reserved() > reg['maximum_gpu_reserved_bytes']):
            raise ValueError('Training-cache qualification resource budget exceeded')
    def progress(n):
        bounded()
        if n % 27 == 0:
            print('Exact CUDA training cache', n, '/783', flush=True)
    try:
        cache = GPUCachedTrainingView(source, reg['maximum_gpu_allocated_bytes'],
            reg['maximum_gpu_reserved_bytes'], progress)
        if cache.cached_tensor_bytes != reg['expected_training_tensor_bytes']:
            raise ValueError('Exact declared training tensor footprint differs')
        source.close()
        # Once cached, accidental archive access invalidates this qualification.
        def audit(event, args):
            if event == 'open' and isinstance(args[0], (str, bytes)):
                if Path(os.fsdecode(args[0])).resolve() == archive.resolve():
                    raise PermissionError('Training cache must not reread its archive')
        sys.addaudithook(audit)
        totals = {m: 0. for m in METRICS}
        model.zero_grad(set_to_none=True)
        for index in range(783):
            row, sample, response = cache.sample(index)
            values = metrics(response.emit(sample, model(sample['features'])), sample, optimization)
            (sum(values.values()) * cache.sample_gradient_weight).backward()
            for m, value in values.items():
                totals[m] += float(value.detach()) / 783
            del values
            bounded()
        fixed = cache.fixed_objective(optimization)
        objective = cache.active_weight * sum(totals.values()) + fixed['objective']
        if not math.isclose(objective, qualified['full148scene_objective'], rel_tol=1e-10, abs_tol=1e-30):
            raise ValueError('Cached full objective differs')
        gradients = {}
        for n, p in model.named_parameters():
            if p.grad is None or not bool(p.grad.isfinite().all()):
                raise ValueError('Complete finite cached gradients required')
            actual = dict(norm=float(p.grad.double().norm()), maximum=float(p.grad.abs().max()))
            if any(not math.isclose(v, qualified['gradients'][n][k], rel_tol=1e-6, abs_tol=1e-18) for k, v in actual.items()):
                raise ValueError('Cached full gradient differs: ' + n)
            gradients[n] = actual
        if inventory(model) != initial:
            raise ValueError('Cached objective changed model weights')
        result.update(status='passed_all783_CUDA_cache_full148scene_objective_and_gradients',
            cached_training_tensor_bytes=cache.cached_tensor_bytes, complete_active_samples=783,
            training_scenes=148, active_scenes=29, fixed_controls=fixed,
            full148scene_objective=objective, gradients=gradients,
            archive_closed_and_rereads_forbidden_during_forward_backward=True,
            zero_update_parameters_unchanged=True, memory_counters=process_memory(),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-started, source_sha256=reg['source_sha256'])
    except BaseException as error:
        result.update(status='failed', error=repr(error), memory_counters=process_memory(),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        raise
    finally:
        source.close()
        if cache is not None:
            cache.close()
        (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
