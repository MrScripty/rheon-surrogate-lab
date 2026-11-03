"""Full training-only per-metric CUDA gradients, with no optimizer or CPU forward."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from capture_memory_telemetry_v1 import process_memory
from internal_residual_dataset_v1 import TrainingView, METRICS
from internal_residual_loss_v1 import metrics
from internal_residual_network_v2 import InternalGroupResidual, active_input
from internal_residual_weighted_loss_v1 import validate, weighted_sum, fixed_objective
from managed_input_stream_v1 import streamed_identity
from qualify_internal_objective_v2 import inventory

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def run(args):
    reg = read(args.protocol)
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only diagnostic required')
    for name, expected in reg['source_sha256'].items():
        if streamed_identity(ROOT / name)['sha256'] != expected:
            raise ValueError('Frozen diagnostic source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest_path = ROOT / reg['manifest']
    manifest = read(manifest_path)
    archive = manifest_path.parent / manifest['archive']['path']
    if streamed_identity(archive) != {k: manifest['archive'][k] for k in ('bytes', 'sha256')}:
        raise ValueError('Training archive identity changed')
    source = read(ROOT / reg['initial_qualification'])
    weights = validate(reg['metric_weights'])
    optimization = read(manifest_path.parent / 'optimization-normalization.json')
    model = InternalGroupResidual(normalization=read(manifest_path.parent / 'network-normalization.json'))
    initial = inventory(model)
    if initial != source['parameter_inventory']:
        raise ValueError('Initial model changed')
    versions = {n: p._version for n, p in model.named_parameters()}
    params = list(model.named_parameters())
    gradients = {m: {n: torch.zeros_like(p, dtype=torch.float64) for n, p in params} for m in METRICS}
    totals = {m: 0. for m in METRICS}
    rows, quiet = [], []
    view = TrainingView(manifest_path, archive)
    fixed = fixed_objective(view, optimization, weights)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    try:
        for index in range(783):
            row, sample, response = view.sample(index)
            values = metrics(response.emit(sample, model(sample['features'])), sample, optimization)
            if not bool(active_input(sample['features'])):
                quiet.append(index)
                if any(float(v.detach()) != 0 for v in values.values()):
                    raise ValueError('Quiescent control has nonzero loss')
            local = {}
            for mi, metric in enumerate(METRICS):
                value = values[metric]
                if not bool(torch.isfinite(value)):
                    raise ValueError('Nonfinite metric')
                derivatives = torch.autograd.grad(value, [p for _, p in params],
                    retain_graph=mi != len(METRICS) - 1, allow_unused=True)
                squared = torch.zeros((), device='cuda', dtype=torch.float64)
                for (name, _), grad in zip(params, derivatives):
                    if grad is not None:
                        if not bool(torch.isfinite(grad).all()):
                            raise ValueError('Nonfinite metric gradient')
                        g = grad.double()
                        gradients[metric][name].add_(g, alpha=view.sample_gradient_weight)
                        squared += g.square().sum()
                local[metric] = dict(loss=float(value.detach()), sample_gradient_norm=float(squared.sqrt()))
                totals[metric] += local[metric]['loss'] / 783
                del derivatives
            rows.append(dict(index=index, identifier=row['identifier'], start=row['start'],
                group_index=row['group_index'], cells_per_metre=row['cells_per_metre'],
                counts_xyz=row['counts_xyz'], metrics=local))
            del row, sample, response, values
            if (process_memory()['peak_since_exec_bytes'] > reg['maximum_peak_rss_bytes'] or
                    torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes'] or
                    time.monotonic() - started > reg['maximum_wall_seconds']):
                raise ValueError('Finite diagnostic resource budget exceeded')
            if (index + 1) % 27 == 0:
                write(args.output / 'progress.json', dict(completed=index + 1, total=783))
                print('Per-metric CUDA gradients', index + 1, '/783', flush=True)
        if quiet != source['inactive_indices']:
            raise ValueError('Quiescent control coverage changed')
        for m in METRICS:
            if not math.isclose(totals[m], source['optimization_metric_means_active'][m], rel_tol=1e-10, abs_tol=1e-30):
                raise ValueError('Complete initial objective changed')
        norms = {m: math.sqrt(sum(float(g.square().sum()) for g in gs.values())) for m, gs in gradients.items()}
        by_param = {m: {n: float(g.norm()) for n, g in gs.items()} for m, gs in gradients.items()}
        weighted_norms = {m: norms[m] * weights[m] for m in METRICS}
        cosines = {}
        for m in METRICS:
            for other in METRICS:
                dot = sum(float((gradients[m][n] * gradients[other][n]).sum()) for n, _ in params)
                cosines[m + ':' + other] = dot / max(norms[m] * norms[other], 1e-300)
        combined = {n: sum(gradients[m][n] for m in METRICS) for n, _ in params}
        weighted = {n: sum(gradients[m][n] * weights[m] for m in METRICS) for n, _ in params}
        if inventory(model) != initial or versions != {n: p._version for n, p in params}:
            raise ValueError('Zero-update gradient diagnostic changed model')
        result = dict(status='passed_full148scene_per_metric_CUDA_gradient_diagnostic',
            training_scenes=148, active_scenes=29, active_local_samples=783, fixed_scenes=119,
            fixed_windows=1071, inactive_indices=quiet, metric_weights=weights,
            optimization_metric_means_active=totals, full148scene_weighted_objective=view.active_weight * weighted_sum(totals, weights) + fixed['objective'],
            full148scene_unweighted_objective=source['full148scene_objective'],
            fixed_controls=fixed, per_metric_full_objective_gradient_norm=norms,
            per_metric_weighted_full_objective_gradient_norm=weighted_norms,
            per_metric_parameter_gradient_norm=by_param, gradient_cosines=cosines,
            combined_gradient_norm=math.sqrt(sum(float(g.square().sum()) for g in combined.values())),
            weighted_combined_gradient_norm=math.sqrt(sum(float(g.square().sum()) for g in weighted.values())),
            rows=rows, zero_update_parameters_unchanged=True, optimizer_updates=0, new_parameter_fits=0,
            cpu_model_forward=False, inference_device='cuda', display_label='GPU (CUDA)',
            expanded_reserved52_references_opened=0, full_generator_target_pass=False,
            parameter_inventory=initial, memory_counters=process_memory(),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-started,
            source_sha256=reg['source_sha256'], protocol_sha256=streamed_identity(args.protocol)['sha256'])
        write(args.output / 'results.json', result)
        print(json.dumps(dict(status=result['status'], norms=norms, weighted_norms=weighted_norms)), flush=True)
    except BaseException as error:
        write(args.output / 'results.json', dict(status='failed', error=repr(error), completed_samples=len(rows),
            optimizer_updates=0, new_parameter_fits=0, cpu_model_forward=False,
            expanded_reserved52_references_opened=0, full_generator_target_pass=False))
        raise
    finally:
        view.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
