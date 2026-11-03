"""Predeclared full-objective finite differences; no optimizer or selected probe model."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from capture_memory_telemetry_v1 import process_memory
from internal_residual_dataset_v1 import TrainingView, METRICS
from internal_residual_network_v2 import InternalGroupResidual
from internal_residual_loss_v1 import metrics
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
        raise ValueError('Fresh CUDA-only finite-difference diagnostic required')
    for p, expected in reg['source_sha256'].items():
        if streamed_identity(ROOT / p)['sha256'] != expected:
            raise ValueError('Frozen diagnostic input changed: ' + p)
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
        raise ValueError('Training archive changed')
    optimization = read(manifest_path.parent / 'optimization-normalization.json')
    model = InternalGroupResidual(normalization=read(manifest_path.parent / 'network-normalization.json'))
    qualified = read(ROOT / reg['initial_qualification'])
    if inventory(model) != qualified['parameter_inventory']:
        raise ValueError('Exact qualified initial weights required')
    initial = {n: v.detach().clone() for n, v in model.state_dict().items()}
    restored = read(ROOT / reg['restored_receipt'])
    if not restored['typed_cpu_weight_restore_verified'] or restored['cpu_model_forward_executed']:
        raise ValueError('Exact managed CPU parameter restore required')
    path = ROOT / reg['restored_weights']
    if streamed_identity(path) != restored['outputs']['weights.npz']:
        raise ValueError('Managed-restored weights changed')
    with np.load(path, allow_pickle=False) as captured:
        if set(captured.files) != set(initial):
            raise ValueError('Exact restored weight inventory required')
        trained = {}
        for n, value in initial.items():
            a = captured[n]
            if a.dtype.str != '<f4' or a.shape != tuple(value.shape):
                raise ValueError('Typed restored state differs')
            trained[n] = torch.from_numpy(a.copy()).to('cuda')
    model.load_state_dict(trained, strict=True)
    if inventory(model) != restored['final_inventory']:
        raise ValueError('Restored CUDA state differs from managed artifact')
    states = {'initial': initial, 'S1': trained}
    view = TrainingView(manifest_path, archive)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', rows=[], inference_device='cuda', display_label='GPU (CUDA)',
        training_scenes=148, active_scenes=29, active_samples=783, fixed_control_scenes=119,
        fixed_control_windows=1071, optimizer_updates=0, new_parameter_fits=0,
        cpu_model_forward=False, expanded_reserved52_references_opened=0,
        probe_weights_selected_or_saved=False, full_generator_target_pass=False)
    params = dict(model.named_parameters())
    def bounded():
        if (time.monotonic() - started > reg['maximum_wall_seconds'] or
                process_memory()['peak_since_exec_bytes'] > reg['maximum_peak_rss_bytes'] or
                torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
            raise ValueError('Finite full-objective diagnostic resource budget exceeded')
    def full(weights, backward, label):
        model.zero_grad(set_to_none=True)
        total = {m: 0. for m in METRICS}
        fixed = fixed_objective(view, optimization, weights)
        with torch.enable_grad() if backward else torch.no_grad():
            for index in range(783):
                row, sample, response = view.sample(index)
                values = metrics(response.emit(sample, model(sample['features'])), sample, optimization)
                if backward:
                    (weighted_sum(values, weights) * view.sample_gradient_weight).backward()
                for m, value in values.items():
                    scalar = float(value.detach())
                    if not math.isfinite(scalar):
                        raise ValueError('Nonfinite physical training metric')
                    total[m] += scalar / 783
                del row, sample, response, values
                bounded()
                if (index + 1) % 261 == 0:
                    write(args.output / 'progress.json', dict(case=label, samples=index + 1, total_samples=783))
                    print('Full-objective finite differences', label, index + 1, '/783', flush=True)
        return dict(objective=view.active_weight * weighted_sum(total, weights) + fixed['objective'],
            active_metric_means=total, fixed_objective=fixed, complete_active_samples=783)
    try:
        for case in reg['cases']:
            source = states[case['state']]
            model.load_state_dict(source, strict=True)
            before = inventory(model)
            weights = validate(case['metric_weights'])
            base = full(weights, True, case['name'] + ':gradient')
            gradients = {}
            for n, p in params.items():
                if p.grad is None or not bool(p.grad.isfinite().all()):
                    raise ValueError('Complete finite full-objective gradients required')
                gradients[n] = p.grad.detach().double().clone()
            norm = math.sqrt(sum(float(g.square().sum()) for g in gradients.values()))
            if norm <= 0 or not math.isfinite(norm):
                raise ValueError('Nonzero full-objective descent direction required')
            direction = {n: (g / norm).float() for n, g in gradients.items()}
            epsilon = reg['epsilon_parameter_L2']
            probes = {}
            actual_states = {}
            for sign in (-1, 1):
                model.load_state_dict(source, strict=True)
                with torch.no_grad():
                    for n, p in params.items():
                        p.copy_(source[n] + sign * epsilon * direction[n])
                actual_states[sign] = {n: p.detach().double().clone() for n, p in params.items()}
                probes[str(sign)] = full(weights, False, case['name'] + ':probe' + str(sign))
            observed = (probes['1']['objective'] - probes['-1']['objective']) / (2 * epsilon)
            predicted = sum(float((gradients[n] * (actual_states[1][n] - actual_states[-1][n]) / (2 * epsilon)).sum()) for n in params)
            error = abs(observed - predicted)
            allowed = reg['derivative_absolute_tolerance'] + reg['derivative_relative_tolerance'] * abs(predicted)
            model.load_state_dict(source, strict=True)
            if inventory(model) != before:
                raise ValueError('Finite-difference probes did not restore original weights exactly')
            row = dict(name=case['name'], state=case['state'], metric_weights=weights,
                baseline=base, gradient_norm=norm, per_parameter_gradient_norm={n: float(g.norm()) for n, g in gradients.items()},
                epsilon_parameter_L2=epsilon, probes=probes, observed_central_derivative=observed,
                predicted_derivative_accounting_F32_parameter_rounding=predicted,
                derivative_absolute_error=error, allowed_derivative_error=allowed,
                derivative_pass=error <= allowed,
                negative_gradient_probe_reduces_loss=probes['-1']['objective'] < base['objective'],
                original_weights_restored_exactly=True, original_inventory=before)
            result['rows'].append(row)
            write(args.output / 'results.json', result)
            if error > allowed:
                raise ValueError('Full-objective central difference disagrees with gradient')
            print(json.dumps(dict(name=case['name'], derivative_pass=True,
                negative_gradient_probe_reduces_loss=row['negative_gradient_probe_reduces_loss'])), flush=True)
        result.update(status='completed_full148scene_finite_difference_descent_diagnostic',
            all_derivatives_pass=True, all_probe_weights_discarded=True,
            memory_counters=process_memory(), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            elapsed_seconds=time.monotonic() - started, source_sha256=reg['source_sha256'],
            protocol_sha256=streamed_identity(args.protocol)['sha256'])
    except BaseException as error:
        result.update(status='failed', error=repr(error), memory_counters=process_memory())
        raise
    finally:
        model.load_state_dict(initial, strict=True)
        if inventory(model) != qualified['parameter_inventory']:
            raise ValueError('Diagnostic failed to restore initial weights')
        view.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
