"""Evaluate captured C106/C107 pressure proposals on their 128 requests on CUDA.

Reuse the original checkpoint/source loader and model's supported CUDA device
configuration. No architecture/weights/policy changes, CPU model forwards,
training, reference labels passed to inference, or latency claim. One model per
process preserves the captured modules' global configuration ownership.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ('base', 'native_input_residual', 'cell_size_m', 'rho', 'dt')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(identifier, output, host_cpu):
    if output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only accuracy execution required')
    os.sched_setaffinity(0, {host_cpu})
    sys.path.insert(0, str(ROOT / 'experiments/full-ai-next-priorities-v1'))
    from study_support import load_checkpoint, historical_cases
    state = load_checkpoint(identifier)
    torch = state.torch
    before = state.data.helpers().state_inventory(state.network)
    architecture = copy.deepcopy(state.model.ARCHITECTURE)
    state.model.configure('cuda')
    state.network.to('cuda').eval()
    if any(p.device.type != 'cuda' for p in state.network.parameters()):
        raise ValueError('Every neural parameter must reside on CUDA')
    cases = historical_cases(state)
    if len(cases) != 128:
        raise ValueError('Original complete 128-request population required')
    output.mkdir(parents=True)
    hooks, neural_calls = [], dict(cuda=0, cpu=0)
    def check_device(module, args):
        tensors = [v for v in args if isinstance(v, torch.Tensor)]
        if not tensors or any(v.device.type != 'cuda' for v in tensors):
            neural_calls['cpu'] += 1
            raise ValueError('CPU or unclassified neural forward is forbidden')
        neural_calls['cuda'] += 1
    for module in state.network.modules():
        if module is state.network or isinstance(module, (torch.nn.Conv3d, torch.nn.Linear)):
            hooks.append(module.register_forward_pre_hook(check_device))
    restore = ROOT / state.entry['restore']
    result = dict(schema='rheon.captured-pressure-GPU-accuracy.v1', status='running',
                  identifier=identifier, rows=[], active=None, inference_device='cuda',
                  display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  pressure_dtype='float64', network_dtype='float32', tf32=False, autocast=False,
                  source_entry=state.entry, weights_sha256=sha(restore / 'weights.npz'),
                  restore_receipt_sha256=sha(restore / 'receipt.json'),
                  source_sha256={str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
                  input_names=list(INPUTS), inference_label_fields=[], host_cpu=host_cpu,
                  threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                  CPU_model_forward=False, neural_module_forward_calls=neural_calls,
                  optimizer_updates=0, latency_claim=False, independent_confirmation=False,
                  scope='Captured original pressure proposal on original prepared consumed requests. Labels remain on CPU for scoring. CUDA uses the captured training/inference implementation; CPU scoring uses its numerical equation operator. No recurrence, fallback or policy replacement.',
                  pressure_metric='Mean-free pressure error L2 / max(mean-free reference L2, 1e-30); absolute RMS and spatial maximum retained.',
                  equation_metric='Captured full_operator pressure residual L2 / max(RHS L2, 1e-30).')
    arrays = {}
    started = time.monotonic()
    def tensor_sha(value):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    try:
        with torch.inference_mode():
            for index, (identity, sample) in enumerate(cases):
                if time.monotonic() - started > 1200:
                    raise TimeoutError('Finite pressure accuracy execution deadline')
                result['active'] = dict(index=index, **identity)
                host_inputs = {name: sample[name] for name in INPUTS}
                original_inputs = {name: tensor_sha(v) for name, v in host_inputs.items()}
                # Reference pressure and RHS are never forwarded to prediction.
                device_inputs = {name: value.to('cuda') for name, value in host_inputs.items()}
                prediction = state.model.predict_pressure(state.network, device_inputs, state.normalization)
                if prediction.device.type != 'cuda' or prediction.dtype != torch.float64:
                    raise ValueError('Captured complete CUDA float64 pressure required')
                pressure = prediction.detach().cpu()
                torch.cuda.synchronize()
                if not bool(torch.isfinite(pressure).all()):
                    raise ValueError('Nonfinite pressure prediction')
                if any(tensor_sha(value) != original_inputs[name] for name, value in host_inputs.items()):
                    raise ValueError('Caller pressure inputs changed')
                residual = state.model.full_operator(pressure, sample['cell_size_m'], sample['rho']) - sample['rhs']
                difference = pressure - sample['reference']
                difference = difference - difference.mean()
                reference = sample['reference'] - sample['reference'].mean()
                rhs_norm = float(torch.linalg.vector_norm(sample['rhs']))
                pressure_norm = float(torch.linalg.vector_norm(reference))
                values = dict(equation_relative_l2=float(torch.linalg.vector_norm(residual)) / max(rhs_norm, 1e-30),
                              pressure_mean_free_relative_l2=float(torch.linalg.vector_norm(difference)) / max(pressure_norm, 1e-30),
                              pressure_absolute_rms=float(difference.square().mean().sqrt()),
                              pressure_absolute_max=float(difference.abs().max()),
                              equation_rhs_l2=rhs_norm, pressure_reference_mean_free_l2=pressure_norm,
                              equation_floor_active=rhs_norm < 1e-30,
                              pressure_floor_active=pressure_norm < 1e-30)
                if any(not math.isfinite(v) for v in values.values()):
                    raise ValueError('Nonfinite pressure metrics')
                result['rows'].append(dict(**identity, metrics=values, output_sha256=tensor_sha(pressure),
                                           input_sha256=original_inputs,
                                           reference_pressure_sha256=tensor_sha(sample['reference']),
                                           reference_rhs_sha256=tensor_sha(sample['rhs'])))
                arrays['case%03d_pressure' % index] = pressure.numpy().copy()
                if index % 32 == 31:
                    write(output / 'results.json', result)
                    print(identifier, index + 1, 'GPU pressure requests scored', flush=True)
        if state.data.helpers().state_inventory(state.network) != before or state.model.ARCHITECTURE != architecture:
            raise ValueError('Weights or captured architecture changed')
        state.np.savez_compressed(output / 'pressure-outputs.npz', **arrays)
        result.update(status='completed_captured_pressure_GPU_accuracy', active=None, cases=len(result['rows']),
                      learned_parameters_unchanged=True, architecture_unchanged=True,
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      diagnostic_elapsed_seconds=time.monotonic() - started,
                      output_archive=dict(path='pressure-outputs.npz', sha256=sha(output / 'pressure-outputs.npz')),
                      worst={key:max(r['metrics'][key] for r in result['rows']) for key in
                             ('equation_relative_l2', 'pressure_mean_free_relative_l2',
                              'pressure_absolute_rms', 'pressure_absolute_max')})
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for hook in hooks:
            hook.remove()
        write(output / 'results.json', result)
    print(identifier, result['status'], result['worst'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identifier', choices=('C106', 'C107'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host-cpu', type=int, default=4)
    args = parser.parse_args()
    run(args.identifier, args.output, args.host_cpu)
