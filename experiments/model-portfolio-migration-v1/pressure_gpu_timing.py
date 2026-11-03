"""Complete native CPU-array -> captured CUDA pressure -> owned CPU-array timing.

One checkpoint per process preserves captured module ownership. Device hooks
qualify each identical case before measurement; measured calls are uninstrumented.
Every returned value must retain the independent accuracy capture's exact hash.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('base', 'native_input_residual', 'cell_size_m', 'rho', 'dt')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def resources():
    def query(args):
        return subprocess.check_output(['nvidia-smi', *args, '--format=csv,noheader,nounits'],
                                       text=True, timeout=20).strip()
    compute = query(['--query-compute-apps=pid,used_memory'])
    pids = [int(row.split(',')[0].strip()) for row in compute.splitlines() if row.strip()]
    return dict(created_unix=time.time(), compute_processes=compute,
                other_compute_pids=[pid for pid in pids if pid != os.getpid()],
                gpu=query(['--query-gpu=index,name,uuid,driver_version,temperature.gpu,power.draw,'
                           'clocks.current.graphics,clocks.current.memory,utilization.gpu,memory.used']),
                load_average=list(os.getloadavg()))


def run(protocol_path, identifier, output):
    if output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA timing output required; never replace a prior campaign')
    protocol = read(protocol_path)
    if identifier not in protocol['models']:
        raise ValueError('Model is outside the pilot protocol')
    for name, expected in protocol['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Preregistered timing source changed: ' + name)
    os.sched_setaffinity(0, {protocol['host_cpu']})
    setup_started = time.perf_counter()
    sys.path.insert(0, str(ROOT / 'experiments/full-ai-next-priorities-v1'))
    import study_support
    import pressure_cases
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    counts = dict(cuda_module_calls=0, forbidden_module_calls=0)

    def check_device(module, args):
        tensors = [value for value in args if isinstance(value, torch.Tensor)]
        if not tensors or any(value.device.type != 'cuda' for value in tensors):
            counts['forbidden_module_calls'] += 1
            raise ValueError('CPU or unclassified neural module execution forbidden')
        counts['cuda_module_calls'] += 1

    hook = torch.nn.modules.module.register_module_forward_pre_hook(check_device)
    result = dict(status='starting', pid=os.getpid(), identifier=identifier,
                  protocol_sha256=sha(protocol_path), created_unix=time.time(), rows=[],
                  inference_device='cuda', timing_units='ms/pressure_request',
                  timing_scope=protocol['timing_scope'], new_optimizer_updates=0,
                  new_reference_generation=False, independent_confirmation=False,
                  samples_per_case=protocol['samples_per_case'], warmups_per_case=protocol['warmups_per_case'],
                  timed_module_hooks_enabled=False, device_qualification=counts,
                  CPU_model_forward=False, full_simulation_step_latency=False,
                  full_portfolio_complete=False, resource_observations=[])
    output.mkdir(parents=True)
    try:
        state = study_support.load_checkpoint(identifier)
        np = state.np
        initial_inventory = state.data.helpers().state_inventory(state.network)
        architecture = copy.deepcopy(state.model.ARCHITECTURE)
        expected = protocol['accuracy'][identifier]
        reference_path = ROOT / expected['path']
        if sha(reference_path) != expected['sha256']:
            raise ValueError('Original accuracy capture changed')
        reference = read(reference_path)
        if sha(ROOT / state.entry['restore'] / 'weights.npz') != reference['weights_sha256']:
            raise ValueError('Timing selects another checkpoint')
        state.model.configure('cuda')
        state.network.to('cuda').eval()
        if any(value.device.type != 'cuda' for value in state.network.parameters()):
            raise ValueError('All neural parameters must reside on CUDA')
        cases = pressure_cases.load(torch, np)
        frozen_path = ROOT / protocol['frozen_cases']['path']
        if sha(frozen_path) != protocol['frozen_cases']['sha256']:
            raise ValueError('Frozen native pressure contract changed')
        frozen = read(frozen_path)
        if len(cases) != 128 or len(frozen['cases']) != 128 or len(reference['rows']) != 128:
            raise ValueError('Complete128case pressure population required')
        result.update(gpu_model=torch.cuda.get_device_name(), torch_version=torch.__version__,
                      cuda_version=torch.version.cuda, source_entry=state.entry,
                      weights_sha256=reference['weights_sha256'],
                      host_cpu=protocol['host_cpu'], host_affinity=sorted(os.sched_getaffinity(0)),
                      torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                      tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32,
                                cudnn=torch.backends.cudnn.allow_tf32),
                      cudnn_benchmark=torch.backends.cudnn.benchmark,
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      network_dtype='float32', pressure_dtype='float64', autocast=False)
        if any(result['tf32'].values()) or result['cudnn_benchmark'] or not result['deterministic_algorithms']:
            raise ValueError('Captured deterministic precision settings required')

        def complete(native):
            if set(native) != set(FIELDS):
                raise ValueError('Exactly the five native input fields required; no reference labels')
            packed = {}
            for name in FIELDS[:2]:
                array = np.asarray(native[name])
                if (array.dtype != np.dtype('<f8') or array.ndim != 3
                        or array.shape != native['base'].shape or not np.isfinite(array).all()):
                    raise ValueError('Finite float64 native pressure arrays required')
                packed[name] = torch.from_numpy(np.ascontiguousarray(array))[None, None]
            for name in FIELDS[2:]:
                value = float(native[name])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('Positive finite native scalar required')
                packed[name] = torch.tensor([value], dtype=torch.float64)
            device = {name: value.to('cuda') for name, value in packed.items()}
            prediction = state.model.predict_pressure(state.network, device, state.normalization)
            if (prediction.device.type != 'cuda' or prediction.dtype != torch.float64
                    or tuple(prediction.shape) != (1, 1, *native['base'].shape)):
                raise ValueError('Captured CUDA pressure output differs')
            # The NumPy view retains its new exclusive CPU Tensor owner; it never
            # aliases a caller input, including a zero-residual return path.
            owned = prediction.detach().to('cpu', copy=True).numpy()[0, 0]
            torch.cuda.synchronize()
            return owned

        def tensor_sha(value):
            return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

        def check_output(value, expected_sha, native):
            if (value.dtype != np.dtype('<f8') or value.shape != native['base'].shape
                    or not np.isfinite(value).all()
                    or hashlib.sha256(value.tobytes()).hexdigest() != expected_sha
                    or any(np.shares_memory(value, native[name]) for name in FIELDS[:2])):
                raise ValueError('Complete pressure output differs from independent retained accuracy')

        result['setup_wall_seconds'] = time.perf_counter() - setup_started
        began = time.monotonic()
        with torch.inference_mode():
            for index, ((identity, sample), baseline, case) in enumerate(zip(cases, reference['rows'], frozen['cases'], strict=True)):
                if time.monotonic() - began > protocol['worker_deadline_seconds']:
                    raise TimeoutError('Finite pilot worker deadline reached')
                inputs = {name: tensor_sha(sample[name]) for name in FIELDS}
                if (inputs != baseline['input_sha256'] or inputs != case['original_case']['input_sha256']
                        or any(identity[key] != baseline[key] for key in identity)):
                    raise ValueError('Timing inputs differ from original scored request')
                native = {name: sample[name].numpy()[0, 0].copy() for name in FIELDS[:2]}
                native.update({name: float(sample[name].item()) for name in FIELDS[2:]})
                before_inputs = {name: hashlib.sha256(native[name].tobytes()).hexdigest() for name in FIELDS[:2]}
                if index % 32 == 0:
                    resource = resources()
                    result['resource_observations'].append(resource)
                    if resource['other_compute_pids']:
                        raise RuntimeError('Other GPU compute observed; retain as contended, not qualified')
                call_started = time.perf_counter_ns()
                value = complete(native)
                first_ms = (time.perf_counter_ns() - call_started) / 1e6
                check_output(value, baseline['output_sha256'], native)
                del value
                for _ in range(protocol['warmups_per_case']):
                    value = complete(native)
                    check_output(value, baseline['output_sha256'], native)
                    del value
                hook.remove()
                hook = None
                samples = []
                # Same deterministic code/inputs and CUDA weights as the
                # immediately preceding device-qualified calls. Verification
                # hooks and scalar hashing remain outside the timed interval.
                for _ in range(protocol['samples_per_case']):
                    call_started = time.perf_counter_ns()
                    value = complete(native)
                    samples.append(time.perf_counter_ns() - call_started)
                    check_output(value, baseline['output_sha256'], native)
                    del value
                hook = torch.nn.modules.module.register_module_forward_pre_hook(check_device)
                if any(hashlib.sha256(native[name].tobytes()).hexdigest() != before_inputs[name] for name in FIELDS[:2]):
                    raise ValueError('Native caller arrays changed')
                result['rows'].append(dict(case_id=case['id'], **identity,
                    cells_per_metre=case['cells_per_metre'], input_dt_s=case['input_dt_s'],
                    raw_wall_nanoseconds=samples, median_ms=statistics.median(samples) / 1e6,
                    minimum_ms=min(samples) / 1e6, maximum_ms=max(samples) / 1e6,
                    instrumented_first_call_ms=first_ms, input_sha256=inputs,
                    output_sha256=baseline['output_sha256'], output_independent_of_inputs=True))
                if index % 32 == 31:
                    save(output / 'results.json', result)
                    print(identifier, index + 1, 'complete CUDA pressure requests timed', flush=True)
        result['resource_observations'].append(resources())
        if any(item['other_compute_pids'] for item in result['resource_observations']):
            raise RuntimeError('Other GPU compute observed; latency is contended')
        if state.data.helpers().state_inventory(state.network) != initial_inventory or state.model.ARCHITECTURE != architecture:
            raise ValueError('Captured weights or architecture changed')
        result.update(status='complete_pressure_GPU_timing_pilot', cases=len(result['rows']),
                      task_calls_timed=len(result['rows']) * protocol['samples_per_case'],
                      case_median_median_ms=statistics.median(row['median_ms'] for row in result['rows']),
                      worst_case_median_ms=max(row['median_ms'] for row in result['rows']),
                      finished_unix=time.time(), learned_parameters_unchanged=True,
                      architecture_unchanged=True, all_output_hashes_exact=True,
                      complete_pressure_latency_observed=True)
    except BaseException as error:
        result.update(status='failed_timing_pilot', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if hook is not None:
            hook.remove()
        save(output / 'results.json', result)
    print(json.dumps(dict(identifier=identifier, status=result['status'], cases=result['cases'],
                          median_ms=result['case_median_median_ms'], worst_case_median_ms=result['worst_case_median_ms'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--identifier', choices=('C106', 'C107'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.identifier, args.output.resolve())
