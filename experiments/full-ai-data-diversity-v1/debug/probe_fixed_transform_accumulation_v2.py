"""DEBUG ONLY: bounded generic GPU contractions against actual native DCT stages.

Numerical response values here are isolated native-reference controls. They
never enter a candidate; this script changes no model or learned coefficient.
"""
import argparse
import ctypes as C
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT, read, write, digest
from cuda_kernel_runtime_v1 import CudaKernels, check
from probe_high_volume_dct_boundaries_v1 import forward_trace, inverse_trace
from probe_high_volume_first_step_v1 import difference
import numerics


def bit_difference(a, b):
    return int((a.detach().cpu().contiguous().reshape(-1).view(torch.int64)
                != b.detach().cpu().contiguous().reshape(-1).view(torch.int64)).sum())


def run(args):
    if args.output.exists():
        raise ValueError('Fresh generic accumulation evidence required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('CUDA diagnostics required')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen diagnostic dependency changed: ' + path)
    os.sched_setaffinity(0, {5})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    case = reg['case']
    scene = read(ROOT / case['receipt'])['scene']
    shape = tuple(reversed(scene['counts']))
    spec = next(s for s in case['states'] if s['step'] == reg['endpoint_step'])
    with np.load(ROOT / spec['path'], allow_pickle=False) as archive:
        residual = torch.from_numpy(archive['native_input_residual'].copy()).reshape(1, 1, *shape)
        base = torch.from_numpy(archive['pressure_base'].copy()).reshape_as(residual)
        expected = torch.from_numpy(archive['pressure'].copy()).reshape_as(residual)
    h = torch.tensor([1 / scene['cells_per_metre']], dtype=torch.float64)
    rho = torch.tensor([1000.], dtype=torch.float64)
    native = base + numerics.correction(residual, h, rho, None)
    native = native - native[:, :, :1, :1, :1].clone()
    if bit_difference(native, expected):
        raise ValueError('Native fixed-transform control does not reproduce actual pressure bytes')
    axes, response = numerics.geometry(shape, None)
    q = residual * (rho / h).reshape(-1, 1, 1, 1, 1)
    forward = forward_trace(q, axes)
    modes = forward[-1] / response
    modes[:, 0, 0, 0] = 0.
    inverse = inverse_trace(modes, axes)
    inputs = [q[:, 0], *forward[:2], modes, *inverse[:2]]
    truths = [*forward, *inverse]
    stages = [(2, False, axes[0]), (1, False, axes[1]), (0, False, axes[2]),
              (0, True, axes[2]), (1, True, axes[1]), (2, True, axes[0])]
    # Establish the Torch primary CUDA context before direct driver loading.
    context_token = torch.empty(0, dtype=torch.uint8, device='cuda')
    kernels = CudaKernels(reg['debug_build'])
    function = C.c_void_p()
    check(kernels.driver.cuModuleGetFunction(C.byref(function), kernels.module,
                                            b'fixed_axis_transform'), 'get fixed transform diagnostic')
    kernels.functions['fixed_axis_transform'] = function
    gpu_axes = [basis.to('cuda').contiguous() for basis in axes]
    ptr = lambda value: C.c_void_p(value.data_ptr())

    def contract(value, axis, inverse_axis, basis, group, fused):
        value = value.reshape(shape).contiguous()
        output = torch.empty_like(value)
        nx, ny, nz = reversed(shape)
        parameters = [ptr(value), ptr(basis), ptr(output), C.c_int(nx), C.c_int(ny), C.c_int(nz),
                      C.c_int(axis), C.c_int(int(inverse_axis)), C.c_int(group), C.c_int(int(fused))]
        kernels.launch('fixed_axis_transform', value.numel(), parameters)
        return output

    args.output.mkdir(parents=True)
    rows = []
    try:
        with torch.inference_mode():
            for scheme in reg['schemes']:
                local = []
                for index, (source, truth, (axis, inverse_axis, _)) in enumerate(zip(inputs, truths, stages)):
                    result = contract(source.to('cuda'), axis, inverse_axis, gpu_axes[2 - axis],
                                      scheme['group'], scheme['fused'])
                    local.append(dict(stage=index, axis=axis, inverse=inverse_axis,
                                      difference=difference(result, truth),
                                      different_bits=bit_difference(result, truth)))
                value = q.to('cuda').reshape(shape)
                for axis in (2, 1, 0):
                    value = contract(value, axis, False, gpu_axes[2 - axis], scheme['group'], scheme['fused'])
                value = value / response.to('cuda')
                value[0, 0, 0] = 0.
                for axis in (0, 1, 2):
                    value = contract(value, axis, True, gpu_axes[2 - axis], scheme['group'], scheme['fused'])
                value = value.reshape_as(base)
                value = value - value[:, :, :1, :1, :1]
                pressure = base.to('cuda') + value
                pressure = pressure - pressure[:, :, :1, :1, :1]
                rows.append(dict(scheme=scheme, local_axes=local,
                                 native_pressure_difference=difference(pressure, expected),
                                 native_pressure_different_bits=bit_difference(pressure, expected)))
                print('Generic GPU accumulation', scheme, 'local axis differing bits',
                      [r['different_bits'] for r in local], 'pressure differing bits',
                      rows[-1]['native_pressure_different_bits'], flush=True)
        write(args.output / 'results.json', dict(
            status='completed_bounded_generic_fixed_transform_accumulation_probe', rows=rows,
            cpu_configuration=torch.__config__.show(),
            native_numerical_oracles_diagnostic_only=True, candidate_changed=False,
            learned_model_forward_executed=False, cpu_model_forward=False,
            optimizer_updates=0, latency_claim=False,
            protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256']))
    finally:
        kernels.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
