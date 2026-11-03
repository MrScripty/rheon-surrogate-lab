"""Read exact geometry kernels' driver resources without inference or timing."""
import argparse
import ctypes as C
import os
from pathlib import Path
import torch
from support import ROOT, read, write, digest
from cuda_kernel_runtime_v1 import CudaKernels, check


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only diagnostic required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen resource diagnostic dependency changed: ' + name)
    torch.empty(0, device='cuda')
    rows = []
    for build in reg['builds']:
        kernels = CudaKernels(build)
        try:
            driver = kernels.driver
            driver.cuFuncGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
            for name in reg['kernels']:
                function = C.c_void_p()
                check(driver.cuModuleGetFunction(C.byref(function), kernels.module, name.encode()), 'geometry kernel lookup')
                attrs = {}
                for key, enum in {'max_threads_per_block': 0, 'static_shared_bytes_per_block': 1,
                                  'local_bytes_per_thread': 3, 'registers_per_thread': 4}.items():
                    value = C.c_int()
                    check(driver.cuFuncGetAttribute(C.byref(value), enum, function), 'resource attribute ' + key)
                    attrs[key] = value.value
                active = C.c_int()
                check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(C.byref(active), function, 128, 0), 'static occupancy')
                rows.append(dict(build=build, kernel=name, block_threads=128, driver_resources=attrs,
                                 theoretical_max_active_blocks_per_SM=active.value,
                                 theoretical_max_active_threads_per_SM=active.value * 128))
        finally:
            kernels.close()
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Diagnostic dependency changed during execution: ' + name)
    args.output.mkdir(parents=True)
    write(args.output / 'results.json', dict(status='completed_geometry_kernel_static_resource_probe', rows=rows,
        gpu_model=torch.cuda.get_device_name(), inference_device='cuda', display_label='GPU (CUDA)',
        scope='Driver-compiled resource attributes and theoretical occupancy at the actual 128-thread block size. These do not establish achieved occupancy, register spills, causal timing effects or speed.',
        optimizer_updates=0, cpu_model_forward=False, learned_model_forward=False, latency_claim=False,
        expanded_reserved52_references_opened=0, protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256']))
    for row in rows:
        print(row, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
