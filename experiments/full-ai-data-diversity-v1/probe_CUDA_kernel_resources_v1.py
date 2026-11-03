"""Read driver-compiled kernel resource attributes; no model forward or timings."""
import argparse
import ctypes as C
from pathlib import Path
import torch
from support import ROOT, read, write, digest
from cuda_kernel_runtime_v1 import CudaKernels, check


def run(args):
    if args.output.exists():
        raise ValueError('Fresh diagnostic evidence required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen diagnostic source changed: ' + name)
    torch.empty(0, device='cuda')  # Establish the qualified primary CUDA context.
    rows = []
    for build in reg['builds']:
        kernels = CudaKernels(build)
        try:
            driver = kernels.driver
            driver.cuFuncGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
            for name in ('learned_transport', 'learned_linear'):
                function = kernels.functions[name]
                attrs = {}
                for key, enum in {'max_threads_per_block': 0, 'static_shared_bytes_per_block': 1,
                                  'local_bytes_per_thread': 3, 'registers_per_thread': 4}.items():
                    value = C.c_int()
                    check(driver.cuFuncGetAttribute(C.byref(value), enum, function), 'resource attribute ' + key)
                    attrs[key] = value.value
                active = C.c_int()
                check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(C.byref(active), function, 128, 0), 'static occupancy query')
                rows.append(dict(build=build, kernel=name, block_threads=128, driver_resources=attrs,
                    theoretical_max_active_blocks_per_SM=active.value,
                    theoretical_max_active_threads_per_SM=active.value * 128))
        finally:
            kernels.close()
    args.output.mkdir(parents=True)
    write(args.output/'results.json', dict(status='completed_driver_compiled_kernel_resource_probe',
        rows=rows, gpu_model=torch.cuda.get_device_name(), inference_device='cuda', display_label='GPU (CUDA)',
        scope='Driver-compiled static attributes and theoretical occupancy only. No dynamic counter, model forward or timing measured.',
        optimizer_updates=0, cpu_model_forward=False, learned_model_forward=False, expanded_reserved52_references_opened=0,
        protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256']))
    for row in rows: print(row, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
