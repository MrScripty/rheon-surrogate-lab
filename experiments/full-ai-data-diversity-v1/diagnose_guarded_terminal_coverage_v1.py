"""Count guarded-terminal decisions during two exact actual GPU model outputs."""
import argparse
import ctypes as C
import hashlib
import math
import os
from pathlib import Path
import numpy as np
import torch
from support import ROOT, read, write, digest
from complete_native_entry_guarded_terminal_v1 import GuardedTerminalEntry
from cuda_native_transfer_v1 import FIELDS
from cuda_kernel_runtime_v1 import check
from diagnose_causal_composition import inventory


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only coverage diagnosis required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen diagnostic dependency changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bank = {r['identifier']: r for r in read(ROOT / reg['bank'])['trajectories']}
    qualified = read(ROOT / reg['baseline_outputs'])
    if qualified['status'] != 'completed_expanded_cuda_full_state_block_evaluation' or not qualified['every_output_bitwise_equal_corrected_baseline']:
        raise ValueError('Fully qualified baseline bytes required')
    golden = {(r['identifier'], r['start'], r['K'], r['lead']): r['output_sha256'] for r in qualified['rows']}
    model = GuardedTerminalEntry(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model)
    rows = []
    try:
        kernels = model._generator.model.kernels
        driver = kernels.driver
        driver.cuModuleGetGlobal_v2.argtypes = [C.POINTER(C.c_uint64), C.POINTER(C.c_size_t), C.c_void_p, C.c_char_p]
        driver.cuMemsetD8_v2.argtypes = [C.c_uint64, C.c_ubyte, C.c_size_t]
        driver.cuMemcpyDtoH_v2.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t]
        address, size = C.c_uint64(), C.c_size_t()
        check(driver.cuModuleGetGlobal_v2(C.byref(address), C.byref(size), kernels.module, b'terminal_coverage'), 'coverage symbol')
        if size.value != 8 * 8:
            raise ValueError('Exact eight-counter diagnostic layout required')
        with torch.inference_mode():
            for case in reg['cases']:
                selected = bank[case['identifier']]
                if digest(ROOT / selected['receipt']) != selected['receipt_sha256']:
                    raise ValueError('Actual scene changed')
                scene = read(ROOT / selected['receipt'])['scene']
                spec = next(s for s in selected['states'] if s['step'] == case['start'])
                if digest(ROOT / spec['path']) != spec['sha256']:
                    raise ValueError('Actual model input changed')
                with np.load(ROOT / spec['path'], allow_pickle=False) as archive:
                    native = {name: torch.from_numpy(archive[name].copy()) for name in FIELDS}
                n, k, start = selected['reference_substeps'], case['K'], case['start']
                model.prepare_native(native, scene, start, k, n)
                torch.cuda.synchronize()
                check(driver.cuMemsetD8_v2(address, 0, size), 'clear diagnostic counters')
                outputs = model.predict_native(native, scene, start, k, n)
                actual = [{name: hashlib.sha256(v.numpy().tobytes()).hexdigest() for name, v in row.items()} for row in outputs]
                if actual != [golden[case['identifier'], start, k, lead] for lead in range(1, k + 1)]:
                    raise ValueError('Counter instrumentation changed actual complete model output')
                values = (C.c_uint64 * 8)()
                check(driver.cuMemcpyDtoH_v2(values, address, size), 'read diagnostic counters')
                counters = dict(zip(reg['counter_names'], values))
                nx, ny, nz = scene['counts']
                carrier_values = (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1) + math.prod(scene['counts'])
                if counters['total'] != carrier_values * n * k or sum(list(values)[1:]) != counters['total']:
                    raise ValueError('Exactly every actual terminal sample must take one counted decision')
                rows.append(dict(identifier=case['identifier'], counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                                 internal_substeps=n, K=k, start=start, counters=counters,
                                 fast_path_fraction=counters['accepted'] / counters['total'],
                                 every_actual_complete_output_bitwise_equal=True, output_sha256=actual))
                model.release_graphs()
        if inventory(model) != before:
            raise ValueError('Coverage diagnostic changed learned values')
        for name, expected in reg['source_sha256'].items():
            if digest(ROOT / name) != expected:
                raise ValueError('Coverage diagnostic source changed during execution: ' + name)
    finally:
        model.close()
    args.output.mkdir(parents=True)
    write(args.output / 'results.json', dict(status='completed_actual_model_guarded_terminal_coverage', rows=rows,
        inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
        internal_sampler_precision='F64 middle/coordinates; pairedFP32 terminal enclosure and original F64 fallback',
        instrumentation_only=True, latency_claim=False, new_arithmetic_candidate=False,
        cpu_model_forward=False, optimizer_updates=0, expanded_reserved52_references_opened=0,
        all_actual_outputs_exact=True, source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol)))
    for row in rows:
        print(row['identifier'], row['counters'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
