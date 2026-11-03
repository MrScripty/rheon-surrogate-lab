"""One paired profile at the real complete native entry seam; no admission."""
import argparse
import ctypes as C
from cuda_kernel_runtime_v1 import check
import collections
import hashlib
import os
from pathlib import Path
import numpy as np
import torch
from support import ROOT, read, write, digest
from cell_coordinate_pressure_graph_v1 import CellCoordinatePressureNativeEntry as ParentEntry
from alpha_argument_pressure_graph_v1 import AlphaArgumentPressureNativeEntry
from cuda_native_transfer_v1 import FIELDS, upload_state, download_outputs
from diagnose_causal_composition import inventory


def hashes(rows):
    return [{n: hashlib.sha256(v.numpy().tobytes()).hexdigest() for n, v in row.items()} for row in rows]


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only paired native-entry profile required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT / p) != h:
            raise ValueError('Frozen paired entry profile dependency changed: ' + p)
    for proof in reg['execution_equivalence']:
        q = read(ROOT / proof['path'])
        if q['status'] != proof['status'] or len(q['rows']) != proof['complete_outputs'] or not q[proof['required_true']]:
            raise ValueError('All full entry accuracy/equivalence gates required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline = read(ROOT / reg['baseline_outputs'])
    candidate = read(ROOT/reg['candidate_outputs'])
    golden = {mode: {(r['identifier'],r['start'],r['K'],r['lead']):r['output_sha256'] for r in source['rows']}
              for mode,source in [('cell_coordinate_entry',baseline),('alpha_argument_entry',candidate)]}
    population = {r['identifier']: r for r in read(ROOT / reg['bank'])['trajectories']}
    models = dict(cell_coordinate_entry=ParentEntry(reg['transport_pair'], reg['shared_pair'], reg['baseline_build'], reg['control_build']),
                  alpha_argument_entry=AlphaArgumentPressureNativeEntry(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build']))
    before = {mode: inventory(m) for mode, m in models.items()}
    if len({str(v) for v in before.values()}) != 1:
        raise ValueError('Identical learned inventories required')
    resources = []
    for mode, model in models.items():
        kernels=model._generator.nested_kernels;driver=kernels.driver
        driver.cuFuncGetAttribute.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p]
        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes=[C.POINTER(C.c_int),C.c_void_p,C.c_int,C.c_size_t]
        for name in ('learned_transport','geometry_transport_32_32_32'):
            function=C.c_void_p();check(driver.cuModuleGetFunction(C.byref(function),kernels.module,name.encode()),'resource kernel lookup')
            attrs={}
            for key,enum in {'max_threads_per_block':0,'static_shared_bytes_per_block':1,'local_bytes_per_thread':3,'registers_per_thread':4}.items():
                value=C.c_int();check(driver.cuFuncGetAttribute(C.byref(value),enum,function),'resource attribute');attrs[key]=value.value
            active=C.c_int();check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(C.byref(active),function,128,0),'theoretical occupancy')
            resources.append(dict(mode=mode,kernel=name,driver_resources=attrs,theoretical_active_blocks_per_SM=active.value,block_threads=128,achieved_occupancy_or_spill_claim=False))
    rows = []
    args.output.mkdir(parents=True)
    try:
        with torch.inference_mode():
            for index, case in enumerate(reg['cases']):
                selected = population[case['identifier']]
                scene = read(ROOT / selected['receipt'])['scene']
                spec = next(s for s in selected['states'] if s['step'] == case['start'])
                if digest(ROOT / spec['path']) != spec['sha256']:
                    raise ValueError('Actual profile native input changed')
                with np.load(ROOT / spec['path'], allow_pickle=False) as archive:
                    current = {n: archive[n].copy() for n in FIELDS}
                native = {n: torch.from_numpy(v) for n, v in current.items()}
                start, k, n = case['start'], case['K'], selected['reference_substeps']
                modes = list(models) if index % 2 == 0 else list(reversed(models))
                for mode in modes:
                    expected = [golden[mode][case['identifier'],start,k,lead] for lead in range(1,k+1)]
                    model = models[mode]
                    model.prepare_native(native, scene, start, k, n)
                    def call():
                        # Same actual NumPy -> native views -> full transfer,
                        # generation, native reconstruction and sync scope.
                        fields = {name: torch.from_numpy(v) for name, v in current.items()}
                        return model.predict_native(fields, scene, start, k, n)
                    for _ in range(reg['warmup_calls']):
                        copied = call()
                        if hashes(copied) != expected:
                            raise ValueError('Actual full-entry warmup output changed')
                    # Retain only the same previous CPU result as the real
                    # call site. Do not accumulate hidden GPU output owners.
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(reg['profiled_calls']):
                            with torch.profiler.record_function('complete_native_entry_call'):
                                copied = call()
                            if hashes(copied) != expected:
                                raise ValueError('Actual profiled native output changed')
                        torch.cuda.synchronize()
                    families = collections.defaultdict(lambda: dict(calls=0, instrumented_ms=0.))
                    allocations = []
                    cuda_names = collections.defaultdict(lambda: dict(calls=0,instrumented_ms=0.))
                    for event in prof.events():
                        if event.device_type == torch.autograd.DeviceType.CUDA:
                            family = next((f for f in ('learned_transport', 'learned_gather', 'fixed_axis_transform',
                                                       'known_axes_dynamic', 'known_control_dynamic') if f in event.name), 'other_CUDA_activity')
                            if 'geometry_transport_' in event.name:
                                family = 'learned_transport'
                            if 'fft' in event.name.lower(): family = 'FFT_kernels'
                            cuda_names[event.name]['calls'] += 1
                            cuda_names[event.name]['instrumented_ms'] += float(event.device_time_total)/1000
                            families[family]['calls'] += 1
                            families[family]['instrumented_ms'] += float(event.device_time_total) / 1000
                        if event.name in ('cudaMalloc', 'cudaHostAlloc'):
                            parents = []
                            parent = event.cpu_parent
                            while parent is not None and len(parents) < 20:
                                parents.append(parent.name)
                                parent = parent.cpu_parent
                            allocations.append(dict(name=event.name, self_cpu_ms=float(event.self_cpu_time_total) / 1000, ancestors=parents))
                    cuda_events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    rows.append(dict(mode=mode, identifier=case['identifier'], counts_xyz=scene['counts'],
                        cells_per_metre=scene['cells_per_metre'], internal_substeps=n, K=k,
                        profiled_calls=reg['profiled_calls'], warmup_calls=reg['warmup_calls'], families=dict(families),
                        cuda_activities_per_generated_output=len(cuda_events) / (reg['profiled_calls'] * k),
                        complete_entry_instrumented_ms=sum(float(e.device_time_total) / 1000 for e in cuda_events
                                                           if e.name == 'complete_native_entry_call'),
                        cpu_allocation_events=allocations, actual_output_sha256=expected,
                        top_CUDA_activities=[dict(name=name,**values) for name,values in sorted(cuda_names.items(),key=lambda pair:pair[1]['instrumented_ms'],reverse=True)[:20]],
                        every_actual_profile_and_fullentry_warmup_output_exact=True,
                        top_CPU_activities=[dict(name=e.key, calls=int(e.count), self_cpu_ms=float(e.self_cpu_time_total) / 1000)
                                            for e in sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)[:15]]))
                    model.release_graphs()
                    print('Paired complete native entry profile', mode, case['identifier'], flush=True)
            if any(inventory(m) != before[mode] for mode, m in models.items()):
                raise ValueError('Profile changed learned weights')
        for p, h in reg['source_sha256'].items():
            if digest(ROOT / p) != h:
                raise ValueError('Profile source changed during execution: ' + p)
        write(args.output / 'results.json', dict(status='completed_paired_complete_native_entry_GPU_profile', rows=rows,
            inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
            internal_sampler_precision_by_mode=dict(cell_coordinate_entry='float64', alpha_argument_entry='SameF64sampler/pressure/parameters;existingvalidatedalphaargument;originalN1unchanged'),
            diagnostic_only=True, latency_acceptance_claim=False, components_not_assumed_additive=True, static_kernel_resources=resources,
            actual_fullentry_scope_and_ownership_matched=True, cpu_model_forward=False, optimizer_updates=0,
            expanded_reserved52_references_opened=0, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256']))
    finally:
        for model in models.values():
            model.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
