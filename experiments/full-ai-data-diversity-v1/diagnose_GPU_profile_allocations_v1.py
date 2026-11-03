"""One bounded allocation-custody diagnostic; no model or timing admission.

Keep all frozen profile failures. Compare warmup transfer/ownership at the real
complete entry seam, retaining the original profile loop's overlapping objects.
"""
import argparse
import hashlib
import os
import time
from pathlib import Path
import numpy as np
import torch
from support import ROOT, read, write, digest
from debug.gather_fused_graph_v1 import GatherFusedGraph
from debug.geometry_specialized_graph_v1 import GeometrySpecializedGraph
from cuda_native_transfer_v1 import FIELDS, upload_state, download_outputs
from diagnose_causal_composition import inventory


def hashes(outputs):
    return [{n: hashlib.sha256(v.numpy().tobytes()).hexdigest() for n, v in r.items()} for r in outputs]


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only diagnostic required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT / p) != h:
            raise ValueError('Frozen diagnostic dependency changed: ' + p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    case = reg['case']
    scene = read(ROOT / case['receipt'])['scene']
    with np.load(ROOT / case['state']['path'], allow_pickle=False) as archive:
        host = {n: torch.from_numpy(archive[n].copy()) for n in FIELDS}
    baseline = read(ROOT / reg['baseline_outputs'])
    golden = [r['output_sha256'] for r in baseline['rows'] if r['identifier'] == case['identifier']
              and r['start'] == case['start'] and r['K'] == 1 and r['lead'] == 1]
    if len(golden) != 1:
        raise ValueError('Exact actual complete-output baseline required')
    args.output.mkdir(parents=True)
    result = dict(status='running', rows=[], active=None, inference_device='cuda', display_label='GPU (CUDA)',
                  diagnostic_only=True, latency_acceptance_claim=False, past_failed_gates_unchanged=True,
                  cpu_model_forward=False, optimizer_updates=0, expanded_reserved52_references_opened=0,
                  source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for cell in reg['cells']:
                if time.monotonic() - started > reg['maximum_wall_seconds']:
                    raise TimeoutError('Finite allocation diagnostic deadline')
                result['active'] = cell
                model_type = GatherFusedGraph if cell['model'] == 'baseline' else GeometrySpecializedGraph
                build = reg['baseline_build'] if cell['model'] == 'baseline' else reg['geometry_build']
                model = model_type(reg['transport_pair'], reg['shared_pair'], build, reg['control_build'])
                try:
                    before = inventory(model)
                    resident = upload_state(host)
                    model.prepare_geometry(tuple(scene['counts']))
                    model.prepare_block(resident, scene, case['start'], 1, 1)
                    torch.cuda.synchronize()
                    # Same owned-process cache reset in all cells. This is a
                    # causal diagnostic, never the latency acceptance harness.
                    torch.cuda.empty_cache()
                    retained = None
                    def warm_call(upload):
                        inputs = upload_state(host) if upload else resident
                        outputs = model.predict(inputs, scene, case['start'], 1, 1)
                        copied = download_outputs(outputs)
                        if hashes(copied) != golden:
                            raise ValueError('Warmup output bytes changed')
                        return inputs, outputs
                    for _ in range(reg['warmup_calls']):
                        if cell['warmup'] == 'resident_single':
                            warm_call(False)
                        elif cell['warmup'] == 'full_entry_single':
                            warm_call(True)
                        elif cell['warmup'] == 'full_entry_double':
                            retained = warm_call(True)
                        else:
                            raise ValueError('Registered custody intervention required')
                    retained = None
                    torch.cuda.synchronize()
                    uploaded = outputs = None
                    copied_outputs = []
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(reg['profiled_calls']):
                            with torch.profiler.record_function('diagnostic_native_upload'):
                                uploaded = upload_state(host)
                            with torch.profiler.record_function('diagnostic_generator'):
                                outputs = model.predict(uploaded, scene, case['start'], 1, 1)
                            with torch.profiler.record_function('diagnostic_native_download'):
                                copied_outputs.append(download_outputs(outputs))
                        torch.cuda.synchronize()
                    if any(hashes(out) != golden for out in copied_outputs) or inventory(model) != before:
                        raise ValueError('Every diagnostic output/learned tensor must remain exact')
                    allocations = []
                    gpu = []
                    for event in prof.events():
                        if event.name in ('cudaMalloc', 'cudaHostAlloc'):
                            parent = event.cpu_parent
                            ancestry = []
                            while parent is not None and len(ancestry) < 20:
                                ancestry.append(parent.name)
                                parent = parent.cpu_parent
                            allocations.append(dict(name=event.name, self_cpu_ms=float(event.self_cpu_time_total) / 1000,
                                                    ancestors=ancestry))
                        if event.device_type == torch.autograd.DeviceType.CUDA:
                            gpu.append(dict(name=event.name, instrumented_ms=float(event.device_time_total) / 1000))
                    generator_allocations = [a for a in allocations if a['name'] == 'cudaMalloc'
                                             and 'diagnostic_generator' in a['ancestors']]
                    result['rows'].append(dict(**cell, actual_output_sha256=golden, allocations=allocations,
                        generator_cudaMalloc_calls=len(generator_allocations),
                        generator_cudaMalloc_self_cpu_ms=sum(a['self_cpu_ms'] for a in generator_allocations),
                        generator_instrumented_ms=sum(e['instrumented_ms'] for e in gpu if e['name'] == 'diagnostic_generator'),
                        sampler_instrumented_ms=sum(e['instrumented_ms'] for e in gpu
                                                   if e['name'] == 'learned_transport' or 'geometry_transport_' in e['name']),
                        profiled_calls=reg['profiled_calls'], warmup_calls=reg['warmup_calls'],
                        all_actual_outputs_exact=True))
                    print('Allocation custody diagnostic', cell, 'generator cudaMalloc', len(generator_allocations), flush=True)
                    if torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']:
                        raise ValueError('Allocation diagnostic GPU memory cap')
                finally:
                    model.close()
                del model, resident, uploaded, outputs, copied_outputs, retained, warm_call, prof
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                write(args.output / 'results.json', result)
        if len(result['rows']) != 6:
            raise ValueError('All six registered diagnostic cells required')
        for p, h in reg['source_sha256'].items():
            if digest(ROOT / p) != h:
                raise ValueError('Diagnostic source changed during execution: ' + p)
        result.update(status='completed_paired_GPU_allocation_custody_diagnostic', active=None,
                      gpu_model=torch.cuda.get_device_name(), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      elapsed_diagnostic_seconds=time.monotonic() - started)
    except BaseException as error:
        result.update(status='failed', active=None, error=repr(error))
        raise
    finally:
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
