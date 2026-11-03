"""Diagnostic complete Native-input/output GPU latency, even after accuracy failure.

No acceptance claim. Includes H2D, known forcing, every learned/physical field
and complete typed D2H outputs. CPU reference scoring/hashing is outside timing.
"""
import argparse
import collections
import hashlib
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from support import ROOT,read,write,digest
from internal_residual_fused_graph_v3 import InternalResidualFusedGraphGenerator as InternalResidualGenerator
from cuda_native_transfer_v1 import FIELDS
from capture_memory_telemetry_v1 import process_memory


def hashes(fields):return {n:hashlib.sha256(v.numpy().tobytes()).hexdigest() for n,v in fields.items()}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Fresh diagnostic CUDA timing required')
    reg=read(args.protocol)
    for name,expected in reg['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen timing dependency changed: '+name)
    native=read(ROOT/reg['native_results'])
    if native['status']!='completed_internal_residual_full_state_native_blocks':raise ValueError('Actual native result population required')
    qualified={(r['identifier'],r['start'],r['K'],r['lead']):r['output_sha256'] for r in native['rows']}
    cases={c['identifier']:c for c in read(ROOT/reg['case_selection'])['cases']}
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True);began=time.monotonic();setup=time.perf_counter()
    model=InternalResidualGenerator(reg['residual_pair'],reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build'])
    setup=time.perf_counter()-setup;inputs={};rows=[]
    try:
        setup_by_case=[]
        for identifier in reg['identifiers']:
            case=cases[identifier]
            if tuple(case['counts_xyz'])!=(32,32,32):raise ValueError('Registered32cubed reference grids required')
            spec=next(s for s in case['states'] if s['step']==reg['start'])
            if digest(ROOT/spec['path'])!=spec['sha256'] or digest(ROOT/case['receipt'])!=case['receipt_sha256']:raise ValueError('Frozen consumed current state differs')
            with np.load(ROOT/spec['path'],allow_pickle=False) as captured:fields={n:torch.from_numpy(captured[n].copy()) for n in FIELDS}
            scene=read(ROOT/case['receipt'])['scene'];before=hashes(fields);tick=time.perf_counter()
            for k in (1,2,3,4):
                output=model.predict_native(fields,scene,reg['start'],k,case['reference_substeps'])
                if [hashes(o) for o in output]!=[qualified[(identifier,reg['start'],k,lead)] for lead in range(1,k+1)]:raise ValueError('Graph warm-up differs from scored outputs')
            setup_by_case.append(dict(identifier=identifier,cells_per_metre=scene['cells_per_metre'],counts_xyz=scene['counts'],setup_and_warmup_seconds=time.perf_counter()-tick))
            for repeat in range(5):
                for k in ((1,2,3,4) if repeat%2==0 else (4,3,2,1)):
                    torch.cuda.synchronize();tick=time.perf_counter_ns()
                    output=model.predict_native(fields,scene,reg['start'],k,case['reference_substeps'])
                    elapsed=(time.perf_counter_ns()-tick)/1e6
                    if len(output)!=k or hashes(fields)!=before or [hashes(o) for o in output]!=[qualified[(identifier,reg['start'],k,lead)] for lead in range(1,k+1)]:raise ValueError('Timed complete output or input ownership differs')
                    rows.append(dict(identifier=identifier,repeat=repeat,K=k,cells_per_metre=scene['cells_per_metre'],counts_xyz=scene['counts'],wall_block_ms=elapsed,wall_ms_per_generated_step=elapsed/k))
                    if time.monotonic()-began>reg['maximum_wall_seconds'] or process_memory()['peak_since_exec_bytes']>reg['maximum_peak_rss_bytes'] or torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes']:raise ValueError('Diagnostic GPU timing resource budget exceeded')
        grouped=collections.defaultdict(list)
        for row in rows:grouped[(row['identifier'],row['cells_per_metre'],row['K'])].append(row)
        summaries=[]
        for (identifier,cpm,k),records in sorted(grouped.items()):
            values=[r['wall_block_ms'] for r in records]
            summaries.append(dict(identifier=identifier,cells_per_metre=cpm,counts_xyz=[32,32,32],K=k,samples=len(values),
                median_block_ms=statistics.median(values),p95_block_ms=float(np.percentile(values,95)),maximum_block_ms=max(values),
                median_ms_per_generated_step=statistics.median(values)/k,p95_ms_per_generated_step=float(np.percentile(values,95))/k,
                maximum_ms_per_generated_step=max(values)/k))
        if len(rows)!=len(reg['identifiers'])*4*5:raise ValueError('Complete fixed timing population required')
        write(args.output/'results.json',dict(status='completed_diagnostic_trained_internal_residual_GPU_native_entry_timing',
            rows=rows,summaries=summaries,setup_seconds=setup,setup_by_case=setup_by_case,inference_device='cuda',display_label='GPU (CUDA)',
            gpu_model=torch.cuda.get_device_name(),host_cpu=reg['host_cpu'],batch_size=1,balanced_serial_repeats=5,
            scope='Complete Native input/output API: pinned H2D, known forcing, all learned compute and intermediate physical states, typed pinned D2H and synchronization. Hashes/scoring excluded. Warm fused CUDAgraph per domain/density; case order is sequential and K order alternates between repeats. Case setup/warm-up is recorded separately, excluded from steady-state timing.',
            accuracy_failure_preserved=not native['all_accuracy_pass'],acceptance_latency_qualification=False,full_generator_target_pass=False,
            expanded_reserved52_references_opened=0,cpu_model_forward=False,new_parameter_fits=0,
            memory_counters=process_memory(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))
    finally:model.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
