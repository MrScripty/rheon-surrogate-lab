"""Finite balanced full-entry GPU benchmark with exact timed-output proof.

First pre-warmup calls are retained. Five complete-entry warmups precede five
balanced serial repeats; both adapters share the identical qualified GPU model.
No selection or additional repeats occur after observing measured maxima.
"""
import argparse
import collections
import hashlib
import os
import statistics
import subprocess
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from support import ROOT,read,write,digest
from fused_cuda_graph_v1 import FusedCudaGraphGenerator
from cuda_known_controls_v3 import CudaKnownControls
from cuda_native_transfer_v1 import upload_state,download_outputs
from expanded_cuda_generator_v1 import FIELDS
from diagnose_causal_composition import inventory

def hashes(outputs):return [{name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in row.items()} for row in outputs]
def run(args):
    if args.output.exists():raise ValueError('Fresh finite benchmark evidence required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('GPU-only inference required')
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen benchmark dependency changed: '+path)
    for proof in reg['qualifications']:
        q=read(ROOT/proof)
        if not q['all_accuracy_pass'] or not q['every_output_bitwise_equal_corrected_baseline']:raise ValueError('Complete preceding full-state qualification required')
    baseline=read(ROOT/reg['baseline_outputs'])
    expected={(r['identifier'],r['start'],r['K'],r['lead']):r['output_sha256'] for r in baseline['rows']}
    if len(expected)!=960 or not baseline['all_accuracy_pass']:raise ValueError('Complete original corrected baseline required')
    population={r['identifier']:r for r in read(ROOT/reg['bank'])['trajectories']};windows={}
    for source in reg['window_sources']:
        for row in read(ROOT/source['controller'])['completed']:
            if row['identifier'] in windows:raise ValueError('Unique current native window population required')
            windows[row['identifier']]=row
    if set(windows)!=set(population) or len(population)!=24:raise ValueError('All24development scenes required')
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.manual_seed(reg['seed']);tick=time.monotonic()
    model=FusedCudaGraphGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],'transport_and_linear')
    controls=CudaKnownControls(reg['control_build']);before=inventory(model);torch.cuda.synchronize()
    setup_ms=1000*(time.monotonic()-tick);args.output.mkdir(parents=True)
    result=dict(status='running',active=None,rows=[],first_pre_warmup_calls=[],summaries=[],every_actual_output_hash_checked=True,
        complete_timed_outputs=0,complete_pre_warmup_and_warmup_outputs=0,inference_device='cuda',display_label='GPU (CUDA)',
        gpu_model=torch.cuda.get_device_name(),gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        driver_version=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout.strip(),
        host_cpu=reg['host_cpu'],pressure_dtype='float64',carrier_dtype='float32',tf32=False,autocast=False,
        constructor_and_weight_loading_setup_ms=setup_ms,geometry_setup=[],source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),
        optimizer_updates=0,cpu_model_forward=False,expanded_reserved52_references_opened=0,new_independent_confirmation=False,
        full_generator_target_verified=False,cpu_latency_target_verified=False,
        scope='Complete current native NumPy fields -> CPU tensor views -> upload -> known scalar controls/GPU phase tables/forces -> all learned/internal stages -> independent K complete native CPU physical outputs. Pinned packing/allocation/copies and final synchronization included. Dataset reads, model/geometry/graph setup and explicitly recorded complete-entry warmups excluded.',
        maximum_is_observed_finite_population_not_real_time_guarantee=True)
    write(args.output/'results.json',result);started=time.monotonic();window_index=0
    def execute(mode,current,scene,start,k,n):
        native={name:torch.from_numpy(current[name]) for name in FIELDS}
        uploaded=upload_state(native) if mode=='packed_pinned' else {name:value.to('cuda') for name,value in native.items()}
        external=controls.generate(scene,start,k,n)
        emitted=model.predict(uploaded,scene,start,k,external,n)
        copied=download_outputs(emitted) if mode=='packed_pinned' else [{name:value.detach().cpu().contiguous() for name,value in row.items()} for row in emitted]
        torch.cuda.synchronize()
        return copied
    def verify(outputs,identifier,start,k):
        if len(outputs)!=k:raise ValueError('Every complete intermediate physical state required')
        actual=hashes(outputs)
        required=[expected[(identifier,start,k,lead)] for lead in range(1,k+1)]
        if actual!=required:raise ValueError('Actual complete timed/warmup outputs changed native qualified bytes')
        return actual
    try:
        with torch.inference_mode():
            for identifier,selected in population.items():
                if digest(ROOT/selected['receipt'])!=selected['receipt_sha256']:raise ValueError('Frozen native scene receipt changed')
                receipt=read(ROOT/selected['receipt']);scene=receipt['scene'];n=selected['reference_substeps']
                tick=time.perf_counter();model.prepare_geometry(tuple(scene['counts']));torch.cuda.synchronize()
                result['geometry_setup'].append(dict(identifier=identifier,ms=1000*(time.perf_counter()-tick)))
                for window in windows[identifier]['windows']:
                    if time.monotonic()-started>reg['maximum_wall_seconds']:raise TimeoutError('Finite balanced measurement budget exhausted')
                    start=window['start'];path=ROOT/window['current_state_path']
                    if digest(path)!=window['current_state_sha256']:raise ValueError('Current native state archive changed')
                    with np.load(path,allow_pickle=False) as a:current={name:a[name].copy() for name in FIELDS}
                    resident={name:torch.from_numpy(value).to('cuda') for name,value in current.items()}
                    for k in reg['horizons']:model.prepare_block(resident,scene,start,k,controls.generate(scene,start,k,n),n)
                    result['active']=dict(identifier=identifier,start=start);write(args.output/'progress.json',dict(status='running',active=result['active'],completed_timed_calls=len(result['rows'])))
                    for k in reg['horizons']:
                        modes=reg['modes'][window_index%2:]+reg['modes'][:window_index%2]
                        for mode in modes:
                            torch.cuda.synchronize();tick=time.perf_counter();copied=execute(mode,current,scene,start,k,n);elapsed=1000*(time.perf_counter()-tick)
                            verify(copied,identifier,start,k);result['complete_pre_warmup_and_warmup_outputs']+=k
                            result['first_pre_warmup_calls'].append(dict(identifier=identifier,start=start,K=k,mode=mode,full_block_ms=elapsed,ms_per_published_step=elapsed/k,internal_substeps=n))
                        for warmup in range(reg['complete_entry_warmup_calls']):
                            for mode in modes:
                                copied=execute(mode,current,scene,start,k,n);verify(copied,identifier,start,k);result['complete_pre_warmup_and_warmup_outputs']+=k
                    for repeat in range(reg['repetitions']):
                        shift=(repeat+window_index)%4;horizons=reg['horizons'][shift:]+reg['horizons'][:shift]
                        for k in horizons:
                            shift_mode=(repeat+window_index+k)%2;modes=reg['modes'][shift_mode:]+reg['modes'][:shift_mode]
                            for mode in modes:
                                torch.cuda.synchronize();tick=time.perf_counter();copied=execute(mode,current,scene,start,k,n);elapsed=1000*(time.perf_counter()-tick)
                                actual=verify(copied,identifier,start,k);result['complete_timed_outputs']+=k
                                result['rows'].append(dict(identifier=identifier,start=start,K=k,repeat=repeat,mode=mode,counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],internal_substeps=n,
                                    internal_advances_per_block=k*n,full_block_ms=elapsed,ms_per_published_step=elapsed/k,actual_complete_output_sha256=actual))
                    window_index+=1
                model.release_graphs();write(args.output/'results.json',result)
                print('Balanced complete GPU entry',identifier,'both adapters/allK/four windows',flush=True)
            if len(result['rows'])!=reg['expected_timed_calls'] or result['complete_timed_outputs']!=reg['expected_timed_outputs'] or inventory(model)!=before:
                raise ValueError('Full registered measurement/output population and unchanged weights required')
        grouped=collections.defaultdict(list)
        for row in result['rows']:grouped[(row['mode'],row['K'],row['cells_per_metre'],tuple(row['counts_xyz']),row['internal_substeps'])].append(row)
        result['summaries']=[dict(mode=mode,K=k,cells_per_metre=density,counts_xyz=list(counts),internal_substeps=n,measured_calls=len(rows),
            median_ms_per_published_step=statistics.median(r['ms_per_published_step'] for r in rows),maximum_ms_per_published_step=max(r['ms_per_published_step'] for r in rows),
            p95_ms_per_published_step=float(np.percentile([r['ms_per_published_step'] for r in rows],95))) for (mode,k,density,counts,n),rows in sorted(grouped.items())]
        for path,want in reg['source_sha256'].items():
            if digest(ROOT/path)!=want:raise ValueError('Benchmark source/evidence changed: '+path)
        result.update(status='completed_balanced_complete_gpu_entry_benchmark',active=None,all_actual_timed_and_warmup_output_hashes_equal_qualified_baseline=True,
            graph_setup_observations=model.setup_observations,diagnostic_elapsed_seconds=time.monotonic()-started,
            elapsed_wall_is_not_latency=True,learned_inventory_before_and_after=before)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        controls.close();model.close();result['active']=None;write(args.output/'results.json',result)
        write(args.output/'progress.json',dict(status=result['status'],active=None,completed_timed_calls=len(result['rows'])))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
