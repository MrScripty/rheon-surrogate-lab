"""Complete GPU state/block accuracy and explicitly scoped full-generator timing."""
import argparse
import collections
import hashlib
import os
from pathlib import Path
import statistics
import subprocess
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from support import ROOT,read,write,digest
from fused_cuda_graph_v1 import FusedCudaGraphGenerator
from cuda_known_controls_v3 import CudaKnownControls
from cuda_native_transfer_v1 import upload_state,download_outputs
from expanded_cuda_generator_v1 import FIELDS,prescribed_controls
from diagnose_causal_composition import inventory,measure


def host(values):return {name:value.detach().cpu().contiguous() for name,value in values.items()}
def hashes(values):return {name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in values.items()}
def stats(values):return dict(median=statistics.median(values),p95=float(np.percentile(values,95)),maximum=max(values),samples=len(values))


def run(args):
    if args.output.exists():raise ValueError('Fresh expanded generator evaluation required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('GPU-only inference required')
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen expanded generator dependency changed: '+path)
    bank=read(ROOT/reg['bank']);population={r['identifier']:r for r in bank['trajectories']}
    if bank['scene_count']!=24 or bank['split']!='development':raise ValueError('Full development population required')
    windows={}
    for spec in reg['window_sources']:
        controller=read(ROOT/spec['controller']);audit=read(ROOT/spec['integrity'])
        if (audit['status']!='completed_development_window_archive_integrity_audit'
                or len(audit['rows'])!=len(controller['completed']) or audit['reserved_references_opened']!=0):
            raise ValueError('Actual complete-subset archive verification required')
        for row in controller['completed']:
            if row['identifier'] in windows or row['split']!='development':raise ValueError('Whole development scene split/uniqueness required')
            windows[row['identifier']]=row
    if set(windows)!=set(population):raise ValueError('All24development scenes require complete labels')
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(reg['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    baseline=read(ROOT/reg['baseline_outputs'])
    if not baseline['all_accuracy_pass'] or len(baseline['rows'])!=reg['expected_complete_states']:
        raise ValueError('Complete corrected baseline outputs required')
    expected_outputs={(r['identifier'],r['K'],r['start'],r['lead']):r['output_sha256'] for r in baseline['rows']}
    if len(expected_outputs)!=reg['expected_complete_states']:raise ValueError('Unique baseline output population required')
    model=FusedCudaGraphGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['fusion_mode']);before=inventory(model)
    if any(value.device.type!='cuda' for value in model.state_dict().values()):raise ValueError('Every learned stage must reside onGPU')
    controls=CudaKnownControls(reg['control_build'])
    control_qualification=read(ROOT/reg['control_qualification'])
    if not control_qualification['all_accuracy_pass']:raise ValueError('Known CUDA controls must qualify first')
    if read(ROOT/reg['transfer_qualification'])['status']!='passed_actual_native_transfer_bytes_and_ownership':
        raise ValueError('Exact native transfer and ownership qualification required')
    args.output=args.output.resolve();args.output.mkdir(parents=True)
    props=torch.cuda.get_device_properties(torch.cuda.current_device())
    result=dict(status='running',active=None,rows=[],timings=[],summaries=[],snapshots=[],
        device=dict(display_label='GPU (CUDA)',inference_device='cuda',gpu_model=props.name,gpu_uuid=str(getattr(props,'uuid','unavailable')),
            device_index=torch.cuda.current_device(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
            driver_version=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True).stdout.strip(),
            pressure_dtype='float64',carrier_dtype='float32',tf32=False,autocast=False,host_cpu=reg['host_cpu'],batch_size=1),
        model_identity=model.asset_identity,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),source_evidence={},
        optimizer_updates=0,cpu_model_forward=False,own_history_inside_each_block=True,own_history_240step_rollout=False,
        output_dt_s=1/60,rendering_fps_and_physical_frequencies_are_separate=True,expanded_reserved52_references_opened=0,
        cpu_latency_target_verified=False,full_generator_target_pass=False,
        internal_substeps_are_counted_work=True,every_intermediate_published_state_required=True)
    write(args.output/'results.json',result);norm=read(ROOT/reg['normalization']);began=time.monotonic()
    try:
        with torch.inference_mode():
            for identifier,selected in population.items():
                row=windows[identifier];path=ROOT/selected['receipt']
                if digest(path)!=selected['receipt_sha256'] or row['source_receipt_sha256']!=selected['receipt_sha256']:
                    raise ValueError('Exact development native lineage required')
                receipt=read(path);scene=receipt['scene'];substeps=selected['reference_substeps']
                model.prepare_geometry(tuple(scene['counts']));saved={}
                def load_native(source,expected):
                    if digest(source)!=expected:raise ValueError('Native field archive changed')
                    with np.load(source,allow_pickle=False) as archive:values={name:archive[name].copy() for name in (*FIELDS,'rhs')}
                    result['source_evidence'][str(source.relative_to(ROOT))]=expected
                    return values
                for window in row['windows']:
                    start=window['start']
                    if start not in reg['starts']:raise ValueError('Registered mature/initial windows required')
                    initial=load_native(ROOT/window['current_state_path'],window['current_state_sha256'])
                    truth={value['lead']:load_native(ROOT/value['path'],value['sha256']) for value in window['outputs']}
                    resident={name:torch.from_numpy(initial[name]).to('cuda') for name in FIELDS};prefix=None
                    for k in reg['horizons']:
                        if time.monotonic()-began>reg['maximum_wall_seconds']:raise TimeoutError('Finite complete generator evaluation deadline reached')
                        result['active']=dict(identifier=identifier,start=start,K=k,internal_substeps=substeps);write(args.output/'results.json',result)
                        external=controls.generate(scene,start,k,substeps)
                        model.prepare_block(resident,scene,start,k,external,substeps)
                        for _ in range(reg['warmup_blocks']):model.predict(resident,scene,start,k,external,substeps)
                        event_times=[];resident_times=[];control_resident_times=[];full_times=[];canonical=None;scored=None
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        for repeat in range(reg['repetitions']):
                            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize();tick=time.perf_counter();begin.record()
                            outputs=model.predict(resident,scene,start,k,external,substeps)
                            end.record();end.synchronize();resident_times.append(1000*(time.perf_counter()-tick));event_times.append(begin.elapsed_time(end))
                            if len(outputs)!=k or any(value.device.type!='cuda' for output in outputs for value in output.values()):raise ValueError('Every complete physical output must be generated onGPU')
                            copied=[host(output) for output in outputs];identities=[hashes(output) for output in copied]
                            if canonical is None:canonical=identities;scored=copied
                            elif identities!=canonical:raise ValueError('Repeated measured complete CUDA outputs changed')
                            torch.cuda.synchronize();tick=time.perf_counter()
                            uploaded=upload_state({name:torch.from_numpy(initial[name]) for name in FIELDS})
                            supplied=controls.generate(scene,start,k,substeps)
                            full=model.predict(uploaded,scene,start,k,supplied,substeps)
                            downloaded=download_outputs(full);torch.cuda.synchronize();full_times.append(1000*(time.perf_counter()-tick))
                            if [hashes(output) for output in downloaded]!=canonical:raise ValueError('Host-input and resident complete outputs differ')
                            torch.cuda.synchronize();tick=time.perf_counter()
                            supplied=controls.generate(scene,start,k,substeps)
                            with_controls=model.predict(resident,scene,start,k,supplied,substeps)
                            torch.cuda.synchronize();control_resident_times.append(1000*(time.perf_counter()-tick))
                            if [hashes(output) for output in map(host,with_controls)]!=canonical:raise ValueError('Generated-control resident outputs differ')
                        if prefix is not None and canonical[:len(prefix)]!=prefix:raise ValueError('Causal unrolled generator changed its common output prefix acrossK')
                        prefix=canonical
                        for lead,output in enumerate(scored,1):
                            if set(output)!=set(FIELDS):raise ValueError('Complete pressure/velocity/tracer output required')
                            for name,value in output.items():
                                if value.dtype!=(torch.float64 if name=='pressure' else torch.float32) or tuple(value.shape)!=truth[lead][name].shape or not bool(value.isfinite().all()):
                                    raise ValueError('Complete typed finite current physical outputs required')
                            bitwise_baseline=canonical[lead-1]==expected_outputs[(identifier,k,start,lead)]
                            metrics=measure(output,truth[lead],scene,norm);metrics['all_accuracy_criteria_pass']=all(value<=reg['maximum_error'] for value in metrics['relative'].values())
                            result['rows'].append(dict(identifier=identifier,K=k,start=start,step=start+lead,lead=lead,internal_substeps=substeps,
                                counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],inference_device='cuda',
                                metrics=metrics,output_sha256=canonical[lead-1],bitwise_equal_corrected_baseline=bitwise_baseline,all_measured_repeats_and_full_transfer_calls_bitwise_match=True))
                            if k==4:
                                for name,value in output.items():saved[f'start{start}_lead{lead}_{name}']=value.numpy().copy()
                        result['timings'].append(dict(identifier=identifier,K=k,start=start,internal_substeps=substeps,internal_advances_per_block=k*substeps,
                            counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],inference_device='cuda',display_label='GPU (CUDA)',
                            full_block_ms=dict(cuda_event=stats(event_times),synchronized_resident_wall=stats(resident_times),synchronized_host_input_and_output_wall=stats(full_times),resident_state_with_known_control_generation_wall=stats(control_resident_times)),
                            amortized_ms_per_published_step=dict(cuda_event=stats([v/k for v in event_times]),synchronized_resident_wall=stats([v/k for v in resident_times]),synchronized_host_input_and_output_wall=stats([v/k for v in full_times]),resident_state_with_known_control_generation_wall=stats([v/k for v in control_resident_times])),
                            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),repetitions=reg['repetitions'],
                            full_scope='All learned stages and internal substeps. CUDA-event/provided-control resident scopes exclude control construction; resident-state-with-known-control includes all scalar table construction/upload, GPU forcing and prediction. Full host I/O adds current-state upload and every intermediate download. Setup/loading/capture/warm-up separate.'))
                    write(args.output/'results.json',result)
                artifact=args.output/(identifier+'-complete-k4-windows.npz');np.savez_compressed(artifact,**saved)
                result['snapshots'].append(dict(path=artifact.name,sha256=digest(artifact),bytes=artifact.stat().st_size))
                if sum(value['bytes'] for value in result['snapshots'])>reg['maximum_output_bytes']:raise ValueError('Finite complete-state snapshot budget reached')
                model.release_graphs()
                print('GPU full state scored',identifier,'allK, four windows; internal substeps',substeps,flush=True)
            if len(result['rows'])!=reg['expected_complete_states'] or inventory(model)!=before:raise ValueError('Full registered complete-output population and unchanged weights required')
        grouped=collections.defaultdict(list)
        for row in result['rows']:grouped[(row['K'],row['cells_per_metre'],tuple(row['counts_xyz']),row['internal_substeps'])].append(row)
        result['summaries']=[dict(K=k,cells_per_metre=density,counts_xyz=list(counts),internal_substeps=substeps,complete_outputs=len(rows),
            all_accuracy_pass=all(r['metrics']['all_accuracy_criteria_pass'] for r in rows),
            worst_error={field:max(r['metrics']['relative'][field] for r in rows) for field in ('equation','pressure','velocity','tracer')})
            for (k,density,counts,substeps),rows in sorted(grouped.items())]
        for path,expected in reg['source_sha256'].items():
            if digest(ROOT/path)!=expected:raise ValueError('Expanded generator source changed during evaluation: '+path)
        result.update(status='completed_expanded_cuda_full_state_block_evaluation',active=None,
            all_accuracy_pass=all(row['metrics']['all_accuracy_criteria_pass'] for row in result['rows']),
            every_measured_complete_output_hash_verified=True,every_output_bitwise_equal_corrected_baseline=all(r['bitwise_equal_corrected_baseline'] for r in result['rows']),baseline_different_complete_outputs=sum(not r['bitwise_equal_corrected_baseline'] for r in result['rows']),baseline_outputs=reg['baseline_outputs'],known_control_identity=controls.identity,transfer_adapter='cuda_native_transfer_v1:packed_pinned_upload_and_async_native_download',every_intermediate_output_scored=True,common_prefix_acrossK_bitwise_equal=True,
            graph_setup_observations=model.setup_observations,diagnostic_elapsed_seconds=time.monotonic()-began,
            diagnostic_elapsed_time_is_not_inference_latency=True,learned_inventory_before_and_after=before)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        controls.close();model.close();result['active']=None;write(args.output/'results.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
