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
from complete_native_entry_guarded_terminal_v1 import GuardedTerminalEntry as CompleteCudaGraphGenerator
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
    entry_gate=read(ROOT/reg['entry_component_qualification'])
    if (entry_gate['status']!='passed_all26_complete_native_entry_bytes_ownership_failure_custody'
            or not entry_gate['every_output_exact_to_qualified_GPU_baseline'] or len(entry_gate['guards'])!=26):
        raise ValueError('All native entry primitive and ownership gates required')
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
    model=CompleteCudaGraphGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']);before=inventory(model)
    if any(value.device.type!='cuda' for value in model.state_dict().values()):raise ValueError('Every learned stage must reside onGPU')
    controls=model._generator.controls
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
                    native_inputs={name:torch.from_numpy(initial[name]) for name in FIELDS};prefix=None
                    for k in reg['horizons']:
                        if time.monotonic()-began>reg['maximum_wall_seconds']:raise TimeoutError('Finite complete generator evaluation deadline reached')
                        result['active']=dict(identifier=identifier,start=start,K=k,internal_substeps=substeps);write(args.output/'results.json',result)
                        
                        model.prepare_native(native_inputs,scene,start,k,substeps)
                        for _ in range(reg['warmup_blocks']):model.predict_native(native_inputs,scene,start,k,substeps)
                        canonical=None;scored=None
                        for repeat in range(reg['repetitions']):
                            copied=model.predict_native(native_inputs,scene,start,k,substeps)
                            identities=[hashes(value) for value in copied]
                            if canonical is None:canonical=identities;scored=copied
                            elif identities!=canonical:raise ValueError('Repeated GPU output bytes changed')
                            uploaded=upload_state({name:torch.from_numpy(initial[name]) for name in FIELDS})
                            downloaded=download_outputs(model.predict(uploaded,scene,start,k,substeps))
                            if [hashes(value) for value in downloaded]!=canonical:raise ValueError('Packed I/O output differs from resident output')
                        if prefix is not None and canonical[:len(prefix)]!=prefix:raise ValueError('Causal unrolled generator changed its common output prefix acrossK')
                        prefix=canonical
                        for lead,output in enumerate(scored,1):
                            if set(output)!=set(FIELDS):raise ValueError('Complete pressure/velocity/tracer output required')
                            for name,value in output.items():
                                if value.dtype!=(torch.float64 if name=='pressure' else torch.float32) or tuple(value.shape)!=truth[lead][name].shape or not bool(value.isfinite().all()):
                                    raise ValueError('Complete typed finite current physical outputs required')
                            bitwise_baseline=canonical[lead-1]==expected_outputs[(identifier,k,start,lead)]
                            native_hashes={name:hashlib.sha256(truth[lead][name].tobytes()).hexdigest() for name in FIELDS}
                            bitwise_native=canonical[lead-1]==native_hashes
                            if scene['counts']==[64,64,64]:
                                if not bitwise_native:raise ValueError('Every repaired64output must match every native field byte')
                            elif not bitwise_baseline:raise ValueError('Every non64output must retain qualified baseline bytes')
                            metrics=measure(output,truth[lead],scene,norm);metrics['all_accuracy_criteria_pass']=all(value<=reg['maximum_error'] for value in metrics['relative'].values())
                            result['rows'].append(dict(identifier=identifier,K=k,start=start,step=start+lead,lead=lead,internal_substeps=substeps,
                                counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],inference_device='cuda',
                                metrics=metrics,output_sha256=canonical[lead-1],bitwise_equal_corrected_baseline=bitwise_baseline,bitwise_equal_native=bitwise_native,all_repeats_and_full_transfer_calls_bitwise_match=True))
                            if k==4:
                                for name,value in output.items():saved[f'start{start}_lead{lead}_{name}']=value.numpy().copy()
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
            every_actual_complete_output_hash_verified=True,
            all_non64_outputs_bitwise_equal_corrected_baseline=all(r['bitwise_equal_corrected_baseline'] for r in result['rows'] if r['counts_xyz']!=[64,64,64]),
            all64_outputs_bitwise_equal_native=all(r['bitwise_equal_native'] for r in result['rows'] if r['counts_xyz']==[64,64,64]),
            latency_claim=False,timing_observations_skipped=True,every_output_bitwise_equal_corrected_baseline=all(r['bitwise_equal_corrected_baseline'] for r in result['rows']),baseline_different_complete_outputs=sum(not r['bitwise_equal_corrected_baseline'] for r in result['rows']),baseline_outputs=reg['baseline_outputs'],known_control_identity=controls.identity,transfer_adapter='complete_native_entry_v1:exclusive_upload_replay_borrowed_GPU_download_owned_CPU',every_intermediate_output_scored=True,common_prefix_acrossK_bitwise_equal=True,
            graph_setup_observations=model.setup_observations,diagnostic_elapsed_seconds=time.monotonic()-began,
            diagnostic_elapsed_time_is_not_inference_latency=True,learned_inventory_before_and_after=before)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        model.close();result['active']=None;write(args.output/'results.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
