"""Full240-step GPU own histories against one uninterrupted streamed native teacher.

Native references are CPU-only numerical teachers. Candidate inference is CUDA.
Every physical step for K1-4 is scored; no reference reset or future label is a
candidate input. One registered scene per owned invocation bounds memory/time.
"""
import argparse
import hashlib
import os
import resource
import shutil
import signal
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from support import ROOT,read,write,digest
from continuous_reference_client import ContinuousReferenceClient
from fft3d_pressure_graph_v3 import FFT3DPressureGraph as CompleteCudaGraphGenerator
from cuda_native_transfer_v1 import upload_state,download_outputs
from diagnose_causal_composition import inventory,measure
from qualify_streamed_reference_v1 import resident_bytes
FIELDS=('pressure','vx','vy','vz','tracer')

def run(args):
    if args.output.exists():raise ValueError('Fresh streamed full-history evidence required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('GPU-only learned inference required')
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen streamed evaluation dependency changed: '+path)
    if read(ROOT/reg['native_stream_qualification'])['status']!='passed_streamed_native_reference_typed_byte_and_custody_qualification':raise ValueError('Actual streamed reference lineage qualification required')
    blocks=read(ROOT/reg['block_qualification'])
    if blocks['status']!='completed_FFT3D_pressure_native_scored_blocks' or not blocks['all_accuracy_pass'] or len(blocks['rows'])!=1018 or not blocks['N1_all_output_bits_unchanged']:raise ValueError('Complete native block gate required')
    case=next((c for c in reg['cases'] if c['identifier']==args.identifier),None)
    if case is None:raise ValueError('Exactly one registered nonreserved scene required')
    preceding=reg.get('preceding_native_lineage_results',{}).get(case['identifier'])
    previous_native={r['step']:r['actual_field_sha256'] for r in read(ROOT/preceding)['reference_rows']} if preceding else {}
    if case['split'] not in ('train','development'):raise ValueError('Reserved references stay unopened')
    if digest(ROOT/case['receipt'])!=case['receipt_sha256']:raise ValueError('Native source receipt changed')
    receipt=read(ROOT/case['receipt']);scene=dict(receipt['scene'],reference_substeps=case['reference_substeps'],forcing_control_substeps=1);n=case['reference_substeps']
    if scene['steps']!=reg['rollout_steps'] or any(reg['rollout_steps']%k for k in reg['horizons']):raise ValueError('Complete aligned physical endpoint required')
    host_cpu=reg.get('host_cpu_by_identifier',{}).get(args.identifier,reg['host_cpu'])
    os.sched_setaffinity(0,{host_cpu});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True);active=None;model=None;started=time.monotonic();specs={r['step']:r for r in case['states']}
    result=dict(status='running',active=None,identifier=case['identifier'],source_split=case['split'],counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],internal_substeps=n,
        inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),host_cpu=host_cpu,cpu_model_forward=False,optimizer_updates=0,
        native_reference_method='Owned uninterrupted native integration with offline CPU spectral pressure teacher; no restore resets',
        candidate_own_history_without_teacher_resets=True,every_intermediate_physical_output_scored=True,rollout_steps=reg['rollout_steps'],output_dt_s=1/60,
        physical_frequency_and_render_rate_are_separate=True,rows=[],reference_rows=[],old_checkpoint_proofs=[],native_process=None,
        newly_independent_holdout=False,expanded_reserved52_references_opened=0,latency_claim=False,full_generator_targets_verified=False,
        no_full_state_archives_retained=True,protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'])
    def save():write(args.output/'results.json',result)
    def stop(*_):raise InterruptedError('Stop streamed evaluation and reap owned native child')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);save()
    def verify_old(step,truth):
        if step not in specs:return
        spec=specs[step];path=ROOT/spec['path']
        if digest(path)!=spec['sha256']:raise ValueError('Original full/native-window archive changed')
        with np.load(path,allow_pickle=False) as old:
            matches={name:truth[name].dtype==old[name].dtype and truth[name].shape==old[name].shape and truth[name].tobytes()==old[name].tobytes() for name in truth}
            diagnostics={name:dict(maximum_absolute_error=float(np.max(np.abs(truth[name].astype(np.float64)-old[name].astype(np.float64)))),old_sha256=hashlib.sha256(old[name].tobytes()).hexdigest(),new_sha256=hashlib.sha256(truth[name].tobytes()).hexdigest()) for name,passed in matches.items() if not passed}
        result['old_checkpoint_proofs'].append(dict(step=step,path=str(path.relative_to(ROOT)),all_typed_bytes_match=all(matches.values()),fields=matches,diagnostics=diagnostics))
        if not all(matches.values()):save();raise ValueError('Uninterrupted native teacher differs from old typed reference lineage')
    try:
        if shutil.disk_usage(args.output).free<reg['minimum_free_disk_bytes']:raise ValueError('Streaming free-disk floor reached')
        model=CompleteCudaGraphGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']);before=inventory(model)
        norm=read(ROOT/reg['normalization']);model.prepare_geometry(tuple(scene['counts']))
        active=ContinuousReferenceClient(scene,args.output/'native-reference',io_timeout_s=120.)
        packet,initial=active.inspect()
        if packet['step']!=0 or packet['time_s']!=0:raise ValueError('Actual native initial state required')
        zero=np.zeros_like(initial['pressure']);verify_old(0,dict(initial,rhs=zero,native_input_residual=zero,pressure_base=zero))
        current=upload_state({name:torch.from_numpy(value) for name,value in initial.items()})
        states={k:current for k in reg['horizons']};queues={k:[] for k in reg['horizons']}
        with torch.inference_mode():
            for k in reg['horizons']:model.prepare_block(current,scene,0,k,n)
            for step in range(1,reg['rollout_steps']+1):
                if time.monotonic()-started>reg['maximum_wall_seconds_per_scene']:raise TimeoutError('Finite full-history scene deadline reached')
                result['active']=dict(step=step,K=None)
                for k in reg['horizons']:
                    if not queues[k]:
                        outputs=model.predict(states[k],scene,step-1,k,n)
                        if len(outputs)!=k:raise ValueError('All complete physical outputs required')
                        queues[k]=list(outputs);states[k]=outputs[-1]
                frame=active.advance()
                if not frame['frame']['advanced'] or frame['frame']['step']!=step:raise ValueError('Native reference rejected actual physical advance')
                packet,truth=active.inspect()
                if packet['step']!=step or packet['time_s']!=step/60:raise ValueError('Actual reference physical clock mismatch')
                truth.update(rhs=active.last_request['rhs'],native_input_residual=active.last_request['native_input_residual'],pressure_base=active.last_request['base'])
                verify_old(step,truth)
                result['reference_rows'].append(dict(step=step,reference_native_step=step*n,frame_quality=frame['frame']['quality'],
                    actual_field_sha256={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in truth.items()}))
                for k in reg['horizons']:
                    result['active']['K']=k;output=queues[k].pop(0)
                    if set(output)!=set(FIELDS) or any(v.device.type!='cuda' for v in output.values()):raise ValueError('Complete physical fields must be generated onGPU')
                    emitted=download_outputs([output])[0]
                    metrics=measure(emitted,truth,scene,norm);metrics['all_accuracy_criteria_pass']=all(v<=reg['maximum_error'] for v in metrics['relative'].values())
                    result['rows'].append(dict(K=k,step=step,metrics=metrics,output_sha256={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in emitted.items()}))
                if previous_native and result['reference_rows'][-1]['actual_field_sha256']!=previous_native[step]:
                    raise ValueError('New native teacher differs from original full failure reference bytes')
                cpu_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+resident_bytes(active.child.pid);gpu_peak=torch.cuda.max_memory_allocated()
                if cpu_peak>reg['maximum_combined_peak_rss_bytes'] or gpu_peak>reg['maximum_gpu_allocated_bytes']:raise ValueError('Predeclared streamed CPU/GPU memory ceiling reached')
                result.update(observed_combined_peak_rss_bytes=cpu_peak,observed_peak_gpu_allocated_bytes=gpu_peak)
                if step%reg['progress_every_steps']==0:
                    save();print('Streamed expanded own-history',case['identifier'],step,'allK physical outputs scored; accuracy failures',sum(not r['metrics']['all_accuracy_criteria_pass'] for r in result['rows']),flush=True)
                if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>reg['maximum_output_bytes']:raise ValueError('Finite streaming diagnostic storage ceiling reached')
            if any(queues.values()) or len(result['rows'])!=reg['rollout_steps']*len(reg['horizons']) or inventory(model)!=before:raise ValueError('Full physical endpoint and unchanged learned tensors required')
        result['native_process']=active.close();active=None
        if not result['native_process']['reaped'] or result['native_process']['returncode']!=0 or not result['native_process']['graceful_protocol_complete']:raise ValueError('Native process must complete and reap')
        for path,want in reg['source_sha256'].items():
            if digest(ROOT/path)!=want:raise ValueError('Streamed evaluation dependency changed: '+path)
        result.update(status='completed_streamed_expanded_gpu_240step_scene',active=None,all_accuracy_pass=all(r['metrics']['all_accuracy_criteria_pass'] for r in result['rows']),
            accuracy_failing_outputs=sum(not r['metrics']['all_accuracy_criteria_pass'] for r in result['rows']),
            worst_error={name:max(r['metrics']['relative'][name] for r in result['rows']) for name in ('equation','pressure','velocity','tracer')},
            worst_error_byK={str(k):{name:max(r['metrics']['relative'][name] for r in result['rows'] if r['K']==k) for name in ('equation','pressure','velocity','tracer')} for k in reg['horizons']},
            graph_setup_observations=model.setup_observations,elapsed_diagnostic_seconds=time.monotonic()-started,diagnostic_elapsed_time_is_not_inference_latency=True,
            preceding_full_reference_every_frame_bitwise_equal=bool(previous_native),
            learned_inventory_before_and_after=before)
    except BaseException as error:result.update(status='failed',error=repr(error));raise
    finally:
        try:
            if active is not None:
                try:result['native_process']=active.close()
                except BaseException as cleanup:
                    result['native_process']=active.last_custody;result['native_cleanup_error']=repr(cleanup)
                    result['status']='failed'
        finally:
            try:
                if model is not None:model.close()
            finally:result['active']=None;save()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--identifier',required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
