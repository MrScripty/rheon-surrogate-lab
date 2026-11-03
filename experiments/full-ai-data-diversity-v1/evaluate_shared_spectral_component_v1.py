"""GPU restored-probe parity and full-pressure development component accuracy.

Native same-request RHS is a component-test input. This is not a complete-state
simulation rollout, and component timings cannot represent full-step latency.
"""
import argparse
import collections
import hashlib
import os
import subprocess
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from support import ROOT,read,write,digest
from shared_spectral_model_v1 import SharedSpectralResponse,objective
from train_shared_spectral_v1 import load,inventory
from modal_basis import bases
from numerics import fine_action


def run(args):
    if args.output.exists():raise ValueError('Fresh GPU component evaluation required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Actual GPU-only inference required')
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen component evidence changed: '+path)
    pair=ROOT/reg['pair'];receipt=read(pair/'receipt.json');restored=read(pair/'restore/receipt.json');trained=read(pair/'train/receipt.json')
    if (receipt['status']!='completed' or not all(receipt['artifact_parity'].values()) or receipt['optimizer_updates']!=160
            or not restored['typed_cpu_weight_restore_verified'] or restored['optimizer_updates']!=0
            or restored['cpu_model_forward_executed'] or not restored['gpu_probe_copied_without_cpu_forward']):
        raise ValueError('Completed managed CUDA fit and exact typed CPU weight restore required')
    for key in ('training_run_id','restore_run_id'):
        terminal=read(pair/('terminal-'+receipt[key]+'.json'))
        if terminal['state']!='completed' or terminal['custody_phase']!='drained':raise ValueError('All managed owners must complete and drain')
    for name in ('weights.npz','probe.npy'):
        if digest(pair/'restore'/name)!=restored['outputs'][name]['sha256']:raise ValueError('Restored artifact changed')
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(reg['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=SharedSpectralResponse('cuda');load(model,pair/'restore/weights.npz',restored['final_inventory']);model.eval().requires_grad_(False)
    before=inventory(model);view=ROOT/reg['training_view'];manifest=read(view/'manifest.json');part=view/manifest['parts'][0]['path']
    if digest(part)!=manifest['parts'][0]['sha256']:raise ValueError('Training probe inputs changed')
    with np.load(part,allow_pickle=False) as archive:
        batch={name:torch.from_numpy(archive[name].copy()).to('cuda') for name in ('features','p','q','normalizer','example_weight')}
    expected=np.load(pair/'restore/probe.npy',allow_pickle=False)
    with torch.inference_mode():observed=model(batch['features']).cpu().numpy().copy();training_objective=float(objective(model,batch))
    if expected.dtype.str!='<f8' or observed.dtype.str!='<f8' or not np.array_equal(expected.view('<u8'),observed.view('<u8')):
        raise ValueError('CUDA re-evaluation of exact CPU-restored weights differs from originating CUDA probe')
    bank=read(ROOT/reg['development_bank']);train_bank=read(ROOT/reg['training_bank'])
    if bank['split']!='development' or bank['scene_count']!=24 or any(row['split']!='development' for row in bank['trajectories']):
        raise ValueError('All24development scenes required; no reserved references')
    training_shapes={tuple(row['counts_xyz']) for row in train_bank['trajectories']}
    norm=read(ROOT/reg['normalization']);pressure_floor=norm['relative_denominator_rms_floor']['pressure']
    args.output=args.output.resolve();args.output.mkdir(parents=True)
    result=dict(status='running',active=None,rows=[],summaries=[],pressure_snapshots=[],inference_device='cuda',display_label='GPU (CUDA)',
        device_index=torch.cuda.current_device(),gpu_uuid=str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()),'uuid','unavailable')),
        driver_version=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True).stdout.strip(),
        gpu_model=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        precision=dict(pressure='float64',tf32=False,autocast=False),
        restored_cuda_probe_bitwise_verified=True,training_objective=training_objective,
        own_history_rollout=False,full_state_accuracy_claim=False,cpu_latency_target_verified=False,
        native_same_request_rhs_used_as_component_input=True,pressure_timings_are_component_only=True,
        expanded_reserved52_references_opened=0,cpu_model_forward_executed=False,optimizer_updates=0,
        protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],source_evidence={})
    write(args.output/'results.json',result);geometry={}
    try:
        with torch.inference_mode():
            for selected in bank['trajectories']:
                path=ROOT/selected['receipt']
                if digest(path)!=selected['receipt_sha256']:raise ValueError('Development native receipt changed')
                native=read(path);scene=native['scene'];shape=tuple(reversed(scene['counts']));saved={}
                if shape not in geometry:
                    axes=bases(shape,'cuda');response=model.response_grid(shape)
                    if not bool(response.flatten()[1:].isfinite().all()) or not bool((response.flatten()[1:]>0).all()):
                        raise ValueError('Finite positive learned response required for every nonconstant mode')
                    geometry[shape]=(axes,response)
                axes,response=geometry[shape];states={entry['step']:entry for entry in selected['states']}
                result['active']=dict(identifier=selected['identifier']);write(args.output/'results.json',result)
                for step in reg['states']:
                    spec=states[step];source=ROOT/spec['path']
                    if digest(source)!=spec['sha256']:raise ValueError('Development native archive changed')
                    with np.load(source,allow_pickle=False) as archive:
                        p=archive['pressure'].copy();rhs=archive['rhs'].copy()
                    for name,value in [('pressure',p),('rhs',rhs)]:
                        if value.dtype.str!='<f8' or hashlib.sha256(value.tobytes()).hexdigest()!=spec['fields'][name]['sha256']:
                            raise ValueError('Actual complete development label changed')
                    q=torch.from_numpy(rhs).to('cuda').reshape(1,1,*shape)*(1000*scene['cells_per_metre'])
                    for _ in range(reg['warmup_calls']):model.decode(q,axes,response)
                    event_ms=[];wall_ms=[];canonical=None
                    for _ in range(reg['repetitions']):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize();tick=time.perf_counter();begin.record()
                        output=model.decode(q,axes,response)
                        end.record();end.synchronize();wall_ms.append(1000*(time.perf_counter()-tick));event_ms.append(begin.elapsed_time(end))
                        actual=output.cpu().flatten();identity=hashlib.sha256(actual.numpy().tobytes()).hexdigest()
                        if canonical is None:canonical=identity
                        elif identity!=canonical:raise ValueError('Timed GPU pressure output bytes changed')
                    target=torch.from_numpy(p);rhs_cpu=torch.from_numpy(rhs)
                    pressure_error=float(((actual-actual.mean())-(target-target.mean())).square().mean().sqrt()/target.sub(target.mean()).square().mean().sqrt().clamp_min(pressure_floor))
                    h=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
                    action=fine_action(actual.reshape(1,1,*shape),h,rho).flatten()
                    equation_error=float((action-rhs_cpu).norm()/rhs_cpu.norm().clamp_min(1e-30))
                    if not np.isfinite([pressure_error,equation_error]).all():raise ValueError('Finite actual component metrics required')
                    result['rows'].append(dict(identifier=selected['identifier'],counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],step=step,
                        reference_substeps=selected['reference_substeps'],shape_seen_in_training=tuple(scene['counts']) in training_shapes,
                        pressure_error=pressure_error,equation_error=equation_error,pressure_and_equation_pass=max(pressure_error,equation_error)<=reg['maximum_error'],
                        output_sha256=canonical,inference_device='cuda',component_gpu_event_ms=event_ms,component_synchronized_resident_wall_ms=wall_ms,
                        timing_scope='Pressure decoder only with resident q, cached fixed basis and learned response; excludes known-input generation, complete-state stages, transfers, setup and scoring.'))
                    saved[f'step{step}_pressure']=actual.numpy().copy()
                    result['source_evidence'][spec['path']]=spec['sha256']
                    if digest(source)!=spec['sha256']:raise ValueError('Development archive changed during evaluation')
                artifact=args.output/(selected['identifier']+'-pressure.npz');np.savez_compressed(artifact,**saved)
                result['pressure_snapshots'].append(dict(path=artifact.name,sha256=digest(artifact),bytes=artifact.stat().st_size))
                if sum(row['bytes'] for row in result['pressure_snapshots'])>reg['maximum_output_bytes']:raise ValueError('Finite pressure snapshot budget reached')
                print('GPU pressure component scored',selected['identifier'],len(reg['states']),'complete pressure fields',flush=True)
            if len(result['rows'])!=24*len(reg['states']) or inventory(model)!=before:raise ValueError('All registered fields and unchanged weights required')
        groups=collections.defaultdict(list)
        for row in result['rows']:groups[(row['cells_per_metre'],tuple(row['counts_xyz']))].append(row)
        result['summaries']=[dict(cells_per_metre=density,counts_xyz=list(counts),pressure_fields=len(rows),
            worst_pressure_error=max(r['pressure_error'] for r in rows),worst_equation_error=max(r['equation_error'] for r in rows),
            all_pressure_and_equation_pass=all(r['pressure_and_equation_pass'] for r in rows),shape_seen_in_training=rows[0]['shape_seen_in_training'])
            for (density,counts),rows in sorted(groups.items())]
        for path,expected in reg['source_sha256'].items():
            if digest(ROOT/path)!=expected:raise ValueError('Component dependency changed during evaluation: '+path)
        result.update(status='completed_gpu_shared_pressure_component_development_evaluation',active=None,
            all_pressure_and_equation_pass=all(r['pressure_and_equation_pass'] for r in result['rows']),
            worst_pressure_error=max(r['pressure_error'] for r in result['rows']),worst_equation_error=max(r['equation_error'] for r in result['rows']),
            learned_inventory_before_and_after=before,full_generator_target_pass=False)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        result['active']=None;write(args.output/'results.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
