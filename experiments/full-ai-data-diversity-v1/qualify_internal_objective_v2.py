"""Actual normalized all148scene CUDA objective, gradients and CPU state restore.

No optimizer and no learned CPU forward. Local fine learned labels remain
training supervision; this does not qualify native rollout accuracy or latency.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from capture_memory_telemetry_v1 import process_memory
from internal_residual_dataset_v1 import TrainingView,METRICS
from internal_residual_network_v2 import InternalGroupResidual,PARAMETERS,active_input
from internal_residual_network_v1 import InternalGroupResidual as UngatedResidual
from internal_residual_loss_v1 import metrics
from managed_input_stream_v1 import streamed_identity


ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def inventory(model):
    return {n:dict(shape=list(v.shape),dtype='<f4',sha256=hashlib.sha256(v.detach().cpu().numpy().astype('<f4',copy=False).tobytes()).hexdigest())
            for n,v in model.state_dict().items()}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':
        raise ValueError('Fresh CUDA-only qualification required')
    reg=read(args.protocol)
    for name,expected in reg['source_sha256'].items():
        if streamed_identity(ROOT/name)['sha256']!=expected:raise ValueError('Frozen source changed: '+name)
    view_manifest=ROOT/reg['manifest'];manifest=read(view_manifest)
    archive=view_manifest.parent/manifest['archive']['path']
    if streamed_identity(archive)!={k:manifest['archive'][k] for k in ('bytes','sha256')}:
        raise ValueError('Complete archive identity differs')
    if read(ROOT/reg['artifact_audit'])['status']!='verified_training_artifact_integrity_producer_resource_gate_failed':
        raise ValueError('Original resource failure and streaming artifact recovery must be retained')
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    args.output.mkdir(parents=True)
    normalization=read(view_manifest.parent/'network-normalization.json')
    optimization=read(view_manifest.parent/'optimization-normalization.json')
    original=read(ROOT/manifest['original_eval_normalization'])
    model=InternalGroupResidual(normalization=normalization)
    before=inventory(model);versions={n:p._version for n,p in model.named_parameters()}
    view=TrainingView(view_manifest,archive)
    fixed=view.fixed_objective(optimization)
    model.zero_grad(set_to_none=True)
    inactive=[];probe_rows=[];probe=copy.deepcopy(model)
    with torch.no_grad():
        for layer in (*probe.velocity,probe.tracer):
            layer.weight.copy_(torch.linspace(-.001,.001,layer.weight.numel(),device='cuda').reshape_as(layer.weight));layer.bias.fill_(.0001)
    ungated=UngatedResidual(normalization=normalization);ungated.load_state_dict(probe.state_dict());ungated.eval().requires_grad_(False)
    rows=[];started=time.monotonic();total={n:0. for n in METRICS};worst={n:0. for n in METRICS}
    result=dict(status='running',optimizer_updates=0,new_parameter_fits=0,cpu_model_forward=False,
                expanded_reserved52_references_opened=0,native_accuracy_claim=False,latency_claim=False,
                full_generator_target_pass=False,inference_device='cuda',display_label='GPU (CUDA)')
    def bounded():
        if (process_memory()['peak_since_exec_bytes']>reg['maximum_peak_rss_bytes'] or
                torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or
                time.monotonic()-started>reg['maximum_wall_seconds']):raise ValueError('Objective qualification resource cap exceeded')
    try:
        for index in range(783):
            row,sample,response=view.sample(index)
            delta=model(sample['features'])
            if any(bool(value.detach().count_nonzero()) for value in delta):raise ValueError('Initial normalized residual is not exactly zero')
            output=response.emit(sample,delta)
            measured=metrics(output,sample,optimization)
            loss=sum(measured.values())
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite complete coupled objective')
            if not bool(active_input(sample['features'])):
                if float(loss.detach())!=0.:raise ValueError('Inactive input requires a nonzero physical correction')
                with torch.no_grad():
                    gated=probe(sample['features']);raw=ungated(sample['features'])
                    if any(bool(x.count_nonzero()) for x in gated):raise ValueError('Still-scene correction is nonzero')
                    if not any(bool(x.count_nonzero()) for x in raw):raise ValueError('Diagnostic ungated counterexample is vacuous')
                inactive.append(index)
            elif index in reg['nonzero_probe_indices']:
                probe.zero_grad(set_to_none=True)
                signed=probe(sample['features'])
                with torch.no_grad():
                    raw=ungated(sample['features'])
                    if any(not torch.equal(a,b) for a,b in zip(signed,raw)):raise ValueError('Activity gate changed active model fields')
                diagnostic_loss=sum(metrics(response.emit(sample,signed),sample,optimization).values());diagnostic_loss.backward()
                norm=float(probe.conv1.weight.grad.double().norm())
                if not math.isfinite(norm) or norm<=0.:raise ValueError('Actual normalized trunk gradient is zero/nonfinite')
                probe_rows.append(dict(index=index,trunk_gradient_norm=norm,active_output_bytes_exact=True))
                del signed,raw,diagnostic_loss
            (loss*view.sample_gradient_weight).backward()
            with torch.no_grad():
                evaluated={n:math.sqrt(max(0.,float(v))) for n,v in metrics(output,sample,original).items()}
                per_metric={n:float(v.detach()) for n,v in measured.items()}
            for n in METRICS:total[n]+=per_metric[n]/783;worst[n]=max(worst[n],evaluated[n])
            rows.append(dict(index=index,identifier=row['identifier'],start=row['start'],group_index=row['group_index'],
                width=row['width'],counts_xyz=row['counts_xyz'],cells_per_metre=row['cells_per_metre'],
                optimization_squared_relative=per_metric,original_local_relative=evaluated))
            del row,sample,response,delta,output,measured,loss
            bounded()
            if (index+1)%27==0:
                write(args.output/'progress.json',dict(result,active_samples=index+1,total_samples=783,rows=rows))
                print('Normalized full-grid objective samples',index+1,'/783',flush=True)
        if len(inactive)!=reg['expected_inactive_samples'] or len(probe_rows)!=len(reg['nonzero_probe_indices']):
            raise ValueError('Complete still-scene/probe coverage differs')
        row,sample,response=view.sample(inactive[0])
        changed=sample['features'].clone();changed[:,21,0,0,0]=.01
        probe.eval().requires_grad_(False)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):probe(changed)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph();static=changed.clone()
        with torch.cuda.graph(graph):captured=probe(static)
        for label,features in [('inactive',sample['features']),('changed_known_force',changed)]:
            static.copy_(features);graph.replay();expected=probe(features)
            if any(not torch.equal(a,b) for a,b in zip(captured,expected)):
                raise ValueError('Dynamic active/inactive CUDAgraph differs from eager')
            if label=='changed_known_force' and not any(bool(x.count_nonzero()) for x in captured):
                raise ValueError('Changed-force graph control is vacuous')
        del sample,response,changed,static,captured,expected,graph
        gradients={}
        for name,p in model.named_parameters():
            if p.grad is None or not bool(p.grad.isfinite().all()):raise ValueError('Finite complete CUDA gradients required')
            gradients[name]=dict(norm=float(p.grad.double().norm()),maximum=float(p.grad.abs().max()))
        if sum(v['norm'] for v in gradients.values())<=0.:raise ValueError('Residual heads have zero complete-objective gradient')
        if inventory(model)!=before or versions!={n:p._version for n,p in model.named_parameters()}:
            raise ValueError('Zero-update objective changed model state')
        np.savez(args.output/'untrained-initial-weights.npz',**{n:v.detach().cpu().numpy().astype('<f4',copy=True) for n,v in model.state_dict().items()})
        restored=InternalGroupResidual(device='cpu',normalization=normalization)
        with np.load(args.output/'untrained-initial-weights.npz',allow_pickle=False) as captured:
            if set(captured.files)!=set(restored.state_dict()):raise ValueError('Complete CPUrestore inventory required')
            state={}
            for name,current in restored.state_dict().items():
                value=captured[name]
                if value.dtype.str!='<f4' or tuple(value.shape)!=tuple(current.shape):raise ValueError('Typed CPUrestore mismatch')
                state[name]=torch.from_numpy(value.copy())
        restored.load_state_dict(state,strict=True)
        if inventory(restored)!=before:raise ValueError('Exact CPUrestore failed')
        bounded()
        result.update(status='passed_activity_preserving_normalized_all148scene_CUDA_objective_and_CPUrestore',
            inactive_indices=inactive,signed_probe_controls=probe_rows,dynamic_activity_CUDAgraph_bitwise_exact=True,
            active_local_samples=783,active_scenes=29,fixed_controls=fixed,training_scenes=148,
            all148training_scenes_in_objective=True,parameters=PARAMETERS,gradient_weight_per_sample=view.sample_gradient_weight,
            optimization_metric_means_active=total,full148scene_objective=view.active_weight*sum(total.values())+fixed['objective'],
            original_local_worst=worst,gradients=gradients,rows=rows,parameter_inventory=before,
            zero_update_parameters_unchanged=True,typed_initial_CPUrestore_exact=True,
            actual_managed_fitted_artifact_CPUrestore_still_required=True,
            memory_counters=process_memory(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            elapsed_seconds=time.monotonic()-started,gpu_model=torch.cuda.get_device_name(),
            protocol_sha256=streamed_identity(args.protocol)['sha256'],source_sha256=reg['source_sha256'],
            label_scope=manifest['label_scope'])
        write(args.output/'results.json',result)
    except BaseException as error:
        result.update(status='failed',error=repr(error),completed_samples=len(rows),rows=rows,memory_counters=process_memory())
        write(args.output/'results.json',result);raise
    finally:
        view.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
