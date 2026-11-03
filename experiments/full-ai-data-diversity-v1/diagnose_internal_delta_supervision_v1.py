"""Training-only direct residual target fit/gradient diagnostic; no optimizer."""
import argparse,hashlib,json,math,os
from pathlib import Path
import time
import numpy as np
import torch
from internal_residual_network_v2 import InternalGroupResidual
from internal_residual_dataset_v1 import TrainingView
from managed_input_stream_v1 import streamed_identity
from qualify_internal_objective_v2 import inventory
from capture_memory_telemetry_v1 import process_memory
ROOT=Path(__file__).resolve().parents[2]
HEADS=('vx','vy','vz','tracer')
def read(p):return json.loads(Path(p).read_text())
def run(args):
    reg=read(args.protocol)
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Fresh CUDA-only residual-label diagnostic required')
    for p,h in reg['source_sha256'].items():
        if streamed_identity(ROOT/p)['sha256']!=h:raise ValueError('Frozen diagnostic source changed')
    mp=ROOT/reg['manifest'];m=read(mp);archive=mp.parent/m['archive']['path']
    if streamed_identity(archive)!={k:m['archive'][k] for k in ('bytes','sha256')}:raise ValueError('Original train-only archive differs')
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    models={};before={}
    for label,pair in reg['pairs'].items():
        restored=read(ROOT/pair/'restore/receipt.json');path=ROOT/pair/'restore/weights.npz'
        if not restored['typed_cpu_weight_restore_verified'] or streamed_identity(path)!=restored['outputs']['weights.npz']:raise ValueError('Exact managed-restored component required')
        model=InternalGroupResidual(normalization=read(ROOT/pair/'restore/normalization.json'))
        with np.load(path,allow_pickle=False) as a:
            if set(a.files)!=set(model.state_dict()):raise ValueError('Exact complete component inventory required')
            model.load_state_dict({n:torch.from_numpy(a[n].copy()).to('cuda') for n in a.files},strict=True)
        model.eval();model.requires_grad_(label==reg['gradient_pair']);models[label]=model;before[label]=inventory(model)
    args.output.mkdir(parents=True);view=TrainingView(mp,archive);started=time.monotonic()
    sums={label:{n:0. for n in HEADS} for label in ('zero',*models)};rows=[];result=dict(status='running',optimizer_updates=0,new_parameter_fits=0,cpu_model_forward=False,expanded_reserved52_references_opened=0,full_generator_target_pass=False,inference_device='cuda',display_label='GPU (CUDA)')
    try:
        models[reg['gradient_pair']].zero_grad(set_to_none=True)
        for i,spec in enumerate(m['parts']):
            a=view.arrays(spec['member']);features=torch.from_numpy(a['features']).to('cuda')
            target=[torch.from_numpy(a['target_delta_'+n]).to('cuda') for n in HEADS]
            scales=models[reg['gradient_pair']].output_rms.double()
            baseline={n:float((value/scales[c]).square().mean()) for c,(n,value) in enumerate(zip(HEADS,target))}
            measured={}
            for label,model in models.items():
                with torch.set_grad_enabled(label==reg['gradient_pair']):
                    output=model(features);values={n:((value.double()-t)/scales[c]).square().mean() for c,(n,value,t) in enumerate(zip(HEADS,output,target))}
                    if label==reg['gradient_pair']:(sum(values.values())*(29./(148*783))).backward()
                    measured[label]={n:float(v.detach()) for n,v in values.items()}
            for n in HEADS:
                sums['zero'][n]+=baseline[n]/783
                for label in models:sums[label][n]+=measured[label][n]/783
            rows.append(dict(identifier=spec['identifier'],start=spec['start'],group_index=spec['group_index'],cells_per_metre=spec['cells_per_metre'],counts_xyz=spec['counts_xyz'],zero=baseline,measured=measured))
            del a,features,target,output,values
            if (process_memory()['peak_since_exec_bytes']>reg['maximum_peak_rss_bytes'] or torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or time.monotonic()-started>reg['maximum_wall_seconds']):raise ValueError('Finite direct-label diagnosis budget exceeded')
            if (i+1)%261==0:print('Direct residual supervision diagnostic',i+1,'/783',flush=True)
        if any(p.grad is None or not bool(p.grad.isfinite().all()) for p in models[reg['gradient_pair']].parameters()):raise ValueError('Finite complete direct-label gradients required')
        gradients={n:float(p.grad.double().norm()) for n,p in models[reg['gradient_pair']].named_parameters()}
        if any(inventory(model)!=before[n] for n,model in models.items()):raise ValueError('Diagnostic changed learned weights')
        result.update(status='completed_training_only_direct_delta_supervision_diagnostic',rows=rows,mean_normalized_delta_MSE=sums,
            explained_fraction_vs_zero={label:{n:1-sums[label][n]/max(sums['zero'][n],1e-300) for n in HEADS} for label in models},
            full148scene_direct_delta_objective={label:(29./148)*sum(values.values()) for label,values in sums.items()},
            gradient_pair=reg['gradient_pair'],direct_delta_parameter_gradient_norm=gradients,
            direct_delta_total_gradient_norm=math.sqrt(sum(v*v for v in gradients.values())),
            scope='Offline target_delta labels only supervise/score outputs, never input features. No optimizer or selected candidate; fixed119control contribution is zero for exact still-network branch.',
            weights_unchanged=True,training_scenes=148,active_samples=783,memory_counters=process_memory(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,source_sha256=reg['source_sha256'])
    except BaseException as error:result.update(status='failed',error=repr(error),rows=rows);raise
    finally:view.close();(args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
