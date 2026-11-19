"""Managed full-grid CUDA residual fit and exact artifact-bound CPU restore."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import numpy as np
import torch

from capture_memory_telemetry_v1 import process_memory
from internal_residual_dataset_v1 import TrainingView,METRICS,FIELDS
from internal_residual_loss_v1 import metrics
from internal_residual_network_v2 import InternalGroupResidual,PARAMETERS
from internal_residual_weighted_loss_v1 import validate,weighted_sum,fixed_objective
from managed_input_stream_v1 import validate_inputs,streamed_identity
from managed_fit_artifacts import inventory,save_weights,load_weights,restore_custody


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return streamed_identity(path)['sha256']


def evaluate(model,view,optimization,original,weights):
    validate(weights)
    total={n:0. for n in METRICS};worst={n:0. for n in METRICS};groups={}
    with torch.no_grad():
        for index in range(783):
            row,sample,response=view.sample(index);output=response.emit(sample,model(sample['features']))
            values=metrics(output,sample,optimization);measured=metrics(output,sample,original)
            key=str(row['cells_per_metre'])+'@'+'x'.join(map(str,row['counts_xyz']))
            group=groups.setdefault(key,dict(cells_per_metre=row['cells_per_metre'],counts_xyz=row['counts_xyz'],
                samples=0,worst={n:0. for n in METRICS}))
            group['samples']+=1
            for name in METRICS:
                value=float(values[name]);relative=math.sqrt(max(0.,float(measured[name])))
                if not math.isfinite(value+relative):raise ValueError('Nonfinite full training evaluation')
                total[name]+=value/783;worst[name]=max(worst[name],relative);group['worst'][name]=max(group['worst'][name],relative)
    fixed=fixed_objective(view,optimization,weights)
    return dict(objective=view.active_weight*weighted_sum(total,weights)+fixed['objective'],
                active_metric_means=total,original_local_worst=worst,by_density_and_shape=groups,
                fixed_objective=fixed,training_scenes=148,local_samples=783)


def run(args):
    started=time.monotonic();config=read(args.config);weights=validate(config['metric_weights']);device='cuda' if args.mode=='train' else 'cpu'
    if os.environ.get('TULDOK_TORCH_DEVICE')!=device:raise ValueError('Actual managed device differs')
    if (config['role']!='internal_group_nonlinear_residual_component' or config['seed']!=1729
            or type(config['epochs']) is not int or not 1<=config['epochs']<=3
            or config['optimizer_updates']!=783*config['epochs'] or config['selection']!='minimum_full148scene_training_objective_earlier_epoch_tie'):
        raise ValueError('Exact finite first residual fit configuration required')
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):raise ValueError('Empty managed output directory required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    evidence,identities=validate_inputs(args);manifest=read(args.manifest);normalization=read(args.normalization)
    optimization=read(args.optimization_normalization);runtime=read(args.runtime_manifest)
    if runtime['packages']['torch']!=torch.__version__ or runtime['packages']['numpy']!=np.__version__:
        raise ValueError('Exact qualified runtime versions required')
    if sha(args.normalization)!=manifest['network_normalization_sha256'] or sha(args.optimization_normalization)!=manifest['optimization_normalization_sha256']:
        raise ValueError('Frozen training normalization changed')
    if not manifest['all148training_scenes_in_objective'] or manifest['development_or_reserved_used']:
        raise ValueError('Complete148scene train-only view required')
    closure=dict(config_sha256=sha(args.config),inputs=identities,
        sources={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},role=config['role'])
    model=InternalGroupResidual(device=device,seed=config['seed'],normalization=normalization)
    initial=inventory(model)
    if args.mode=='train' and initial!=config['qualified_initial_inventory']:raise ValueError('Actual initial model differs from qualified CUDA objective')
    origin=None;view=None
    if args.mode=='train':
        if len(args.shards)!=1 or identities['part_00']!={k:manifest['archive'][k] for k in ('bytes','sha256')}:
            raise ValueError('Exactly one complete typed training archive required')
        save_weights(model,args.output_dir/'initial-weights.npz')
        view=TrainingView(args.manifest,args.shards[0]);fixed=fixed_objective(view,optimization,weights)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],eps=config['adam_epsilon'],weight_decay=0.)
        best=config['qualified_initial_objective'];best_epoch=0
        best_state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        history=[];updates=0;coverage=[]
        def bounded():
            if (time.monotonic()-started>config['wall_seconds']-30
                    or process_memory()['peak_since_exec_bytes']>config['maximum_peak_rss_bytes']
                    or torch.cuda.max_memory_allocated()>config['maximum_gpu_allocated_bytes']):
                raise RuntimeError('Finite managed training resource budget exceeded')
        try:
            for epoch in range(config['epochs']):
                order=np.random.default_rng(config['seed']+epoch).permutation(783).tolist()
                observed=set();model.train()
                for index in order:
                    bounded();row,sample,response=view.sample(index)
                    fraction=updates/max(1,config['optimizer_updates']-1)
                    lr=config['minimum_learning_rate']+.5*(config['learning_rate']-config['minimum_learning_rate'])*(1.+math.cos(math.pi*fraction))
                    optimizer.param_groups[0]['lr']=lr
                    optimizer.zero_grad(set_to_none=True)
                    output=response.emit(sample,model(sample['features']))
                    value=weighted_sum(metrics(output,sample,optimization),weights)*view.active_weight+fixed['objective']
                    if not bool(value.isfinite()):raise ValueError('Nonfinite full-grid coupled loss')
                    value.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)
                    optimizer.step();updates+=1;observed.add(index)
                    if any(not bool(p.isfinite().all()) for p in model.parameters()):raise ValueError('Nonfinite learned parameter')
                    if updates==1 or updates%27==0 or updates==config['optimizer_updates']:
                        item=dict(update=updates,epoch=epoch+1,index=index,loss=float(value.detach()),gradient_norm=float(norm),learning_rate=lr,
                                  cells_per_metre=row['cells_per_metre'],counts_xyz=row['counts_xyz'])
                        history.append(item)
                        print(json.dumps(dict(format='tuldok_progress_v1',stage='internal-residual-full-grid',completed=updates,
                            total=config['optimizer_updates'],metrics={'loss':item['loss']})),flush=True)
                    del row,sample,response,output,value,norm
                if len(observed)!=783:raise ValueError('Every active scene/window/group must be consumed in each epoch')
                coverage.append(dict(epoch=epoch+1,local_samples=783,active_scenes=29,fixed_scenes=119,fixed_windows=1071))
                model.eval();scored=evaluate(model,view,optimization,config['evaluation_normalization'],weights);bounded()
                history.append(dict(epoch=epoch+1,complete_full148scene_evaluation=scored))
                if scored['objective']<best:
                    best=scored['objective'];best_epoch=epoch+1;best_state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
            model.load_state_dict(best_state,strict=True);model.eval().requires_grad_(False)
            final=inventory(model);save_weights(model,args.output_dir/'weights.npz')
            row,sample,response=view.sample(0)
            with torch.no_grad():output=response.emit(sample,model(sample['features']))
            vector=torch.cat([output[n].double().flatten() for n in FIELDS])
            np.save(args.output_dir/'probe.npy',vector.cpu().numpy().astype('<f8',copy=False))
            probe_spec=dict(index=0,field_order=list(FIELDS),shapes={n:list(output[n].shape) for n in FIELDS},elements=vector.numel(),dtype='<f8')
            write(args.output_dir/'history.json',dict(initial_inventory=initial,final_inventory=final,
                history=history,epoch_coverage=coverage,optimizer_updates=updates,selected_epoch=best_epoch,
                initial_full148scene_objective=config['qualified_initial_objective'],selected_full148scene_objective=best,
                probe=probe_spec,selection=config['selection'],train_only=True))
            execution=dict(device='cuda',gpu=torch.cuda.get_device_name(),all_learning_and_forward_on_cuda=True,
                residual_dtype='float32',coupled_loss_dtype='float64',tf32=False,amp=False,
                memory_counters=process_memory(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        finally:view.close()
        cpu_restore=False;probe_copied=False
    else:
        if args.shards:raise ValueError('CPU restore must not consume the unused training archive')
        origin=restore_custody(evidence,args);trained=read(args.trained_receipt);saved=read(args.trained_history)
        if (trained['mode']!='train' or trained['schema']!='rheon-internal-residual-managed-v2'
                or trained['optimizer_updates']!=config['optimizer_updates']
                or trained['closure']['config_sha256']!=closure['config_sha256']
                or trained['closure']['sources']!=closure['sources']):
            raise ValueError('Exact originating managed CUDA closure required')
        if any(trained['closure']['inputs'].get(n)!=v for n,v in identities.items()):raise ValueError('Static input differs from CUDA fit')
        if trained['closure']['inputs']['part_00']!={k:manifest['archive'][k] for k in ('bytes','sha256')}:
            raise ValueError('Originating archive identity differs')
        load_weights(model,args.trained_initial,saved['initial_inventory'])
        if inventory(model)!=config['qualified_initial_inventory']:raise ValueError('Restored initial model differs from qualified CUDA state')
        load_weights(model,args.trained_weights,saved['final_inventory'])
        initial=saved['initial_inventory'];final=inventory(model);probe_spec=saved['probe']
        for path,name in [(args.trained_initial,'initial-weights.npz'),(args.trained_weights,'weights.npz'),
                          (args.trained_history,'history.json'),(args.trained_probe,'probe.npy')]:
            if streamed_identity(path)!=trained['outputs'][name]:raise ValueError('Bound CUDA artifact changed')
            shutil.copyfile(path,args.output_dir/name)
        probe=np.load(args.trained_probe,allow_pickle=False)
        if probe.dtype.str!='<f8' or probe.shape!=(probe_spec['elements'],) or not np.isfinite(probe).all():
            raise ValueError('Typed originating complete-field CUDA probe required')
        if torch.cuda.is_initialized():raise ValueError('CPU restore initialized CUDA')
        execution=dict(device='cpu',model_forward_executed=False,cuda_initialized=False)
        updates=0;cpu_restore=True;probe_copied=True;best_epoch=saved['selected_epoch'];best=saved['selected_full148scene_objective']
    for path,name in [(args.normalization,'normalization.json'),(args.optimization_normalization,'optimization-normalization.json')]:
        shutil.copyfile(path,args.output_dir/name)
    outputs={p.name:streamed_identity(p) for p in args.output_dir.iterdir() if p.is_file()}
    write(args.output_dir/'receipt.json',dict(status='complete',schema='rheon-internal-residual-managed-v2',mode=args.mode,
        role=config['role'],training_metric_weights=weights,parameters=PARAMETERS,optimizer_updates=updates,closure=closure,initial_inventory=initial,final_inventory=final,
        execution=execution,outputs=outputs,originating_training_run=origin,typed_cpu_weight_restore_verified=cpu_restore,
        gpu_probe_copied_without_cpu_forward=probe_copied,full_cpu_probe_bitwise_verified=False,
        selected_epoch=best_epoch,selected_full148scene_objective=best,probe=probe_spec,
        cpu_model_forward_executed=False,training_scenes=148,expanded_reserved52_references_opened=0,
        native_accuracy_claim=False,latency_claim=False,scientific_acceptance_or_promotion=False,elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('train','restore'))
    for name in ('config','manifest','normalization','optimization-normalization','runtime-manifest','input-evidence','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shards',nargs='*',type=Path,default=[])
    for name in ('trained-weights','trained-initial','trained-history','trained-probe','trained-receipt'):
        parser.add_argument('--'+name,type=Path)
    run(parser.parse_args())
