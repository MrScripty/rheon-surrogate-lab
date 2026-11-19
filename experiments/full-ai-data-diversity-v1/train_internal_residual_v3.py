"""Finite managed full-objective LBFGS warm start using exact compact CUDA data."""
import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from managed_input_stream_v1 import validate_inputs,streamed_identity
from managed_fit_artifacts import inventory,save_weights,load_weights,restore_custody
from capture_memory_telemetry_v1 import process_memory
from internal_residual_network_v2 import InternalGroupResidual,PARAMETERS
from internal_residual_loss_v1 import metrics
from internal_residual_dataset_v1 import METRICS,FIELDS
from internal_residual_cached_view_v1 import GPUCachedTrainingView
from compact_internal_training_view_v1 import CompactTrainingView


def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):return streamed_identity(p)['sha256']


def load_warm(model,manifest,expected):
    raw=base64.b64decode(manifest['inline_warm_weights_npz_base64'],validate=True)
    receipt=manifest['inline_warm_restore_receipt']
    if hashlib.sha256(raw).hexdigest()!=receipt['outputs']['weights.npz']['sha256'] or not receipt['typed_cpu_weight_restore_verified'] or receipt['cpu_model_forward_executed']:
        raise ValueError('Exact managed warm-start parameter restore required')
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        if set(archive.files)!=set(model.state_dict()):raise ValueError('Warm-start inventory differs')
        values={}
        for n,current in model.state_dict().items():
            a=archive[n]
            if a.dtype.str!='<f4' or tuple(a.shape)!=tuple(current.shape):raise ValueError('Typed warm-start field differs')
            values[n]=torch.from_numpy(a.copy()).to(current.device)
    model.load_state_dict(values,strict=True)
    if inventory(model)!=expected:raise ValueError('Qualified warm-start state differs')


def prepare_cache(args,config,manifest):
    allowed=[args.config,args.manifest,args.normalization,args.optimization_normalization,args.runtime_manifest,args.teacher_code,*args.shards]
    source=CompactTrainingView(args.manifest,args.shards[0],args.teacher_code,args.output_dir.parent,Path(__file__).parent,allowed)
    def progress(n):
        if n%27==0:print(json.dumps(dict(format='tuldok_progress_v1',stage='compact-CUDA-cache',completed=n,total=783)),flush=True)
    try:
        cache=GPUCachedTrainingView(source,config['maximum_gpu_allocated_bytes'],config['maximum_gpu_reserved_bytes'],progress)
        source.finish_generation()
        if cache.cached_tensor_bytes!=config['expected_cached_tensor_bytes']:raise ValueError('Complete cached tensor footprint differs')
        return source,cache
    except BaseException:source.close();raise


def full_objective(model,view,optimization,backward,bounded):
    model.zero_grad(set_to_none=True);total={n:0. for n in METRICS}
    with torch.enable_grad() if backward else torch.no_grad():
        for index in range(783):
            row,sample,response=view.sample(index)
            measured=metrics(response.emit(sample,model(sample['features'])),sample,optimization)
            if backward:(sum(measured.values())*view.sample_gradient_weight).backward()
            for name,value in measured.items():
                scalar=float(value.detach())
                if not math.isfinite(scalar):raise ValueError('Nonfinite complete coupled objective')
                total[name]+=scalar/783
            del measured
            if index%27==0:bounded()
    fixed=view.fixed_objective(optimization)
    return dict(objective=view.active_weight*sum(total.values())+fixed['objective'],active_metric_means=total,
        fixed_controls=fixed,active_samples=783,training_scenes=148)


class ClosureBudget(Exception):pass


def run(args):
    started=time.monotonic();config=read(args.config);device='cuda' if args.mode=='train' else 'cpu'
    if os.environ.get('TULDOK_TORCH_DEVICE')!=device:raise ValueError('Explicit managed device required')
    if config['optimizer']!='full_batch_LBFGS' or config['role']!='internal_group_nonlinear_residual_component' or config['maximum_optimizer_iterations']!=8 or config['maximum_closure_evaluations']!=18:
        raise ValueError('Exact finite warm-start program required')
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):raise ValueError('Empty managed output directory required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    evidence,identities=validate_inputs(args);manifest=read(args.manifest);normalization=read(args.normalization);optimization=read(args.optimization_normalization)
    runtime=read(args.runtime_manifest)
    if runtime['packages']['torch']!=torch.__version__ or runtime['packages']['numpy']!=np.__version__:raise ValueError('Exact qualified runtime required')
    if sha(args.normalization)!=manifest['network_normalization_sha256'] or sha(args.optimization_normalization)!=manifest['optimization_normalization_sha256']:raise ValueError('Frozen normalization differs')
    if not manifest['all148training_scenes_in_objective'] or manifest['development_or_reserved_used']:raise ValueError('All148training scenes only')
    if sha(args.teacher_code)!=manifest['teacher_code']['sha256']:raise ValueError('Captured teacher code differs')
    closure=dict(config_sha256=sha(args.config),inputs=identities,role=config['role'],sources={p.name:sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file() and p.suffix in ('.py','.zip')})
    model=InternalGroupResidual(device=device,seed=config['seed'],normalization=normalization)
    source=cache=None;origin=None
    if args.mode=='train':
        if len(args.shards)!=1 or identities['part_00']!={k:manifest['archive'][k] for k in ('bytes','sha256')}:raise ValueError('Complete compact captured input required')
        load_warm(model,manifest,config['qualified_initial_inventory']);initial=inventory(model);save_weights(model,args.output_dir/'initial-weights.npz')
        source,cache=prepare_cache(args,config,manifest)
        optimizer=torch.optim.LBFGS(model.parameters(),lr=config['initial_step_learning_rate'],max_iter=1,max_eval=3,
            tolerance_grad=1e-12,tolerance_change=1e-15,history_size=5,line_search_fn='strong_wolfe')
        best=config['qualified_initial_objective'];best_eval=0;calls=0;history=[]
        best_state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};last_state=best_state
        def bounded():
            if (time.monotonic()-started>config['wall_seconds']-30 or process_memory()['peak_since_exec_bytes']>config['maximum_peak_rss_bytes'] or torch.cuda.max_memory_allocated()>config['maximum_gpu_allocated_bytes'] or torch.cuda.max_memory_reserved()>config['maximum_gpu_reserved_bytes']):
                raise RuntimeError('Finite registered training budget exceeded')
        def objective_closure():
            nonlocal calls,best,best_eval,best_state,last_state
            if calls>=config['maximum_closure_evaluations']:raise ClosureBudget()
            bounded();calls+=1
            scored=full_objective(model,cache,optimization,True,bounded)
            if calls==1 and not math.isclose(scored['objective'],config['qualified_initial_objective'],rel_tol=1e-10,abs_tol=1e-30):raise ValueError('Actual initial full objective differs')
            if any(p.grad is None or not bool(p.grad.isfinite().all()) for p in model.parameters()):raise ValueError('Complete finite full gradients required')
            item=dict(evaluation=calls,**scored,gradient_norm=math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters())))
            history.append(item);last_state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
            if scored['objective']<best:best=scored['objective'];best_eval=calls;best_state=last_state
            print(json.dumps(dict(format='tuldok_progress_v1',stage='full148scene-LBFGS',completed=calls,total=config['maximum_closure_evaluations'],metrics={'loss':scored['objective']})),flush=True)
            return torch.tensor(scored['objective'],device='cuda',dtype=torch.float64)
        budget_exhausted=False
        try:
            for outer in range(config['maximum_optimizer_iterations']):
                optimizer.param_groups[0]['lr']=config['initial_step_learning_rate'] if outer==0 else config['subsequent_step_learning_rate']
                optimizer.step(objective_closure)
                if calls>=config['maximum_closure_evaluations']:break
        except ClosureBudget:budget_exhausted=True
        finally:
            if source is not None:source.close()
        updates=int(optimizer.state[next(iter(model.parameters()))].get('n_iter',0))
        model.load_state_dict(last_state,strict=True);save_weights(model,args.output_dir/'last-evaluated-weights.npz')
        model.load_state_dict(best_state,strict=True);model.eval().requires_grad_(False);final=inventory(model)
        save_weights(model,args.output_dir/'weights.npz');row,sample,response=cache.sample(0)
        with torch.no_grad():output=response.emit(sample,model(sample['features']))
        vector=torch.cat([output[n].double().flatten() for n in FIELDS]);np.save(args.output_dir/'probe.npy',vector.cpu().numpy().astype('<f8',copy=False))
        probe_spec=dict(index=0,field_order=list(FIELDS),shapes={n:list(output[n].shape) for n in FIELDS},elements=vector.numel(),dtype='<f8')
        write(args.output_dir/'history.json',dict(initial_inventory=initial,final_inventory=final,history=history,optimizer_updates=updates,
            optimizer_update_definition='LBFGS internal iteration counter, including the last searched iteration if the global closure budget interrupted its line search.',
            closure_evaluations=calls,closure_budget_exhausted=budget_exhausted,selected_evaluation=best_eval,
            initial_full148scene_objective=config['qualified_initial_objective'],selected_full148scene_objective=best,
            probe=probe_spec,selection=config['selection'],train_only=True))
        execution=dict(device='cuda',gpu=torch.cuda.get_device_name(),all_learning_and_forward_on_cuda=True,tf32=False,amp=False,
            cached_tensor_bytes=cache.cached_tensor_bytes,memory_counters=process_memory(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),ambient_repository_reads_denied=source.denied)
        cache.close();cpu_restore=False;probe_copied=False
    else:
        if args.shards:raise ValueError('CPUrestore must omit unused large compact training input')
        origin=restore_custody(evidence,args);trained=read(args.trained_receipt);saved=read(args.trained_history)
        if trained['schema']!='rheon-internal-residual-managed-v3' or trained['mode']!='train' or trained['closure']['config_sha256']!=closure['config_sha256'] or trained['closure']['sources']!=closure['sources']:
            raise ValueError('Exact originating managed CUDA source closure required')
        if any(trained['closure']['inputs'].get(n)!=v for n,v in identities.items()):raise ValueError('Static input differs from originating CUDAfit')
        if trained['closure']['inputs']['part_00']!={k:manifest['archive'][k] for k in ('bytes','sha256')}:raise ValueError('Originating compact input identity differs')
        if not 0<=trained['optimizer_updates']<=config['maximum_optimizer_iterations'] or not 1<=saved['closure_evaluations']<=config['maximum_closure_evaluations']:raise ValueError('Finite training budget identity differs')
        load_weights(model,args.trained_initial,saved['initial_inventory'])
        if inventory(model)!=config['qualified_initial_inventory']:raise ValueError('CPUrestored initial warm start differs')
        load_weights(model,args.trained_weights,saved['final_inventory']);initial=saved['initial_inventory'];final=inventory(model);probe_spec=saved['probe']
        for path,name in [(args.trained_initial,'initial-weights.npz'),(args.trained_weights,'weights.npz'),(args.trained_history,'history.json'),(args.trained_probe,'probe.npy')]:
            if streamed_identity(path)!=trained['outputs'][name]:raise ValueError('Bound GPU artifact changed')
            shutil.copyfile(path,args.output_dir/name)
        probe=np.load(args.trained_probe,allow_pickle=False)
        if probe.dtype.str!='<f8' or probe.shape!=(probe_spec['elements'],) or not np.isfinite(probe).all():raise ValueError('Complete typed originating GPU probe required')
        if torch.cuda.is_initialized():raise ValueError('CPUrestore initializedCUDA')
        execution=dict(device='cpu',model_forward_executed=False,cuda_initialized=False);updates=0;cpu_restore=True;probe_copied=True
        best_eval=saved['selected_evaluation'];best=saved['selected_full148scene_objective'];calls=saved['closure_evaluations']
    for path,name in [(args.normalization,'normalization.json'),(args.optimization_normalization,'optimization-normalization.json')]:shutil.copyfile(path,args.output_dir/name)
    outputs={p.name:streamed_identity(p) for p in args.output_dir.iterdir() if p.is_file()}
    write(args.output_dir/'receipt.json',dict(status='complete',schema='rheon-internal-residual-managed-v3',mode=args.mode,role=config['role'],parameters=PARAMETERS,
        optimizer='full_batch_LBFGS',optimizer_updates=updates,closure_evaluations=calls,closure=closure,initial_inventory=initial,final_inventory=final,
        execution=execution,outputs=outputs,originating_training_run=origin,typed_cpu_weight_restore_verified=cpu_restore,gpu_probe_copied_without_cpu_forward=probe_copied,
        full_cpu_probe_bitwise_verified=False,cpu_model_forward_executed=False,selected_evaluation=best_eval,selected_full148scene_objective=best,probe=probe_spec,
        training_scenes=148,expanded_reserved52_references_opened=0,native_accuracy_claim=False,latency_claim=False,scientific_acceptance_or_promotion=False,elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('train','restore'))
    for n in ('config','manifest','normalization','optimization-normalization','runtime-manifest','input-evidence','output-dir','teacher-code'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--shards',nargs='*',type=Path,default=[])
    for n in ('trained-weights','trained-initial','trained-history','trained-probe','trained-receipt'):parser.add_argument('--'+n,type=Path)
    run(parser.parse_args())
