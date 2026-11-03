"""Managed CUDA-only native supervision and exact typed CPU artifact restore."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from shared_spectral_model_v1 import SharedSpectralResponse,objective
from managed_fit_artifacts import validate_inputs,restore_custody


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def inventory(model):
    result={}
    for name,value in model.state_dict().items():
        array=value.detach().cpu().numpy()
        if array.dtype.str!='<f8':raise ValueError('Exact double learned parameters required')
        result[name]=dict(dtype=array.dtype.str,shape=list(array.shape),sha256=hashlib.sha256(array.tobytes()).hexdigest())
    return result

def save(model,path):np.savez(path,**{name:value.detach().cpu().numpy() for name,value in model.state_dict().items()})
def load(model,path,expected):
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(model.state_dict()) or set(expected)!=set(archive.files):raise ValueError('Complete typed parameter inventory required')
        values={}
        for name,current in model.state_dict().items():
            value=archive[name]
            if (value.dtype.str!='<f8' or list(value.shape)!=expected[name]['shape'] or tuple(value.shape)!=tuple(current.shape)
                    or hashlib.sha256(value.tobytes()).hexdigest()!=expected[name]['sha256']):raise ValueError('Typed learned parameter identity changed')
            values[name]=torch.from_numpy(value.copy())
    model.load_state_dict(values,strict=True)
    if inventory(model)!=expected:raise ValueError('Exact typed restore identity failed')


def run(args):
    began=time.perf_counter();config=read(args.config)
    device='cuda' if args.mode=='train' else 'cpu'
    if os.environ.get('TULDOK_TORCH_DEVICE')!=device:raise ValueError('Actual managed device differs')
    if (config['identifier']!='expanded-shared-spectral-01' or config['optimizer_calls']!=160
            or config['objective']!='equal_scene_native_response_residual_v1' or config['seed']!=2026102071):
        raise ValueError('Exact registered finite component fit required')
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):raise ValueError('Empty managed output directory required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    evidence,identities=validate_inputs(args);manifest=read(args.manifest);runtime=read(args.runtime_manifest)
    if runtime['packages']['torch']!=torch.__version__ or runtime['packages']['numpy']!=np.__version__:raise ValueError('Qualified managed runtime versions required')
    if (manifest['status']!='verified_all148train_fixed_shared_spectral_view' or manifest['training_scenes']!=148
            or manifest['development_or_reserved_used'] or len(args.shards)!=1):raise ValueError('All148train-only supervised view required')
    if sha(args.normalization)!=manifest['normalization_sha256'] or sha(args.shards[0])!=manifest['parts'][0]['sha256']:
        raise ValueError('Typed native view identity changed')
    closure=dict(config_sha256=sha(args.config),inputs=identities,
        sources={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},role=config['role'])
    model=SharedSpectralResponse(device=device);parameters=sum(p.numel() for p in model.parameters())
    if parameters!=4:raise ValueError('Exactly four learned coefficients required')
    origin=None
    if args.mode=='train':
        initial=inventory(model);save(model,args.output_dir/'initial-weights.npz')
        if any(p.device.type!='cuda' for p in model.parameters()):raise ValueError('Actual managed CUDA learning required')
        with np.load(args.shards[0],allow_pickle=False) as archive:
            if set(archive.files)!=set(manifest['arrays']):raise ValueError('Complete source arrays required')
            batch={}
            for name,spec in manifest['arrays'].items():
                value=archive[name]
                if value.dtype.str!=spec['dtype'] or list(value.shape)!=spec['shape'] or not np.isfinite(value).all():raise ValueError('Exact finite supervised input required')
                batch[name]=torch.from_numpy(value.copy()).to('cuda')
        if int(batch['scene_index'].unique().numel())!=148:raise ValueError('Every expanded train scene must contribute')
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=1,max_eval=25,
            tolerance_grad=1e-24,tolerance_change=1e-30,history_size=20,line_search_fn='strong_wolfe')
        torch.cuda.reset_peak_memory_stats();history=[];durations=[];changed_steps=0;closure_calls=0;started=time.time()
        def calculate():
            nonlocal closure_calls
            optimizer.zero_grad(set_to_none=True);loss=objective(model,batch)
            if not bool(loss.isfinite()):raise ValueError('Nonfinite managed CUDA supervised loss')
            loss.backward();closure_calls+=1
            if any(p.grad is None or p.grad.device.type!='cuda' or not bool(p.grad.isfinite().all()) for p in model.parameters()):
                raise ValueError('Finite CUDA gradients required')
            return loss
        for call in range(1,config['optimizer_calls']+1):
            if time.perf_counter()-began>config['wall_seconds']-30:raise TimeoutError('Finite managed training deadline reached')
            previous=model.coefficients.detach().clone();tick=time.perf_counter();optimizer.step(calculate)
            if not bool(model.coefficients.isfinite().all()):raise ValueError('Nonfinite learned response')
            if not torch.equal(previous,model.coefficients):changed_steps+=1
            torch.cuda.synchronize();durations.append(time.perf_counter()-tick)
            if call==1 or call%10==0 or call==config['optimizer_calls']:
                with torch.no_grad():value=float(objective(model,batch))
                history.append(dict(optimizer_call=call,objective=value,closure_calls=closure_calls))
                print(json.dumps(dict(format='tuldok_progress_v1',stage='shared-native-spectral-response',completed=call,total=config['optimizer_calls'],metrics={'loss':value})),flush=True)
        model.eval().requires_grad_(False);final=inventory(model);save(model,args.output_dir/'weights.npz')
        with torch.inference_mode():probe=model(batch['features']).cpu().numpy().copy()
        if probe.dtype.str!='<f8' or probe.shape!=(manifest['mode_rows'],) or not np.isfinite(probe).all():raise ValueError('Complete CUDA response probe required')
        np.save(args.output_dir/'probe.npy',probe,allow_pickle=False)
        write(args.output_dir/'history.json',dict(history=history,initial_inventory=initial,final_inventory=final,
            training_started_unix=started,training_finished_unix=time.time(),optimizer_updates=config['optimizer_calls'],
            optimizer_step_calls=config['optimizer_calls'],actual_parameter_changed_steps=changed_steps,closure_evaluations=closure_calls,
            per_update_seconds=durations,training_scene_count=148,train_only=True))
        execution=dict(device='cuda',cuda_initialized=True,gpu=torch.cuda.get_device_name(),dtype='float64',tf32=False,amp=False,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),all_learning_and_forward_on_cuda=True)
        updates=config['optimizer_calls'];cpu_restore=False;probe_copied=False
    else:
        origin=restore_custody(evidence,args);trained=read(args.trained_receipt);saved=read(args.trained_history)
        if trained['mode']!='train' or trained['optimizer_updates']!=config['optimizer_calls'] or trained['closure']!=closure:
            raise ValueError('Exact originating managed CUDA closure required')
        initial=saved['initial_inventory'];load(model,args.trained_initial,initial);load(model,args.trained_weights,saved['final_inventory']);final=inventory(model)
        for path,name in [(args.trained_initial,'initial-weights.npz'),(args.trained_weights,'weights.npz'),(args.trained_history,'history.json'),(args.trained_probe,'probe.npy')]:
            if sha(path)!=trained['outputs'][name]['sha256']:raise ValueError('Originating CUDA artifact identity changed')
            shutil.copyfile(path,args.output_dir/name)
        probe=np.load(args.trained_probe,allow_pickle=False)
        if probe.dtype.str!='<f8' or probe.shape!=(manifest['mode_rows'],) or not np.isfinite(probe).all():raise ValueError('Typed originating CUDA probe required')
        execution=dict(device='cpu',cuda_initialized=torch.cuda.is_initialized(),dtype='float64',model_forward_executed=False)
        if execution['cuda_initialized']:raise ValueError('CPU weight restore initializedCUDA')
        updates=0;cpu_restore=True;probe_copied=True
    shutil.copyfile(args.normalization,args.output_dir/'normalization.json')
    outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in args.output_dir.iterdir() if p.is_file()}
    write(args.output_dir/'receipt.json',dict(status='complete',schema='rheon-shared-spectral-managed-result-v1',mode=args.mode,role=config['role'],
        optimizer_updates=updates,parameters=parameters,trainable_parameters=parameters,closure=closure,initial_inventory=initial,final_inventory=final,
        execution=execution,outputs=outputs,originating_training_run=origin,typed_cpu_weight_restore_verified=cpu_restore,
        full_cpu_probe_bitwise_verified=False,gpu_probe_copied_without_cpu_forward=probe_copied,
        probe_scope='Learned response on every535394native training feature row, generated onCUDA; CPU restore copies the bound GPU probe without a forward pass.',
        cpu_model_forward_executed=False,causal_full_state_generator=False,scientific_acceptance_or_promotion=False,elapsed_s=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('train','restore'))
    for name in ('config','manifest','normalization','runtime-manifest','input-evidence','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shards',nargs='+',type=Path,required=True)
    for name in ('trained-weights','trained-initial','trained-history','trained-probe','trained-receipt'):parser.add_argument('--'+name,type=Path)
    run(parser.parse_args())
