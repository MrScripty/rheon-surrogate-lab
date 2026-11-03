"""Managed causal request component fit; future reference requests are labels only."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from managed_fit_artifacts import digest,read,write,inventory,save_weights,load_weights,validate_inputs,restore_custody
from pressure_request_model import CausalRequestNet
from pressure_request_data import load,features,inputs
from pressure_request_objective import loss


def construct(args,device):
    config=read(args.config)
    if config['schema']!='rheon-causal-request-fit-v1'or config['horizons']!=1 or config['updates']!=10000:
        raise ValueError('Registered bounded K1 component fit required')
    if config['seed']!=2026101994 or config['sampling_seed']!=2026101995:
        raise ValueError('Registered causal request seeds required')
    if not config['component_only']or config['future_request_input']or config['current_pressure_input']:
        raise ValueError('Causal component contract changed')
    torch.manual_seed(config['seed'])
    manifest,norm,parents=load(args.manifest,args.normalization,args.shards,device)
    network=CausalRequestNet(config['horizons'],norm['scales']['velocity'],norm['scales']['q']).to(device)
    caches=[features(parent,norm['scales'],config['horizons'],device)for parent in parents]
    return config,manifest,norm,parents,caches,network


def sample_window(rng):
    parent=int(rng.integers(16));bucket=float(rng.random())
    step=0 if bucket<.25 else int(rng.integers(1,9))if bucket<.5 else int(rng.integers(9,121))
    return parent,step


def forward_objective(network,parent,cache,step):
    predicted=network(*inputs(parent,cache,step,network.horizons,next(network.parameters()).device))
    target=parent['q'][step+1:step+network.horizons+1].unsqueeze(0)
    return loss(predicted,target)


def CPU_probe(network,parents,norm):
    network.eval().cpu();pieces=[]
    with torch.inference_mode():
        for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
            parent=next(parent for parent in parents if parent['scene']['counts']==counts)
            current=dict(parent,velocity=parent['velocity'].cpu(),q=None)
            cache=features(current,norm['scales'],network.horizons,'cpu')
            for step in (0,8,120):
                pieces.append(network(*inputs(current,cache,step,network.horizons,'cpu')).numpy().reshape(-1).copy())
    return np.concatenate(pieces).astype('<f4',copy=False)


def run(args):
    started=time.perf_counter();device='cuda'if args.mode=='train'else'cpu'
    if os.environ.get('TULDOK_TORCH_DEVICE')!=device:raise ValueError('Managed runtime device differs')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not args.output_dir.is_dir()or any(args.output_dir.iterdir()):raise ValueError('Empty managed output directory required')
    evidence,input_identities=validate_inputs(args)
    config,manifest,norm,parents,caches,network=construct(args,device)
    runtime=read(args.runtime_manifest)
    if runtime['profile']!='managed_torch_python_v1' or runtime['packages']['torch']!=torch.__version__ or runtime['packages']['numpy']!=np.__version__:
        raise ValueError('Approved Torch/NumPy runtime identity differs')
    sources={path.name:digest(path.read_bytes())for path in sorted(Path(__file__).parent.glob('*.py'))}
    closure=dict(config_sha256=digest(Path(args.config).read_bytes()),inputs=input_identities,sources=sources,
        horizons=config['horizons'],normalization_sha256=digest(Path(args.normalization).read_bytes()),task='causal_pressure_request_component')
    initial=inventory(network);parameters=sum(p.numel()for p in network.parameters());output=args.output_dir
    setup_seconds=time.perf_counter()-started
    if device=='cuda':
        optimizer=torch.optim.Adam(network.parameters(),lr=config['learning_rate'])
        rng=np.random.Generator(np.random.PCG64(config['sampling_seed']))
        save_weights(network,output/'initial-weights.npz');save_weights(network,output/'checkpoint-000.npz')
        torch.cuda.reset_peak_memory_stats();history=[];samples=[];step_times=[]
        work=dict(network_forwards=0,predicted_request_fields=0,input_cells=0)
        training_started=time.time()
        for update in range(1,config['updates']+1):
            if time.perf_counter()-started>config['wall_seconds']-30:raise TimeoutError('Bounded component fitting deadline')
            for change in config['schedule']:
                if update==change['update']:
                    for group in optimizer.param_groups:group['lr']=change['learning_rate']
            parent_index,step=sample_window(rng);parent=parents[parent_index]
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.perf_counter()
            objective=forward_objective(network,parent,caches[parent_index],step)
            if not bool(torch.isfinite(objective)):raise ValueError('Nonfinite request objective')
            objective.backward()
            if any(p.grad is None or p.grad.device.type!='cuda'or not bool(torch.isfinite(p.grad).all())for p in network.parameters()):
                raise ValueError('Incomplete finite CUDA gradient ownership')
            torch.nn.utils.clip_grad_norm_(network.parameters(),config['gradient_norm_cap']);optimizer.step()
            if any(not bool(torch.isfinite(p).all())or optimizer.state[p]['exp_avg'].device.type!='cuda'for p in network.parameters()):
                raise ValueError('Nonfinite component weights or misplaced Adam moments')
            torch.cuda.synchronize();step_times.append(time.perf_counter()-tick);samples.append((parent_index,step))
            work['network_forwards']+=1;work['predicted_request_fields']+=config['horizons']
            work['input_cells']+=int(np.prod(parent['velocity'].shape[2:]))
            if update in (2000,10000):save_weights(network,output/f'checkpoint-{update:03d}.npz')
            if update==1 or update%100==0:
                value=float(objective.detach());history.append(dict(update=update,loss=value))
                print(__import__('json').dumps(dict(format='tuldok_progress_v1',stage='causal-pressure-request-fitting',
                    completed=update,total=config['updates'],metrics=dict(loss=value))),flush=True)
        training_finished=time.time();final=inventory(network);save_weights(network,output/'weights.npz')
        with np.load(output/'checkpoint-10000.npz',allow_pickle=False)as snapshot:
            if set(snapshot.files)!=set(final)or any(digest(snapshot[name].tobytes())!=spec['sha256']for name,spec in final.items()):
                raise ValueError('Final snapshot tensors differ from published model')
        device_evidence=dict(device='cuda',cuda_initialized=torch.cuda.is_initialized(),gpu=torch.cuda.get_device_name(0),dtype='float32',
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            optimizer='Adam on CUDA',all_optimizer_moments_cuda=True)
        write(output/'history.json',dict(schema='rheon-causal-request-history-v1',initial_inventory=initial,final_inventory=final,
            history=history,updates=config['updates'],sample_order_sha256=digest(np.asarray(samples,dtype='<i8').tobytes()),
            per_update_seconds=step_times,optimizer_work=work,valid_parent_count=16,train_only=True,
            setup_seconds=setup_seconds,training_started_unix=training_started,training_finished_unix=training_finished,
            component_only=True,own_history_supported=False,future_request_input=False,
            sampling='25percentstep0,25percentsteps1..8,50percentsteps9..120',
            analytical_pressure_inverse='Offline training loss only'))
        origin=None;updates=config['updates']
    else:
        origin=restore_custody(evidence,args);trained=read(args.trained_receipt);saved=read(args.trained_history)
        if trained['mode']!='train'or trained['optimizer_updates']!=config['updates']or trained['closure']!=closure:
            raise ValueError('Exact trained component closure required')
        if initial!=saved['initial_inventory']:raise ValueError('CPU seeded component initialization differs')
        load_weights(network,args.trained_initial,initial);load_weights(network,args.trained_weights,saved['final_inventory'])
        final=inventory(network)
        for source,name in ((args.trained_weights,'weights.npz'),(args.trained_initial,'initial-weights.npz'),(args.trained_history,'history.json')):
            shutil.copyfile(source,output/name)
        updates=0;device_evidence=dict(device='cpu',cuda_initialized=torch.cuda.is_initialized(),dtype='float32')
        if device_evidence['cuda_initialized']:raise ValueError('CPU restoration initialized CUDA')
    probe=CPU_probe(network,parents,norm)
    if args.mode=='restore':
        reference=np.load(args.trained_probe,allow_pickle=False)
        if not np.array_equal(probe.view('<u4'),reference.view('<u4')):raise ValueError('Independent actual CPU component probe differs')
    np.save(output/'probe.npy',probe,allow_pickle=False);shutil.copyfile(args.normalization,output/'normalization.json')
    outputs={path.name:dict(bytes=path.stat().st_size,sha256=digest(path.read_bytes()))for path in output.iterdir()if path.is_file()}
    write(output/'receipt.json',dict(schema='rheon-causal-request-training-result-v1',status='complete',mode=args.mode,
        optimizer_updates=updates,parameters=parameters,horizons=config['horizons'],closure=closure,
        initial_inventory=initial,final_inventory=final,execution=device_evidence,outputs=outputs,
        full_cpu_probe_bitwise_verified=args.mode=='restore',originating_training_run=origin,
        elapsed_s=time.perf_counter()-started,component_only=True,full_state_temporal_candidate=False,
        candidate_numerical_pressure_solve=False,scientific_acceptance_or_promotion=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','restore'])
    for name in ('config','manifest','normalization','runtime-manifest','input-evidence','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shards',nargs='+',type=Path,required=True)
    for name in ('trained-weights','trained-initial','trained-history','trained-probe','trained-receipt'):
        parser.add_argument('--'+name,type=Path)
    run(parser.parse_args())
