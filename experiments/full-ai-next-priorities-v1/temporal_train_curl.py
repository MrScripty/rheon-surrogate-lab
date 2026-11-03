"""Standalone managed CUDA temporal fitting and independently verified CPU restore."""
import argparse
from functools import partial
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from bundle_model_curl import BundleNet
from temporal_data import load,make_features,feature_at
from temporal_objective_increment import loss as supervised_loss
from temporal_feedback import next_input

def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def inventory(network):
    return {name:dict(shape=list(value.shape),dtype='<f4',sha256=digest(value.detach().cpu().numpy().astype('<f4',copy=False).tobytes()))
            for name,value in network.state_dict().items()}

def save_weights(network,path):
    np.savez(path,**{name:value.detach().cpu().numpy().astype('<f4',copy=False)for name,value in network.state_dict().items()})

def load_weights(network,path,expected):
    with np.load(path,allow_pickle=False)as source:
        if set(source.files)!=set(network.state_dict()):raise ValueError('Complete parameter inventory required')
        values={}
        for name,current in network.state_dict().items():
            value=source[name]
            if value.dtype.str!='<f4'or tuple(value.shape)!=tuple(current.shape):raise ValueError('Parameter shape/dtype changed')
            if digest(value.tobytes())!=expected[name]['sha256']:raise ValueError('Parameter bytes changed')
            values[name]=torch.from_numpy(value.copy())
    network.load_state_dict(values,strict=True)
    if inventory(network)!=expected:raise ValueError('Loaded parameter inventory differs')

def validate_inputs(args):
    evidence=read(args.input_evidence)
    static={'manifest':args.manifest,'normalization':args.normalization,'runtime_manifest':args.runtime_manifest,
            **{f'part_{index:02d}':path for index,path in enumerate(args.shards)}}
    owners=set();identities={}
    expected=set(static)|({'trained_weights','trained_initial','trained_history','trained_probe','trained_receipt'}if args.mode=='restore'else set())
    if set(evidence)!=expected:raise ValueError('Unexpected or missing typed fitting/restore inputs')
    for slot,path in static.items():
        row=evidence[slot];raw=Path(path).read_bytes();measured=digest(raw)
        if row['reference_id']!=slot or row['measured_sha256']!=measured:raise ValueError('Typed static input identity failed')
        owners.add((row['project_id'],row['manifest_sha256']));identities[slot]=dict(bytes=len(raw),sha256=measured)
    if len(owners)!=1:raise ValueError('Mixed Project inputs rejected')
    return evidence,identities

def restore_custody(evidence,args):
    path=Path(__file__).with_name('custody_validator.py')
    spec=importlib.util.spec_from_file_location('temporal_custody_validator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    owner=None
    expected={'trained_weights':('weights.npz','checkpoint'),'trained_initial':('initial-weights.npz','checkpoint'),
              'trained_history':('history.json','training_diagnostics'),'trained_probe':('probe.npy','predictions'),
              'trained_receipt':('receipt.json','training_result')}
    for slot in ('trained_weights','trained_initial','trained_history','trained_probe','trained_receipt'):
        raw=Path(getattr(args,slot)).read_bytes();row=evidence[slot]
        if row['kind']!='run_artifact':raise ValueError('CPU restore requires managed Run artifacts')
        observed=module.validate_artifact_source(row['source'],digest(raw),len(raw))
        if (observed['artifact_name'],observed['artifact_role'])!=expected[slot]:raise ValueError('Restore artifact role differs')
        key=tuple(observed[k]for k in ('catalog_id','run_id','attempt_id','request_sha256','material_sha256'))
        if owner is not None and owner!=key:raise ValueError('Mixed training Run artifacts')
        owner=key
    return dict(zip(('catalog_id','run_id','attempt_id','request_sha256','material_sha256'),owner))

def construct_network(config,device,statistics_path=None):
    path=Path(statistics_path)if statistics_path is not None else Path(__file__).with_name('step-scales.json')
    if digest(path.read_bytes())!=config['step_scale_statistics_sha256']:
        raise ValueError('Fixed training-only step statistics changed')
    stats=read(path)
    if stats['status']!='verified_train_only_statistics'or stats['development_or_reserved_used']or stats['training_parent_count']!=16:
        raise ValueError('Verified train-only step scales required')
    if stats['original_normalization_sha256']!=config['original_normalization_sha256']:
        raise ValueError('Physical state normalization changed')
    return BundleNet(config['horizons'],config['width'],config['architecture'],
        forcing_scale=stats['normalized_prescribed_force_rms'],increment_scales=stats['normalized_increment_rms']).to(device)


def construct_objective(config,network):
    mode=config['supervision_mode']
    if mode not in ('full_state','increments'):raise ValueError('Registered MSE supervision mode required')
    return partial(supervised_loss,increment_scales=network.increment_scales,mode=mode)


def run(args):
    started=time.perf_counter();config=read(args.config);device='cuda'if args.mode=='train'else'cpu'
    if os.environ.get('TULDOK_TORCH_DEVICE')!=device:raise ValueError('Managed runtime device differs')
    if config.get('feedback_protocol')!=2:raise ValueError('Deployment-equivalent feedback protocol required')
    if not 1<=config['updates']<=10000 or config['horizons']not in (1,2,3,4):raise ValueError('Finite registered fit required')
    torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output=args.output_dir
    if not output.is_dir()or any(output.iterdir()):raise ValueError('Empty managed output directory required')
    evidence,inputs=validate_inputs(args)
    runtime=read(args.runtime_manifest)
    if runtime['profile']!='managed_torch_python_v1' or runtime['packages']['torch']!=torch.__version__ or runtime['packages']['numpy']!=np.__version__:
        raise ValueError('Approved Torch and NumPy runtime versions differ')
    manifest,norm,parents=load(args.manifest,args.normalization,args.shards,device)
    if len(parents)!=16:raise ValueError('All16 training scenes required')
    sources={p.name:digest(p.read_bytes())for p in sorted(Path(__file__).parent.glob('*.py'))}
    closure=dict(config_sha256=digest(Path(args.config).read_bytes()),inputs=inputs,sources=sources,
                 horizons=config['horizons'],normalization_sha256=digest(Path(args.normalization).read_bytes()))
    network=construct_network(config,device)
    loss=construct_objective(config,network)
    initial=inventory(network);parameters=sum(p.numel()for p in network.parameters())
    horizon=config['horizons'];features=[make_features(parent,norm['scales'],horizon,device)for parent in parents]
    setup_seconds=time.perf_counter()-started
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats();optimizer=torch.optim.Adam(network.parameters(),lr=config['learning_rate'])
        rng=np.random.Generator(np.random.PCG64(config['sampling_seed']))
        history=[];samples=[];step_times=[]
        work=dict(network_forwards=0,encoder_cells=0,predicted_physical_states=0,encoder_calls=0,latent_updates=0,endpoint_predictions=0,intermediate_states=0,enabled=False)
        def count_forward(module,arguments):
            if work['enabled']:
                state=arguments[0];work['network_forwards']+=1
                work['encoder_cells']+=state.shape[0]*math.prod(state.shape[2:])
                work['predicted_physical_states']+=state.shape[0]*horizon
                work['encoder_calls']+=1
                work['latent_updates']+=(horizon-1)if config['architecture']=='reuse'else 0
                work['endpoint_predictions']+=1 if config['architecture']=='endpoint'else 0
                work['intermediate_states']+=(horizon-1)if config['architecture']=='endpoint'else 0
        hook=network.register_forward_pre_hook(count_forward)
        training_started=time.time()
        for update in range(1,config['updates']+1):
            if time.perf_counter()-started>config['wall_seconds']-30:raise TimeoutError('Finite fitting deadline; no result substitution')
            parent_index=int(rng.integers(16));step=int(rng.integers(121))
            parent=parents[parent_index];target=parent['states'];counts=parent['scene']['counts']
            optimizer.zero_grad(set_to_none=True)
            for change in config['schedule']:
                if update==change['update']:
                    for group in optimizer.param_groups:group['lr']=change['learning_rate']
            torch.cuda.synchronize();tick=time.perf_counter();work['enabled']=True
            predicted=network(target[step:step+1],feature_at(features[parent_index],step,horizon))
            first_target=target[step+1:step+horizon+1].unsqueeze(0)
            objective=loss(predicted,first_target,counts,norm['scales']['tracer'])
            blocks=1
            if update>config['single_block_updates']:
                second=network(next_input(predicted,counts,norm['scales']),feature_at(features[parent_index],step+horizon,horizon))
                second_target=target[step+horizon+1:step+2*horizon+1].unsqueeze(0)
                objective=(objective+loss(second,second_target,counts,norm['scales']['tracer']))/2
                blocks=2
            if not bool(torch.isfinite(objective)):raise ValueError('Temporal objective nonfinite')
            objective.backward()
            if any(p.grad is None or p.grad.device.type!='cuda'or not bool(torch.isfinite(p.grad).all())for p in network.parameters()):
                raise ValueError('Incomplete finite CUDA gradient ownership')
            torch.nn.utils.clip_grad_norm_(network.parameters(),1.)
            optimizer.step();work['enabled']=False
            if any(not bool(torch.isfinite(p).all())for p in network.parameters()):raise ValueError('Nonfinite temporal parameters')
            if any(optimizer.state[p]['exp_avg'].device.type!='cuda'for p in network.parameters()):raise ValueError('Adam moment device differs')
            torch.cuda.synchronize();step_times.append(time.perf_counter()-tick)
            samples.append((parent_index,step,blocks))
            if update==1 or update%100==0 or update==config['updates']:
                value=float(objective.detach());history.append(dict(update=update,loss=value,blocks=blocks))
                print(json.dumps(dict(format='tuldok_progress_v1',stage='temporal-bundle-fitting',completed=update,total=config['updates'],metrics=dict(loss=value))),flush=True)
        training_finished=time.time();hook.remove();work.pop('enabled')
        device_evidence=dict(device='cuda',cuda_initialized=True,gpu=torch.cuda.get_device_name(0),dtype='float32',
            tf32=False,amp=False,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        network=network.cpu().eval();final=inventory(network)
        write(output/'history.json',dict(history=history,initial_inventory=initial,final_inventory=final,
            training_started_unix=training_started,training_finished_unix=training_finished,
            per_update_seconds=step_times,optimizer_work=work,
            sample_order_sha256=digest(np.asarray(samples,dtype='<i8').tobytes()),
            curriculum=dict(single_block_updates=config['single_block_updates'],later_blocks=2,second_block_uses_own_predicted_history=True,feedback_encoding='physical_reconstruction_then_deployed_mean_centering_v2'),
            valid_parent_count=16,train_only=True,setup_seconds=setup_seconds))
        # Keep the exact seeded initialization as a separately declared artifact.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed']);initial_network=construct_network(config,'cpu')
        if inventory(initial_network)!=initial:raise ValueError('Seeded initial inventory changed')
        save_weights(initial_network,output/'initial-weights.npz')
        save_weights(network,output/'weights.npz')
        optimizer_updates=config['updates'];origin=None
    else:
        origin=restore_custody(evidence,args);trained=read(args.trained_receipt);saved=read(args.trained_history)
        if trained['mode']!='train'or trained['optimizer_updates']!=config['updates']or trained['closure']!=closure:
            raise ValueError('Exact trained closure required')
        if initial!=saved['initial_inventory']:raise ValueError('CPU seeded initialization differs')
        load_weights(network,args.trained_initial,initial)
        load_weights(network,args.trained_weights,saved['final_inventory']);network.eval();final=inventory(network)
        for source,name in ((args.trained_weights,'weights.npz'),(args.trained_initial,'initial-weights.npz'),(args.trained_history,'history.json')):
            shutil.copyfile(source,output/name)
        optimizer_updates=0;device_evidence=dict(device='cpu',cuda_initialized=torch.cuda.is_initialized(),dtype='float32')
        if device_evidence['cuda_initialized']:raise ValueError('CPU restore initialized CUDA')
    # Both training publication and fresh restore execute the actual CPU model.
    current=parents[0]['states'][0:1].cpu()
    cpu_features=make_features(dict(parents[0],states=parents[0]['states'].cpu()),norm['scales'],horizon,'cpu')
    with torch.no_grad():probe=network(current,feature_at(cpu_features,0,horizon)).numpy().copy()
    if args.mode=='restore':
        reference=np.load(args.trained_probe,allow_pickle=False)
        if not np.array_equal(probe.view('<u4'),reference.view('<u4')):raise ValueError('Fresh CPU joint-output probe differs')
    np.save(output/'probe.npy',probe,allow_pickle=False)
    shutil.copyfile(args.normalization,output/'normalization.json')
    outputs={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes()))for p in output.iterdir()if p.is_file()}
    receipt=dict(schema='rheon-temporal-training-result-v1',status='complete',mode=args.mode,
        optimizer_updates=optimizer_updates,parameters=parameters,horizons=horizon,closure=closure,
        initial_inventory=initial,final_inventory=final,execution=device_evidence,outputs=outputs,
        full_cpu_probe_bitwise_verified=args.mode=='restore',originating_training_run=origin,
        elapsed_s=time.perf_counter()-started,scientific_acceptance_or_promotion=False)
    write(output/'receipt.json',receipt)
    print(json.dumps(dict(status='completed',mode=args.mode,updates=optimizer_updates,horizons=horizon,parameters=parameters)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','restore'])
    for name in ('config','manifest','normalization','runtime-manifest','input-evidence','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shards',nargs='+',type=Path,required=True)
    for name in ('trained-weights','trained-initial','trained-history','trained-probe','trained-receipt'):
        parser.add_argument('--'+name,type=Path)
    run(parser.parse_args())
