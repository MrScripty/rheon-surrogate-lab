"""Managed neural-head calibration with immutable fine weights and exact CPU restore."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import model
from head_objective import freeze_fine,head_name,loss
from head_utils import digest,read,write,inventory,save_weights,load_weights,restore_custody

STATIC=('teacher','teacher_manifest','initial_weights','initial_inventory','normalization','runtime_manifest')


def construct(args,device):
    config=read(args.config);context=read(Path(__file__).with_name('config.json'))
    if config['schema']!='rheon-neural-tail-calibration-v1'or not 1<=config['updates']<=10000 or config['batch_size']!=256:
        raise ValueError('Bounded registered head calibration required')
    if context['training']['updates']!=config['updates']:raise ValueError('Model and calibration update clocks differ')
    model.configure(device);torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    initial=read(args.initial_inventory)
    if digest(Path(args.initial_weights).read_bytes())!=initial['weights_sha256']or digest(Path(args.normalization).read_bytes())!=initial['normalization_sha256']:
        raise ValueError('Exact selected-parent plus seeded-tail initialization and normalization required')
    model.select(context);network=model.PressureCorrectionNet().to(device)
    load_weights(network,args.initial_weights,initial['inventory']);freeze_fine(network)
    manifest=read(args.teacher_manifest);raw=Path(args.teacher).read_bytes()
    if manifest['status']!='generated_offline_teacher_only'or digest(raw)!=manifest['archive_sha256']:
        raise ValueError('Qualified offline teacher identity differs')
    with np.load(args.teacher,allow_pickle=False)as source:arrays={name:source[name].copy()for name in source.files}
    if set(arrays)!=set(manifest['inventory']):raise ValueError('Teacher fields differ')
    for name,value in arrays.items():
        spec=manifest['inventory'][name]
        if value.dtype.str!=spec['dtype']or list(value.shape)!=spec['shape']or digest(value.tobytes())!=spec['sha256']:
            raise ValueError('Full teacher array identity differs')
    features=torch.from_numpy(arrays['features']).to(device);target=torch.from_numpy(arrays['log_gain_fraction']).to(device)
    small=np.flatnonzero((arrays['split']==0)&(arrays['tier_small']==1));large=np.flatnonzero((arrays['split']==0)&(arrays['tier_small']==0))
    development=np.flatnonzero(arrays['split']==1)
    if not len(small)or len(large)or not len(development):raise ValueError('Whole-geometry small-grid-only teacher splits required')
    return config,network,initial,features,target,small,large,development


def probe(network,normalization):
    generator=torch.Generator(device='cpu').manual_seed(2026101951);values=[]
    with torch.inference_mode():
        for shape in ((16,16,16),(11,11,9)):
            residual=torch.randn((1,1,*shape),dtype=torch.float64,generator=generator)*1e-5
            residual=residual-residual.mean()
            sample=dict(base=torch.zeros_like(residual),native_input_residual=residual,
                cell_size_m=torch.tensor([1/16],dtype=torch.float64),rho=torch.tensor([1000.],dtype=torch.float64),
                dt=torch.tensor([1/60],dtype=torch.float64))
            pressure=model.predict_pressure(network,sample,normalization)
            if not bool(torch.isfinite(pressure).all())or pressure[0,0,0,0,0]!=0:raise ValueError('Actual CPU pressure probe contract failed')
            values.append(pressure.numpy().reshape(-1).copy())
    return np.concatenate(values)


def run(args):
    started=time.perf_counter();device='cuda'if args.mode=='train'else'cpu'
    if os.environ.get('TULDOK_TORCH_DEVICE')!=device:raise ValueError('Managed device differs')
    if not args.output_dir.is_dir()or any(args.output_dir.iterdir()):raise ValueError('Empty managed output required')
    evidence=read(args.input_evidence);expected=set(STATIC)|({'trained_weights','trained_initial','trained_history','trained_probe','trained_receipt'}if args.mode=='restore'else set())
    if set(evidence)!=expected:raise ValueError('Closed fitting/restore inputs required')
    owners=set();inputs={}
    for slot in STATIC:
        path=Path(getattr(args,slot));raw=path.read_bytes();observed=evidence[slot]
        if observed['reference_id']!=slot or observed['measured_sha256']!=digest(raw):raise ValueError('Typed static input identity differs')
        owners.add((observed['project_id'],observed['manifest_sha256']));inputs[slot]=dict(bytes=len(raw),sha256=digest(raw))
    if len(owners)!=1:raise ValueError('Mixed Project static inputs')
    runtime=read(args.runtime_manifest)
    if runtime['profile']!='managed_torch_python_v1'or runtime['packages']['torch']!=torch.__version__ or runtime['packages']['numpy']!=np.__version__:
        raise ValueError('Approved runtime versions differ')
    config,network,initial,features,target,small,large,development=construct(args,device)
    initial_inventory=inventory(network);fine_initial={k:v for k,v in initial_inventory.items()if not head_name(k)}
    closure=dict(source_witness_sha256=digest(Path(__file__).with_name('source-witness.json').read_bytes()),
        experiment_sha256=digest(Path(__file__).with_name('config.json').read_bytes()),calibration_sha256=digest(Path(args.config).read_bytes()),
        inputs=inputs,sources={p.name:digest(p.read_bytes())for p in sorted(Path(__file__).parent.glob('*.py'))})
    trainable=[p for p in network.parameters()if p.requires_grad];parameters=sum(p.numel()for p in network.parameters())
    if parameters!=128938 or sum(p.numel()for p in trainable)!=33982:raise ValueError('Selected frozen parent and all33982 tail parameters required')
    output=args.output_dir;setup_seconds=time.perf_counter()-started
    if args.mode=='train':
        torch.cuda.reset_peak_memory_stats();optimizer=torch.optim.Adam(trainable,lr=config['learning_rate'])
        rng=np.random.Generator(np.random.PCG64(config['sampling_seed']));sample_hash=__import__('hashlib').sha256()
        history=[];step_times=[];train_started=time.time()
        for update in range(1,config['updates']+1):
            if time.perf_counter()-started>config['wall_seconds']-30:raise TimeoutError('Bounded head fit deadline')
            for change in config['schedule']:
                if update==change['update']:
                    for group in optimizer.param_groups:group['lr']=change['learning_rate']
            indices=rng.choice(small,256,replace=True).astype('<i8')
            sample_hash.update(indices.tobytes());selected=torch.from_numpy(indices).to(device)
            tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            objective=loss(network,features[selected],target[selected])
            if not bool(torch.isfinite(objective)):raise ValueError('Nonfinite head objective')
            objective.backward()
            if any(p.grad is None or p.grad.device.type!='cuda'or not bool(torch.isfinite(p.grad).all())for p in trainable):
                raise ValueError('Every trainable head tensor needs a finite CUDA gradient')
            if any(p.grad is not None for name,p in network.named_parameters()if not head_name(name)):
                raise ValueError('Frozen fine network acquired a gradient')
            torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step();torch.cuda.synchronize()
            if any(not bool(torch.isfinite(p).all())or optimizer.state[p]['exp_avg'].device.type!='cuda'for p in trainable):
                raise ValueError('Head parameter or Adam moment ownership differs')
            step_times.append(time.perf_counter()-tick)
            if update==1 or update%1000==0 or update==config['updates']:
                with torch.no_grad():development_loss=float(loss(network,features[development],target[development]))
                history.append(dict(update=update,training_loss=float(objective.detach()),teacher_development_loss=development_loss))
                print(__import__('json').dumps(dict(format='tuldok_progress_v1',stage='neural-head-calibration',completed=update,total=config['updates'],metrics=dict(loss=float(objective.detach())))),flush=True)
        execution=dict(device='cuda',cuda_initialized=True,gpu=torch.cuda.get_device_name(0),dtype='float32',tf32=False,amp=False,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        network=network.cpu().eval();final_inventory=inventory(network)
        if {k:v for k,v in final_inventory.items()if not head_name(k)}!=fine_initial:raise ValueError('Fine C107 tensor bytes changed')
        shutil.copyfile(args.initial_weights,output/'initial-weights.npz');save_weights(network,output/'weights.npz')
        checkpoints={}
        for update,source_name in ((0,'initial-weights.npz'),(config['updates'],'weights.npz')):
            name=f'checkpoint-{update:03d}.npz';shutil.copyfile(output/source_name,output/name)
            checkpoints[name]=dict(update=update,inventory=initial_inventory if update==0 else final_inventory,
                bytes=(output/name).stat().st_size,sha256=digest((output/name).read_bytes()))
        write(output/'history.json',dict(history=history,initial_inventory=initial_inventory,final_inventory=final_inventory,checkpoints=checkpoints,
            fine_tensors_bitwise_unchanged=True,trainable_parameters=33982,frozen_parameters=94956,
            optimizer_work=dict(head_forwards=config['updates'],fine_spatial_forwards_during_fitting=0,teacher_examples=256*config['updates']),
            sampled_geometry_order_sha256=sample_hash.hexdigest(),per_update_seconds=step_times,setup_seconds=setup_seconds,
            training_started_unix=train_started,training_finished_unix=time.time(),physical_reference_fields_used=False))
        origin=None;optimizer_updates=config['updates']
    else:
        origin=restore_custody(evidence,args);trained=read(args.trained_receipt);saved=read(args.trained_history)
        if trained['mode']!='train'or trained['closure']!=closure or trained['optimizer_updates']!=config['updates']:
            raise ValueError('Exact trained head closure required')
        if saved['initial_inventory']!=initial_inventory:raise ValueError('C107 initialization differs on CPU restore')
        load_weights(network,args.trained_initial,initial_inventory);load_weights(network,args.trained_weights,saved['final_inventory'])
        network.eval();final_inventory=inventory(network)
        if {k:v for k,v in final_inventory.items()if not head_name(k)}!=fine_initial:raise ValueError('Restored fine C107 tensors changed')
        for path,name in ((args.trained_weights,'weights.npz'),(args.trained_initial,'initial-weights.npz'),(args.trained_history,'history.json')):
            shutil.copyfile(path,output/name)
        for update,source_name in ((0,'initial-weights.npz'),(config['updates'],'weights.npz')):
            name=f'checkpoint-{update:03d}.npz';shutil.copyfile(output/source_name,output/name)
            if digest((output/name).read_bytes())!=saved['checkpoints'][name]['sha256']:raise ValueError('Restored snapshot alias differs')
        optimizer_updates=0;execution=dict(device='cpu',cuda_initialized=torch.cuda.is_initialized(),dtype='float32')
        if execution['cuda_initialized']:raise ValueError('CPU restore initialized CUDA')
    current_probe=probe(network,read(args.normalization))
    if args.mode=='restore'and not np.array_equal(current_probe.view('u1'),np.load(args.trained_probe,allow_pickle=False).view('u1')):
        raise ValueError('Fresh CPU pressure probes differ')
    np.save(output/'probe.npy',current_probe,allow_pickle=False);shutil.copyfile(args.normalization,output/'normalization.json')
    outputs={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes()))for p in output.iterdir()if p.is_file()}
    receipt=dict(schema='rheon-neural-head-training-result-v1',status='complete',mode=args.mode,optimizer_updates=optimizer_updates,
        task='pressure_tail_calibration',pressure_stages=2,learned_global_blocks=3,parameters=parameters,trainable_parameters=33982,architecture=model.ARCHITECTURE,
        closure=closure,initial_inventory=initial_inventory,final_inventory=final_inventory,execution=execution,outputs=outputs,
        full_cpu_probe_bitwise_verified=args.mode=='restore',originating_training_run=origin,
        fine_tensors_bitwise_unchanged=True,candidate_numerical_pressure_solve=False,
        elapsed_s=time.perf_counter()-started,scientific_acceptance_or_promotion=False)
    write(output/'receipt.json',receipt)
    print(__import__('json').dumps(dict(status='completed',mode=args.mode,optimizer_updates=optimizer_updates,parameters=parameters)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['train','restore'])
    for name in ('config',*STATIC,'input_evidence','output_dir'):
        parser.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    for name in ('trained_weights','trained_initial','trained_history','trained_probe','trained_receipt'):
        parser.add_argument('--'+name.replace('_','-'),dest=name,type=Path)
    run(parser.parse_args())
