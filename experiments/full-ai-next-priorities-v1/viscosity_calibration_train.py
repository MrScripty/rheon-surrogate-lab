"""Managed linear viscosity calibration and typed exact CPU restoration."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from viscosity_calibration_model import ViscosityNet, objective
from managed_fit_artifacts import digest, read, write, inventory, save_weights, load_weights, restore_custody

STATIC = ('teacher', 'teacher_manifest', 'runtime_manifest')


def construct(args, device):
    config = read(args.config)
    if config['schema'] != 'rheon-viscosity-calibration-v1' or config['updates'] != 1000 or config['batch_size'] != 8:
        raise ValueError('Registered viscosity calibration required')
    torch.set_num_threads(1); torch.manual_seed(config['seed']); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    manifest = read(args.teacher_manifest)
    if manifest['status'] != 'native_qualified' or digest(Path(args.teacher).read_bytes()) != manifest['archive_sha256']:
        raise ValueError('Exact qualified viscosity teacher required')
    with np.load(args.teacher, allow_pickle=False) as data:
        arrays = {name:data[name].copy() for name in data.files}
    if set(arrays) != set(manifest['inventory']):
        raise ValueError('Teacher inventory differs')
    for name, value in arrays.items():
        spec = manifest['inventory'][name]
        if value.dtype.str != spec['dtype'] or list(value.shape) != spec['shape'] or digest(value.tobytes()) != spec['sha256']:
            raise ValueError('Teacher array identity differs')
    if manifest['train_indices'] != list(range(64)) or manifest['development_indices'] != list(range(64,80)):
        raise ValueError('Synthetic split changed')
    network = ViscosityNet().double().to(device)
    velocity = torch.from_numpy(arrays['velocity']).double().to(device)
    target = torch.from_numpy(arrays['target']).to(device)
    mask = torch.from_numpy(arrays['mask']).to(device)
    return config, network, velocity, target, mask


def errors(network, velocity, target, mask):
    with torch.no_grad():
        prediction = network(velocity)
        error = ((prediction.double()-target.double())*mask).flatten(1).norm(dim=1)
        reference = (target.double()*mask).flatten(1).norm(dim=1).clamp_min(1e-30)
        return (error/reference).cpu().numpy()


def cpu_probe(network, velocity):
    network = network.cpu().float().eval()
    with torch.inference_mode():
        value = network(velocity[:2].cpu().float()).numpy().copy()
    if not np.isfinite(value).all():
        raise ValueError('CPU viscosity probe is nonfinite')
    return value


def run(args):
    started = time.perf_counter(); device = 'cuda' if args.mode == 'train' else 'cpu'
    if os.environ.get('TULDOK_TORCH_DEVICE') != device:
        raise ValueError('Managed device differs')
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
        raise ValueError('Empty managed output required')
    evidence = read(args.input_evidence)
    expected = set(STATIC) | ({'trained_weights','trained_initial','trained_history','trained_probe','trained_receipt'} if args.mode == 'restore' else set())
    if set(evidence) != expected:
        raise ValueError('Closed typed inputs required')
    inputs = {}; owners = set()
    for slot in STATIC:
        raw = Path(getattr(args,slot)).read_bytes(); row = evidence[slot]
        if row['reference_id'] != slot or row['measured_sha256'] != digest(raw):
            raise ValueError('Typed input identity differs')
        owners.add((row['project_id'],row['manifest_sha256']))
        inputs[slot] = dict(bytes=len(raw),sha256=digest(raw))
    if len(owners) != 1:
        raise ValueError('Mixed Project inputs')
    runtime = read(args.runtime_manifest)
    if runtime['profile'] != 'managed_torch_python_v1' or runtime['packages']['torch'] != torch.__version__ or runtime['packages']['numpy'] != np.__version__:
        raise ValueError('Approved runtime differs')
    config, network, velocity, target, mask = construct(args,device)
    if sum(p.numel() for p in network.parameters()) != 243:
        raise ValueError('Point layer parameter count differs')
    closure = dict(config_sha256=digest(Path(args.config).read_bytes()), inputs=inputs,
        sources={p.name:digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*.py'))})
    initial_inventory = inventory(network); output = args.output_dir
    if args.mode == 'train':
        save_weights(network,output/'initial-weights.npz')
        optimizer = torch.optim.SGD(network.parameters(),lr=config['learning_rate'])
        rng = np.random.Generator(np.random.PCG64(config['sampling_seed']))
        sample_hash = hashlib.sha256(); history = []
        torch.cuda.reset_peak_memory_stats()
        for update in range(1,config['updates']+1):
            if time.perf_counter()-started > config['wall_seconds']-30:
                raise TimeoutError('Bounded viscosity fit deadline')
            indices = rng.choice(64,size=config['batch_size'],replace=False).astype('<i8')
            sample_hash.update(indices.tobytes()); index = torch.from_numpy(indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(network,velocity[index],target[index],mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Point objective is nonfinite')
            loss.backward()
            if any(p.grad is None or p.grad.device.type != device or not bool(torch.isfinite(p.grad).all()) for p in network.parameters()):
                raise FloatingPointError('Finite CUDA viscosity gradients required')
            optimizer.step()
            if update == 1 or update % 100 == 0:
                development = errors(network,velocity[64:],target[64:],mask)
                row = dict(update=update,loss=float(loss.detach()),worst_development_relative_error=float(development.max()))
                history.append(row)
                print(json.dumps(dict(format='tuldok_progress_v1',stage='viscosity-calibration-calibration',completed=update,total=config['updates'],metrics=row)),flush=True)
        double_errors = errors(network,velocity[64:],target[64:],mask)
        network = network.cpu().float()
        float_errors = errors(network,velocity[64:].cpu().float(),target[64:].cpu(),mask.cpu())
        save_weights(network,output/'weights.npz'); final_inventory = inventory(network)
        write(output/'history.json',dict(history=history,initial_inventory=initial_inventory,final_inventory=final_inventory,
            sampled_pattern_order_sha256=sample_hash.hexdigest(),optimizer='SGDfloat64',learning_rate=config['learning_rate'],
            worst_development_relative_error_float64=float(double_errors.max()),
            worst_development_relative_error_published_float32=float(float_errors.max()),
            coefficient_accuracy_pass=bool(double_errors.max() <= config['double_tolerance'] and float_errors.max() <= config['published_tolerance']),
            optimizer_updates=config['updates'],physical_states_predicted=0,temporal_scenes_used=False))
        execution = dict(device='cuda',cuda_initialized=True,tf32=False,amp=False,training_dtype='float64',published_dtype='float32',
            gpu=torch.cuda.get_device_name(0),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        optimizer_updates = config['updates']; origin = None
    else:
        origin = restore_custody(evidence,args); trained = read(args.trained_receipt); saved = read(args.trained_history)
        if trained['mode'] != 'train' or trained['closure'] != closure or trained['optimizer_updates'] != config['updates'] or saved['initial_inventory'] != initial_inventory:
            raise ValueError('Exact training closure required')
        network = network.float()
        load_weights(network,args.trained_initial,initial_inventory)
        load_weights(network,args.trained_weights,saved['final_inventory'])
        final_inventory = inventory(network)
        for source, name in ((args.trained_initial,'initial-weights.npz'),(args.trained_weights,'weights.npz'),(args.trained_history,'history.json')):
            shutil.copyfile(source,output/name)
        optimizer_updates = 0; execution = dict(device='cpu',cuda_initialized=torch.cuda.is_initialized(),dtype='float32')
        if execution['cuda_initialized']:
            raise ValueError('CPU restore initialized CUDA')
    probe = cpu_probe(network,velocity)
    if args.mode == 'restore' and not np.array_equal(probe.view('u1'),np.load(args.trained_probe,allow_pickle=False).view('u1')):
        raise ValueError('Full CPU viscosity probe differs')
    np.save(output/'probe.npy',probe,allow_pickle=False)
    write(output/'normalization.json',dict(input='dimensionless velocity',target='dimensionless viscosity update',channel_order='vx_vy_vz'))
    outputs = {p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in output.iterdir() if p.is_file()}
    write(output/'receipt.json',dict(schema='rheon-viscosity-calibration-fit-v1',status='complete',mode=args.mode,optimizer_updates=optimizer_updates,
        parameters=243,trainable_parameters=243,task='learned_viscosity_coefficients',closure=closure,initial_inventory=initial_inventory,
        final_inventory=final_inventory,execution=execution,outputs=outputs,originating_training_run=origin,
        full_cpu_probe_bitwise_verified=args.mode=='restore',elapsed_s=time.perf_counter()-started,
        numerical_pressure_solves=0,native_sampling_at_candidate_inference=False,full_state_forecast=False))
    print('Completed managed viscosity calibration',args.mode,optimizer_updates,'updates',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['train','restore'])
    for name in ('config',*STATIC,'input_evidence','output_dir'):
        parser.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    for name in ('trained_weights','trained_initial','trained_history','trained_probe','trained_receipt'):
        parser.add_argument('--'+name.replace('_','-'),dest=name,type=Path)
    run(parser.parse_args())
