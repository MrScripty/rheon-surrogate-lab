"""Managed CUDA supervised coefficient fit and exact typed double CPU restore."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from learned_modal_pressure import LearnedModalPressure
from modal_basis import bases, transform
from modal_precision_supervision import prepare as prepare_labels, objective as supervised_objective, initialize
from managed_fit_artifacts import validate_inputs, restore_custody


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inventory(model):
    result = {}
    for name, value in model.state_dict().items():
        array = value.detach().cpu().numpy()
        if array.dtype.str != '<f8':
            raise ValueError('Every learned pressure parameter must remain double')
        result[name] = dict(shape=list(array.shape), dtype=array.dtype.str, sha256=digest(array.tobytes()))
    return result


def save(model, path):
    np.savez(path, **{name: value.detach().cpu().numpy() for name, value in model.state_dict().items()})


def load(model, path, expected):
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != set(model.state_dict()):
            raise ValueError('Complete typed double parameter inventory required')
        values = {}
        for name, current in model.state_dict().items():
            value = source[name]
            if value.dtype.str != '<f8' or tuple(value.shape) != tuple(current.shape) or digest(value.tobytes()) != expected[name]['sha256']:
                raise ValueError('Double parameter identity changed')
            values[name] = torch.from_numpy(value.copy())
    model.load_state_dict(values, strict=True)
    if inventory(model) != expected:
        raise ValueError('Loaded double parameter inventory differs')


def data(args, manifest, device):
    if len(args.shards) != len(manifest['geometries']):
        raise ValueError('Every registered geometry shard required')
    groups = []
    for path, row in zip(args.shards, manifest['geometries']):
        if digest(path.read_bytes()) != row['sha256']:
            raise ValueError('Fixed native modal shard changed')
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != {'q_coefficients', 'pressure_coefficients'}:
                raise ValueError('Native projected inputs/labels only')
            arrays = [source[key].copy() for key in ('q_coefficients', 'pressure_coefficients')]
        if any(value.dtype.str != '<f8' or list(value.shape) != [len(row['samples']), *row['counts_zyx']] or not np.isfinite(value).all() for value in arrays):
            raise ValueError('Complete finite double native training coefficients required')
        groups.append(dict(key=row['key'], q=torch.from_numpy(arrays[0]).to(device), p=torch.from_numpy(arrays[1]).to(device)))
    return groups


def run(args):
    started = time.perf_counter(); config = read(args.config)
    device = 'cuda' if args.mode == 'train' else 'cpu'
    if os.environ.get('TULDOK_TORCH_DEVICE') != device:
        raise ValueError('Actual managed device differs')
    if config['updates'] != 80 or config['objective'] != 'native_mixed_query_weighted_precision_v3' or config['role'] != 'pressure_component_precision_refinement':
        raise ValueError('Exact registered finite component fit required')
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
        raise ValueError('Empty managed output directory required')
    torch.set_num_threads(1); torch.manual_seed(config['seed']); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    evidence, identities = validate_inputs(args)
    runtime = read(args.runtime_manifest)
    if runtime['profile'] != 'managed_torch_python_v1' or runtime['packages']['torch'] != torch.__version__ or runtime['packages']['numpy'] != np.__version__:
        raise ValueError('Approved runtime versions differ')
    manifest = read(args.manifest)
    if manifest['status'] != 'verified_train_only_fixed_transform_view' or manifest['scenes'] != 16 or manifest['samples'] != 272 or manifest.get('native_training_samples') != 144 or manifest.get('augmented_component_queries') != 128 or manifest['development_or_reserved_used']:
        raise ValueError('Exact original train-only fixed basis view required')
    if digest(args.normalization.read_bytes()) != manifest['normalization_sha256']:
        raise ValueError('Modal units/basis normalization changed')
    sources = {path.name: digest(path.read_bytes()) for path in sorted(Path(__file__).parent.glob('*.py'))}
    closure = dict(config_sha256=digest(args.config.read_bytes()), inputs=identities, sources=sources,
                   normalization_sha256=digest(args.normalization.read_bytes()), role=config['role'])
    normalization=read(args.normalization)
    model = initialize(LearnedModalPressure(manifest['geometries']),normalization).to(device)
    initial = inventory(model); groups = data(args, manifest, device)
    parameters = sum(p.numel() for p in model.parameters())
    setup_seconds = time.perf_counter() - started
    if args.mode == 'train':
        if not torch.cuda.is_initialized() or any(p.device.type != 'cuda' for p in model.parameters()):
            raise ValueError('All coefficient learning must execute on managed CUDA')
        labels,coverage=prepare_labels(groups,config['mode_scale_floor'])
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=5,max_eval=8,history_size=48,
            tolerance_grad=1e-18,tolerance_change=1e-30,line_search_fn='strong_wolfe')
        torch.cuda.reset_peak_memory_stats();history=[];durations=[];training_started=time.time();closure_calls=0
        for update in range(1,config['updates']+1):
            if time.perf_counter()-started>config['wall_seconds']-30:raise TimeoutError('Finite managed pressure precision deadline')
            tick=time.perf_counter()
            def closure_loss():
                nonlocal closure_calls
                closure_calls+=1;optimizer.zero_grad(set_to_none=True)
                value=supervised_objective(model,labels)
                if not bool(torch.isfinite(value)):raise ValueError('Nonfinite native coefficient precision loss')
                value.backward();return value
            loss=optimizer.step(closure_loss)
            if any(not bool(torch.isfinite(parameter).all())for parameter in model.parameters()):raise ValueError('Nonfinite refined pressure parameter')
            if any(tensor.device.type!='cuda'for state in optimizer.state.values()for tensor in state.values()if isinstance(tensor,torch.Tensor)):
                raise ValueError('Refinement optimizer state escaped managed CUDA')
            torch.cuda.synchronize();durations.append(time.perf_counter()-tick)
            if update==1 or update%10==0:
                value=float(supervised_objective(model,labels).detach())
                history.append(dict(update=update,objective=value,optimizer='LBFGS',closure_calls=closure_calls))
                print(json.dumps(dict(format='tuldok_progress_v1',stage='learned-modal-precision',completed=update,total=config['updates'],metrics=dict(loss=value))),flush=True)
        coverage['optimizer_closure_calls']=closure_calls
        execution = dict(device='cuda', cuda_initialized=True, gpu=torch.cuda.get_device_name(0), dtype='float64',
                         tf32=False, amp=False, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        model = model.cpu().eval(); final = inventory(model)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed']); initial_model = initialize(LearnedModalPressure(manifest['geometries']),normalization)
        if inventory(initial_model) != initial:
            raise ValueError('Seeded double initialization differs')
        save(initial_model, args.output_dir/'initial-weights.npz'); save(model, args.output_dir/'weights.npz')
        write(args.output_dir/'history.json', dict(history=history, initial_inventory=initial, final_inventory=final,
            coverage=coverage, training_started_unix=training_started, training_finished_unix=time.time(),
            per_update_seconds=durations, setup_seconds=setup_seconds, optimizer_updates=config['updates'],
            full_batch_geometry_updates=config['updates']*len(groups), training_scene_count=16, original_native_training_states=144, mixed_reference_queries=128, train_only=True))
        origin = None; updates = config['updates']
    else:
        origin = restore_custody(evidence, args); trained = read(args.trained_receipt); saved = read(args.trained_history)
        if trained['mode'] != 'train' or trained['optimizer_updates'] != config['updates'] or trained['closure'] != closure or initial != saved['initial_inventory']:
            raise ValueError('Exact managed double training closure required')
        load(model, args.trained_initial, initial); load(model, args.trained_weights, saved['final_inventory'])
        model.eval(); final = inventory(model)
        for source, name in ((args.trained_initial,'initial-weights.npz'), (args.trained_weights,'weights.npz'), (args.trained_history,'history.json')):
            shutil.copyfile(source,args.output_dir/name)
        execution = dict(device='cpu', cuda_initialized=torch.cuda.is_initialized(), dtype='float64')
        if execution['cuda_initialized']: raise ValueError('Fresh CPU restore initialized CUDA')
        updates = 0
    with torch.inference_mode():
        probes = []
        for group, row in zip(groups, manifest['geometries']):
            coefficients = model(group['q'][:1].cpu(), group['key'])
            pressure = transform(coefficients, bases(row['counts_zyx']), inverse=True)
            pressure = pressure - pressure[:, :1, :1, :1]
            probes.append(pressure.contiguous().flatten().numpy())
        probe = np.concatenate(probes).astype('<f8', copy=False)
    if args.mode == 'restore':
        expected = np.load(args.trained_probe, allow_pickle=False)
        if expected.dtype.str != '<f8' or not np.array_equal(probe.view('<u8'), expected.view('<u8')):
            raise ValueError('Complete double CPU physical-pressure probe differs')
    np.save(args.output_dir/'probe.npy', probe, allow_pickle=False)
    shutil.copyfile(args.normalization, args.output_dir/'normalization.json')
    outputs = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p.read_bytes())) for p in args.output_dir.iterdir() if p.is_file()}
    write(args.output_dir/'receipt.json', dict(schema='rheon-learned-modal-pressure-result-v1', status='complete', mode=args.mode,
        role=config['role'], optimizer_updates=updates, parameters=parameters, trainable_parameters=parameters,
        closure=closure, initial_inventory=initial, final_inventory=final, execution=execution, outputs=outputs,
        full_cpu_probe_bitwise_verified=args.mode=='restore', originating_training_run=origin,
        causal_full_state_generator=False, scientific_acceptance_or_promotion=False, elapsed_s=time.perf_counter()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'restore'))
    for name in ('config','manifest','normalization','runtime-manifest','input-evidence','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shards',nargs='+',type=Path,required=True)
    for name in ('trained-weights','trained-initial','trained-history','trained-probe','trained-receipt'):
        parser.add_argument('--'+name,type=Path)
    run(parser.parse_args())
