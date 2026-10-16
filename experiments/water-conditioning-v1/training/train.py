#!/usr/bin/env python3
"""Managed CUDA training, portable checkpoints and exact CPU deployment restore."""
import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from policy import HERE, MIB, canonical, decode, execution, experiment, load_module, read, require, sha, write
import data
import model
from prepared import PreparedSample

DATA = data.DATA


def input_slots(config):
    if config['schema'] in ('rheon-water-residual-experiment-v8', 'rheon-water-residual-experiment-v9'):
        import direction_feedback
        return {**DATA, **direction_feedback.SLOTS}
    return DATA


def outputs(config):
    result = {'weights.npz': ('checkpoint', 2*MIB), 'predictions.npz': ('predictions', 2*MIB),
              'normalization.json': ('preprocessing', MIB), 'splits.json': ('training_splits', 4*MIB),
              'history.json': ('training_diagnostics', MIB), 'profile.json': ('training_diagnostics', MIB), 'receipt.json': ('training_result', MIB),
              'summary.json': ('result_summary', 64*1024)}
    for update in config['training']['checkpoints']:
        result[f'checkpoint-{update:03d}.npz'] = ('checkpoint', 2*MIB)
    return result


def trained_inputs(config):
    return {'trained_'+name.replace('-', '_').replace('.', '_'): (name, role, cap)
            for name, (role, cap) in outputs(config).items() if name != 'summary.json'}


def order(parents, config):
    eligible = [i for i, p in enumerate(parents) if p['entry']['split'] == 'train']
    rng = np.random.Generator(np.random.PCG64(config['training']['sampling_seed']))
    result=[]
    for _ in range(config['training']['updates']):
        batch=[]
        for _ in range(config['training']['batch_size']):
            parent=eligible[int(rng.integers(len(eligible)))];frame=int(rng.integers(len(parents[parent]['frames'])))
            batch.append((parent,frame))
        result.append(batch)
    return np.asarray(result,dtype='<i8')



def finite_tensors(tensors):
    """One exact finite reduction and host synchronization across a tensor group."""
    return bool(torch.isfinite(torch.cat([value.detach().reshape(-1) for value in tensors])).all())


def one_update(network, optimizer, parents, indices, normalization, cache=None, strategy=None, prepared=None, analytical_sample=None, analytical_weight=0., supplemental_sample=None, supplemental_weight=0., direction_sample=None, direction_weight=0.):
    network.train(); optimizer.zero_grad(set_to_none=True)
    strategy=strategy or {'finite_checks':'per-tensor','batch_mode':'serial'}
    samples=[cache[(int(parent),int(frame))] if cache is not None else
             data.sample(parents[int(parent)],int(frame),device=next(network.parameters()).device)
             for parent,frame in indices]
    if strategy['batch_mode']=='stacked':
        require(len({tuple(s['base'].shape[2:]) for s in samples})==1,'batch_shape','stack only identical grid dimensions')
        samples=[{key:torch.cat([sample[key] for sample in samples],dim=0) for key in samples[0]}]
    losses = []
    for position,sample in enumerate(samples):
        require(all(value.device == next(network.parameters()).device for value in sample.values()), 'batch_device', 'all conditioning and fields colocated')
        if prepared is not None:
            parent,slot=indices[position]
            loss=prepared[(int(parent),int(slot))].loss(network).mean()
        else:
            loss = model.training_loss(network, sample, normalization).mean()
        require(bool(torch.isfinite(loss)), 'training_nonfinite', 'per-frame loss')
        if loss.requires_grad:
            (loss/len(samples)).backward()
        else:
            require(float(loss) == 0, 'zero_residual_labels', 'bypassed exact solution must have zero loss')
        losses.append(float(loss.detach()))
    if analytical_sample is not None:
        import analytical
        extra=model.training_loss(network,analytical_sample,normalization,
            criterion=lambda prediction:analytical.loss(prediction,analytical_sample,model.full_operator)).mean()
        require(bool(torch.isfinite(extra)), 'training_nonfinite', 'analytical feedback')
        (analytical_weight*extra).backward()
        extra_value=float(extra.detach())*analytical_weight
    else:extra_value=0.
    if supplemental_sample is not None:
        import analytical
        extra=model.training_loss(network,supplemental_sample,normalization,
            criterion=lambda prediction:analytical.loss(prediction,supplemental_sample,model.full_operator)).mean()
        require(bool(torch.isfinite(extra)), 'training_nonfinite', 'supplemental feedback')
        (supplemental_weight*extra).backward()
        extra_value+=float(extra.detach())*supplemental_weight
    if direction_sample is not None:
        import analytical
        extra = model.training_loss(network, direction_sample, normalization,
            criterion=lambda prediction: analytical.loss(prediction, direction_sample, model.full_operator)).mean()
        require(bool(torch.isfinite(extra)), 'training_nonfinite', 'captured direction feedback')
        (direction_weight*extra).backward()
        extra_value += float(extra.detach())*direction_weight
    gradient = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    finite=finite_tensors(network.parameters()) if strategy['finite_checks']=='combined' else all(bool(torch.isfinite(p).all()) for p in network.parameters())
    require(finite, 'training_nonfinite', 'weights')
    return math.fsum(losses)/len(losses)+extra_value, float(gradient)


def evaluate(network, parents, normalization):
    network.eval(); grouped = []
    with torch.no_grad():
        for parent in parents:
            if parent['entry']['split'] != 'validation':
                continue
            values = []
            for frame in range(len(parent['frames'])):
                sample = data.sample(parent, frame, device=next(network.parameters()).device)
                loss = model.pressure_loss(model.predict_pressure(network, sample, normalization), sample, normalization)
                require(bool(torch.isfinite(loss).all()), 'validation_nonfinite', 'loss')
                values.append(float(loss.mean()))
            grouped.append(math.fsum(values)/len(values))
    require(grouped, 'validation', 'held-out whole parents')
    return math.fsum(grouped)/len(grouped)


def cases(parents):
    result=[]
    for split in ('train','validation'):
        eligible=[i for i,p in enumerate(parents) if p['entry']['split']==split]
        chosen=[eligible[int(n)] for n in np.linspace(0,len(eligible)-1,min(12,len(eligible)),dtype=int)]
        for index,parent in enumerate(chosen):
            p=parents[parent];slot=index%len(p['frames'])
            result.append({'parent':parent,'frame':slot,'native_frame':p['frames'][slot]['index'],
                'trajectory_id':p['entry']['trajectory_id'],'split':split,'counts_xyz':p['counts_xyz'],'cell_size_m':p['cell_size_m']})
    return result


def predictions_on_device(network, parents, normalization, population):
    network.eval(); arrays = {}; clocks = []
    with torch.no_grad():
        for i, case in enumerate(population):
            sample = data.sample(parents[case['parent']], case['frame'], labels=False, device=next(network.parameters()).device)
            start = time.perf_counter()
            pressure = model.predict_pressure(network, sample, normalization)
            clocks.append(time.perf_counter()-start)
            require(bool(torch.isfinite(pressure).all()), 'prediction', 'finite per-cell pressure')
            arrays[f'case_{i:03d}'] = pressure[0, 0].cpu().numpy().copy()
    return arrays, clocks



def predictions(network, parents, normalization, population):
    require(next(network.parameters()).device.type=='cpu', 'deployment', 'CPU inference only')
    return predictions_on_device(network, parents, normalization, population)


def verify_cuda_update(network, optimizer, combined=False):
    if combined:
        parameters=list(network.parameters());gradients=[];moments=[]
        for parameter in parameters:
            require(parameter.device.type=='cuda' and parameter.dtype==torch.float32
                    and parameter.grad is not None and parameter.grad.device==parameter.device
                    and parameter.grad.dtype==torch.float32,'cuda_gradient','float32 CUDA parameters and gradients')
            gradients.append(parameter.grad)
            for name in ('exp_avg','exp_avg_sq'):
                value=optimizer.state[parameter][name]
                require(value.device==parameter.device and value.dtype==torch.float32,'cuda_optimizer','float32 CUDA Adam buffers')
                moments.append(value)
        require(finite_tensors(gradients),'cuda_gradient','all finite gradients')
        require(finite_tensors(moments),'cuda_optimizer','all finite Adam buffers')
        return
    for parameter in network.parameters():
        require(parameter.device.type=='cuda' and parameter.dtype==torch.float32
                and parameter.grad is not None and parameter.grad.device==parameter.device
                and bool(torch.isfinite(parameter.grad).all()), 'cuda_gradient', 'actual float32 GPU parameters and finite gradients')
        state=optimizer.state[parameter]
        for name in ('exp_avg','exp_avg_sq'):
            require(name in state and state[name].device==parameter.device and state[name].dtype==torch.float32
                    and bool(torch.isfinite(state[name]).all()), 'cuda_optimizer', 'actual finite GPU Adam moment buffers')


def project_inputs(evidence, inputs, witness):
    owner = None
    for slot, raw in inputs.items():
        entry = evidence[slot]
        require(set(entry) == {'project_id', 'manifest_sha256', 'reference_id', 'path', 'measured_sha256'}
                and entry['reference_id'] == slot and entry['measured_sha256'] == sha(raw)
                and witness['inputs'][slot] == {'sha256': sha(raw), 'bytes': len(raw)}, 'input_evidence', slot)
        path = Path(entry['path'])
        require(not path.is_absolute() and path.as_posix() == entry['path'] and all(p not in ('.', '..') for p in path.parts), 'input_path', slot)
        identity = entry['project_id'], entry['manifest_sha256']
        require(owner is None or owner == identity, 'mixed_project', 'one immutable input project')
        owner = identity


def custody(evidence, trained, declared, validator_path, validator_sha):
    require(sha(read(validator_path, MIB)) == validator_sha, 'custody_source', 'selected Tuldok validator')
    validator = load_module('variable_pressure_custody', validator_path)
    shared = None
    for slot, (name, role, _cap) in declared.items():
        entry = evidence[slot]
        require(set(entry) == {'kind', 'source'} and entry['kind'] == 'run_artifact', 'restore_input', 'typed managed artifact')
        source = validator.validate_artifact_source(entry['source'], sha(trained[slot]), len(trained[slot]))
        require(source['artifact_name'] == name and source['artifact_role'] == role, 'artifact_role', name)
        origin = {key: source[key] for key in ('catalog_id', 'run_id', 'attempt_id', 'request_sha256', 'material_sha256', 'custody_receipt_sha256', 'custody_identity', 'custody_receipt_base64')}
        require(shared is None or shared == origin, 'mixed_run', 'one completed and drained training Run')
        shared = origin
    return shared


def run(args):
    start = time.perf_counter(); helpers = data.helpers()
    config_raw = read(args.config, MIB); config = experiment(decode(config_raw))
    strategy = execution(config)
    model.select(config)
    witness_raw = read(args.source_witness, 2*MIB); witness = decode(witness_raw)
    require(witness['schema'] == 'rheon-water-learning-source-v1' and witness['experiment_sha256'] == sha(config_raw)
            and witness['architecture'] == model.ARCHITECTURE, 'source_config', 'selected model and configuration')
    for name, digest in witness['sources'].items():
        require(Path(name).name == name and sha(read(HERE/name, 4*MIB)) == digest, 'source_identity', name)
    runtime_raw = read(args.runtime_manifest, 16*MIB); runtime = decode(runtime_raw)
    device = 'cuda' if args.mode == 'train' else 'cpu'
    require(sha(runtime_raw) == witness['runtime_manifest_sha256'] and runtime['profile'] == 'managed_torch_python_v1'
            and runtime['packages']['torch'] == torch.__version__ and runtime['packages']['numpy'] == np.__version__
            and runtime['python_version'] == f'{sys.version_info.major}.{sys.version_info.minor}'
            and os.environ.get('TULDOK_TORCH_DEVICE') == device, 'runtime', 'approved installed Torch and explicit device')
    inputs = {slot: read(getattr(args, slot), 4*MIB if slot in ('splits', 'direction_manifest') else 32*MIB) for slot in input_slots(config)}
    evidence = decode(read(args.input_evidence, MIB)); declared = trained_inputs(config) if args.mode == 'restore' else {}
    require(set(evidence) == set(input_slots(config)) | set(declared), 'inputs', 'all typed slots')
    project_inputs(evidence, inputs, witness)
    manifest = decode(inputs['splits']); packet_config, parents = data.load((inputs['dataset'], inputs['dataset_extra']), manifest)
    require(sha(canonical(packet_config)) == witness['packet_experiment_sha256'], 'packet_config', 'unchanged qualified physical experiment')
    model.configure(device); torch.manual_seed(config['training']['seed'])
    normalization = model.fit_normalization(parents,data.fit_normalization)
    require(sha(canonical(normalization)) == witness['normalization_sha256'], 'normalization', 'train-only scales')
    network = model.PressureCorrectionNet().to(device); initial_inventory = helpers.state_inventory(network)
    sample_order = order(parents, config); population = cases(parents)
    closure = {'source_witness_sha256': sha(witness_raw), 'experiment_sha256': sha(config_raw),
               'runtime_manifest_sha256': sha(runtime_raw), 'inputs': witness['inputs'],
               'normalization_sha256': witness['normalization_sha256'], 'sample_order_sha256': sha(sample_order.tobytes())}
    output = args.output_dir.absolute()
    require(output == output.resolve() and output.is_dir() and not any(output.iterdir()), 'output', 'empty owned directory')
    origin = None; checkpoint_specs = {}; history = []; trace = []
    def checkpoint(update):
        name = f'checkpoint-{update:03d}.npz'
        tensors = helpers.save_weights(network, output/name)
        raw = read(output/name)
        checkpoint_specs[name] = {'update': update, 'sha256': sha(raw), 'bytes': len(raw), 'inventory': tensors}
    if args.mode == 'train':
        torch.cuda.reset_peak_memory_stats()
        device_evidence = {'device': str(next(network.parameters()).device), 'gpu': torch.cuda.get_device_name(0), 'model_dtype': 'float32', 'amp': False, 'tf32': False, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
        checkpoint(0); initial_loss = evaluate(network, parents, normalization)
        training_started_unix = time.time()
        cache = {(i,slot):data.sample(parent,slot,device=device) for i,parent in enumerate(parents) if parent['entry']['split']=='train' for slot in range(len(parent['frames']))} if strategy['resident_inputs'] else None
        prepared={key:PreparedSample(sample,normalization) for key,sample in cache.items()} if strategy.get('prepared_fields') else None
        analytical_samples=None;analytical_evidence=None
        feedback=config.get('analytical_feedback')
        if feedback is not None and feedback['weight']>0:
            import analytical
            if config['schema'] in ('rheon-water-residual-experiment-v5','rheon-water-residual-experiment-v6','rheon-water-residual-experiment-v7','rheon-water-residual-experiment-v8','rheon-water-residual-experiment-v9'):
                require(cache,'analytical_training','resident actual training states required')
                amplitude=analytical.training_correction_rms(cache.values())
                analytical_samples,analytical_order,analytical_evidence=analytical.bank(next(iter(cache.values())),feedback,config['training']['updates'],amplitude_pa=amplitude)
            else:
                require(len(cache or {})==1,'analytical_training','one recorded fitting frame for this controlled comparison')
                analytical_samples,analytical_order,analytical_evidence=analytical.bank(next(iter(cache.values())),feedback,config['training']['updates'])
        supplemental_samples=None;supplemental_evidence=None
        supplemental=config.get('supplemental_feedback')
        if supplemental is not None and supplemental['weight']>0:
            supplemental_samples,supplemental_order,supplemental_evidence=analytical.supplemental_bank(
                next(iter(cache.values())),supplemental,config['training']['updates'],
                (analytical_samples,analytical_order,analytical_evidence),feedback,amplitude)
        direction_samples = None
        direction_evidence = None
        direction = config.get('direction_feedback')
        if direction is not None:
            import direction_feedback
            resident = analytical_evidence['resident_tensor_bytes']+supplemental_evidence['resident_tensor_bytes']
            direction_samples, direction_order, direction_evidence = direction_feedback.bank(
                inputs, direction, normalization,
                [p['entry']['trajectory_id'] for p in parents if p['entry']['split'] == 'train'],
                config['training']['updates'], device, resident,
                min(feedback['cache_max_bytes'], supplemental['cache_max_bytes']))
        profile_report = {'schema':'rheon-cuda-training-profile-v2','enabled':False,'execution':strategy}
        def capture_profile(profiler):
            events = profiler.key_averages()
            require(any(getattr(e,'self_device_time_total',0)>0 for e in events),
                    'cuda_profiler','actual device events required')
            def row(event):
                return {'name':event.key,'calls':event.count,'self_cpu_us':event.self_cpu_time_total,
                        'self_device_us':getattr(event,'self_device_time_total',None)}
            profile_report.update(enabled=True,scope='managed CUDA updates after ten-step warm-up',
                top_cpu=[row(e) for e in sorted(events,key=lambda e:e.self_cpu_time_total,reverse=True)[:20]],
                top_device=[row(e) for e in sorted(events,key=lambda e:getattr(e,'self_device_time_total',0),reverse=True)[:20]],
                scalar_extractions=sum(e.count for e in events if e.key=='aten::_local_scalar_dense'),
                scalar_extraction_scope='CPU and CUDA tensor scalar extraction calls',
                tensor_copy_or_conversion_calls=sum(e.count for e in events if e.key=='aten::_to_copy'),
                cuda_stream_synchronizations=sum(e.count for e in events if e.key=='cudaStreamSynchronize'))
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=10,warmup=1,active=strategy['profile_updates'],repeat=1),
            on_trace_ready=capture_profile) if strategy['profile_updates'] else nullcontext()
        optimizer = torch.optim.Adam(network.parameters(), lr=config['training']['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0)
        with profiler:
            for update, indices in enumerate(sample_order, 1):
                require(time.perf_counter()-start < config['resources']['wall_seconds']-10, 'deadline', 'publication reserve')
                helpers.observed_peak_rss_bytes()
                for change in config['training']['schedule']:
                    if change['update'] == update:
                        for group in optimizer.param_groups: group['lr'] = change['learning_rate']
                extra=analytical_samples[int(analytical_order[update-1])] if analytical_samples is not None else None
                second=supplemental_samples[int(supplemental_order[update-1])] if supplemental_samples is not None else None
                third = direction_samples[int(direction_order[update-1])] if direction_samples is not None else None
                loss, gradient = one_update(network, optimizer, parents, indices, normalization, cache, strategy, prepared,
                    analytical_sample=extra,analytical_weight=feedback['weight'] if extra is not None else 0.,
                    supplemental_sample=second,supplemental_weight=supplemental['weight'] if second is not None else 0.,
                    direction_sample=third,direction_weight=direction['weight'] if third is not None else 0.)
                verify_cuda_update(network, optimizer, combined=strategy['finite_checks']=='combined')
                if update % 100 == 0:
                    with torch.no_grad():
                        parent, slot = indices[0]
                        sample = data.sample(parents[int(parent)], int(slot), device=device)
                        predicted = model.predict_pressure(network,sample,normalization)
                        residual = model.full_operator(predicted,sample['cell_size_m'],sample['rho'])-sample['rhs']
                        physical_ratio = float(torch.linalg.vector_norm(residual))/max(float(torch.linalg.vector_norm(sample['rhs'])),1e-30)
                    trace.append({'update': update, 'loss': loss, 'gradient_norm_before_clip': gradient,
                        'gradient_clipped': gradient>1., 'learning_rate': optimizer.param_groups[0]['lr'],
                        'post_update_sample_relative_residual':physical_ratio})
                    helpers.progress('water-learning-sequence', update, config['training']['updates'], loss=loss)
                if update in config['training']['checkpoints']:
                    checkpoint(update); history.append({'update': update, 'loss': loss, 'gradient_norm_before_clip': gradient})
                    helpers.progress('variable-grid-training', update, config['training']['updates'], loss=loss)
                if strategy['profile_updates']: profiler.step()
        training_finished_unix = time.time()
        write(output/'profile.json',profile_report)
        validation = {'initial': initial_loss, 'final': evaluate(network, parents, normalization)}
        device_evidence.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(), parameters_gradients_adam_moments_cuda=True, batches_cuda=True)
        write(output/'history.json', {'history': history, 'checkpoints': checkpoint_specs, 'validation': validation,
                                     'analytical_feedback':analytical_evidence,
                                     **({'supplemental_feedback':supplemental_evidence} if supplemental_evidence is not None else {}),
                                     **({'direction_feedback':direction_evidence} if direction_evidence is not None else {}),
                                     'trace': trace, 'scope': 'final update candidate; no optimizer resume; loss before update, physical ratio after update',
                                     'training_started_unix':training_started_unix,'training_finished_unix':training_finished_unix,
                                     'execution_strategy':strategy,'optimizer_updates_per_s':config['training']['updates']/(training_finished_unix-training_started_unix)})
        final_name = f'checkpoint-{config["training"]["updates"]:03d}.npz'
        (output/'weights.npz').write_bytes(read(output/final_name))
        gpu_predicted, _ = predictions_on_device(network, parents, normalization, population)
        network = network.cpu()
        require(helpers.state_inventory(network) == checkpoint_specs[final_name]['inventory'], 'cpu_transfer', 'every float32 weight bit preserved')
        predicted, clocks = predictions(network, parents, normalization, population)
        device_evidence['cuda_cpu_pressure_max_abs_difference'] = max(float(np.max(np.abs(gpu_predicted[key]-value))) for key,value in predicted.items())
        with (output/'predictions.npz').open('xb') as stream:
            np.savez(stream, **predicted)
        prediction_inventory = data.inventory(predicted)
        write(output/'normalization.json', normalization); (output/'splits.json').write_bytes(inputs['splits'])
    else:
        device_evidence = {'device': 'cpu', 'cuda_initialized': torch.cuda.is_initialized(), 'optimizer_updates': 0}
        raw_trained = {slot: read(getattr(args, slot), cap) for slot, (_name, _role, cap) in declared.items()}
        origin = custody(evidence, raw_trained, declared, HERE/'custody-validator.py', witness['custody_validator_sha256'])
        trained = decode(raw_trained['trained_receipt_json'])
        require(trained['schema'] == 'rheon-water-learning-training-v1' and trained['status'] == 'complete'
                and trained['mode'] == 'train' and trained['optimizer_updates'] == config['training']['updates']
                and trained['closure'] == closure and trained['cases'] == population, 'restore_identity', 'same final trained state')
        for slot, (name, _role, _cap) in declared.items():
            if name != 'receipt.json':
                require(trained['outputs'][name]['sha256'] == sha(raw_trained[slot]) and trained['outputs'][name]['bytes'] == len(raw_trained[slot]), 'trained_hash', name)
        require(raw_trained['trained_normalization_json'] == canonical(normalization) and raw_trained['trained_splits_json'] == inputs['splits'], 'restore_inputs', 'exact preprocessing and population')
        observed = decode(raw_trained['trained_history_json']); checkpoint_specs = observed['checkpoints']
        updates = list(config['training']['checkpoints'])
        require(set(checkpoint_specs) == {f'checkpoint-{u:03d}.npz' for u in updates}
                and [item['update'] for item in observed['history']] == updates[1:], 'restore_history', 'all observed states')
        for update in updates:
            name = f'checkpoint-{update:03d}.npz'; raw = raw_trained['trained_'+name.replace('-', '_').replace('.', '_')]; spec = checkpoint_specs[name]
            require(spec['update'] == update and spec['sha256'] == sha(raw) and spec['bytes'] == len(raw), 'checkpoint', name)
            require(set(spec['inventory']) == set(initial_inventory) and all(spec['inventory'][key]['shape'] == initial_inventory[key]['shape'] for key in initial_inventory), 'state_shape', name)
            helpers.read_npz(raw, spec['inventory'], 2*MIB)
        require(checkpoint_specs['checkpoint-000.npz']['inventory'] == initial_inventory, 'restore_initialization', 'same seeded initial state')
        final_name = f'checkpoint-{config["training"]["updates"]:03d}.npz'
        require(raw_trained['trained_weights_npz'] == raw_trained['trained_'+final_name.replace('-', '_').replace('.', '_')], 'candidate', 'final update only')
        helpers.load_weights(network, args.trained_weights_npz, checkpoint_specs[final_name]['inventory'])
        predicted, clocks = predictions(network, parents, normalization, population); prediction_inventory = data.inventory(predicted)
        require(prediction_inventory == trained['prediction_inventory'], 'restore_prediction', 'fresh exact numeric parity on every shape')
        helpers.read_npz(raw_trained['trained_predictions_npz'], prediction_inventory, 2*MIB)
        for slot, (name, _role, _cap) in declared.items():
            if name != 'receipt.json':
                (output/name).write_bytes(raw_trained[slot])
        validation = observed['validation']
    write(output/'summary.json', {'mode': args.mode, 'optimizer_updates': config['training']['updates'] if args.mode == 'train' else 0,
          'candidate_update': config['training']['updates'], 'validation': validation, 'cubic_cells': True,
          'shapes_xyz': [p['counts_xyz'] for p in parents], 'solver_acceleration_claim': False})
    artifact_specs = {name: {'sha256': sha(read(output/name, cap)), 'bytes': len(read(output/name, cap)), 'role': role}
                      for name, (role, cap) in outputs(config).items() if name != 'receipt.json'}
    receipt = {'schema': 'rheon-water-learning-training-v1', 'status': 'complete', 'mode': args.mode,
               'optimizer_updates': config['training']['updates'] if args.mode == 'train' else 0,
               'closure': closure, 'cases': population, 'prediction_inventory': prediction_inventory,
               'per_case_inference_s': clocks, 'outputs': artifact_specs, 'originating_training_run': origin,
               'architecture': model.ARCHITECTURE, 'execution': device_evidence, 'elapsed_s': time.perf_counter()-start,
               'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(), 'peak_rss_bytes': helpers.observed_peak_rss_bytes(), 'inference_device': 'cpu'}}
    require(sum(s['bytes'] for s in artifact_specs.values())+len(canonical(receipt)) <= config['resources']['max_output_bytes'], 'output_budget', 'complete history and publication')
    write(output/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=('train', 'restore'))
    for name in (*DATA, 'config', 'runtime-manifest', 'source-witness', 'input-evidence', 'output-dir'):
        parser.add_argument('--'+name.replace('_', '-'), dest=name.replace('-', '_'), type=Path, required=True)
    known, _unknown = parser.parse_known_args()
    config = experiment(decode(read(known.config)))
    for slot in set(input_slots(config))-set(DATA):
        parser.add_argument('--'+slot.replace('_', '-'), dest=slot, type=Path, required=True)
    for name in ('weights.npz', 'predictions.npz', 'normalization.json', 'splits.json', 'history.json', 'profile.json', 'receipt.json', *(f'checkpoint-{u:03d}.npz' for u in config['training']['checkpoints'])):
        slot = 'trained_'+name.replace('-', '_').replace('.', '_'); parser.add_argument('--'+slot.replace('_', '-'), dest=slot, type=Path)
    args = parser.parse_args()
    require(args.mode != 'restore' or all(getattr(args, slot) is not None for slot in trained_inputs(config)), 'restore_slots', 'all originating artifacts')
    require(args.mode != 'train' or all(value is None for key, value in vars(args).items() if key.startswith('trained_')), 'train_slots', 'fresh training only')
    run(args)


if __name__ == '__main__':
    main()
