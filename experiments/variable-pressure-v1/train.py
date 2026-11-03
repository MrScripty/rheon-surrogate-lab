#!/usr/bin/env python3
"""Managed heterogeneous-grid CPU training and exact zero-update state restore."""
import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from policy import HERE, MIB, canonical, decode, experiment, load_module, read, require, sha, write
import data
import model

DATA = {'dataset': 'dataset.npz', 'splits': 'splits.json'}


def outputs(config):
    result = {'weights.npz': ('checkpoint', 2*MIB), 'predictions.npz': ('predictions', 2*MIB),
              'normalization.json': ('preprocessing', MIB), 'splits.json': ('training_splits', 4*MIB),
              'history.json': ('training_diagnostics', MIB), 'receipt.json': ('training_result', MIB),
              'summary.json': ('result_summary', 64*1024)}
    for update in range(0, config['training']['updates']+1, config['training']['checkpoint_interval']):
        result[f'checkpoint-{update:03d}.npz'] = ('checkpoint', 2*MIB)
    return result


def trained_inputs(config):
    return {'trained_'+name.replace('-', '_').replace('.', '_'): (name, role, cap)
            for name, (role, cap) in outputs(config).items() if name != 'summary.json'}


def order(parents, config):
    eligible = [i for i, p in enumerate(parents) if p['entry']['split'] == 'train']
    rng = np.random.Generator(np.random.PCG64(config['training']['seed']+1))
    return np.asarray([[(eligible[int(rng.integers(len(eligible)))], int(rng.integers(24)))
                        for _ in range(config['training']['batch_size'])]
                       for _ in range(config['training']['updates'])], dtype='<i8')


def one_update(network, optimizer, parents, indices, normalization):
    network.train(); optimizer.zero_grad(set_to_none=True)
    losses = []
    for parent, frame in indices:
        sample = data.sample(parents[int(parent)], int(frame))
        loss = model.pressure_loss(model.predict_pressure(network, sample, normalization), sample, normalization).mean()
        require(bool(torch.isfinite(loss)), 'training_nonfinite', 'per-frame loss')
        if loss.requires_grad:
            (loss/len(indices)).backward()
        else:
            require(float(loss) == 0, 'zero_residual_labels', 'bypassed exact solution must have zero loss')
        losses.append(float(loss.detach()))
    gradient = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in network.parameters()), 'training_nonfinite', 'weights')
    return math.fsum(losses)/len(losses), float(gradient)


def evaluate(network, parents, normalization):
    network.eval(); grouped = []
    with torch.no_grad():
        for parent in parents:
            if parent['entry']['split'] != 'validation':
                continue
            values = []
            for frame in range(24):
                sample = data.sample(parent, frame)
                loss = model.pressure_loss(model.predict_pressure(network, sample, normalization), sample, normalization)
                require(bool(torch.isfinite(loss).all()), 'validation_nonfinite', 'loss')
                values.append(float(loss.mean()))
            grouped.append(math.fsum(values)/24)
    require(grouped, 'validation', 'held-out whole parents')
    return math.fsum(grouped)/len(grouped)


def cases(parents):
    return [{'parent': i, 'frame': frame, 'trajectory_id': p['entry']['trajectory_id'],
             'split': p['entry']['split'], 'counts_xyz': p['counts_xyz'], 'spacing_m': p['spacing_m']}
            for i, p in enumerate(parents) for frame in (0, 23)]


def predictions(network, parents, normalization, population):
    network.eval(); arrays = {}; clocks = []
    with torch.no_grad():
        for i, case in enumerate(population):
            sample = data.sample(parents[case['parent']], case['frame'], labels=False)
            start = time.perf_counter()
            pressure = model.predict_pressure(network, sample, normalization)
            clocks.append(time.perf_counter()-start)
            require(bool(torch.isfinite(pressure).all()), 'prediction', 'finite per-cell pressure')
            arrays[f'case_{i:03d}'] = pressure[0, 0].numpy().copy()
    return arrays, clocks


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
    witness_raw = read(args.source_witness, 2*MIB); witness = decode(witness_raw)
    require(witness['schema'] == 'rheon-variable-pressure-source-v1' and witness['experiment_sha256'] == sha(config_raw)
            and witness['architecture'] == model.ARCHITECTURE, 'source_config', 'selected model and configuration')
    for name, digest in witness['sources'].items():
        require(Path(name).name == name and sha(read(HERE/name, 4*MIB)) == digest, 'source_identity', name)
    runtime_raw = read(args.runtime_manifest, 16*MIB); runtime = decode(runtime_raw)
    require(sha(runtime_raw) == witness['runtime_manifest_sha256'] and runtime['profile'] == 'captured_cpu_python_v1'
            and runtime['packages']['torch'] == torch.__version__ == '2.8.0+cpu'
            and runtime['packages']['numpy'] == np.__version__ == '2.3.5' and torch.version.cuda is None
            and runtime['python_version'] == f'{sys.version_info.major}.{sys.version_info.minor}', 'runtime', 'captured eager CPU runtime')
    inputs = {slot: read(getattr(args, slot), 32*MIB if slot == 'dataset' else 4*MIB) for slot in DATA}
    evidence = decode(read(args.input_evidence, MIB)); declared = trained_inputs(config) if args.mode == 'restore' else {}
    require(set(evidence) == set(DATA) | set(declared), 'inputs', 'all typed slots')
    project_inputs(evidence, inputs, witness)
    manifest = decode(inputs['splits']); packet_config, parents = data.load(inputs['dataset'], manifest)
    require(canonical(packet_config) == config_raw, 'packet_config', 'same selected experiment')
    model.configure_cpu(); torch.manual_seed(config['training']['seed'])
    normalization = data.fit_normalization(parents)
    require(sha(canonical(normalization)) == witness['normalization_sha256'], 'normalization', 'train-only scales')
    network = model.PressureCorrectionNet(); initial_inventory = helpers.state_inventory(network)
    sample_order = order(parents, config); population = cases(parents)
    closure = {'source_witness_sha256': sha(witness_raw), 'experiment_sha256': sha(config_raw),
               'runtime_manifest_sha256': sha(runtime_raw), 'inputs': witness['inputs'],
               'normalization_sha256': witness['normalization_sha256'], 'sample_order_sha256': sha(sample_order.tobytes())}
    output = args.output_dir.absolute()
    require(output == output.resolve() and output.is_dir() and not any(output.iterdir()), 'output', 'empty owned directory')
    origin = None; checkpoint_specs = {}; history = []
    def checkpoint(update):
        name = f'checkpoint-{update:03d}.npz'
        tensors = helpers.save_weights(network, output/name)
        raw = read(output/name)
        checkpoint_specs[name] = {'update': update, 'sha256': sha(raw), 'bytes': len(raw), 'inventory': tensors}
    if args.mode == 'train':
        checkpoint(0); initial_loss = evaluate(network, parents, normalization)
        optimizer = torch.optim.Adam(network.parameters(), lr=config['training']['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0)
        for update, indices in enumerate(sample_order, 1):
            require(time.perf_counter()-start < config['resources']['wall_seconds']-10, 'deadline', 'publication reserve')
            helpers.observed_peak_rss_bytes()
            loss, gradient = one_update(network, optimizer, parents, indices, normalization)
            if update % config['training']['checkpoint_interval'] == 0:
                checkpoint(update); history.append({'update': update, 'loss': loss, 'gradient_norm_before_clip': gradient})
                helpers.progress('variable-grid-training', update, config['training']['updates'], loss=loss)
        validation = {'initial': initial_loss, 'final': evaluate(network, parents, normalization)}
        write(output/'history.json', {'history': history, 'checkpoints': checkpoint_specs, 'validation': validation,
                                     'scope': 'final update candidate; no optimizer resume'})
        final_name = f'checkpoint-{config["training"]["updates"]:03d}.npz'
        (output/'weights.npz').write_bytes(read(output/final_name))
        predicted, clocks = predictions(network, parents, normalization, population)
        with (output/'predictions.npz').open('xb') as stream:
            np.savez(stream, **predicted)
        prediction_inventory = data.inventory(predicted)
        write(output/'normalization.json', normalization); (output/'splits.json').write_bytes(inputs['splits'])
    else:
        raw_trained = {slot: read(getattr(args, slot), cap) for slot, (_name, _role, cap) in declared.items()}
        origin = custody(evidence, raw_trained, declared, args.input_evidence.with_name('recipe_process.py'), witness['custody_validator_sha256'])
        trained = decode(raw_trained['trained_receipt_json'])
        require(trained['schema'] == 'rheon-variable-pressure-training-v1' and trained['status'] == 'complete'
                and trained['mode'] == 'train' and trained['optimizer_updates'] == config['training']['updates']
                and trained['closure'] == closure and trained['cases'] == population, 'restore_identity', 'same final trained state')
        for slot, (name, _role, _cap) in declared.items():
            if name != 'receipt.json':
                require(trained['outputs'][name]['sha256'] == sha(raw_trained[slot]) and trained['outputs'][name]['bytes'] == len(raw_trained[slot]), 'trained_hash', name)
        require(raw_trained['trained_normalization_json'] == canonical(normalization) and raw_trained['trained_splits_json'] == inputs['splits'], 'restore_inputs', 'exact preprocessing and population')
        observed = decode(raw_trained['trained_history_json']); checkpoint_specs = observed['checkpoints']
        updates = list(range(0, config['training']['updates']+1, config['training']['checkpoint_interval']))
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
    receipt = {'schema': 'rheon-variable-pressure-training-v1', 'status': 'complete', 'mode': args.mode,
               'optimizer_updates': config['training']['updates'] if args.mode == 'train' else 0,
               'closure': closure, 'cases': population, 'prediction_inventory': prediction_inventory,
               'per_case_inference_s': clocks, 'outputs': artifact_specs, 'originating_training_run': origin,
               'architecture': model.ARCHITECTURE, 'elapsed_s': time.perf_counter()-start,
               'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(), 'peak_rss_bytes': helpers.observed_peak_rss_bytes()}}
    require(sum(s['bytes'] for s in artifact_specs.values())+len(canonical(receipt)) <= config['resources']['max_output_bytes'], 'output_budget', 'complete history and publication')
    write(output/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=('train', 'restore'))
    for name in (*DATA, 'config', 'runtime-manifest', 'source-witness', 'input-evidence', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('weights.npz', 'predictions.npz', 'normalization.json', 'splits.json', 'history.json', 'receipt.json', *(f'checkpoint-{u:03d}.npz' for u in range(201))):
        slot = 'trained_'+name.replace('-', '_').replace('.', '_'); parser.add_argument('--'+slot.replace('_', '-'), dest=slot, type=Path)
    args = parser.parse_args(); config = experiment(decode(read(args.config)))
    require(args.mode != 'restore' or all(getattr(args, slot) is not None for slot in trained_inputs(config)), 'restore_slots', 'all originating artifacts')
    require(args.mode != 'train' or all(value is None for key, value in vars(args).items() if key.startswith('trained_')), 'train_slots', 'fresh training only')
    run(args)


if __name__ == '__main__':
    main()
