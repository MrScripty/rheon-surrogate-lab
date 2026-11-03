"""Managed full-population guided CUDA fit and exact typed CPU restoration.

Qualification performs no parameter updates. Only the managed training mode
evaluates optimizer trials. Every score covers all24 training masters/all12
configurations; canonical selection ranks worst field error before mean loss.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from managed_input_stream_v1 import streamed_identity
from managed_fit_artifacts import restore_custody
from portable_shared_runtime_v1 import PortableSharedRuntime

CHECKPOINTS = {'trained_weights': 'weights.npz', 'trained_mean': 'best-mean-weights.npz',
    'trained_initial': 'initial-weights.npz', 'trained_last': 'last-evaluated-weights.npz'}
ARTIFACTS = dict(CHECKPOINTS, trained_history='history.json', trained_probe='probe.npy', trained_receipt='receipt.json')
FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')


def read(path): return json.loads(Path(path).read_text())
def sha(path): return streamed_identity(path)['sha256']


def write(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def cpu_state(network):
    return {n: v.detach().cpu().clone() for n, v in network.state_dict().items()}


def state_inventory(state):
    return {n: dict(dtype=v.numpy().dtype.str, shape=list(v.shape),
        sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for n, v in state.items()}


def save_checkpoint(path, state):
    temporary = Path(path).with_name('.' + Path(path).name + '.writing')
    with temporary.open('wb') as stream:
        np.savez(stream, **{'w4.' + n: v.numpy() for n, v in state.items()})
    os.replace(temporary, path)


def restore_network(network, path, expected):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'w4.' + n for n in network.state_dict()}:
            raise ValueError('Exact complete checkpoint keys required')
        values = {}
        for name, current in network.state_dict().items():
            value = archive['w4.' + name]
            if (value.dtype.str != '<f4' or tuple(value.shape) != tuple(current.shape)
                    or not np.isfinite(value).all() or hashlib.sha256(value.tobytes()).hexdigest() != expected[name]['sha256']):
                raise ValueError('Exact finite typed CPU checkpoint differs')
            values[name] = torch.from_numpy(value.copy())
    network.load_state_dict(values, strict=True)
    if state_inventory(cpu_state(network)) != expected:
        raise ValueError('Restored checkpoint inventory differs')


def run(args):
    began = time.monotonic()
    config, meta, normalization = read(args.config), read(args.manifest), read(args.normalization)
    device = 'cpu' if args.mode == 'restore' else 'cuda'
    if os.environ.get('TULDOK_TORCH_DEVICE') != device or not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
        raise ValueError('Explicit runtime device and empty output directory required')
    if (config['widths'] != [4] or config['maximum_iterations'] != 4
            or config['optimizer'] != 'full_population_quadratic_ray' or config['probe_normalized_step'] != .001):
        raise ValueError('Exact registered finite single-width fit required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    static = dict(manifest=args.manifest, normalization=args.normalization,
        runtime_manifest=args.runtime_manifest, model_source=args.model_source)
    if args.mode != 'restore':
        if len(args.shards) != 1:
            raise ValueError('Exactly one captured complete24 input required')
        static['part_00'] = args.shards[0]
    elif args.shards:
        raise ValueError('CPU restoration cannot consume training data')
    evidence, identities, owners = read(args.input_evidence), {}, set()
    if set(evidence) != set(static) | (set(ARTIFACTS) if args.mode == 'restore' else set()):
        raise ValueError('Exact typed input inventory required')
    for slot, path in static.items():
        row, identity = evidence[slot], streamed_identity(path)
        if row['reference_id'] != slot or row['measured_sha256'] != identity['sha256']:
            raise ValueError('Typed static input identity differs')
        identities[slot] = identity
        owners.add((row['project_id'], row['manifest_sha256']))
    if len(owners) != 1:
        raise ValueError('Mixed Project inputs rejected')
    runtime = read(args.runtime_manifest)
    if runtime['packages']['torch'] != torch.__version__ or runtime['packages']['numpy'] != np.__version__:
        raise ValueError('Exact qualified runtime required')
    if sha(args.normalization) != meta['normalization_sha256'] or sha(args.model_source) != meta['model_source_sha256']:
        raise ValueError('Frozen source or loss normalization differs')
    qualification, eligibility, index = meta['objective_qualification'], meta['reference_eligibility'], meta['qualified_index']
    if (qualification['status'] != 'passed_portable_guided_full24_all12_zero_heads_and_gradients'
            or not qualification['correction_units_guide_RMS'] or qualification['unexpected_ambient_reads'] != 0
            or eligibility['status'] != 'qualified_selected240_reference_epochs_for_full36_training'
            or not index['training_ready_full36'] or index['independent_masters_per_split'] != {'train': 24, 'development': 12}):
        raise ValueError('Qualified full36 references and guide-RMS full24 objective required')
    closure = dict(config_sha256=sha(args.config), inputs=identities,
        sources={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')})
    allowed = [*static.values(), args.config, args.input_evidence,
        *[getattr(args, n) for n in ARTIFACTS if getattr(args, n) is not None],
        *[Path(__file__).with_name(n) for n in ('custody_validator.py', 'managed_fit_artifacts.py')]]
    model = bank = None
    try:
        with PortableSharedRuntime(args.model_source, meta['source_manifest'], meta['forbidden_repository_root'], allowed, args.output_dir) as portable:
            from captured_parameter_inventory_v1 import inventory
            from shared_temporal_network_v1 import SharedTemporalResidual
            if args.mode == 'restore':
                origin = restore_custody({n: v for n, v in evidence.items() if n not in ('trained_mean', 'trained_last')}, args)
                spec = importlib.util.spec_from_file_location('guided_extra_custody', Path(__file__).with_name('custody_validator.py'))
                validator = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validator)
                for slot in ('trained_mean', 'trained_last'):
                    path, row = getattr(args, slot), evidence[slot]
                    observed = validator.validate_artifact_source(row['source'], sha(path), path.stat().st_size)
                    if (row['kind'] != 'run_artifact' or observed['artifact_name'] != ARTIFACTS[slot]
                            or observed['artifact_role'] != 'checkpoint'
                            or any(observed[n] != origin[n] for n in origin)):
                        raise ValueError('Additional checkpoint custody differs from originating fit')
                trained, history = read(args.trained_receipt), read(args.trained_history)
                if (trained['schema'] != 'rheon.guided-shared-managed-fit.v1' or trained['mode'] != 'train'
                        or trained['closure']['sources'] != closure['sources']
                        or trained['closure']['config_sha256'] != closure['config_sha256']
                        or any(trained['closure']['inputs'][n] != v for n, v in identities.items())):
                    raise ValueError('Exact originating managed CUDA fit closure required')
                for slot, name in CHECKPOINTS.items():
                    path = getattr(args, slot)
                    restore_network(SharedTemporalResidual(hidden=4, device='cpu'), path, history['checkpoint_inventories'][name])
                for slot, name in ARTIFACTS.items():
                    if slot == 'trained_receipt':
                        continue
                    path = getattr(args, slot)
                    if sha(path) != trained['outputs'][name]['sha256']:
                        raise ValueError('Originating GPU artifact identity differs')
                    shutil.copyfile(path, args.output_dir / name)
                probe = np.load(args.trained_probe, allow_pickle=False)
                if probe.dtype.str != '<f8' or list(probe.shape) != history['probe_shape'] or not np.isfinite(probe).all():
                    raise ValueError('Copied complete GPU probe differs')
                if torch.cuda.is_initialized():
                    raise ValueError('CPU restore unexpectedly initialized CUDA')
                receipt = dict(schema='rheon.guided-shared-managed-fit.v1', mode='restore', closure=closure,
                    originating_training_run=origin, optimizer_updates=0, parameter_fits=0,
                    typed_cpu_weight_restore_verified=True, restored_checkpoints=sorted(CHECKPOINTS.values()),
                    gpu_probe_copied_without_cpu_forward=True, cpu_model_forward_executed=False,
                    execution=dict(device='cpu', cuda_initialized=False))
            else:
                from captured_long_training_bank_v1 import CapturedLongTrainingBank
                from guided_shared_fit_core_v3 import make_model, full_objective, physical_block
                from guided_full_population_optimizer_v1 import snapshot, normalized_direction, parameter_trial, quadratic_trial_scale
                if identities['part_00'] != meta['archive']:
                    raise ValueError('Complete captured native training archive differs')
                bank = CapturedLongTrainingBank(args.shards[0], meta['archive'])
                if hashlib.sha256(json.dumps(bank.manifest, sort_keys=True).encode()).hexdigest() != meta['bank_semantic_sha256']:
                    raise ValueError('Captured scene/target semantics differ')
                for identifier, case in bank.manifest['masters'].items():
                    if (index['masters'][identifier]['split'] != 'train'
                            or case['native_receipt_sha256'] != eligibility['masters'][identifier]['receipt_sha256']):
                        raise ValueError('Training bank differs from qualified population')
                model = make_model(meta['source_manifest']['configuration'], 4)
                if inventory(model) != config['initial_model_inventory']:
                    raise ValueError('Qualified complete initial model inventory differs')
                initial = cpu_state(model.network)
                window = next(w for w in bank.windows if w['identifier'] == config['probe_identifier'] and w['start_fine_step'] == config['probe_start_fine_step'])
                case = bank.manifest['masters'][window['identifier']]
                if case['scene']['counts'] != [32, 32, 32]:
                    raise ValueError('Explicit32cube probe required before optimization')
                native = bank.load(window['identifier'], window['start_fine_step'])
                probe_fields = {n: torch.from_numpy(native[n]).to('cuda') for n in FIELDS}
                vector_size = 4 * sum(v.numel() for v in probe_fields.values())
                if vector_size != 667648 or 3 * vector_size * 8 + 4096 > config['maximum_probe_bytes']:
                    raise ValueError('Preflight probe shape/byte budget differs')

                def bounded():
                    if (time.monotonic() - began > config['wall_seconds'] - 30
                            or torch.cuda.max_memory_allocated() > config['maximum_gpu_allocated_bytes']
                            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > config['maximum_peak_rss_bytes']):
                        raise ValueError('Finite guided fit resource budget exceeded')

                if args.mode == 'qualify':
                    item = bank.cuda_window(bank.windows[0])
                    measured, maxima = 0., {}
                    for example in bank.windows[0]['configurations']:
                        model.network.zero_grad(set_to_none=True)
                        diagnostics = {}
                        loss, unused = physical_block(model, item['current'], item['scene'], example, item['targets'], normalization,
                            certify_zero_heads=True, diagnostics=diagnostics)
                        measured += float(loss.detach()) / 12
                        loss.backward()
                        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.network.parameters()):
                            raise ValueError('Nonfinite actual entrypoint gradient')
                        for n, v in diagnostics['worst_relative'].items(): maxima[n] = max(v, maxima.get(n, 0.))
                    expected = next(r for r in qualification['rows'] if r['hidden_width'] == 4)['rows'][0]['objective']
                    if not math.isclose(measured, expected, rel_tol=1e-10, abs_tol=1e-30) or cpu_state(model.network).keys() != initial.keys():
                        raise ValueError('Actual zero-update entrypoint objective differs')
                    if state_inventory(cpu_state(model.network)) != state_inventory(initial):
                        raise ValueError('Qualification changed initial parameters')
                    receipt = dict(schema='rheon.guided-shared-managed-fit.v1', mode='qualify', closure=closure,
                        parameter_fits=0, optimizer_updates=0, cpu_model_forward_executed=False,
                        first_window_objective=measured, first_window_worst_relative=maxima, configurations=12,
                        probe_preflight=dict(counts=[32,32,32], shape=[3,vector_size], payload_bytes=3*vector_size*8),
                        execution=dict(device='cuda', ambient_repository_reads=len(portable.denied),
                            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                        checkpoint_inventories={'initial-weights.npz': state_inventory(initial)})
                    save_checkpoint(args.output_dir / 'initial-weights.npz', initial)
                else:
                    states = {'initial-weights.npz': initial, 'weights.npz': initial,
                        'best-mean-weights.npz': initial, 'last-evaluated-weights.npz': initial}
                    history = dict(iterations=[], trials=[], checkpoint_inventories={}, independent_training_masters=24,
                        initial_objective=config['initial_objective'], maximum_iterations=config['maximum_iterations'],
                        selected_worst_trial='initial', best_mean_trial='initial', last_evaluated_trial='initial',
                        parameter_trial_updates=0, accepted_steps=0, full_gradient_evaluations=0, full_forward_evaluations=0,
                        probe_preflight=dict(identifier=window['identifier'], counts=[32,32,32], shape=[3,vector_size], payload_bytes=3*vector_size*8),
                        guide_autograd=False, correction_units_guide_RMS=True)
                    best_worst = best_mean = None

                    def persist():
                        for name, state in states.items():
                            save_checkpoint(args.output_dir / name, state)
                            history['checkpoint_inventories'][name] = state_inventory(state)
                        write(args.output_dir / 'history.json', history)

                    def remember(score, label):
                        nonlocal best_worst, best_mean
                        value = cpu_state(model.network)
                        states['last-evaluated-weights.npz'] = value
                        history['last_evaluated_trial'] = label
                        rank = (score['maximum_relative_error'], score['objective'])
                        if best_worst is None or rank < best_worst:
                            best_worst, states['weights.npz'] = rank, value
                            history.update(selected_worst_trial=label, selected_worst_score=score)
                        if best_mean is None or score['objective'] < best_mean:
                            best_mean, states['best-mean-weights.npz'] = score['objective'], value
                            history.update(best_mean_trial=label, best_mean_score=score)
                        history.pop('pending_trial', None)
                        persist()
                        (args.output_dir / 'pending-trial-weights.npz').unlink(missing_ok=True)

                    def record_pending(label, step):
                        save_checkpoint(args.output_dir / 'pending-trial-weights.npz', cpu_state(model.network))
                        history['pending_trial'] = dict(label=label, step=step, fully_scored=False)
                        persist()

                    def score(backward, label, receiver=None, zero=False):
                        def progress(n, count, value):
                            if n % 24 == 0:
                                print(json.dumps(dict(format='tuldok_progress_v1', stage=label, completed=n,
                                    total=count, metrics={'loss': value})), flush=True)
                        value = full_objective(model, bank, normalization, backward, bounded, progress,
                            certify_zero_heads=zero, gradient_receiver=receiver)
                        history['full_gradient_evaluations' if backward else 'full_forward_evaluations'] += 1
                        return value

                    for iteration in range(config['maximum_iterations']):
                        gradients = {}
                        current = score(True, f'gradient-{iteration+1}', gradients, zero=iteration == 0)
                        if iteration == 0:
                            if not math.isclose(current['objective'], config['initial_objective'], rel_tol=1e-9, abs_tol=1e-20):
                                raise ValueError('Managed full-population baseline differs from portable qualification')
                            history['initial_full_score'] = current
                            remember(current, 'initial')
                        base = snapshot(model.network)
                        direction, norm = normalized_direction(model.network, gradients)
                        if direction is None:
                            history['stop_reason'] = 'zero_complete_population_gradient'
                            break
                        step = config['probe_normalized_step']
                        parameter_trial(model.network, base, direction, step)
                        history['parameter_trial_updates'] += 1
                        record_pending(f'probe-{iteration+1}', step)
                        probe_score = score(False, f'probe-{iteration+1}')
                        history['trials'].append(dict(label=f'probe-{iteration+1}', step=step, score=probe_score))
                        remember(probe_score, f'probe-{iteration+1}')
                        proposal = quadratic_trial_scale(current['objective'], probe_score['objective'], step, norm)
                        candidate_step = proposal['proposed_step']
                        if candidate_step == step:
                            candidate = probe_score
                        else:
                            parameter_trial(model.network, base, direction, candidate_step)
                            history['parameter_trial_updates'] += 1
                            record_pending(f'candidate-{iteration+1}', candidate_step)
                            candidate = score(False, f'candidate-{iteration+1}')
                            history['trials'].append(dict(label=f'candidate-{iteration+1}', step=candidate_step, score=candidate))
                            remember(candidate, f'candidate-{iteration+1}')
                        accepted = candidate['objective'] < current['objective']
                        history['iterations'].append(dict(iteration=iteration+1, current=current,
                            gradient_norm=norm, proposal=proposal, candidate=candidate, accepted=accepted))
                        if accepted:
                            history['accepted_steps'] += 1
                        else:
                            model.network.load_state_dict(base, strict=True)
                            history['stop_reason'] = 'full_population_candidate_did_not_improve'
                        persist()
                        if not accepted:
                            break
                    history.setdefault('stop_reason', 'completed_registered_iteration_limit')
                    persist()
                    vectors = []
                    for name in ('weights.npz', 'best-mean-weights.npz', 'last-evaluated-weights.npz'):
                        model.network.load_state_dict(states[name], strict=True)
                        with torch.inference_mode():
                            outputs = model.predict(probe_fields, case['scene'], window['start_fine_step'], 60, 4)
                            vector = torch.cat([o[n].double().flatten() for o in outputs for n in FIELDS])
                            if vector.numel() != vector_size or not torch.isfinite(vector).all():
                                raise ValueError('Finite complete bounded GPU probe required')
                            vectors.append(vector.cpu().numpy())
                    probe = np.stack(vectors)
                    if probe.nbytes + 4096 > config['maximum_probe_bytes']:
                        raise ValueError('Probe serialization budget exceeded')
                    np.save(args.output_dir / 'probe.npy', probe)
                    history.update(probe_shape=list(probe.shape), probe_labels=['best-worst','best-mean','last-evaluated'])
                    persist()
                    receipt = dict(schema='rheon.guided-shared-managed-fit.v1', mode='train', closure=closure,
                        parameter_fits=int(history['parameter_trial_updates'] > 0), optimizer_updates=history['parameter_trial_updates'],
                        accepted_steps=history['accepted_steps'], checkpoint_inventories=history['checkpoint_inventories'],
                        cpu_model_forward_executed=False, model_family='guided_shared_residual_v3',
                        full_generator_target_pass=False, training_master_count=24, reserved52_references_opened=0,
                        execution=dict(device='cuda', gpu=torch.cuda.get_device_name(),
                            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                            ambient_repository_reads=len(portable.denied)))
            if portable.denied:
                raise ValueError('Actual entrypoint attempted ambient scientific reads')
    finally:
        if model is not None: model.close()
        if bank is not None: bank.close()
    shutil.copyfile(args.normalization, args.output_dir / 'normalization.json')
    receipt['outputs'] = {p.name: streamed_identity(p) for p in args.output_dir.iterdir() if p.is_file()}
    receipt['elapsed_seconds'] = time.monotonic()-began
    write(args.output_dir / 'receipt.json', receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train','restore','qualify'))
    for name in ('config','manifest','normalization','runtime-manifest','model-source','input-evidence','output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--shards', nargs='*', type=Path, default=[])
    for name in ARTIFACTS:
        p.add_argument('--'+name.replace('_','-'), type=Path)
    run(p.parse_args())
