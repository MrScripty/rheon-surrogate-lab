"""Managed independent width4/8 CUDA fits with one captured native data copy.

Qualification checks integration on the first complete12configuration window;
the bound complete24scene objective/gradient evidence supplies the broad check.
Only managed training mode creates an optimizer. CPU mode restores typed
weights and copies the original GPU probe without executing a learned forward.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import resource

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from managed_input_stream_v1 import streamed_identity
from managed_fit_artifacts import restore_custody
from portable_shared_runtime_v1 import PortableSharedRuntime


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return streamed_identity(path)['sha256']


def save_pair(path, states):
    np.savez(path, **{f'w{width}.{name}': value.numpy() for width, state in states.items() for name, value in state.items()})


def cpu_state(network):
    return {n: v.detach().cpu().clone() for n, v in network.state_dict().items()}


def validate(args):
    static = dict(manifest=args.manifest, normalization=args.normalization,
        runtime_manifest=args.runtime_manifest, model_source=args.model_source)
    if args.mode != 'restore':
        if len(args.shards) != 1:
            raise ValueError('Exactly one captured complete24scene input required')
        static['part_00'] = args.shards[0]
    elif args.shards:
        raise ValueError('CPU restore must not consume the training archive')
    evidence = read(args.input_evidence)
    artifacts = {'trained_weights', 'trained_initial', 'trained_history', 'trained_probe', 'trained_receipt'} if args.mode == 'restore' else set()
    if set(evidence) != set(static) | artifacts:
        raise ValueError('Exact typed input inventory required')
    owners, identities = set(), {}
    for slot, path in static.items():
        row = evidence[slot]
        identities[slot] = streamed_identity(path)
        if row['reference_id'] != slot or row['measured_sha256'] != identities[slot]['sha256']:
            raise ValueError('Typed captured input identity differs')
        owners.add((row['project_id'], row['manifest_sha256']))
    if len(owners) != 1:
        raise ValueError('Mixed Project inputs rejected')
    return evidence, identities


def run(args):
    started = time.monotonic()
    config, manifest, normalization = read(args.config), read(args.manifest), read(args.normalization)
    device = 'cpu' if args.mode == 'restore' else 'cuda'
    if os.environ.get('TULDOK_TORCH_DEVICE') != device:
        raise ValueError('Explicit qualified runtime device required')
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
        raise ValueError('Empty managed output directory required')
    if config['widths'] != [4, 8] or config['epochs_per_width'] != 4 or config['optimizer'] != 'Adam':
        raise ValueError('Exact predeclared dualwidth program required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    evidence, identities = validate(args)
    runtime = read(args.runtime_manifest)
    if runtime['packages']['torch'] != torch.__version__ or runtime['packages']['numpy'] != np.__version__:
        raise ValueError('Exact qualified runtime package versions required')
    if sha(args.normalization) != manifest['normalization_sha256'] or sha(args.model_source) != manifest['model_source_sha256']:
        raise ValueError('Captured model source or frozen physical normalization differs')
    eligibility, index = manifest['reference_eligibility'], manifest['qualified_index']
    if (eligibility['status'] != 'qualified_selected240_reference_epochs_for_full36_training'
            or not index['training_ready_full36'] or len(index['masters']) != 36
            or index['independent_masters_per_split'] != {'train': 24, 'development': 12}
            or set(eligibility['masters']) != set(index['masters'])):
        raise ValueError('Complete explicitly qualified36scene reference set required')
    qualification = manifest['objective_qualification']
    if qualification['status'] != 'passed_portable_full24_all12_physical_objective_and_gradients':
        raise ValueError('Actual complete captured24scene objective qualification required')
    closure = dict(config_sha256=sha(args.config), inputs=identities,
        sources={p.name: sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file() and p.suffix == '.py'})
    allowed = [args.config, args.manifest, args.normalization, args.runtime_manifest, args.model_source,
               args.input_evidence, *args.shards, *[Path(__file__).with_name(n) for n in
               ('custody_validator.py', 'managed_fit_artifacts.py')]]
    updates = {4: 0, 8: 0}
    model = bank = None
    completed = False
    try:
        with PortableSharedRuntime(args.model_source, manifest['source_manifest'],
                manifest['forbidden_repository_root'], allowed, args.output_dir) as portable:
            from captured_parameter_inventory_v1 import inventory
            from shared_temporal_network_v1 import SharedTemporalResidual
            if args.mode == 'restore':
                origin = restore_custody(evidence, args)
                trained, history = read(args.trained_receipt), read(args.trained_history)
                if (trained['schema'] != 'rheon.shared-temporal-managed-pair.v1' or trained['mode'] != 'train'
                        or trained['closure']['config_sha256'] != closure['config_sha256']
                        or trained['closure']['sources'] != closure['sources']
                        or {n: v for n, v in trained['closure']['inputs'].items() if n != 'part_00'} != identities):
                    raise ValueError('Exact originating CUDA fit closure required')
                expected_keys = set()
                for kind, path in [('initial_inventory', args.trained_initial), ('final_inventory', args.trained_weights)]:
                    with np.load(path, allow_pickle=False) as archive:
                        for width in config['widths']:
                            network = SharedTemporalResidual(hidden=width, device='cpu')
                            values = {}
                            for n, current in network.state_dict().items():
                                key = f'w{width}.{n}'
                                expected_keys.add(key)
                                a = archive[key]
                                if a.dtype.str != '<f4' or tuple(a.shape) != tuple(current.shape):
                                    raise ValueError('Complete typed CPU parameter restoration required')
                                values[n] = torch.from_numpy(a.copy())
                            network.load_state_dict(values, strict=True)
                            if inventory(network) != history[kind][str(width)]:
                                raise ValueError('CPU restored parameter bytes differ')
                        if set(archive.files) != expected_keys:
                            raise ValueError('Unexpected or missing restored parameters')
                for source, name in [(args.trained_weights, 'weights.npz'), (args.trained_initial, 'initial-weights.npz'),
                        (args.trained_history, 'history.json'), (args.trained_probe, 'probe.npy')]:
                    if sha(source) != trained['outputs'][name]['sha256']:
                        raise ValueError('Originating GPU artifact identity differs')
                    shutil.copyfile(source, args.output_dir / name)
                probe = np.load(args.trained_probe, allow_pickle=False)
                if probe.dtype.str != '<f8' or list(probe.shape) != history['probe_shape']:
                    raise ValueError('Copied complete GPU probe differs')
                if torch.cuda.is_initialized():
                    raise ValueError('CPU restore unexpectedly initialized CUDA')
                receipt = dict(schema='rheon.shared-temporal-managed-pair.v1', mode='restore', closure=closure,
                    optimizer_updates=0, originating_training_run=origin, typed_cpu_weight_restore_verified=True,
                    gpu_probe_copied_without_cpu_forward=True, cpu_model_forward_executed=False,
                    execution=dict(device='cpu', cuda_initialized=False))
            else:
                from captured_long_training_bank_v1 import CapturedLongTrainingBank
                from shared_temporal_fit_core_v2 import make_model, full_objective, physical_block
                from shared_temporal_optimizer_v1 import epoch
                if identities['part_00'] != manifest['archive']:
                    raise ValueError('Complete captured native training archive differs')
                bank = CapturedLongTrainingBank(args.shards[0], manifest['archive'])
                if hashlib.sha256(json.dumps(bank.manifest, sort_keys=True).encode()).hexdigest() != manifest['bank_semantic_sha256']:
                    raise ValueError('Captured training recipes or target identities differ')
                for identifier, case in bank.manifest['masters'].items():
                    if (index['masters'][identifier]['split'] != 'train'
                            or case['native_receipt_sha256'] != eligibility['masters'][identifier]['receipt_sha256']):
                        raise ValueError('Training bank differs from qualified physical-scene population')
                initial, final, last, initial_inventory, final_inventory = {}, {}, {}, {}, {}
                history = dict(widths={}, initial_inventory=initial_inventory, final_inventory=final_inventory,
                    source_qualification_sha256=config['initial_qualification_sha256'], independent_training_masters=24)
                probes = []

                def bounded():
                    if (time.monotonic() - started > config['wall_seconds'] - 20
                            or torch.cuda.max_memory_allocated() > config['maximum_gpu_allocated_bytes']
                            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > config['maximum_peak_rss_bytes']):
                        raise ValueError('Registered finite fit resource budget exceeded')

                for width in config['widths']:
                    model = make_model(manifest['source_manifest']['configuration'], width)
                    if inventory(model) != config['initial_inventory_by_width'][str(width)]:
                        raise ValueError('Qualified initial model parameters differ')
                    initial[width] = cpu_state(model.network)
                    initial_inventory[str(width)] = inventory(model.network)
                    if args.mode == 'qualify':
                        window = bank.windows[0]
                        item = bank.cuda_window(window)
                        measured = 0.
                        for example in window['configurations']:
                            model.network.zero_grad(set_to_none=True)
                            loss, unused = physical_block(model, item['current'], item['scene'], example, item['targets'], normalization)
                            measured += float(loss.detach()) / 12
                            loss.backward()
                            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.network.parameters()):
                                raise ValueError('Nonfinite integration-probe gradient')
                        expected = next(r for r in qualification['rows'] if r['hidden_width'] == width)['rows'][0]['objective']
                        if not math.isclose(measured, expected, rel_tol=1e-10, abs_tol=1e-30):
                            raise ValueError('Actual entrypoint physical objective differs from full qualified core')
                        if inventory(model.network) != initial_inventory[str(width)]:
                            raise ValueError('Zero-update entrypoint qualification changed parameters')
                        history['widths'][str(width)] = dict(first_window_mean_objective=measured, configurations=12, optimizer_updates=0)
                    else:
                        class CountedAdam(torch.optim.Adam):
                            def step(self, *a, **kw):
                                value = super().step(*a, **kw)
                                updates[width] += 1
                                return value
                        optimizer = CountedAdam(model.network.parameters(), lr=config['learning_rate'],
                            betas=tuple(config['optimizer_betas']), eps=config['optimizer_epsilon'],
                            weight_decay=config['weight_decay'], foreach=config['foreach'])
                        best = config['initial_objective_by_width'][str(width)]
                        selected_epoch, best_state, epochs = 0, initial[width], []
                        for iteration in range(config['epochs_per_width']):
                            def progress(n, total, value):
                                if n % 24 == 0:
                                    print(json.dumps(dict(format='tuldok_progress_v1', stage=f'width{width}-epoch{iteration + 1}',
                                        completed=n, total=total, metrics={'loss': value})), flush=True)
                            trained_epoch = epoch(model, bank, normalization, optimizer,
                                config['shuffle_seed'] + iteration, bounded, progress)
                            scored = full_objective(model, bank, normalization, False, bounded)
                            epochs.append(dict(epoch=iteration + 1, training=trained_epoch, scored=scored))
                            if scored['objective'] < best:
                                best, selected_epoch, best_state = scored['objective'], iteration + 1, cpu_state(model.network)
                            print(json.dumps(dict(format='tuldok_progress_v1', stage=f'width{width}-selection',
                                completed=iteration + 1, total=config['epochs_per_width'], metrics={'loss': scored['objective']})), flush=True)
                        if updates[width] != config['expected_updates_per_width']:
                            raise ValueError('Incomplete all24scene finite optimizer schedule')
                        last[width] = cpu_state(model.network)
                        model.network.load_state_dict(best_state, strict=True)
                        history['widths'][str(width)] = dict(initial_objective=config['initial_objective_by_width'][str(width)],
                            selected_objective=best, selected_epoch=selected_epoch, optimizer_updates=updates[width], epochs=epochs)
                        window = bank.windows[0]
                        item = bank.cuda_window(window)
                        with torch.inference_mode():
                            outputs = model.predict(item['current'], item['scene'], window['start_fine_step'], 60, 4)
                            vector = torch.cat([output[n].double().flatten() for output in outputs
                                for n in ('pressure', 'vx', 'vy', 'vz', 'tracer')])
                            probes.append(vector.cpu().numpy().astype('<f8', copy=False))
                    final[width] = cpu_state(model.network)
                    final_inventory[str(width)] = inventory(model.network)
                    model.close()
                    model = None
                    bounded()
                save_pair(args.output_dir / 'initial-weights.npz', initial)
                save_pair(args.output_dir / 'weights.npz', final)
                if args.mode == 'train':
                    save_pair(args.output_dir / 'last-evaluated-weights.npz', last)
                    probe = np.stack(probes)
                    np.save(args.output_dir / 'probe.npy', probe)
                    history['probe_shape'] = list(probe.shape)
                write(args.output_dir / 'history.json', history)
                receipt = dict(schema='rheon.shared-temporal-managed-pair.v1', mode=args.mode, closure=closure,
                    optimizer_updates=sum(updates.values()), updates_by_width=updates,
                    parameter_fits=sum(v > 0 for v in updates.values()),
                    qualification_scope='first complete12configuration window perwidth; complete24scene core evidence bound separately' if args.mode == 'qualify' else None,
                    initial_inventory=initial_inventory, final_inventory=final_inventory,
                    cpu_model_forward_executed=False, execution=dict(device='cuda', gpu=torch.cuda.get_device_name(),
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), ambient_repository_reads=len(portable.denied)),
                    training_master_count=24, complete_reference_master_count=36, reserved52_references_opened=0,
                    combined_goal_met=False)
            if portable.denied:
                raise ValueError('Actual entrypoint attempted ambient scientific reads')
    finally:
        if model is not None:
            model.close()
        if bank is not None:
            bank.close()
    shutil.copyfile(args.normalization, args.output_dir / 'normalization.json')
    receipt['outputs'] = {p.name: streamed_identity(p) for p in args.output_dir.iterdir() if p.is_file()}
    receipt['elapsed_seconds'] = time.monotonic() - started
    write(args.output_dir / 'receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'restore', 'qualify'))
    for name in ('config', 'manifest', 'normalization', 'runtime-manifest', 'model-source', 'input-evidence', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--shards', nargs='*', type=Path, default=[])
    for name in ('trained-weights', 'trained-initial', 'trained-history', 'trained-probe', 'trained-receipt'):
        parser.add_argument('--' + name, type=Path)
    run(parser.parse_args())
