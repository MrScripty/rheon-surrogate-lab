"""Managed zero-update recovery of completed fits after probe packaging failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from managed_input_stream_v1 import streamed_identity
from managed_fit_artifacts import restore_custody
from portable_shared_runtime_v1 import PortableSharedRuntime


def read(p): return json.loads(Path(p).read_text())
def write(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')
def sha(p): return streamed_identity(p)['sha256']


def restore_network(network, path, width):
    with np.load(path, allow_pickle=False) as source:
        values = {}
        for n, current in network.state_dict().items():
            value = source[f'w{width}.{n}']
            if value.dtype.str != '<f4' or tuple(value.shape) != tuple(current.shape) or not np.isfinite(value).all():
                raise ValueError('Exact finite typed recovered weights required')
            values[n] = torch.from_numpy(value.copy()).to(current.device)
    network.load_state_dict(values, strict=True)


def run(args):
    meta = read(args.manifest)
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
        raise ValueError('Fresh managed recovery output required')
    device = 'cuda' if args.mode == 'recover' else 'cpu'
    if os.environ.get('TULDOK_TORCH_DEVICE') != device:
        raise ValueError('Explicit recovery runtime device required')
    static = dict(manifest=args.manifest, normalization=args.normalization,
        runtime_manifest=args.runtime_manifest, model_source=args.model_source)
    if args.mode == 'recover':
        static.update(original_weights=args.original_weights, original_initial=args.original_initial,
            original_last=args.original_last, probe_state=args.probe_state)
    evidence, identities, owners = read(args.input_evidence), {}, set()
    artifacts = {'trained_weights', 'trained_initial', 'trained_last', 'trained_history', 'trained_probe', 'trained_receipt'} if args.mode == 'restore' else set()
    if set(evidence) != set(static) | artifacts:
        raise ValueError('Exact recovery input inventory required')
    for slot, path in static.items():
        identities[slot] = streamed_identity(path)
        row = evidence[slot]
        if row['reference_id'] != slot or row['measured_sha256'] != identities[slot]['sha256']:
            raise ValueError('Recovery static Project identity differs')
        owners.add((row['project_id'], row['manifest_sha256']))
    if len(owners) != 1 or meta['original_terminal']['state'] != 'failed' or meta['original_terminal']['custody_phase'] != 'drained':
        raise ValueError('Same Project and authoritatively drained original failure required')
    runtime = read(args.runtime_manifest)
    if runtime['packages']['torch'] != torch.__version__ or runtime['packages']['numpy'] != np.__version__:
        raise ValueError('Qualified recovery runtime differs')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    closure = dict(inputs=identities, sources={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')})
    allowed = [*static.values(), args.input_evidence,
        *[Path(__file__).with_name(n) for n in ('custody_validator.py', 'managed_fit_artifacts.py')]]
    began = time.monotonic()
    with PortableSharedRuntime(args.model_source, meta['source_manifest'], meta['forbidden_repository_root'], allowed, args.output_dir) as portable:
        from captured_parameter_inventory_v1 import inventory
        from shared_temporal_network_v1 import SharedTemporalResidual
        if args.mode == 'recover':
            for slot in ('original_weights', 'original_initial', 'original_last'):
                if identities[slot] != meta['original_checkpoint_identities'][slot]:
                    raise ValueError('Preserved failed-Run checkpoint differs')
            for width, scores in meta['epoch_scores'].items():
                if len(scores) != 4 or any(score < meta['config']['initial_objective_by_width'][width] for score in scores):
                    raise ValueError('Recovery assumes logged selection retained original initial weights')
            if sha(args.original_weights) != sha(args.original_initial):
                raise ValueError('Selected initial checkpoint recovery differs from recorded selection')
            with np.load(args.probe_state, allow_pickle=False) as archive:
                fields = {n: torch.from_numpy(archive[n].copy()).to('cuda') for n in ('pressure', 'vx', 'vy', 'vz', 'tracer')}
            scene = meta['probe_scene']
            if scene['counts'] != [32, 32, 32]:
                raise ValueError('Explicit32cube bounded complete probe required')
            from shared_temporal_fit_core_v2 import make_model
            history = dict(initial_inventory={}, final_inventory={}, last_inventory={}, epoch_scores=meta['epoch_scores'],
                original_training_run=meta['original_terminal']['id'], original_training_Run_state='failed',
                recovered_completed_parameter_fits=2, original_updates_by_width={'4': 960, '8': 960},
                detailed_epoch_window_history_available=False,
                detail='Original trainer completed all8epochs and wrote checkpoints, then exceeded its probe file bound before serializing in-memory detailed history. Original scalar epoch scores and exact checkpoints retained; no retraining.',
                probe_labels=[], selected_epoch_by_width={'4': 0, '8': 0})
            vectors = []
            for width in (4, 8):
                model = make_model(meta['source_manifest']['configuration'], width)
                try:
                    for kind, path in [('initial', args.original_initial), ('selected', args.original_weights), ('last', args.original_last)]:
                        restore_network(model.network, path, width)
                        history[{'initial': 'initial_inventory', 'selected': 'final_inventory', 'last': 'last_inventory'}[kind]][str(width)] = inventory(model.network)
                        if kind != 'initial':
                            with torch.inference_mode():
                                outputs = model.predict(fields, scene, meta['probe_start_fine_step'], 60, 4)
                                vector = torch.cat([output[n].double().flatten() for output in outputs for n in ('pressure', 'vx', 'vy', 'vz', 'tracer')])
                                if not torch.isfinite(vector).all():
                                    raise ValueError('Nonfinite recovered full GPU probe')
                                vectors.append(vector.cpu().numpy())
                                history['probe_labels'].append(f'width{width}-{kind}')
                finally:
                    model.close()
            probe = np.stack(vectors)
            if probe.nbytes + 1024 > meta['maximum_probe_bytes']:
                raise ValueError('Explicit probe budget exceeded before writing')
            np.save(args.output_dir / 'probe.npy', probe)
            history['probe_shape'] = list(probe.shape)
            for source, name in [(args.original_weights, 'weights.npz'), (args.original_initial, 'initial-weights.npz'),
                                 (args.original_last, 'last-evaluated-weights.npz')]:
                shutil.copyfile(source, args.output_dir / name)
            write(args.output_dir / 'history.json', history)
            receipt = dict(schema='rheon.shared-temporal-managed-recovery.v1', mode='recover', closure=closure,
                original_training_run=meta['original_terminal']['id'], new_parameter_fits=0, optimizer_updates=0,
                recovered_parameter_fits=2, recovered_original_optimizer_updates=1920,
                original_Run_failed_after_checkpoint_export=True, final_inventory=history['final_inventory'],
                last_inventory=history['last_inventory'], execution=dict(device='cuda', cpu_model_forward=False))
        else:
            origin = restore_custody({n: v for n, v in evidence.items() if n != 'trained_last'}, args)
            import importlib.util
            spec = importlib.util.spec_from_file_location('recovery_extra_custody', Path(__file__).with_name('custody_validator.py'))
            validator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(validator)
            row = evidence['trained_last']
            extra = validator.validate_artifact_source(row['source'], sha(args.trained_last), Path(args.trained_last).stat().st_size)
            if (extra['run_id'] != origin['run_id'] or extra['artifact_name'] != 'last-evaluated-weights.npz' or extra['artifact_role'] != 'checkpoint'):
                raise ValueError('Last checkpoint recovery custody differs')
            trained, history = read(args.trained_receipt), read(args.trained_history)
            if (trained['schema'] != 'rheon.shared-temporal-managed-recovery.v1' or trained['mode'] != 'recover'
                    or trained['closure']['sources'] != closure['sources']
                    or any(trained['closure']['inputs'][n] != v for n, v in identities.items())):
                raise ValueError('Same originating managed GPU recovery closure required')
            for kind, path in [('initial_inventory', args.trained_initial), ('final_inventory', args.trained_weights), ('last_inventory', args.trained_last)]:
                for width in (4, 8):
                    network = SharedTemporalResidual(hidden=width, device='cpu')
                    restore_network(network, path, width)
                    if inventory(network) != history[kind][str(width)]:
                        raise ValueError('Typed CPU recovered checkpoint differs')
            for source, name in [(args.trained_weights, 'weights.npz'), (args.trained_initial, 'initial-weights.npz'),
                                 (args.trained_last, 'last-evaluated-weights.npz'), (args.trained_history, 'history.json'), (args.trained_probe, 'probe.npy')]:
                if sha(source) != trained['outputs'][name]['sha256']:
                    raise ValueError('Managed recovered artifact identity differs')
                shutil.copyfile(source, args.output_dir / name)
            if torch.cuda.is_initialized():
                raise ValueError('CPU recovery restore initialized CUDA')
            receipt = dict(schema='rheon.shared-temporal-managed-recovery.v1', mode='restore', closure=closure,
                originating_GPU_recovery_run=origin, original_training_run=meta['original_terminal']['id'],
                new_parameter_fits=0, optimizer_updates=0, typed_cpu_weight_restore_verified=True,
                gpu_probe_copied_without_cpu_forward=True, cpu_model_forward_executed=False,
                execution=dict(device='cpu', cuda_initialized=False))
        if portable.denied:
            raise ValueError('Recovery attempted ambient scientific reads')
    shutil.copyfile(args.normalization, args.output_dir / 'normalization.json')
    receipt['outputs'] = {p.name: streamed_identity(p) for p in args.output_dir.iterdir() if p.is_file()}
    receipt['elapsed_seconds'] = time.monotonic() - began
    write(args.output_dir / 'receipt.json', receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('recover', 'restore'))
    for name in ('manifest', 'normalization', 'runtime-manifest', 'model-source', 'input-evidence', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('original-weights', 'original-initial', 'original-last', 'probe-state', 'trained-weights', 'trained-initial', 'trained-last', 'trained-history', 'trained-probe', 'trained-receipt'):
        p.add_argument('--' + name, type=Path)
    run(p.parse_args())
