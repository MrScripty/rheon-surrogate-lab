"""Zero-update checks of the actual fitting objective and typed double artifacts."""
import argparse
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from profile_frozen_components import ROOT, read, write, digest
from learned_modal_pressure import LearnedModalPressure
from modal_precision_supervision import prepare, objective, initialize
from modal_basis import bases, transform
from train_modal_augmented import inventory, save, load, data
import torch


def run(args):
    if args.output.exists(): raise ValueError('Fresh actual objective qualification required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected: raise ValueError('Qualified objective source/data changed: ' + path)
    if os.environ.get('TULDOK_TORCH_DEVICE') != args.device: raise ValueError('Runtime device differs')
    os.sched_setaffinity(0, {reg['cpu']}); torch.set_num_threads(1)
    torch.manual_seed(reg['seed']); torch.use_deterministic_algorithms(True)
    manifest_path = ROOT / reg['manifest']; manifest = read(manifest_path)
    config = read(ROOT / reg['config'])
    normalization=read(manifest_path.parent/'normalization.json')
    model = initialize(LearnedModalPressure(manifest['geometries']),normalization).to(args.device); before = inventory(model)
    groups = data(SimpleNamespace(shards=[manifest_path.parent/r['path'] for r in manifest['geometries']]),manifest,args.device)
    label_scope='Actual272original-plus-mixed pressure pairs and weighted objective, zero optimizer updates'
    labels, coverage = prepare(groups, config['mode_scale_floor'])
    value = objective(model, labels); value.backward()
    if not bool(torch.isfinite(value)) or not bool(value > 0): raise ValueError('Actual objective failed finite/nontrivial control')
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) or not bool(p.grad.norm() > 0) for p in model.parameters()):
        raise ValueError('Actual objective failed every-geometry gradient control')
    if inventory(model) != before: raise ValueError('Objective qualification changed parameters')
    model = model.cpu()
    with tempfile.TemporaryDirectory(prefix='rheon-double-artifact-', dir='/tmp') as directory:
        path = Path(directory)/'weights.npz'; save(model, path)
        restored = LearnedModalPressure(manifest['geometries']); load(restored, path, before)
        if inventory(restored) != before: raise ValueError('Actual typed double save/load failed')
    probe_errors = []
    with torch.inference_mode():
        for row in manifest['geometries']:
            q = torch.randn((1, *row['counts_zyx']), dtype=torch.float64)
            axis = bases(row['counts_zyx']); reconstructed = transform(transform(q, axis), axis, inverse=True)
            error = float((reconstructed-q).norm()/q.norm())
            if error > 1e-11: raise ValueError('Actual CPU physical decoder round-trip failed')
            probe_errors.append(error)
    write(args.output, dict(status='passed', device=args.device, objective=float(value.detach()),
        parameters=sum(p.numel() for p in model.parameters()), optimizer_updates=0, label_scope=label_scope,
        native_data_fitted=False, coverage=coverage, actual_typed_double_save_load_verified=True,
        physical_decoder_roundtrip_relative_l2=probe_errors, weights_unchanged=True,
        cuda_initialized=torch.cuda.is_initialized(), protocol_sha256=digest(args.protocol),
        source_sha256=reg['source_sha256'], model_source_sha256=digest(Path(__file__).with_name('learned_modal_pressure.py')),
        objective_source_sha256=digest(Path(__file__).with_name('modal_precision_supervision.py')),
        trainer_source_sha256=digest(Path(__file__).with_name('train_modal_augmented.py'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    run(parser.parse_args())
