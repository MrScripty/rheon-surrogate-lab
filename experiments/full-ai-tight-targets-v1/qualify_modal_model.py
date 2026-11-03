"""Qualify double modal forward/backward without updating any parameter."""
import argparse
import hashlib
import os
from pathlib import Path
from profile_frozen_components import ROOT, read, write, digest
from learned_modal_pressure import LearnedModalPressure
import numpy as np
import torch


def inventory(model):
    return {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for name, value in model.state_dict().items()}


def run(args):
    if args.output.exists():
        raise ValueError('Fresh modal qualification required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered modal source/data changed: ' + path)
    if os.environ.get('TULDOK_TORCH_DEVICE') != args.device:
        raise ValueError('Qualified runtime device differs')
    os.sched_setaffinity(0, {reg['cpu']}); torch.set_num_threads(1)
    torch.manual_seed(reg['seed']); torch.use_deterministic_algorithms(True)
    manifest_path = ROOT / reg['manifest']; manifest = read(manifest_path)
    if manifest['status'] != 'verified_train_only_fixed_transform_view' or manifest['development_or_reserved_used']:
        raise ValueError('Fixed train-only native double view required')
    model = LearnedModalPressure(manifest['geometries']).to(args.device)
    before = inventory(model); rows = []
    for row in manifest['geometries']:
        path = manifest_path.parent / row['path']
        if digest(path) != row['sha256']:
            raise ValueError('Fixed basis features changed')
        with np.load(path, allow_pickle=False) as source:
            q = torch.from_numpy(source['q_coefficients'][:1].copy()).to(args.device)
        q.requires_grad_(True); model.zero_grad(set_to_none=True)
        value = model(q, row['key'])
        target = q.detach().clone(); target[:, 0, 0, 0] = 0.
        if not torch.equal(value, target):
            raise ValueError('Unit initial non-DC gains or zero-DC gauge changed')
        loss = value.square().sum() / q.detach().square().sum()
        loss.backward()
        gradient = model.log_gains[row['key']].grad
        if not bool(torch.isfinite(gradient).all()) or not bool(gradient.norm() > 0) or float(gradient[0,0,0]) != 0.:
            raise ValueError('Modal parameter gradient qualification failed')
        expected = 2 * q.detach()[0].square() / q.detach().square().sum()
        expected[0, 0, 0] = 0.
        gradient_error = float((gradient - expected).norm() / expected.norm())
        if gradient_error > 1e-12 or not bool(torch.isfinite(q.grad).all()):
            raise ValueError('Analytic derivative control failed')
        rows.append(dict(key=row['key'], dtype=str(value.dtype), device=str(value.device),
                         parameter_gradient_relative_error=gradient_error, finite_input_gradient=True,
                         forward_unit_gain_exact=True, dc_gradient_zero=True))
    if inventory(model) != before:
        raise ValueError('Qualification modified learned parameters')
    write(args.output, dict(status='passed', device=args.device, rows=rows, optimizer_updates=0,
                            parameters=sum(p.numel() for p in model.parameters()), source_sha256=reg['source_sha256'],
                            protocol_sha256=digest(args.protocol), weights_unchanged=True,
                            cuda_initialized=torch.cuda.is_initialized(), model_accuracy_qualified=False,
                            model_source_sha256=digest(Path(__file__).with_name('learned_modal_pressure.py'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    run(parser.parse_args())
