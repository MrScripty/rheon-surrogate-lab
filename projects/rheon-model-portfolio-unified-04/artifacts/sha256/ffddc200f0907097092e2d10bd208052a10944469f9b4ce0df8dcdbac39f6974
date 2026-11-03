"""Bounded observation of the original optimizer; never changes its decision.

The original 5000-update configuration and all full-length sampling arrays are
preserved. The independent cap stops before another optimizer update. This is a
diagnostic entrypoint, not a shortened completed pressure-model training recipe.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

import torch

PREFIX = '[DEBUG-coarse-gradient-v1]'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def norm(value):
    return number(torch.linalg.vector_norm(value.detach().to(torch.float64)))


def gradient_snapshot(network):
    """Observe unmodified gradients with independent wider scalar reductions."""
    rows = []
    ordinary = []
    squared = 0.0
    all_finite = True
    for name, parameter in network.named_parameters():
        value = parameter.grad
        if value is None:
            rows.append(dict(parameter=name, gradient_present=False))
            all_finite = False
            continue
        detached = value.detach()
        finite = bool(torch.isfinite(detached).all())
        wide = norm(detached)
        ordinary.append(torch.linalg.vector_norm(detached))
        if isinstance(wide, float):
            squared += wide * wide
        else:
            squared = float('nan')
        all_finite = all_finite and finite
        rows.append(dict(parameter=name, gradient_present=True, dtype=str(value.dtype),
            device=str(value.device), finite=finite, maximum_absolute=number(detached.abs().max()),
            ordinary_float32_norm=number(ordinary[-1]), reference_float64_norm=wide))
    return dict(all_gradient_components_finite=all_finite, parameters=rows,
        ordinary_float32_total_norm=number(torch.linalg.vector_norm(torch.stack(ordinary))) if ordinary else None,
        reference_float64_total_norm=number(math.sqrt(squared)),
        gradients_modified=False, optimizer_decision_modified=False)


class DiagnosticLimit(RuntimeError):
    pass


class Probe:
    def __init__(self, trainer, limit, reference, emit):
        self.trainer, self.limit, self.reference, self.emit = trainer, limit, reference, emit
        self.update = 0
        self.completed = 0
        self.started = time.perf_counter()
        self.orders = {}
        self.bank_evidence = {}
        self.bank_cases = {}
        self.current = None
        self.handles = []
        self.network = None
        self.removers = []

    def event(self, kind, **fields):
        value = dict(format='pressure-coarse-gradient-diagnostic-v1', event=kind,
            attempted_update=self.update, completed_updates=self.completed,
            elapsed_s=time.perf_counter()-self.started, **fields)
        self.emit(value)
        return value

    def patch(self, owner, name, value):
        original = getattr(owner, name)
        setattr(owner, name, value)
        self.removers.append(lambda: setattr(owner, name, original))
        return original

    def capture_order(self, label, order, evidence=None):
        identity = digest(order.astype('<i8').tobytes())
        expected = self.reference['sampling_order_sha256'][label]
        if identity != expected:
            raise ValueError('Original full-length sampling sequence differs: ' + label)
        self.orders[label] = order
        if evidence is not None:
            self.bank_evidence[label] = evidence
        self.event('sampling_verified', stream=label, full_order_sha256=identity,
            prefix_sha256=digest(order[:self.limit].astype('<i8').tobytes()),
            full_length=len(order), prefix_length=min(self.limit, len(order)))

    def attach_network(self, network):
        if self.network is not None:
            if self.network is not network:
                raise ValueError('One diagnostic network only')
            return
        self.network = network
        inventory = {name: dict(shape=list(value.shape), sha256=digest(value.detach().cpu().numpy().tobytes()))
                     for name, value in network.state_dict().items()}
        if inventory != self.reference['initial_inventory']:
            raise ValueError('Original seeded initial tensors differ')
        self.event('initial_tensors_verified', parameters=sum(p.numel() for p in network.parameters()),
            dtype=str(next(network.parameters()).dtype), device=str(next(network.parameters()).device))
        self.local = self.coarse = None
        def local(_module, _arguments, result):
            self.local = result.detach()
        def coarse(_module, _arguments, result):
            self.coarse = result.detach()
        def forward(_module, arguments, result):
            if self.current is None:
                return
            self.current['forwards'].append(dict(stream=self.active_stream, correction=self.stage,
                shape_zyx=list(arguments[0].shape[2:]), residual_input_norm=norm(arguments[0]),
                local_output_norm=norm(self.local), coarse_output_norm=norm(self.coarse),
                total_output_norm=norm(result), output_finite=bool(torch.isfinite(result).all())))
            self.stage += 1
        self.handles = [network.head.register_forward_hook(local),
            network.spatial_coarse.register_forward_hook(coarse), network.register_forward_hook(forward)]

    def stream_identity(self, label, values):
        if label == 'real':
            return [dict(parent=int(parent), frame=int(frame),
                trajectory_id=values['parents'][int(parent)]['entry']['trajectory_id'],
                counts_xyz=values['parents'][int(parent)]['counts_xyz'],
                native_frame=values['parents'][int(parent)]['frames'][int(frame)]['index'])
                for parent, frame in values['indices']]
        index = int(self.orders[label][self.update - 1])
        evidence = self.bank_evidence[label]
        result = dict(bank_index=index)
        if 'grids' in evidence:
            size = len(evidence['grids'][0]['bank']['samples'])
            grid, sample = divmod(index, size)
            result.update(grid=evidence['grids'][grid]['grid'],
                          sample=evidence['grids'][grid]['bank']['samples'][sample])
        if label in self.bank_cases:
            case = self.bank_cases[label][index]
            result.update({key:case[key] for key in ('identity','counts_xyz','rho','dt','cell_size_m')})
        return result

    def install(self):
        import analytical
        import direction_feedback
        original_order = self.trainer.order
        def order(parents, config):
            result = original_order(parents, config)
            self.capture_order('real', result)
            return result
        self.patch(self.trainer, 'order', order)
        original_bank = analytical.bank
        def bank(*args, **kwargs):
            result = original_bank(*args, **kwargs)
            seed = args[1]['seed']
            label = self.reference['analytical_seeds'][str(seed)]
            self.capture_order(label, result[1], result[2])
            return result
        self.patch(analytical, 'bank', bank)
        original_direction = direction_feedback.bank
        def direction(*args, **kwargs):
            result = original_direction(*args, **kwargs)
            label = self.reference['direction_seeds'][str(args[1]['seed'])]
            self.bank_cases[label] = json.loads(args[0]['direction_manifest'])['cases']
            self.capture_order(label, result[1], result[2])
            return result
        self.patch(direction_feedback, 'bank', direction)
        original_loss = self.trainer.model.training_loss
        def loss(network, sample, normalization, **kwargs):
            self.active_stream = self.streams.pop(0)
            self.stage = 1
            result = original_loss(network, sample, normalization, **kwargs)
            self.current['losses'].append(dict(stream=self.active_stream,
                finite=bool(torch.isfinite(result).all()), value=number(result.detach().mean())))
            return result
        self.patch(self.trainer.model, 'training_loss', loss)
        original_clip = torch.nn.utils.clip_grad_norm_
        def clip(parameters, max_norm, *args, **kwargs):
            try:
                return original_clip(parameters, max_norm, *args, **kwargs)
            except RuntimeError:
                self.current['gradient_snapshot'] = gradient_snapshot(self.network)
                self.event('clip_failure', update=self.current)
                raise
        self.patch(torch.nn.utils, 'clip_grad_norm_', clip)
        original_update = self.trainer.one_update
        signature = inspect.signature(original_update)
        def update(*args, **kwargs):
            if self.completed >= self.limit:
                self.event('diagnostic_cap_reached', limit=self.limit, numerical_failure=False,
                    final_gradient_snapshot=gradient_snapshot(self.network))
                raise DiagnosticLimit('Diagnostic stop before update beyond explicit cap')
            values = signature.bind(*args, **kwargs)
            values.apply_defaults()
            values = values.arguments
            self.attach_network(values['network'])
            if set(self.orders) != {'real', 'analytical', 'supplemental', 'direction', 'current'}:
                raise ValueError('All five original sampling streams must verify before updates')
            self.update += 1
            self.streams = ['real'] * len(values['indices'])
            self.streams += [label for label, key in [('analytical','analytical_sample'),
                ('supplemental','supplemental_sample'),('direction','direction_sample'),('current','current_sample')]
                if values[key] is not None]
            self.current = dict(update=self.update, learning_rates=[g['lr'] for g in values['optimizer'].param_groups],
                samples={label:self.stream_identity(label, values) for label in set(self.streams)},
                forwards=[], losses=[])
            try:
                result = original_update(*args, **kwargs)
            except BaseException as error:
                self.event('update_failed', exception_type=type(error).__name__, message=str(error), update=self.current)
                raise
            self.completed += 1
            if self.completed % 100 == 0:
                self.event('update_completed', original_gradient_norm=result[1], update=self.current)
            self.current = None
            return result
        self.patch(self.trainer, 'one_update', update)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for remove in reversed(self.removers):
            remove()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gradient-limit', type=int, required=True)
    parser.add_argument('--gradient-reference', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if not 1 <= args.gradient_limit <= 1500:
        raise ValueError('One through 1500 diagnostic updates only')
    reference = json.loads(args.gradient_reference.read_text())
    if reference['format'] != 'pressure-coarse-gradient-reference-v1':
        raise ValueError('Explicit gradient reference required')
    for name, identity in reference['original_sources'].items():
        if Path(name).name != name or digest(Path(__file__).with_name(name).read_bytes()) != identity:
            raise ValueError('Original numerical source changed: ' + name)
    config = Path(remaining[remaining.index('--config') + 1])
    if digest(config.read_bytes()) != reference['configuration_sha256']:
        raise ValueError('Original full-length configuration must remain exact')
    import evidence
    import train
    def emit(value):
        print(PREFIX + json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
    probe = Probe(train, args.gradient_limit, reference, emit)
    probe.install()
    sys.argv[:] = [sys.argv[0], *remaining]
    try:
        evidence.main()
    finally:
        probe.close()


if __name__ == '__main__':
    main()
