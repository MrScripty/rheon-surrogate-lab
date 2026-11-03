"""Exact managed mixed-query pressure checkpoint with uniform restored head precision."""
import torch
from refined_pressure_sampler import load_model as load_refined


def load_model(pair, library_path):
    model = load_refined(pair, library_path, 'runs/full-ai-tight-targets-v1/mixed-modal-pair-11')
    try:
        changes = {}
        step = 2.**-40
        with torch.no_grad():
            for name in ('mid_gate', 'mix'):
                parameter = getattr(model, name)
                rounded = (parameter / step).round() * step
                changes[name] = dict(maximum_absolute_change=float((rounded - parameter).abs().max()),
                                     nonzero_before=int(parameter.count_nonzero()), nonzero_after=int(rounded.count_nonzero()))
                parameter.copy_(rounded)
        model.frozen_versions = {name: value._version for name, value in model.named_parameters()}
        model.asset_identity['transport_precision_policy'] = dict(kind='uniform_absolute_quantization',
            step=step, scope='All48managed-restored double query-head parameters', changes=changes,
            raw_managed_artifacts_unchanged=True, optimizer_updates=0, analytic_coefficients_assigned=False)
        return model
    except BaseException:
        model.close()
        raise
