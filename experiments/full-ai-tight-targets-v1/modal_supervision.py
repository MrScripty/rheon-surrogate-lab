"""Native-data supervised log-coefficient objective; no operator eigenvalues."""
import torch


def prepare(groups, relative_floor):
    labels, coverage = {}, {}
    for group in groups:
        q, p = group['q'], group['p']
        rms = q.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        valid = ((q * p > 0) & (q.abs() > relative_floor * rms))
        valid[:, 0, 0, 0] = False
        counts = valid.sum(dim=0); observed = counts > 0
        if not bool(observed.any()):
            raise ValueError('Observed non-DC native labels required')
        floor = torch.finfo(torch.float64).tiny
        target = p.abs().clamp_min(floor).log() - q.abs().clamp_min(floor).log()
        labels[group['key']] = (target, valid, counts.clamp_min(1), observed.sum())
        coverage[group['key']] = dict(observed_nonDC_modes=int(observed.sum()),
            unobserved_nonDC_modes=int((~observed).sum())-1, valid_sample_mode_labels=int(valid.sum()),
            original_dtype=str(q.dtype), device=str(q.device))
    return labels, coverage


def objective(model, labels):
    components = []
    for key, (target, valid, counts, observed) in labels.items():
        squared = (model.log_gains[key].unsqueeze(0) - target).square()
        components.append(((squared * valid).sum(dim=0) / counts).sum() / observed)
    return torch.stack(components).mean()
