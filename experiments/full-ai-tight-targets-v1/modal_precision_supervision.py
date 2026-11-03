"""Native pressure coefficient least-squares objective, without inverse labels."""
import torch


def prepare(groups,relative_floor):
    labels={};coverage={}
    for group in groups:
        q,p=group['q'],group['p']
        scale=p.square().mean(0).sqrt()
        floor=p.square().mean().sqrt()*relative_floor
        scale=scale.clamp_min(floor.clamp_min(torch.finfo(torch.float64).tiny))
        mask=torch.ones_like(scale,dtype=torch.bool);mask[0,0,0]=False
        labels[group['key']]=(q/scale,p/scale,mask)
        coverage[group['key']]=dict(observed_nonDC_modes=int(mask.sum()),native_samples=len(q),
            normalization='Train-only per-mode pressure RMS with fixed relative floor',
            ratios_or_inverse_coefficient_targets_computed=False,device=str(q.device))
    return labels,coverage


def objective(model,labels):
    terms=[]
    for key,(q,p,mask)in labels.items():
        error=q*model.gains(key)-p
        # Sum modes to keep individual parameter gradients well-conditioned;
        # average native samples without treating low-signal ratios equally.
        terms.append(error.square().mean(0)[mask].sum())
    return torch.stack(terms).sum()


def initialize(model,normalization):
    import hashlib
    with torch.no_grad():
        values={name:torch.tensor(value,dtype=torch.float64)for name,value in normalization['initial_pressure_parameters'].items()}
        if set(values)!=set(model.state_dict()):raise ValueError('Complete inherited learned pressure inventory required')
        for name,value in values.items():
            spec=normalization['initial_pressure_inventory'][name]
            if list(value.shape)!=spec['shape']or hashlib.sha256(value.numpy().tobytes()).hexdigest()!=spec['sha256']:
                raise ValueError('Inherited exact managed pressure parameters changed in transport')
        model.load_state_dict(values,strict=True)
    return model
