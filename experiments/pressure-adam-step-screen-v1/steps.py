"""Independent Adam endpoints and float32 parameter-space interpolation."""
import torch


def adam(before, gradient, group, step_before, moments=None):
    if (group['betas'] != [.9,.999] or group['eps'] != 1e-8 or group['weight_decay'] != 0
            or group['amsgrad'] or group['maximize']):
        raise ValueError('Captured ordinary Adam contract required')
    parameter=torch.nn.Parameter(before.clone())
    optimizer=torch.optim.Adam([parameter],lr=group['lr'],betas=tuple(group['betas']),
        eps=group['eps'],weight_decay=0)
    if moments is not None:
        optimizer.state[parameter]=dict(step=torch.tensor(float(step_before)),
            exp_avg=moments[0].clone(),exp_avg_sq=moments[1].clone())
    elif step_before != 0:
        raise ValueError('No fabricated Adam history')
    parameter.grad=gradient.clone()
    optimizer.step()
    return parameter.detach().clone()


def interpolate(before, endpoint, fraction):
    if fraction not in (0.,.125,.25,.5,1.):
        raise ValueError('Predeclared bounded fractions required')
    if fraction==0:return before.clone()
    if fraction==1:return endpoint.clone()
    return (before.double()+fraction*(endpoint.double()-before.double())).float()
