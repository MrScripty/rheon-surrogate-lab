"""Own the one-example optimization stream and its explicit fitted-case trace."""
import torch
import analytical
import fine_tuning
import model
from gradient_clipping import clip_l2_

SAMPLE = None


def bind(cache):
    if not cache:
        raise ValueError('Original train-only states required for scalar calibration')
    values = list(cache.values())
    amplitude = analytical.training_correction_rms(values)
    case = fine_tuning.SPEC['fitting_case']
    template = values[0]
    shape = tuple(reversed(case['counts_xyz']))
    template = {**template, 'base':torch.zeros((1,1,*shape),dtype=torch.float64,device=template['base'].device),
                'cell_size_m':torch.tensor([1/case['cells_per_metre']],dtype=torch.float64,device=template['base'].device)}
    global SAMPLE
    SAMPLE = analytical.make_sample(template, [(tuple(case['mode_zyx']),1.)], amplitude)
    fine_tuning.STATS['single_case'] = dict(case=case, amplitude_rms_pa=amplitude,
        optimizer_examples=0, optimizer_network_passes_per_example=fine_tuning.SPEC['training_passes'], trace=[])


def loss(network, normalization):
    if SAMPLE is None:
        raise ValueError('Bind the explicit fitting case first')
    return model.training_loss(network,SAMPLE,normalization,
        criterion=lambda prediction:analytical.loss(prediction,SAMPLE,model.full_operator)).mean()


def one_update(network, optimizer, parents, indices, normalization, cache=None, strategy=None, prepared=None, **extras):
    expected = {'analytical_sample','analytical_weight','supplemental_sample','supplemental_weight',
                'direction_sample','direction_weight','current_sample','current_weight'}
    if (set(extras) != expected or prepared is not None or len(indices) != 1
            or any(extras[name] is not None for name in extras if name.endswith('_sample'))
            or any(extras[name] != 0 for name in extras if name.endswith('_weight'))):
        raise ValueError('Exactly one manufactured example, no companion losses')
    network.train()
    optimizer.zero_grad(set_to_none=True)
    value = loss(network,normalization)
    if not bool(torch.isfinite(value)):
        raise ValueError('Nonfinite single-case loss')
    value.backward()
    gradient = clip_l2_(network.parameters(),1.)
    optimizer.step()
    if not bool(torch.isfinite(torch.cat([p.detach().flatten() for p in network.parameters()])).all()):
        raise ValueError('Nonfinite single-case weights')
    fine_tuning.STATS['single_case']['optimizer_examples'] += 1
    return float(value.detach()),float(gradient)


def observe(network, normalization, update):
    if update % 100:
        return
    import physical_residual
    network.eval()
    with torch.no_grad():
        pressure = SAMPLE['base']
        initial = SAMPLE['native_input_residual']
        denominator = torch.linalg.vector_norm(initial)
        rows = []
        for passes in range(1,9):
            residual = initial-model.full_operator(pressure-SAMPLE['base'],SAMPLE['cell_size_m'],SAMPLE['rho'])
            request = {**SAMPLE,'base':pressure,'native_input_residual':residual}
            pressure = physical_residual.predict_pressure(network,request,normalization,model.owner,{'activation':'linear','steps':1})
            if passes in (1,2,4,8):
                remaining = initial-model.full_operator(pressure-SAMPLE['base'],SAMPLE['cell_size_m'],SAMPLE['rho'])
                rows.append(dict(passes=passes,relative_equation_residual=float(torch.linalg.vector_norm(remaining)/denominator)))
        fine_tuning.STATS['single_case']['trace'].append(dict(update=update,measurements=rows))
