"""Own a bounded pure-fundamental optimizer stream and its explicit exposure."""
import math
import numpy as np
import torch
from policy import canonical, sha

SAMPLES = []
ORDER = None


def validate(spec):
    if set(spec) != {'grids','modes_zyx','sampling_seed','cache_max_bytes','amplitude_reference'}:
        raise ValueError('Closed fundamental fitting bank required')
    if not 1 <= len(spec['grids']) <= 8 or not 1 <= len(spec['modes_zyx']) <= 3:
        raise ValueError('Bounded grid/axis combinations required')
    modes=spec['modes_zyx']
    if any(tuple(m) not in [(0,0,1),(0,1,0),(1,0,0)] for m in modes) or len({tuple(m)for m in modes})!=len(modes):
        raise ValueError('Distinct fundamental axes required')
    grids=[]
    for grid in spec['grids']:
        if set(grid)!={'counts_xyz','cells_per_metre'}:
            raise ValueError('Counts and cubic spacing required')
        counts=grid['counts_xyz'];density=grid['cells_per_metre']
        if (len(counts)!=3 or any(type(n)is not int or not 5<=n<=128 for n in counts)
                or math.prod(counts)>65536 or type(density)is not int or not 1<=density<=128):
            raise ValueError('Bounded cubic-voxel grid required')
        grids.append((*counts,density))
    if len(set(grids))!=len(grids) or type(spec['sampling_seed'])is not int:
        raise ValueError('Distinct grids and explicit private sampling seed required')
    projected=len(modes)*sum(32*math.prod(g['counts_xyz'])+24 for g in spec['grids'])
    if (type(spec['cache_max_bytes'])is not int or not 0<projected<=spec['cache_max_bytes']<=8*1024**2
            or spec['amplitude_reference']!='original-train-correction-rms'):
        raise ValueError('Train-only amplitude and bounded resident fitting bank required')
    return projected


def sampling(count, updates, seed):
    if updates%count:
        raise ValueError('Complete balanced sampling cycles required')
    rng=np.random.Generator(np.random.PCG64(seed))
    return np.concatenate([rng.permutation(count)for _ in range(updates//count)]).astype('<i8')


def build(template, spec, amplitude):
    import analytical
    projected=validate(spec)
    samples=[];cases=[]
    for grid in spec['grids']:
        shape=tuple(reversed(grid['counts_xyz']))
        local={**template,'base':torch.zeros((1,1,*shape),dtype=torch.float64,device=template['base'].device),
            'cell_size_m':torch.tensor([1/grid['cells_per_metre']],dtype=torch.float64,device=template['base'].device)}
        for mode in spec['modes_zyx']:
            samples.append(analytical.make_sample(local,[(tuple(mode),1.)],amplitude))
            cases.append(dict(**grid,mode_zyx=mode))
    actual=sum(v.numel()*v.element_size()for sample in samples for v in sample.values())
    if actual!=projected:
        raise ValueError('Exact resident fitting-bank projection required')
    return samples,cases,actual


def bind(cache, updates):
    import analytical
    import fine_tuning
    if not cache:
        raise ValueError('Original train-only states required for calibration')
    spec=fine_tuning.SPEC['fitting_bank'];values=list(cache.values())
    amplitude=analytical.training_correction_rms(values)
    global SAMPLES,ORDER
    SAMPLES,cases,residency=build(values[0],spec,amplitude)
    ORDER=sampling(len(SAMPLES),updates,spec['sampling_seed'])
    fine_tuning.STATS['fundamental_bank']=dict(cases=cases,amplitude_rms_pa=amplitude,
        resident_tensor_bytes=residency,description_sha256=sha(canonical(cases)),
        sampling_order_sha256=sha(ORDER.tobytes()),expected_exposures=np.bincount(ORDER,minlength=len(SAMPLES)).tolist(),
        optimizer_exposures=[0]*len(SAMPLES),optimizer_examples=0,trace=[])


def loss(network, sample, normalization):
    import analytical
    import model
    return model.training_loss(network,sample,normalization,
        criterion=lambda p:analytical.loss(p,sample,model.full_operator)).mean()


def one_update(network, optimizer, parents, indices, normalization, cache=None, strategy=None, prepared=None, **extras):
    import fine_tuning
    from gradient_clipping import clip_l2_
    expected={'analytical_sample','analytical_weight','supplemental_sample','supplemental_weight',
        'direction_sample','direction_weight','current_sample','current_weight'}
    if (set(extras)!=expected or prepared is not None or len(indices)!=1
            or any(extras[n]is not None for n in extras if n.endswith('_sample'))
            or any(extras[n]!=0 for n in extras if n.endswith('_weight'))):
        raise ValueError('One explicit fitting example and no companion losses required')
    stats=fine_tuning.STATS['fundamental_bank'];index=int(ORDER[stats['optimizer_examples']])
    network.train();optimizer.zero_grad(set_to_none=True)
    value=loss(network,SAMPLES[index],normalization)
    if not bool(torch.isfinite(value)):
        raise ValueError('Nonfinite fitting loss')
    value.backward();gradient=clip_l2_(network.parameters(),1.);optimizer.step()
    if not all(bool(torch.isfinite(p).all())for p in network.parameters()):
        raise ValueError('Nonfinite fitted weights')
    stats['optimizer_examples']+=1;stats['optimizer_exposures'][index]+=1
    return float(value.detach()),float(gradient)


def measure(network, normalization, samples):
    import model
    import physical_residual
    network.eval();rows=[]
    with torch.no_grad():
        for index,sample in enumerate(samples):
            pressure=sample['base'];initial=sample['native_input_residual'];denominator=torch.linalg.vector_norm(initial)
            for passes in range(1,9):
                residual=initial-model.full_operator(pressure-sample['base'],sample['cell_size_m'],sample['rho'])
                pressure=physical_residual.predict_pressure(network,{**sample,'base':pressure,'native_input_residual':residual},
                    normalization,model.owner,{'activation':'linear','steps':1})
                if passes in (1,2,4,8):
                    remaining=initial-model.full_operator(pressure-sample['base'],sample['cell_size_m'],sample['rho'])
                    rows.append(dict(case=index,passes=passes,relative_equation_residual=float(torch.linalg.vector_norm(remaining)/denominator)))
    return rows


def observe(network, normalization, update):
    import fine_tuning
    if update%(10*len(SAMPLES)):
        return
    fine_tuning.STATS['fundamental_bank']['trace'].append(dict(update=update,
        exposures_per_case=update//len(SAMPLES),measurements=measure(network,normalization,SAMPLES)))
