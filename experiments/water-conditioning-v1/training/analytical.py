"""Training-only compatible Neumann fields with analytical pressure labels.

These fields exercise the pressure operator, not complete fluid simulations.
Generation uses a private NumPy RNG and CPU float64; it cannot consume model
initialization randomness or read validation fields. Inference never uses labels.
"""
import math
import numpy as np
import torch
from policy import canonical, require, sha


AMPLITUDES=(-2.,-.5,.5,2.)
LOSS_WEIGHTS={'pressure':.25,'gradient':.25,'equation':.5,'amplification':1.}


def eigenmode(shape, frequencies):
    factors=[torch.cos(torch.pi*k*(torch.arange(n,dtype=torch.float64)+.5)/n)
             for n,k in zip(shape,frequencies)]
    pressure=torch.einsum('z,y,x->zyx',*factors)[None,None]
    eigenvalue=math.fsum(2-2*math.cos(math.pi*k/n) for n,k in zip(shape,frequencies))
    return pressure,eigenvalue


def make_sample(template, components, amplitude_pa):
    """Use analytical eigenvalues, independent of the implemented FV operator."""
    shape=template['base'].shape[2:]
    pressure=torch.zeros((1,1,*shape),dtype=torch.float64)
    laplacian=torch.zeros_like(pressure)
    for frequencies,weight in components:
        mode,eigenvalue=eigenmode(shape,frequencies)
        pressure=pressure+weight*mode
        laplacian=laplacian+(weight*eigenvalue)*mode
    pressure=pressure-pressure[:,:,:1,:1,:1].clone()
    gain=amplitude_pa/float(pressure.square().mean().sqrt())
    pressure=pressure*gain;laplacian=laplacian*gain
    scalars={name:template[name].detach().cpu().clone() for name in ('cell_size_m','rho','dt')}
    rhs=laplacian*(scalars['cell_size_m']/scalars['rho']).reshape(-1,1,1,1,1)
    sample={**scalars,'base':torch.zeros_like(pressure),'reference':pressure,
        'rhs':rhs,'native_input_residual':rhs.clone()}
    device=template['base'].device
    return {name:value.to(device) for name,value in sample.items()}


def single_grid_bank(template, specification, updates, amplitude=None):
    rng=np.random.Generator(np.random.PCG64(specification['seed']))
    shape=tuple(template['base'].shape[2:])
    require(template['base'].shape[:2]==(1,1) and min(shape)>=2,
        'analytical_shape','single full grid with at least two cells per axis')
    high=tuple(n-1 for n in shape)
    fixed=[(0,0,1),(0,1,0),(1,0,0),(0,1,2),(3,2,1),(4,4,4),(0,0,high[2]),high]
    fixed=[tuple(min(k,n-1) for k,n in zip(f,shape)) for f in fixed]
    if amplitude is None:amplitude=float((template['reference']-template['base']).square().mean().sqrt())
    require(math.isfinite(amplitude) and amplitude>0,'analytical_amplitude','positive fitted correction RMS')
    samples=[];descriptions=[]
    for index in range(specification['patterns']):
        if index<len(fixed):components=[(fixed[index],1.)]
        else:
            # Alternate arbitrary pure modes and mixtures, spanning all axes.
            components=[]
            for _ in range(1 if index%2 else 3):
                frequencies=tuple(int(rng.integers(n)) for n in shape)
                if not any(frequencies):frequencies=(0,0,1)
                components.append((frequencies,float(rng.uniform(.5,1.5))))
        for multiplier in AMPLITUDES:
            samples.append(make_sample(template,components,amplitude*multiplier))
            descriptions.append({'components_zyx':components,'amplitude_pa':amplitude*multiplier})
    order=rng.integers(len(samples),size=updates,dtype=np.int64)
    evidence={'schema':'rheon-analytical-feedback-v1','source':'train frame shape, scalar physics and correction RMS only',
        'shape_zyx':shape,'samples':descriptions,'sampling_order_sha256':sha(order.astype('<i8').tobytes()),
        'bank_description_sha256':sha(canonical(descriptions)),'loss_weights':LOSS_WEIGHTS,
        'weight':specification['weight'],'additional_examples':updates,
        'scope':'manufactured operator training; not additional CFD simulations'}
    return samples,order,evidence


def loss(prediction,sample,operator):
    error=prediction-sample['reference']
    pressure=error.square().sum()/sample['reference'].square().sum()
    gradient_error=prediction.new_zeros(());gradient_norm=prediction.new_zeros(())
    for axis in (2,3,4):
        gradient_error=gradient_error+torch.diff(error,dim=axis).square().sum()
        gradient_norm=gradient_norm+torch.diff(sample['reference'],dim=axis).square().sum()
    residual=sample['rhs']-operator(prediction,sample['cell_size_m'],sample['rho'])
    gain_squared=residual.square().sum()/sample['rhs'].square().sum()
    w=LOSS_WEIGHTS
    return w['pressure']*pressure+w['gradient']*gradient_error/gradient_norm+w['equation']*gain_squared+w['amplification']*torch.relu(gain_squared-1).square()


def validate_grids(specification):
    """Own uniform-grid sampling and a hard resident tensor budget."""
    grids=specification['grids']
    require(type(grids) is list and 1<=len(grids)<=8,'analytical_grids','one to eight explicit grids')
    require(specification['sampling']=='uniform-balanced-grid',
        'analytical_sampling','equal grid exposure, shuffled independently of real frames')
    seen=set();cells=0
    for grid in grids:
        require(type(grid) is dict and set(grid)=={'counts_xyz','cells_per_metre'},
            'analytical_grid','closed counts and cubic spacing')
        counts=grid['counts_xyz'];density=grid['cells_per_metre']
        require(type(counts) is list and len(counts)==3 and all(type(n) is int and 2<=n<=128 for n in counts)
            and math.prod(counts)<=65536,'analytical_grid','bounded rectangular cubic-voxel grid')
        require(type(density) is int and 1<=density<=128,
            'analytical_spacing','positive integer cells per metre')
        key=(*counts,density)
        require(key not in seen,'analytical_grid','no duplicate grid/spacing')
        seen.add(key);cells+=math.prod(counts)
    # Four dense float64 fields and three scalar tensors per bank entry.
    projected=specification['patterns']*len(AMPLITUDES)*(32*cells+24*len(grids))
    limit=specification['cache_max_bytes']
    require(type(limit) is int and 0<limit<=768*1024**2 and projected<=limit,
        'analytical_cache','resident tensor projection must fit explicit budget')
    return projected


def bank(template,specification,updates):
    if 'grids' not in specification:
        return single_grid_bank(template,specification,updates)
    projected=validate_grids(specification)
    amplitude=float((template['reference']-template['base']).square().mean().sqrt())
    require(math.isfinite(amplitude) and amplitude>0,'analytical_amplitude','positive fitted correction RMS')
    samples=[];banks=[];size=specification['patterns']*len(AMPLITUDES)
    # A separate private stream changes only analytical shape/example selection.
    rng=np.random.Generator(np.random.PCG64(specification['seed']))
    grid_order=np.concatenate([rng.permutation(len(specification['grids']))
        for _ in range(math.ceil(updates/len(specification['grids'])))])[:updates]
    within=rng.integers(size,size=updates,dtype=np.int64)
    order=(grid_order*size+within).astype('<i8')
    for grid in specification['grids']:
        shape=tuple(reversed(grid['counts_xyz']))
        grid_template={**template,'base':torch.zeros((1,1,*shape),dtype=torch.float64,device=template['base'].device),
            'cell_size_m':torch.tensor([1/grid['cells_per_metre']],dtype=torch.float64,device=template['base'].device)}
        values,_,evidence=single_grid_bank(grid_template,specification,1,amplitude=amplitude)
        samples.extend(values);banks.append({'grid':grid,'bank':evidence})
    actual=sum(value.numel()*value.element_size() for sample in samples for value in sample.values())
    require(actual==projected,'analytical_cache','exact resident tensor projection')
    evidence={'schema':'rheon-analytical-feedback-v2','source':'real train frame correction RMS and scalar water physics; explicit grid shapes/spacing',
        'grids':banks,'sampling':specification['sampling'],'grid_exposures':np.bincount(grid_order,minlength=len(banks)).tolist(),
        'sampling_order_sha256':sha(order.tobytes()),'bank_description_sha256':sha(canonical(banks)),
        'loss_weights':LOSS_WEIGHTS,'weight':specification['weight'],'additional_examples':updates,
        'resident_tensor_bytes':actual,'cache_max_bytes':specification['cache_max_bytes'],
        'scope':'manufactured operator training, not additional CFD simulations; normalization remains real-train-only'}
    return samples,order,evidence
