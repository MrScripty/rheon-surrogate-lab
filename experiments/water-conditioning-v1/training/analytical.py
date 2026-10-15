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


def bank(template, specification, updates):
    rng=np.random.Generator(np.random.PCG64(specification['seed']))
    shape=tuple(template['base'].shape[2:])
    require(template['base'].shape[:2]==(1,1) and min(shape)>=2,
        'analytical_shape','single full grid with at least two cells per axis')
    high=tuple(n-1 for n in shape)
    fixed=[(0,0,1),(0,1,0),(1,0,0),(0,1,2),(3,2,1),(4,4,4),(0,0,high[2]),high]
    fixed=[tuple(min(k,n-1) for k,n in zip(f,shape)) for f in fixed]
    amplitude=float((template['reference']-template['base']).square().mean().sqrt())
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
