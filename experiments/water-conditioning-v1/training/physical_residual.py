"""One residual input for the uniform sealed cubic finite-volume operator.

A=(h/rho)L, hence L delta_p=(rho/h)r. Grid shape specifies the sealed
Neumann stencil. This contract does not represent obstacles or free surfaces.
Physical assembly/reconstruction stay f64; learned tensors and output stay f32.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from policy import MAX_CELLS, require

CHANNELS=['dimensionless_laplacian_rhs']
SCHEMA='rheon-physical-residual-normalization-v1'
LOSS_WEIGHTS={'relative_correction_mse':.25,'relative_correction_gradient_mse':.25,
    'relative_equation_mse':.4,'relative_equation_max_squared':.1}


def fit_normalization(parents,previous):
    result=copy.deepcopy(previous)
    train=[p for p in parents if p['entry']['split']=='train']
    cells=sum(p['arrays']['native_input_residual'].size for p in train)
    squared=math.fsum(float(((p['arrays']['native_input_residual']*
        (p['density_kg_m3']/p['cell_size_m']))**2).sum()) for p in train)
    rms=math.sqrt(squared/cells)
    require(math.isfinite(rms) and rms>0,'residual_scale','positive train-only physically scaled RMS')
    result.update(schema=SCHEMA,channels=CHANNELS,
        coordinates='implicit grid indices; sealed stencil; no coordinate input maps',
        operator='A=(h/rho)L; q=(rho/h)r in Pa; scalar cubic h in metres')
    result['scales']['laplacian_rhs']={'value':rms,'units':'Pa','method':'train_cell_rms',
        'source':'native_input_residual * density_kg_m3 / cell_size_m'}
    return result


def scales(normalization):
    require(normalization.get('schema')==SCHEMA and normalization.get('channels')==CHANNELS,
            'residual_normalization','versioned one-channel normalization')
    result=normalization['scales']
    require(set(result)=={'residual','pressure','rhs','rho','dt','laplacian_rhs'},
            'residual_normalization','explicit scale inventory')
    require(all(type(s['value']) in (int,float) and math.isfinite(s['value']) and s['value']>0
                for s in result.values()),'residual_normalization','positive finite scales')
    return {key:value['value'] for key,value in result.items()}


def transform(network,owner,correction):
    original=network.encoders[0].layers[0]
    with torch.random.fork_rng(devices=[]):
        first=owner.ReplicatedConvolution(1,original.out_channels,original.kernel_size,
            padding=original.padding,padding_mode=original.padding_mode)
    with torch.no_grad():
        first.weight.copy_(original.weight[:,:1]);first.bias.copy_(original.bias)
    network.encoders[0].layers[0]=first
    if correction['activation']=='linear':
        for module in network.modules():
            if isinstance(module,nn.Conv3d):module.register_parameter('bias',None)
        for block in [*network.encoders,network.bottleneck,*network.decoders]:
            for index,layer in enumerate(block.layers):
                if isinstance(layer,nn.SiLU):block.layers[index]=nn.Identity()


def forward(network,values,owner):
    require(values.ndim==5 and values.shape[1]==1 and 1<=values.shape[0]<=4
            and min(values.shape[2:])>=1 and math.prod(values.shape[2:])<=MAX_CELLS,
            'residual_input','N,1,Z,Y,X within declared grid budget')
    require(values.dtype==torch.float32 and values.device.type in ('cpu','cuda')
            and bool(torch.isfinite(values).all()),'residual_input','finite f32 CPU/CUDA')
    skips=[]
    for encoder in network.encoders:
        values=encoder(values);skips.append(values)
        kernel=tuple(min(2,n) for n in values.shape[2:])
        values=owner.NonoverlappingAverage.apply(values,kernel) if values.device.type=='cuda' else F.avg_pool3d(values,kernel,stride=kernel,ceil_mode=True,count_include_pad=False)
    values=network.bottleneck(values)
    values=values+network.global_pool(values).expand_as(values)
    for decoder,skip in zip(network.decoders,reversed(skips)):
        values=owner.NearestResize.apply(values,skip.shape[2:]) if values.device.type=='cuda' else F.interpolate(values,size=skip.shape[2:],mode='nearest')
        values=decoder(torch.cat((values,skip),dim=1))
    return network.head(values)


def build_inputs(residual,base,h,rho,dt,normalization,owner):
    owner.field_shape_contract(base);owner.cell_size_contract(h)
    require(residual.shape==base.shape and base.dtype==residual.dtype==torch.float64
            and base.device==residual.device==h.device and h.shape==(base.shape[0],),
            'residual_fields','colocated full-grid base/residual and one cubic h')
    require(bool(torch.isfinite(base).all() and torch.isfinite(residual).all()
            and (base[:,:,0,0,0]==0).all()),'residual_fields','finite fields and cell-zero base gauge')
    for scalar in (rho,dt):
        require(scalar.shape==h.shape and scalar.dtype==torch.float64 and scalar.device==h.device
                and bool(torch.isfinite(scalar).all() and (scalar>0).all()),
                'residual_physics','positive finite per-sample density/timestep')
    q=residual*(rho/h).reshape(-1,1,1,1,1)
    values=(q/scales(normalization)['laplacian_rhs']).to(torch.float32)
    require(bool(torch.isfinite(values).all()),'residual_input','representable f32 physical residual')
    return values


def predict_pressure(network,sample,normalization,owner,correction):
    base=sample['base'];initial=sample['native_input_residual'];pressure=base.clone()
    for step in range(correction['steps']):
        residual=initial if step==0 else initial-owner.full_operator(pressure-base,sample['cell_size_m'],sample['rho'])
        values=build_inputs(residual,pressure,sample['cell_size_m'],sample['rho'],sample['dt'],normalization,owner)
        selected=torch.nonzero((residual.flatten(1)!=0).any(dim=1),as_tuple=False).flatten()
        if selected.numel()==0:break
        raw=network(values.index_select(0,selected))
        require(raw.dtype==torch.float32 and raw.shape==(selected.numel(),1,*base.shape[2:])
                and bool(torch.isfinite(raw).all()),'residual_output','finite f32 same-grid correction')
        delta=owner.frozen.gauge_correction(raw,scales(normalization)['pressure'])
        pressure=pressure.clone().index_copy(0,selected,pressure.index_select(0,selected)+delta)
    return pressure


def pressure_loss(prediction,sample,normalization,owner,objective='canonical'):
    # Keep the old physical objective exactly for the input/linearity ablations.
    scales(normalization)
    previous={**normalization,'schema':'rheon-cubic-pressure-normalization-v2',
        'channels':owner.CHANNELS,'scales':{key:value for key,value in normalization['scales'].items() if key!='laplacian_rhs'}}
    canonical=owner.pressure_loss(prediction,sample,previous)
    if objective=='canonical':return canonical
    require(objective=='correction-gradient-divergence','residual_objective','known physical objective')
    scale=scales(normalization);cells=prediction[0].numel()
    target=sample['reference']-sample['base'];error=prediction-sample['reference']
    floor=(1e-6*math.sqrt(cells)*scale['pressure'])**2
    correction_error=error.flatten(1).square().sum(1)/target.flatten(1).square().sum(1).clamp_min(floor)
    gradient_error=torch.zeros_like(correction_error);gradient_norm=torch.zeros_like(correction_error)
    for axis in (2,3,4):
        gradient_error=gradient_error+torch.diff(error,dim=axis).flatten(1).square().sum(1)
        gradient_norm=gradient_norm+torch.diff(target,dim=axis).flatten(1).square().sum(1)
    residual=owner.full_operator(prediction,sample['cell_size_m'],sample['rho'])-sample['rhs']
    # dt/h^3 converts integrated equation residual to predicted divergence.
    # Relative terms cancel this scalar, retaining each case's physical scale.
    denominator=sample['rhs'].flatten(1).square().sum(1).clamp_min((1e-6*math.sqrt(cells)*scale['rhs'])**2)
    equation=residual.flatten(1).square().sum(1)/denominator
    maximum=residual.flatten(1).abs().amax(1).square()/(denominator/cells)
    w=LOSS_WEIGHTS
    return w['relative_correction_mse']*correction_error+w['relative_correction_gradient_mse']*gradient_error/gradient_norm.clamp_min(floor)+w['relative_equation_mse']*equation+w['relative_equation_max_squared']*maximum
