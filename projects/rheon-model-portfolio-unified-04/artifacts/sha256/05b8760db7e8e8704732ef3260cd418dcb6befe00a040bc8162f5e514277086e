"""Matched two-channel spacing ablation with amplitude-homogeneous features."""
import json
from pathlib import Path
from historical_physical import *
import historical_physical

ENABLED = json.loads(Path(__file__).with_name('spacing-method.json').read_text())['explicit_spacing']


def transform(network,owner,correction):
    historical_physical.transform(network,owner,correction)
    original = network.encoders[0].layers[0]
    with torch.random.fork_rng(devices=[]):
        replacement = owner.ReplicatedConvolution(2,original.out_channels,original.kernel_size,
            padding=original.padding,padding_mode=original.padding_mode,bias=False)
    with torch.no_grad():
        replacement.weight.zero_()
        replacement.weight[:,:1].copy_(original.weight)
    network.encoders[0].layers[0] = replacement


def build_inputs(residual,base,h,rho,dt,normalization,owner):
    values = historical_physical.build_inputs(residual,base,h,rho,dt,normalization,owner)
    extra = values*torch.log2(h/(1/8)).to(values.dtype).reshape(-1,1,1,1,1) if ENABLED else torch.zeros_like(values)
    return torch.cat((values,extra),dim=1)


def forward(network,values,owner):
    require(values.ndim==5 and values.shape[1]==2 and 1<=values.shape[0]<=4
            and min(values.shape[2:])>=1 and math.prod(values.shape[2:])<=MAX_CELLS,
            'spacing_input','N,2,Z,Y,X within declared grid budget')
    require(values.dtype==torch.float32 and values.device.type in ('cpu','cuda')
            and bool(torch.isfinite(values).all()),'spacing_input','finite f32 CPU/CUDA')
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


def predict_pressure(network,sample,normalization,owner,correction,intermediates=None):
    base=sample['base'];initial=sample['native_input_residual'];pressure=base.clone()
    for step in range(correction['steps']):
        residual=initial if step==0 else initial-owner.full_operator(pressure-base,sample['cell_size_m'],sample['rho'])
        values=build_inputs(residual,pressure,sample['cell_size_m'],sample['rho'],sample['dt'],normalization,owner)
        selected=torch.nonzero((residual.flatten(1)!=0).any(dim=1),as_tuple=False).flatten()
        if selected.numel()==0:break
        raw=network(values.index_select(0,selected))
        require(raw.dtype==torch.float32 and raw.shape==(selected.numel(),1,*base.shape[2:])
                and bool(torch.isfinite(raw).all()),'spacing_output','finite f32 same-grid correction')
        delta=owner.frozen.gauge_correction(raw,scales(normalization)[normalization.get('output_scale','pressure')])
        pressure=pressure.clone().index_copy(0,selected,pressure.index_select(0,selected)+delta)
        if intermediates is not None:intermediates.append(pressure)
    return pressure
