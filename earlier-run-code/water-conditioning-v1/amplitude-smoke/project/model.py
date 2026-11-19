"""Controlled feature conditioning over the unchanged seven-channel network."""
import copy
import torch
from policy import HERE, LAB, load_module, require

source=HERE/'portable_model.py'
if not source.exists(): source=LAB/'experiments/cuda-pressure-v1/model.py'
owner=load_module('conditioning_portable_model',source)
CHANNELS=owner.CHANNELS
PressureCorrectionNet=owner.PressureCorrectionNet
configure_cpu=owner.configure_cpu
configure=owner.configure
parameter_count=owner.parameter_count
full_operator=owner.full_operator
pressure_loss=owner.pressure_loss
frozen=owner.frozen
ARCHITECTURE=copy.deepcopy(owner.ARCHITECTURE)
SELECTED=None


def select(config):
    global SELECTED
    require(config['schema']=='rheon-water-conditioning-experiment-v1',
            'conditioning_configuration','explicit versioned model transformation')
    SELECTED=config['conditioning']
    require(SELECTED in ('scalar-units','amplitude-and-units'),'conditioning_configuration','supported variant')
    ARCHITECTURE.update(schema='rheon-conditioned-cubic-groupnorm-v1',conditioning=SELECTED,
        scalar_feature_units={'rho_kg_m3':1000.,'dt_s':1/60},
        amplitude='per-sample residual RMS / training residual RMS; restore output amplitude' if SELECTED=='amplitude-and-units' else 'historical fixed training RMS')


def amplitude(residual,normalization):
    require(SELECTED is not None,'conditioning_configuration','select model before use')
    if SELECTED=='scalar-units': return torch.ones(residual.shape[0],dtype=torch.float64,device=residual.device)
    measured=residual.flatten(1).square().mean(dim=1).sqrt()/owner.scales(normalization)['residual']
    return torch.where(measured>0,measured,torch.ones_like(measured))


def build_inputs(residual,base,cell_size,rho,dt,normalization):
    gain=amplitude(residual,normalization).reshape(-1,1,1,1,1)
    values=owner.build_inputs(residual/gain,base,cell_size,rho,dt,normalization)
    # Existing normalization records remain explicit. These additional fixed
    # physical units are part of this model's declared preprocessing contract.
    scale=owner.scales(normalization)
    values[:,5] *= scale['rho']/1000.
    values[:,6] *= scale['dt']/(1/60)
    return values


def predict_pressure(network,sample,normalization):
    residual=sample['native_input_residual']
    values=build_inputs(residual,sample['base'],sample['cell_size_m'],sample['rho'],sample['dt'],normalization)
    active=(residual.flatten(1)!=0).any(dim=1)
    if not bool(active.any()): return sample['base'].clone()
    selected=torch.nonzero(active,as_tuple=False).flatten()
    raw=network(values.index_select(0,selected))
    require(raw.shape==(selected.numel(),1,*values.shape[2:]) and raw.dtype==torch.float32 and bool(torch.isfinite(raw).all()),
            'conditioning_output','finite same-grid float32 correction')
    correction=frozen.gauge_correction(raw,owner.scales(normalization)['pressure'])
    correction=correction*amplitude(residual,normalization).index_select(0,selected).reshape(-1,1,1,1,1)
    return sample['base'].clone().index_copy(0,selected,sample['base'].index_select(0,selected)+correction)
