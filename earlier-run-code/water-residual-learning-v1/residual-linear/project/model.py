"""Controlled feature conditioning over the unchanged seven-channel network."""
import copy
import torch
from policy import HERE, LAB, load_module, require

source=HERE/'portable_model.py'
if not source.exists(): source=LAB/'experiments/cuda-pressure-v1/model.py'
owner=load_module('conditioning_portable_model',source)
CHANNELS=owner.CHANNELS
configure_cpu=owner.configure_cpu
configure=owner.configure
parameter_count=owner.parameter_count
full_operator=owner.full_operator
pressure_loss=owner.pressure_loss
frozen=owner.frozen
ARCHITECTURE=copy.deepcopy(owner.ARCHITECTURE)
SELECTED=None
NORMALIZATION='groupnorm'
CORRECTION=None


def select(config):
    global SELECTED,NORMALIZATION,CORRECTION
    CORRECTION=None
    if config['schema']=='rheon-water-residual-experiment-v1':
        import physical_residual
        SELECTED='physical-residual';NORMALIZATION='none';CORRECTION=config['correction']
        ARCHITECTURE.clear();ARCHITECTURE.update(copy.deepcopy(owner.ARCHITECTURE))
        ARCHITECTURE.update(schema='rheon-physical-residual-pressure-v1',input_channels=1,
            channels=['dimensionless_laplacian_rhs'],conditioning=SELECTED,
            block_normalization={'name':'Identity'},correction=copy.deepcopy(CORRECTION),
            preprocessing='q=(rho/h)*residual; q / train-only q RMS',
            reconstruction='float32 raw times train pressure RMS, cell-zero gauge, add supplied base',
            initialization='retain original first-convolution residual column and all surviving seeded tensors',
            restrictions='uniform-density cubic sealed Neumann operator; no independently specified geometry masks')
        return
    require(config['schema'] in ('rheon-water-conditioning-experiment-v1','rheon-water-conditioning-experiment-v2','rheon-water-conditioning-experiment-v3','rheon-water-conditioning-experiment-v4'),
            'conditioning_configuration','explicit versioned model transformation')
    SELECTED=config['conditioning']
    NORMALIZATION=config.get('normalization','groupnorm')
    require(SELECTED in ('scalar-units','amplitude-and-units'),'conditioning_configuration','supported variant')
    require(NORMALIZATION in ('groupnorm','none'),'normalization_configuration','supported block intervention')
    ARCHITECTURE.clear();ARCHITECTURE.update(copy.deepcopy(owner.ARCHITECTURE))
    ARCHITECTURE.update(schema='rheon-conditioned-cubic-groupnorm-v1',conditioning=SELECTED,
        scalar_feature_units={'rho_kg_m3':1000.,'dt_s':1/60},
        amplitude='per-sample residual RMS / training residual RMS; restore output amplitude' if SELECTED=='amplitude-and-units' else 'historical fixed training RMS')
    if NORMALIZATION=='none':
        ARCHITECTURE.update(schema='rheon-conditioned-cubic-no-normalization-v1',
            block_normalization={'name':'Identity','position':'replace only parameter-free GroupNorm'})


class PressureCorrectionNet(owner.PressureCorrectionNet):
    def __init__(self):
        super().__init__()
        if NORMALIZATION=='none':
            for block in [*self.encoders,self.bottleneck,*self.decoders]:
                for index,layer in enumerate(block.layers):
                    if isinstance(layer,torch.nn.GroupNorm):block.layers[index]=torch.nn.Identity()
        if CORRECTION is not None:
            import physical_residual
            physical_residual.transform(self,owner,CORRECTION)

    def forward(self,values):
        if CORRECTION is None:return super().forward(values)
        import physical_residual
        return physical_residual.forward(self,values,owner)


def scales(normalization):
    if CORRECTION is None:return owner.scales(normalization)
    import physical_residual
    return physical_residual.scales(normalization)


def fit_normalization(parents,fit):
    result=fit(parents)
    if CORRECTION is not None:
        import physical_residual
        result=physical_residual.fit_normalization(parents,result)
    return result


def amplitude(residual,normalization):
    require(SELECTED is not None,'conditioning_configuration','select model before use')
    if SELECTED in ('scalar-units','physical-residual'): return torch.ones(residual.shape[0],dtype=torch.float64,device=residual.device)
    measured=residual.flatten(1).square().mean(dim=1).sqrt()/owner.scales(normalization)['residual']
    return torch.where(measured>0,measured,torch.ones_like(measured))


def build_inputs(residual,base,cell_size,rho,dt,normalization):
    if CORRECTION is not None:
        import physical_residual
        return physical_residual.build_inputs(residual,base,cell_size,rho,dt,normalization,owner)
    gain=amplitude(residual,normalization).reshape(-1,1,1,1,1)
    values=owner.build_inputs(residual/gain,base,cell_size,rho,dt,normalization)
    # Existing normalization records remain explicit. These additional fixed
    # physical units are part of this model's declared preprocessing contract.
    scale=owner.scales(normalization)
    values[:,5] *= scale['rho']/1000.
    values[:,6] *= scale['dt']/(1/60)
    return values


def predict_pressure(network,sample,normalization):
    if CORRECTION is not None:
        import physical_residual
        return physical_residual.predict_pressure(network,sample,normalization,owner,CORRECTION)
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


def pressure_loss(prediction,sample,normalization):
    if CORRECTION is None:return owner.pressure_loss(prediction,sample,normalization)
    import physical_residual
    return physical_residual.pressure_loss(prediction,sample,normalization,owner)
