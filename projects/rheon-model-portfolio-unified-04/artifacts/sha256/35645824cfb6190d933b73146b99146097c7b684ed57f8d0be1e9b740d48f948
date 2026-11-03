"""Same-equation pressure/gradient supervision on original or post-coarse inputs."""
import torch
SPEC=None
STATS={}
def select(spec):
    if spec['schema']!='rheon-multilevel-smoother-objective-v1' or spec['coarse_training_cycles']not in [0,1]:raise ValueError('Declared smoother objective required')
    global SPEC;SPEC=spec
def initialize(network,path,helpers):
    from policy import read,sha
    if sha(read(path,2*1024**2))!=SPEC['initial_weights_sha256']:raise ValueError('Initialization changed')
    helpers.load_weights(network,path,SPEC['initial_inventory'])
    STATS.clear();STATS.update(loss_calls=0,teacher_forwards=0)
def ratio(value,reference):
    return value.flatten(1).square().sum(1)/reference.flatten(1).square().sum(1).clamp_min(1e-100)
def supervised(model,prediction,sample):
    target=sample['reference']-sample['base'];target=target-target.mean((2,3,4),keepdim=True)
    error=prediction-sample['reference'];error=error-error.mean((2,3,4),keepdim=True)
    pressure=ratio(error,target)
    gradient_error=torch.zeros_like(pressure);gradient_target=torch.zeros_like(pressure)
    for axis in [2,3,4]:
        gradient_error+=torch.diff(error,dim=axis).flatten(1).square().sum(1)
        gradient_target+=torch.diff(target,dim=axis).flatten(1).square().sum(1)
    r=sample['native_input_residual']-model.full_operator(prediction-sample['base'],sample['cell_size_m'],sample['rho'])
    denominator=sample['native_input_residual'].flatten(1).square().sum(1).clamp_min(1e-100)
    equation=r.flatten(1).square().sum(1)/denominator
    maximum=r.flatten(1).abs().amax(1).square()*r[0].numel()/denominator
    return .25*pressure+.25*gradient_error/gradient_target.clamp_min(1e-100)+.4*equation+.1*maximum
def loss(model,network,sample,normalization,criterion=None):
    prediction=model.predict_pressure(network,sample,normalization)
    result=supervised(model,prediction,sample)if criterion is None else criterion(prediction)
    r=sample['native_input_residual']-model.full_operator(prediction-sample['base'],sample['cell_size_m'],sample['rho'])
    denominator=(sample['native_input_residual'].flatten(1).square().sum(1)*1e-12).clamp_min(1e-100)
    STATS['loss_calls']=STATS.get('loss_calls',0)+1
    return result+torch.log1p(r.flatten(1).square().sum(1)/denominator)

