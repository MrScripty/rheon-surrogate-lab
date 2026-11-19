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
    import training_multilevel as mg
    base=sample['base'];h=sample['cell_size_m'];rho=sample['rho']
    initial=sample['native_input_residual'];delta=torch.zeros_like(base)
    residual=sample['rhs']-model.full_operator(base,h,rho)
    if not bool(residual.ne(0).any()):return residual.flatten(1).sum(1)*0
    directions=[];images=[];losses=[]
    for _ in range(SPEC['solver_training_iterations']):
        # Autograd copies preserve the gradient through the exact frozen CPU
        # hierarchy, including CUDA -> CPU -> CUDA. Nothing is detached.
        coarse=mg.correction(residual.to('cpu'),h.to('cpu'),rho.to('cpu')).to(base.device)
        remaining=residual-model.full_operator(coarse,h,rho)
        local=model.predict_pressure(network,dict(sample,base=torch.zeros_like(base),native_input_residual=remaining),normalization)
        d=coarse+local;ad=model.full_operator(d,h,rho)
        for _sweep in range(2):
            for old,old_image in zip(directions,images):
                beta=(ad*old_image).sum();d=d-beta*old;ad=ad-beta*old_image
        length=torch.linalg.vector_norm(ad).clamp_min(1e-100)
        d=d/length;ad=ad/length;alpha=(residual*ad).sum()
        delta=delta+alpha*d;prediction=base+delta
        residual=sample['rhs']-model.full_operator(prediction,h,rho)
        supervised_loss=supervised(model,prediction,sample)if criterion is None else criterion(prediction)
        denom=initial.flatten(1).square().sum(1).clamp_min(1e-100)
        equation=residual.flatten(1).square().sum(1)/denom
        losses.append(supervised_loss+torch.log1p(equation/1e-12))
        directions.append(d);images.append(ad)
        STATS['network_forwards']=STATS.get('network_forwards',0)+1
        STATS['coarse_cycles']=STATS.get('coarse_cycles',0)+1
    STATS['loss_calls']=STATS.get('loss_calls',0)+1
    return torch.stack(losses).mean(0)
