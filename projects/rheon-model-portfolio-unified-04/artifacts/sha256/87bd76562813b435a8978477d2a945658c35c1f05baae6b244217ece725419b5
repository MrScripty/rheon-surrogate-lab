"""Train two differentiable GCR iterations with the actual neural preconditioner."""
import torch
import feedback_objective as feedback

STATS=feedback.STATS
def __getattr__(name):return getattr(feedback,name)
def select(spec):feedback.select(spec)
def initialize(network,path,helpers):
    feedback.initialize(network,path,helpers);STATS.update(convergence_loss_calls=0,convergence_proposals=0)

def predict(network,sample,normalization,owner,steps=2,intermediates=None):
    import physical_residual
    base=sample['base'];initial=sample['native_input_residual'];h=sample['cell_size_m'];rho=sample['rho']
    delta=torch.zeros_like(base);residual=initial;directions=[];images=[]
    for _ in range(steps):
        zero=torch.zeros_like(base)
        proposal=physical_residual.predict_pressure(network,dict(sample,base=zero,native_input_residual=residual),
            normalization,owner,{'activation':'linear','steps':1})
        image=owner.full_operator(proposal,h,rho)
        for _ in range(2):
            for prior,image_prior in zip(directions,images):
                beta=(image*image_prior).flatten(1).sum(1).reshape(-1,1,1,1,1)
                proposal=proposal-beta*prior;image=image-beta*image_prior
        length=image.flatten(1).square().sum(1).clamp_min(1e-60).sqrt().reshape(-1,1,1,1,1)
        proposal=proposal/length;image=image/length
        alpha=(residual*image).flatten(1).sum(1).reshape(-1,1,1,1,1)
        delta=delta+alpha*proposal
        residual=initial-owner.full_operator(delta,h,rho)
        directions.append(proposal);images.append(image)
        if intermediates is not None:intermediates.append(base+delta)
    return base+delta

def loss(model,network,sample,normalization,criterion=None,**kwargs):
    pressures=[]
    prediction=predict(network,sample,normalization,model.owner,feedback.SPEC['training_passes'],pressures)
    result=model.pressure_loss(prediction,sample,normalization)if criterion is None else criterion(prediction)
    values=[]
    for p in pressures:
        residual=sample['native_input_residual']-model.full_operator(p-sample['base'],sample['cell_size_m'],sample['rho'])
        # Combine the target and reference scale before clamping. Successive
        # divisions by 1e-300 and 1e-12 had a finite zero forward value but an
        # overflowing derivative (1e312), producing NaN coarse-head gradients.
        denominator=(sample['native_input_residual'].flatten(1).square().sum(1)
                     *feedback.SPEC['relative_equation_target']**2).clamp_min(1e-300)
        values.append(torch.log1p(residual.flatten(1).square().sum(1)/denominator))
    result=result+values[-1]
    if len(values)>1:result=result+feedback.SPEC['early_weight']*torch.stack(values[:-1]).mean(0)
    STATS['convergence_loss_calls']+=1;STATS['convergence_proposals']+=len(pressures)
    return result
