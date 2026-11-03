"""Frozen full-model correction supervision, counted separately from student work."""
import json
from pathlib import Path
import torch
import feedback_objective as feedback

DISTILL=None;FULL_TEACHER=None
STATS=feedback.STATS
def __getattr__(name):return getattr(feedback,name)

def select(spec):
    global DISTILL
    feedback.select(spec)
    DISTILL=json.loads(Path(__file__).with_name('distillation.json').read_text())
    if DISTILL['weight']not in [0.,1.]or DISTILL['width_fraction']not in [[3,4],[1,2]]:
        raise ValueError('Predeclared distillation ablation required')

def initialize(network,path,helpers,teacher_path=None):
    global FULL_TEACHER
    feedback.initialize(network,path,helpers)
    if teacher_path is None:raise ValueError('Explicit frozen teacher input required')
    from policy import read,sha
    if sha(read(teacher_path,2*1024**2))!=DISTILL['teacher_weights_sha256']:raise ValueError('Full teacher changed')
    import teacher_model
    FULL_TEACHER=teacher_model.PressureCorrectionNet().to(next(network.parameters()).device).eval()
    helpers.load_weights(FULL_TEACHER,teacher_path,DISTILL['teacher_inventory'])
    FULL_TEACHER.requires_grad_(False)
    STATS.update(distillation_teacher_forwards=0,distillation_teacher_forward_cells=0,distillation_loss_calls=0)

def loss(model,network,sample,normalization,criterion=None,**kwargs):
    if not DISTILL['weight']:return feedback.loss(model,network,sample,normalization,criterion,**kwargs)
    if FULL_TEACHER is None:raise ValueError('Frozen teacher not initialized')
    auxiliary=[]
    def supervise(module,inputs,predicted):
        values=inputs[0].detach()
        with torch.no_grad():target=FULL_TEACHER(values).detach()
        difference=(predicted-target).to(torch.float64)
        target=target.to(torch.float64)
        # Gauge does not carry useful pressure information.
        difference=difference-difference.mean(dim=(2,3,4),keepdim=True)
        target=target-target.mean(dim=(2,3,4),keepdim=True)
        relative=difference.flatten(1).square().sum(1)/target.flatten(1).square().sum(1).clamp_min(1e-30)
        auxiliary.append(torch.log1p(relative/1e-12))
        STATS['distillation_teacher_forwards']+=1
        STATS['distillation_teacher_forward_cells']+=values.numel()
    hook=network.register_forward_hook(supervise)
    try:result=feedback.loss(model,network,sample,normalization,criterion,**kwargs)
    finally:hook.remove()
    if not auxiliary:raise ValueError('No correction participated in student supervision')
    STATS['distillation_loss_calls']+=1
    return result+DISTILL['weight']*torch.stack(auxiliary).mean(0)
