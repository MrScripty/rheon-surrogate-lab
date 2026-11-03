"""Qualify disjoint optimizer groups without executing an optimizer step."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha, source_inputs
import qualify_training


def qualify(project,output):
    qualify_training.qualify(project,output)
    import torch
    import model
    import data
    import fine_tuning
    import head_optimizer
    config = read(project/'config.json')
    network = model.PressureCorrectionNet().eval()
    paths = source_inputs(project)
    initial=ROOT/next(row['path']for row in read(project/'project.json')['data']if row['id']=='initial_weights')
    fine_tuning.initialize(network,initial,data.helpers())
    inventory = data.helpers().state_inventory(network)
    groups = head_optimizer.groups(network,config['training']['learning_rate'])
    optimizer = torch.optim.Adam(groups,lr=config['training']['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0)
    ids = [id(p)for group in optimizer.param_groups for p in group['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p)for p in network.parameters()}:
        raise ValueError('Optimizer groups must own every parameter exactly once')
    if optimizer.param_groups[1]['params']!=[network.coarse_logits]:
        raise ValueError('Second group must contain only the added logits')
    schedules=[]
    for rate in [config['training']['learning_rate'],*[r['learning_rate']for r in config['training']['schedule']]]:
        for group in optimizer.param_groups:
            group['lr'] = rate*group['lr_multiplier']
        schedules.append(dict(fine=optimizer.param_groups[0]['lr'],coarse=optimizer.param_groups[1]['lr']))
    _,parents = data.load(tuple(paths[k].read_bytes()for k in ['dataset','dataset_extra']),read(paths['splits']))
    sample = data.sample(parents[0],0)
    normalization = model.fit_normalization(parents,data.fit_normalization)
    loss = model.training_loss(network,sample,normalization).mean()
    loss.backward()
    gradient = network.coarse_logits.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()) or float(gradient.abs().sum())==0:
        raise ValueError('New coarse branch must have a finite nonzero trainable gradient')
    if optimizer.state or data.helpers().state_inventory(network)!=inventory or torch.cuda.is_initialized():
        raise ValueError('Qualification executed an update or changed weights/device')
    result = read(output)
    result['coarse_head_optimizer'] = dict(disjoint_complete_groups=True,optimizer_steps=0,
        schedules=schedules,added_head_gradient_l2=float(torch.linalg.vector_norm(gradient)),
        captured_optimizer_source_sha256=sha(project/'head_optimizer.py'))
    result['source_sha256'] = sha(__file__)
    write(output,result)
    print('Qualified separate coarse-head rates without any optimizer step',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    qualify(args.project.resolve(),args.output.resolve())
