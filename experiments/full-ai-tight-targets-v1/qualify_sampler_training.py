"""Zero-update qualification of the actual transport training graph/artifacts."""
import argparse
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from profile_frozen_components import ROOT,read,write,digest
from learned_sampler_head import LearnedSamplerHead,objective
from train_sampler_head import data,inventory,save,load
from recurrent_sampler_reference import interpolate
import torch


def run(args):
    if args.output.exists():raise ValueError('Fresh actual training qualification required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Training source/data changed: '+path)
    if os.environ.get('TULDOK_TORCH_DEVICE')!=args.device:raise ValueError('Actual qualified runtime device required')
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.manual_seed(reg['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    path=ROOT/reg['manifest'];manifest=read(path);norm=read(path.parent/'normalization.json')
    group=data(SimpleNamespace(shards=[path.parent/r['path']for r in manifest['shards']]),manifest,args.device)
    model=LearnedSamplerHead(norm['inherited_mix']).to(args.device);before=inventory(model)
    scales=torch.tensor(norm['scales'],dtype=torch.float64,device=args.device)
    loss=objective(model,group,scales);loss.backward()
    if not bool(torch.isfinite(loss))or not bool(loss>0):raise ValueError('Actual training loss failed nontrivial control')
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all())or not bool(p.grad.norm()>0)for p in model.parameters()):
        raise ValueError('Every actual learned parameter group needs finite nonzero gradients')
    if inventory(model)!=before:raise ValueError('Qualification changed learned tensors')
    model=model.cpu();source_error=0.
    # Arbitrary nonzero neural gates/mixing, independent scalar interpolation.
    with torch.no_grad():
        torch.manual_seed(reg['seed']);model.mid_gate.uniform_(-.25,.75);model.mix.add_(torch.randn_like(model.mix)*.02)
        for kind in range(4):
            indices=(group['kind']==kind).nonzero().flatten()[:8]
            batch={k:v[indices].cpu()for k,v in group.items()};middle,end=model(batch['point'],batch['patch'],batch['kind'],batch['alpha'])
            for i in range(len(indices)):
                phase=torch.tensor([0. if a==kind else .5 for a in range(3)],dtype=torch.float64)
                query=phase+middle[i]*batch['alpha'][i]
                values=[]
                for c in range(3):
                    offset=torch.tensor([0. if a==c else .5 for a in range(3)],dtype=torch.float64)
                    values.append(interpolate(batch['patch'][i,c],(query-offset+1).reshape(3,1))[0])
                sampled=torch.stack(values);expected=[]
                for a in range(3):
                    v=torch.zeros((),dtype=torch.float64)
                    for c in range(3):v=v+model.mix[kind,a,c]*sampled[c]
                    expected.append(v)
                error=float((end[i]-torch.stack(expected)).abs().max());source_error=max(source_error,error)
        if source_error>1e-15:raise ValueError('Local training graph differs from registered scalar interpolation')
    # Verify exact typed serialization with the trained shape, zero updates.
    checked=inventory(model)
    with tempfile.TemporaryDirectory(prefix='rheon-sampler-head-',dir='/tmp')as directory:
        output=Path(directory)/'weights.npz';save(model,output);restored=LearnedSamplerHead(norm['inherited_mix']);load(restored,output,checked)
        if inventory(restored)!=checked:raise ValueError('Exact typed sampler restore failed')
    write(args.output,dict(status='passed',device=args.device,optimizer_updates=0,parameters=48,objective=float(loss.detach()),
        original_parameters_unchanged_before_seeded_control=True,typed_double_save_load_verified=True,
        maximum_independent_patch_interpolation_absolute_error=source_error,training_points=manifest['points'],
        cuda_initialized=torch.cuda.is_initialized(),protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],
        model_source_sha256=digest(Path(__file__).with_name('learned_sampler_head.py')),
        objective_source_sha256=digest(Path(__file__).with_name('learned_sampler_head.py')),
        trainer_source_sha256=digest(Path(__file__).with_name('train_sampler_head.py'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    run(parser.parse_args())
