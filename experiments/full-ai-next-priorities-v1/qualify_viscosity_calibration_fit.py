"""Exercise the actual viscosity trainer constructor and gradients without updates."""
import argparse
from pathlib import Path
import sys
from study_support import ROOT,HERE,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh viscosity-fit qualification required')
    assembly=read(args.project/'assembly.json')
    for row in assembly['sources']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Captured source changed')
    sys.path.insert(0,str(args.project.resolve()));import viscosity_calibration_train as trainer
    torch=trainer.torch
    context=argparse.Namespace(config=args.project/'config.json',**{slot:ROOT/spec['path']for slot,spec in assembly['static_inputs'].items()})
    config,network,velocity,target,mask=trainer.construct(context,args.device)
    initial=trainer.inventory(network);gradients=[]
    for repeat in range(2):
        network.zero_grad(set_to_none=True)
        loss=trainer.objective(network,velocity[:8],target[:8],mask);loss.backward()
        gradient=network.projection.weight.grad
        if gradient is None or not bool(torch.isfinite(gradient).all())or not bool((gradient!=0).any()):
            raise ValueError('Finite active viscosity gradients required')
        gradients.append(gradient.detach().cpu().clone())
    if not torch.equal(*gradients)or trainer.inventory(network)!=initial:
        raise ValueError('Zero-update gradient repeatability failed')
    probe=None
    if args.device=='cpu':
        probe=trainer.cpu_probe(network,velocity)
        if torch.cuda.is_initialized():raise ValueError('CPU initialized CUDA')
    write(args.output,dict(status='passed',device=args.device,optimizer_updates=0,parameters=243,
        project_sha256=sha(args.project/'project.json'),source_sha256={row['path']:row['sha256']for row in assembly['sources']},
        actual_trainer_constructor_exercised=True,actual_objective_constructor_exercised=True,
        finite_active_repeatable_gradients=True,initial_weights_unchanged=True,objective=float(loss.detach()),
        probe_sha256=trainer.digest(probe.tobytes())if probe is not None else None,
        qualifier_sha256=sha(__file__),full_state_prediction=False))
    print('Qualified actual viscosity trainer',args.device,'243 coefficients; zero updates',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],required=True);run(parser.parse_args())
