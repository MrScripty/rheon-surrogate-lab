"""Compare the actual differentiable Project method to the independent GCR loop."""
import os
import sys
import argparse
from pathlib import Path
from v4_support import ROOT,BASE,read,write,sha

def run(args):
    os.sched_setaffinity(0,{22});project=args.project;sys.path.insert(0,str(project))
    import torch,model,data,fine_tuning,train
    model.configure('cpu');torch.set_num_threads(1);config=read(project/'config.json');model.select(config)
    paths={r['id']:ROOT/r['path']for r in read(project/'project.json')['data']}
    norm=read(project/'fixed-normalization.json');network=model.PressureCorrectionNet()
    fine_tuning.initialize(network,paths['initial_weights'],data.helpers())
    import physical_residual
    from preconditioned import gcr
    generator=torch.Generator().manual_seed(2026101909)
    reference=torch.randn((1,1,9,7,5),generator=generator,dtype=torch.float64);reference=reference-reference[:,:,:1,:1,:1].clone()
    h=torch.tensor([1/16],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64);base=reference*.3
    rhs=model.full_operator(reference,h,rho)
    sample=dict(base=base,reference=reference,rhs=rhs,native_input_residual=rhs-model.full_operator(base,h,rho),
        cell_size_m=h,rho=rho,dt=torch.tensor([1/60],dtype=torch.float64));zero=torch.zeros_like(base)
    def action(p):return model.full_operator(p,h,rho)
    def proposal(r):return physical_residual.predict_pressure(network,dict(sample,base=zero,native_input_residual=r),norm,model.owner,{'activation':'linear','steps':1})
    with torch.no_grad():
        result=model.predict_pressure(network,sample,norm)
        reference,work=gcr(sample['native_input_residual'],action,proposal,lambda r:False,max_iterations=2)
        reference=sample['base']+reference
        difference=float(torch.linalg.vector_norm(result-reference))/max(float(torch.linalg.vector_norm(reference)),1e-300)
        if work['iterations']!=2 or difference>1e-12:raise ValueError('Training GCR differs from independent inference loop')
    empty={k:torch.zeros_like(v)if k in ['base','rhs','reference','native_input_residual']else v for k,v in sample.items()}
    calls=[];hook=network.register_forward_hook(lambda *_:calls.append(1))
    network.zero_grad(set_to_none=True);loss=model.training_loss(network,empty,norm).mean()
    if loss.requires_grad:loss.backward()
    hook.remove()
    # The captured historical predictor explicitly selects only nonzero input
    # residuals. Its fine network is intentionally absent from this zero graph.
    # Missing gradients on that bypass are valid; NaN/Inf gradients are not.
    missing=[name for name,p in network.named_parameters()if p.grad is None]
    bad=[name for name,p in network.named_parameters()if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    passed=float(loss.detach())==0 and not bad and not calls
    write(args.output,dict(status='passed'if passed else'failed_zero_backward',pressure_relative_difference=difference,iterations=2,
        zero_equation_loss=float(loss.detach()),finite_zero_backward=passed,nonfinite_gradient_names=bad,
        skipped_fine_gradient_names=missing,zero_equation_network_forwards=len(calls),optimizer_updates=0,
        project_sha256=sha(project/'project.json'),source_sha256=sha(__file__),preconditioner_source_sha256=sha(__import__('pathlib').Path(__file__).with_name('preconditioned.py'))))
    if not passed:raise ValueError('Zero equation must have finite zero-loss backward behavior')
    print('Differentiable GCR matches independent loop; finite zero-equation backward',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
