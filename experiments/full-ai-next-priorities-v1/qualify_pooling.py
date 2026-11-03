"""Compare real pooling outputs and vector-Jacobian products with the reference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from adaptive_pool import adaptive_pool3d,fixed_pool2
from study_support import write,sha,HERE

def run(args):
    if args.output.exists():raise ValueError('Fresh pooling qualification required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    generator=torch.Generator().manual_seed(2026101924);rows=[]
    for shape in ((1,1,3,3,3),(1,8,9,9,9),(1,12,17,17,17),(1,12,33,33,33),(1,8,9,6,5)):
        reference=torch.randn(shape,generator=generator,requires_grad=True)
        size=(3,3,3)if args.kind=='adaptive'else tuple((n+1)//2 for n in shape[-3:])
        gradient=torch.randn((shape[0],shape[1],*size),generator=generator)
        output=F.adaptive_avg_pool3d(reference,(3,3,3))if args.kind=='adaptive'else F.avg_pool3d(reference,2,ceil_mode=True)
        output.backward(gradient)
        value=reference.detach().to(args.device).requires_grad_(True)
        observed=adaptive_pool3d(value,(3,3,3))if args.kind=='adaptive'else fixed_pool2(value)
        observed.backward(gradient.to(args.device))
        output_error=float(torch.linalg.vector_norm(observed.detach().cpu()-output.detach()))/float(torch.linalg.vector_norm(output.detach()))
        gradient_error=float(torch.linalg.vector_norm(value.grad.cpu()-reference.grad))/float(torch.linalg.vector_norm(reference.grad))
        if output_error>1e-6 or gradient_error>1e-6:raise ValueError('Pooling value or full gradient equivalence failed')
        first=value.grad.detach().clone();value.grad=None
        repeated=adaptive_pool3d(value,(3,3,3))if args.kind=='adaptive'else fixed_pool2(value)
        repeated.backward(gradient.to(args.device))
        if not torch.equal(first,value.grad):raise ValueError('Repeated pooling gradients differ')
        rows.append(dict(shape=list(shape),output_relative_difference=output_error,gradient_relative_difference=gradient_error,repeated_gradient_bitwise_equal=True))
    write(args.output,dict(status='passed',device=args.device,kind=args.kind,rows=rows,relative_tolerance=1e-6,
        deterministic_algorithms=True,tf32=False,optimizer_updates=0,
        source_sha256=sha(__file__),operator_source_sha256=sha(HERE/'adaptive_pool.py')))
    print('Pooling forward and full gradients equivalent; repeated gradients bitwise equal',args.device,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--kind',choices=['adaptive','fixed'],default='adaptive')
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
