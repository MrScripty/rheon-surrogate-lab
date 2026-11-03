"""Compare the actual constant-field failure with its exact analytic zero oracle."""
import argparse,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from compact_fft_dct_v1 import CompactFFTDCT
from qualify_compact_fft_dct_v1 import matrix,matrix_response,error
from types import SimpleNamespace
from support import write
p=argparse.ArgumentParser();p.add_argument('--protocol');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
fft=CompactFFTDCT();results=[]
for shape in [(64,64,64),(128,128,128)]:
 fft.prepare(shape);basis={}
 for n in set(shape):
  j=torch.arange(n,device='cuda',dtype=torch.float64);b=torch.cos((math.pi/n)*j[:,None]*(j[None,:]+.5))*math.sqrt(2/n);b[0]=1/math.sqrt(n);basis[n]=b
 gen=torch.Generator(device='cuda').manual_seed(831277+sum((a+1)*n for a,n in enumerate(shape)))
 divisor=.5+1.5*torch.rand(shape,generator=gen,device='cuda',dtype=torch.float64)
 value=torch.ones(shape,device='cuda',dtype=torch.float64);core=SimpleNamespace(divisors={'g'+'_'.join(map(str,shape)):divisor})
 actual=fft.pressure(core,value[None,None],tuple(reversed(shape)))[0,0]
 coefficients=fft.forward(value);nonDC=coefficients.clone();nonDC[0,0,0]=0.
 unfused=coefficients/divisor;unfused[0,0,0]=0.;unfused=fft.inverse(unfused);unfused=unfused-unfused[:1,:1,:1]
 reference=matrix_response(value,basis,divisor)
 row=dict(shape_zyx=shape,candidate_vs_exact_zero=error(actual,torch.zeros_like(actual)),unfused_vs_exact_zero=error(unfused,torch.zeros_like(unfused)),matrix_vs_exact_zero=error(reference,torch.zeros_like(reference)),candidate_vs_matrix=error(actual,reference),nonDC_coefficient_norm=float(nonDC.norm()),fused_unfused_exact=bool(torch.equal(actual,unfused)),learned_forward=False)
 assert row['candidate_vs_exact_zero']['relative_l2']==0 and row['unfused_vs_exact_zero']['relative_l2']==0 and row['nonDC_coefficient_norm']==0
 assert row['matrix_vs_exact_zero']['relative_l2']>2e-12
 results.append(row)
args.output.mkdir(parents=True);write(args.output/'results.json',dict(status='isolated_explicit_matrix_constant_DC_roundoff',rows=results,learned_forward=False));fft.close();print(results,flush=True)
