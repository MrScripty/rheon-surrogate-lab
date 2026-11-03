"""Isolate forward FFT striding versus phase/packing; no learned inference."""
import argparse
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fft3d_dct_v2 import FFT3DDCT
from qualify_fft_dct_v2 import matrix,error
from support import write
p=argparse.ArgumentParser();p.add_argument('--protocol');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
a=FFT3DDCT();shape=(11,11,9);a.prepare(shape)
v=torch.zeros(shape,device='cuda',dtype=torch.float64);v[5,5,4]=1.
ext=torch.empty(tuple(2*n for n in shape),device='cuda',dtype=torch.float64)
a.launch('dct_pack_even',ext.numel(),[v,ext],shape)
expected=v
for axis in range(3):expected=torch.cat((expected,expected.flip((axis,))),dim=axis)
assert torch.equal(ext,expected),'Packing differs from explicit mirror'
spectrum=torch.fft.rfftn(ext,dim=(0,1,2),norm='backward');fr,fi,_,_=a.constants[shape]
bases={}
for n in set(shape):
 j=torch.arange(n,device='cuda',dtype=torch.float64);b=torch.cos((math.pi/n)*j[:,None]*(j[None,:]+.5))*math.sqrt(2/n);b[0]=1/math.sqrt(n);bases[n]=b
reference=matrix(v,bases,False);results={}
for name,tensor in [('actual_fft_tensor',spectrum),('contiguous_fft_tensor',spectrum.contiguous())]:
 output=torch.empty_like(v);a.launch('dct_unpack_forward',v.numel(),[tensor,fr,fi,output],shape);results[name]=error(output,reference)
rotated=(spectrum[:shape[0],:shape[1],:shape[2]]*torch.complex(fr,fi)).real
result=dict(shape_zyx=shape,fft_shape=list(spectrum.shape),fft_strides=list(spectrum.stride()),fft_is_contiguous=spectrum.is_contiguous(),pack_exact=True,forward_results=results,explicit_tensor_phase=error(rotated,reference),inverse_output_strides=list(torch.fft.irfftn(spectrum,s=tuple(2*n for n in shape),norm='backward').stride()),learned_forward=False)
assert results['actual_fft_tensor']['relative_l2']>1e-2
assert results['contiguous_fft_tensor']['relative_l2']<2e-12
assert result['explicit_tensor_phase']['relative_l2']<2e-12
args.output.mkdir(parents=True);write(args.output/'results.json',result);a.close()
print(result,flush=True)
