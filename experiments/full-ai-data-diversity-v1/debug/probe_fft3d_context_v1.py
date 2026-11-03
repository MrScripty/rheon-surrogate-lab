"""Minimal actual module-loader context regression; no model or numeric transform."""
import argparse
import ctypes as C
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fft3d_dct_v1 import FFT3DDCT
from support import write
p=argparse.ArgumentParser();p.add_argument('--protocol');p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True)
driver=C.CDLL('libcuda.so.1');driver.cuInit(0);driver.cuCtxGetCurrent.argtypes=[C.POINTER(C.c_void_p)]
def context():
 v=C.c_void_p();s=driver.cuCtxGetCurrent(C.byref(v));return dict(status=s,pointer=v.value)
result=dict(before=context(),red_error=None,green=False,learned_forward=False)
try:
 try:
  value=FFT3DDCT()
 except RuntimeError as e:
  result['red_error']=str(e)
 else:
  value.close();raise AssertionError('Context regression no longer reproduces')
 assert result['before']['pointer'] is None
 assert result['red_error']=='cuModuleLoadDataEx failed with CUDA/NVRTC status 201'
 allocation=torch.empty((),device='cuda',dtype=torch.float64)
 result['after_allocation']=context()
 assert result['after_allocation']['pointer'] is not None
 value=FFT3DDCT();value.close();result['green']=True
 print('Same source/PTX constructor: context absent -> status201; one CUDA allocation -> passed',flush=True)
finally:write(a.output/'results.json',result)
