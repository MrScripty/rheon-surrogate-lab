"""Minimal regression for the deterministic CUDA pooling backward used by the model."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import torch
from torch.nn import functional as F

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation',choices=['reference','separable','fixed-reference','fixed-separable'],required=True)
    args=parser.parse_args();torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(2026101923)
    value=torch.randn((1,1,3,3,3),device='cuda',requires_grad=True)
    if args.implementation=='reference':pooled=F.adaptive_avg_pool3d(value,(2,2,2))
    elif args.implementation=='fixed-reference':pooled=F.avg_pool3d(value,2,ceil_mode=True)
    elif args.implementation=='fixed-separable':
        from adaptive_pool import fixed_pool2
        pooled=fixed_pool2(value)
    else:
        from adaptive_pool import adaptive_pool3d
        pooled=adaptive_pool3d(value,(2,2,2))
    pooled.sum().backward()
    if value.grad is None or not bool(torch.isfinite(value.grad).all()):raise ValueError('Pooling gradient missing or nonfinite')
    print('Deterministic CUDA pooling backward passed',args.implementation,flush=True)
