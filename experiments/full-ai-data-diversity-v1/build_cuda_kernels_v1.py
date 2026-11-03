"""Compile a registered precise CUDA execution backend, without model inference."""
import argparse
from pathlib import Path
from cuda_kernel_runtime_v1 import compile_kernels

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    compile_kernels(args.protocol,args.output.resolve())
