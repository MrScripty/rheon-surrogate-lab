"""Match the actual trainer's deterministic CUDA math during zero-update qualification."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import torch
import qualify_temporal_fit_v5
from study_support import read,write,sha

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.device='cuda'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    qualify_temporal_fit_v5.run(args)
    record=read(args.output);record.update(cuda_configuration=dict(deterministic_algorithms=True,tf32=False,amp=False),
        cuda_launcher_sha256=sha(__file__))
    write(args.output,record)
