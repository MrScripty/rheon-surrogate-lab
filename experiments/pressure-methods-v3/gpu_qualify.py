"""Launch only the zero-update GPU qualification in the qualified CUDA runtime."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,environment

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    subprocess.run([runtime['python'],'-B',str(Path(__file__).with_name('qualify.py')),'--project',str(args.project),
        '--output',str(args.output),'--cuda'],env=environment(runtime,'cuda'),check=True,timeout=600)
