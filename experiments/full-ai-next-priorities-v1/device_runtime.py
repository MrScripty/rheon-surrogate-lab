"""Launch a finite command in the existing qualified CPU or CUDA runtime."""
import argparse
import subprocess
from study_support import ROOT,runtime
from common import environment

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--timeout',type=int,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not 1<=args.timeout<=3600:raise ValueError('Finite runtime deadline required')
    selected=runtime()
    raise SystemExit(subprocess.run([selected['python'],'-B',*args.command],cwd=ROOT,
        env=environment(selected,args.device),timeout=args.timeout).returncode)
