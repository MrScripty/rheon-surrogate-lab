"""Run zero-update objective qualification in the existing CUDA runtime."""
import argparse
from pathlib import Path
import subprocess
from v4_support import ROOT,read
from common import environment

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kind',choices=['student','continuation'],default='student')
    args=parser.parse_args();runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    qualifier='qualify_student.py'if args.kind=='student'else'qualify_continuation.py'
    subprocess.run([runtime['python'],'-B',str(Path(__file__).with_name(qualifier)),'--project',str(args.project),'--output',str(args.output),'--cuda'],
        cwd=ROOT,env=environment(runtime,'cuda'),check=True,timeout=600)
