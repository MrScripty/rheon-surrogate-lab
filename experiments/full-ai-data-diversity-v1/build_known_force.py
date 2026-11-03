"""Build strict arithmetic for prescribed schedule inputs, without learned fitting."""
import argparse
from pathlib import Path
import subprocess
from support import ROOT, read, write, digest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh known-force build required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered known-force source changed')
    args.output.mkdir(parents=True)
    binary=args.output/'libscheduled_force.so'
    command=['/usr/bin/g++','-std=c++17','-O3','-fPIC','-shared','-ffp-contract=off',
             '-fno-fast-math','-march=native',str(ROOT/reg['kernel']),'-o',str(binary)]
    with (args.output/'build.log').open('xb')as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    write(args.output/'receipt.json',dict(status='built',binary_sha256=digest(binary),
        source_sha256=digest(ROOT/reg['kernel']),protocol_sha256=digest(args.protocol),
        command=command,optimizer_updates=0,known_external_input_only=True,
        numerical_pressure_solver=False,physical_state_evolution=False))
