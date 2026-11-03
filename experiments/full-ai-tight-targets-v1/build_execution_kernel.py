"""Compile one registered CPU execution kernel with exact recorded flags."""
import argparse
import json
from pathlib import Path
import subprocess
from build_factorized_force import ROOT,digest


def run(args):
    if args.output.exists():raise ValueError('Fresh execution build required')
    reg=json.loads(args.protocol.read_text())
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered execution source changed')
    if reg['extra_flags']!=['-march=native']:raise ValueError('Only the registered host ISA policy is allowed')
    args.output.mkdir(parents=True);source=ROOT/reg['kernel'];binary=args.output/reg['binary_name']
    command=[reg['compiler'],'-std=c++17','-O3','-fPIC','-shared','-ffp-contract=off','-fno-fast-math',*reg['extra_flags'],str(source),'-o',str(binary)]
    with (args.output/'build.log').open('xb')as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
    version=subprocess.run([reg['compiler'],'--version'],capture_output=True,timeout=10,check=True).stdout.decode()
    target=subprocess.run(['lscpu','-J'],capture_output=True,timeout=10,check=True).stdout.decode()
    receipt=dict(status='built',source_sha256=digest(source),builder_sha256=digest(__file__),binary_sha256=digest(binary),
        protocol_sha256=digest(args.protocol),command=command,compiler_version=version,host_cpu=json.loads(target),
        fast_math=False,fused_multiply_add=False,optimizer_updates=0,architecture_specific_binary=True)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
