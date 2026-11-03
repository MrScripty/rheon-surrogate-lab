"""Build a separately registered execution of existing prescribed forcing."""
import argparse
from pathlib import Path
import shutil
import subprocess
from study_support import HERE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh forcing build required')
    protocol_path=DOCS/'fused-force-construction-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered force source changed')
    compiler=shutil.which('c++')
    if compiler is None:raise ValueError('Host C++ compiler required')
    args.output.mkdir(parents=True);binary=args.output/'libprescribed_force.so'
    command=[compiler,'-std=c++17','-O3','-fPIC','-shared','-ffp-contract=off','-fno-fast-math',
        str(HERE/'fused_force_construction.cpp'),'-o',str(binary)]
    with (args.output/'build.log').open('xb')as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
    version=subprocess.run([compiler,'--version'],capture_output=True,timeout=10,check=True).stdout.decode()
    write(args.output/'receipt.json',dict(status='built',source_sha256=sha(HERE/'fused_force_construction.cpp'),
        builder_sha256=sha(__file__),binary_sha256=sha(binary),compiler=compiler,compiler_version=version,
        command=command,protocol_sha256=sha(protocol_path),fast_math=False,fused_multiply_add=False,optimizer_updates=0))
    print('Built prescribed-forcing execution; no model weights changed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
