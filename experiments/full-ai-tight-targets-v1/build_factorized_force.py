"""Compile a source-bound forcing execution kernel without learning parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    if args.output.exists():
        raise ValueError('Fresh build directory required')
    reg = json.loads(args.protocol.read_text())
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered build source changed')
    args.output.mkdir(parents=True)
    source = ROOT / reg['kernel']
    binary = args.output / 'libcompact_force.so'
    command = [reg['compiler'], '-std=c++17', '-O3', '-fPIC', '-shared', '-ffp-contract=off',
               '-fno-fast-math', str(source), '-o', str(binary)]
    with (args.output / 'build.log').open('xb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    version = subprocess.run([reg['compiler'], '--version'], capture_output=True, check=True, timeout=10).stdout.decode()
    receipt = dict(status='built', source_sha256=digest(source), builder_sha256=digest(__file__),
                   protocol_sha256=digest(args.protocol), binary_sha256=digest(binary),
                   compiler_version=version, command=command, fast_math=False,
                   fused_multiply_add=False, optimizer_updates=0)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
