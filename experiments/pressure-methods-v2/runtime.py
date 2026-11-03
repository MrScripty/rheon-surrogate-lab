"""Launch a bounded command in the already qualified managed Python runtime."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import environment

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    runtime=json.loads((ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json').read_text())
    raise SystemExit(subprocess.run([runtime['python'],'-B',*args.command],env=environment(runtime),timeout=args.timeout).returncode)
