#!/usr/bin/env python3
"""Freeze CUDA/CPU recipes or diagnose their retained managed Run artifacts."""
import argparse
from pathlib import Path
import json

from policy import LAB, HERE, load_module


def observer(root):
    module=load_module('cuda_pressure_observer',LAB/'experiments/live-project-v1/observer.py')
    result=module.Observer(root)
    if (root/'frame.json').exists():result.preview=json.loads((root/'frame.json').read_text())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    assemble=commands.add_parser('assemble')
    assemble.add_argument('--packet',type=Path,required=True);assemble.add_argument('--runtime-manifest',type=Path,required=True)
    assemble.add_argument('--tuldok',type=Path,required=True);assemble.add_argument('--output',type=Path,required=True)
    assemble.add_argument('--config',type=Path,default=HERE/'configs/qualification.json')
    evaluate=commands.add_parser('evaluate')
    for name in ('packet','binary','training','restored','output'):evaluate.add_argument('--'+name,type=Path,required=True)
    evaluate.add_argument('--live-root',type=Path,default=LAB/'runs/live-project')
    args=parser.parse_args()
    if args.command=='assemble':
        import project
        result=project.assemble(args.packet.absolute(),args.runtime_manifest.absolute(),args.tuldok.absolute(),args.output.absolute(),args.config)
    else:
        import evaluate
        result=evaluate.run(args.packet.absolute(),args.binary.absolute(),args.training.absolute(),args.restored.absolute(),args.output.absolute(),observer(args.live_root.absolute()))
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
