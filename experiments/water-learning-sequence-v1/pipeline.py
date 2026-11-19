"""Reusable lossless views, measured managed-trial assembly and native evaluation."""
import argparse
import json
from pathlib import Path
from policy import LAB


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    assemble=commands.add_parser('assemble')
    for name in ('packet','validation','reserved','runtime-manifest','tuldok','output','config'):
        assemble.add_argument('--'+name,type=Path,required=True)
    evaluate=commands.add_parser('evaluate')
    for name in ('packet','inference','output','binary'):evaluate.add_argument('--'+name,type=Path,required=True)
    evaluate.add_argument('--training-only',action='store_true')
    evaluate.add_argument('--live-root',type=Path,default=LAB/'runs/live-project')
    args=parser.parse_args()
    if args.command=='assemble':
        import project
        result=project.assemble(args.packet.absolute(),args.validation.absolute(),args.reserved.absolute(),args.runtime_manifest.absolute(),args.tuldok.absolute(),args.output.absolute(),args.config.absolute())
    else:
        import diagnostics
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('sequence_observer',LAB/'experiments/live-project-v1/observer.py');module=module_from_spec(spec);spec.loader.exec_module(module)
        result=diagnostics.run(args.packet.absolute(),args.inference.absolute(),args.output.absolute(),args.binary.absolute(),module.Observer(args.live_root.absolute()),args.training_only)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
