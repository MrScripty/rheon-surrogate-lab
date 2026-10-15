#!/usr/bin/env python3
"""Reusable sealed-water capture, packet, managed project and native evaluation CLI."""
import argparse
import json
from pathlib import Path
from policy import HERE,LAB,decode,read,load_module


def observer(root):
    module=load_module('water_pipeline_observer',LAB/'experiments/live-project-v1/observer.py');value=module.Observer(root)
    if (root/'frame.json').exists():value.preview=decode(read(root/'frame.json',2*1024**2))
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    build=commands.add_parser('build')
    for name in ('config','prepared','cargo','output'):build.add_argument('--'+name,type=Path,required=True)
    native_eval=commands.add_parser('build-evaluation')
    for name in ('prepared','cargo','output'):native_eval.add_argument('--'+name,type=Path,required=True)
    generate=commands.add_parser('generate')
    for name in ('config','binary','output'):generate.add_argument('--'+name,type=Path,required=True)
    generate.add_argument('--fixture',action='store_true')
    generate.add_argument('--qualified-fixture',type=Path)
    generate.add_argument('--pilot-parents',type=int)
    generate.add_argument('--live-root',type=Path,default=LAB/'runs/live-project')
    pack=commands.add_parser('pack')
    for name in ('capture','output'):pack.add_argument('--'+name,type=Path,required=True)
    export=commands.add_parser('export-fields')
    for name in ('capture','tuldok','output'):export.add_argument('--'+name,type=Path,required=True)
    export.add_argument('--population',choices=('train_validation','reserved_test'),required=True)
    assemble=commands.add_parser('assemble')
    for name in ('packet','test-packet','runtime-manifest','tuldok','output'):assemble.add_argument('--'+name,type=Path,required=True)
    assemble.add_argument('--config',type=Path,required=True)
    evaluate=commands.add_parser('evaluate')
    for name in ('packet','inference','output','binary'):evaluate.add_argument('--'+name,type=Path,required=True)
    evaluate.add_argument('--live-root',type=Path,default=LAB/'runs/live-project')
    args=parser.parse_args()
    if args.command=='build':
        import build_native
        result={'binary':str(build_native.build('capture',decode(read(args.config)),args.prepared.absolute(),args.cargo.absolute(),args.output.absolute()))}
    elif args.command=='build-evaluation':
        import evaluation_native
        result={'binary':str(evaluation_native.build('evaluation',{},args.prepared.absolute(),args.cargo.absolute(),args.output.absolute()))}
    elif args.command=='generate':
        import capture
        result=capture.generate(decode(read(args.config)),args.binary.absolute(),args.output.absolute(),fixture=args.fixture,
            observer=observer(args.live_root.absolute()),qualified_fixture=args.qualified_fixture.absolute() if args.qualified_fixture else None,limit=args.pilot_parents)
    elif args.command=='pack':
        import data
        result=data.pack(args.capture.absolute(),args.output.absolute())
    elif args.command=='export-fields':
        import collection
        result=collection.bundles(args.capture.absolute(),args.tuldok.absolute(),args.output.absolute(),args.population)
    elif args.command=='assemble':
        import project
        result=project.assemble(args.packet.absolute(),args.runtime_manifest.absolute(),args.tuldok.absolute(),args.output.absolute(),args.config,args.test_packet.absolute())
    else:
        import diagnostics
        result=diagnostics.run(args.packet.absolute(),args.inference.absolute(),args.output.absolute(),args.binary.absolute(),observer(args.live_root.absolute()))
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
