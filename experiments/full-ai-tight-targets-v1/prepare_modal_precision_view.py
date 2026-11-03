"""Copy original train-only coefficients and exact inherited managed parameters."""
import argparse
from pathlib import Path
import shutil
from profile_frozen_components import ROOT,read,write,digest
from train_modal_pressure import inventory,load
from learned_modal_pressure import LearnedModalPressure


def run(args):
    if args.output.exists():raise ValueError('Fresh pressure precision view required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered pressure refinement source changed: '+path)
    original=ROOT/reg['view'];manifest=read(original/'manifest.json');pair=ROOT/reg['pair'];receipt=read(pair/'restore/receipt.json')
    if manifest['status']!='verified_train_only_fixed_transform_view'or manifest['development_or_reserved_used']or manifest['scenes']!=16:
        raise ValueError('All original native training pairs, no held-out data, required')
    if not receipt['full_cpu_probe_bitwise_verified']or receipt['optimizer_updates']!=0:raise ValueError('Exact inherited managed CPU restoration required')
    model=LearnedModalPressure(manifest['geometries']);load(model,pair/'restore/weights.npz',receipt['final_inventory'])
    args.output.mkdir(parents=True)
    for row in manifest['geometries']:
        path=original/row['path']
        if digest(path)!=row['sha256']:raise ValueError('Native coefficient data changed')
        shutil.copyfile(path,args.output/row['path'])
    normalization=read(original/'normalization.json')
    normalization.update(initial_pressure_parameters={name:value.detach().numpy().tolist()for name,value in model.state_dict().items()},
        initial_pressure_inventory=inventory(model),initial_pressure_pair=reg['pair'],
        initial_pressure_archive_sha256=digest(pair/'restore/weights.npz'),parameter_learning_executed=0)
    write(args.output/'normalization.json',normalization)
    manifest.update(normalization_sha256=digest(args.output/'normalization.json'),
        original_view_manifest_sha256=digest(original/'manifest.json'),initialization_provenance=reg['pair'],
        precision_view_protocol_sha256=digest(args.protocol),precision_view_source_sha256=reg['source_sha256'])
    write(args.output/'manifest.json',manifest)
    print('Preserved144original native coefficient pairs and all38673exact inherited learned pressure parameters; no learning',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
