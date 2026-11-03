"""Assemble learned heads locally; never claim a new managed training Run."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil
import time
import numpy as np
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha
from routed_descriptor import describe


def component(entry_path):
    entry=read(entry_path)
    for name,digest in entry['identities'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Managed CPU component identity changed')
    restore=ROOT/entry['restore'];receipt=read(restore/'receipt.json')
    if receipt['mode']!='restore'or receipt['optimizer_updates']!=0 or receipt['execution']['cuda_initialized']:
        raise ValueError('Actual zero-update CPU restore required for every component')
    if sha(restore/'weights.npz')!=entry['weights_sha256']or receipt['outputs']['weights.npz']['sha256']!=entry['weights_sha256']:
        raise ValueError('Published component weight identity differs')
    with np.load(restore/'weights.npz',allow_pickle=False)as source:arrays={name:source[name].copy()for name in source.files}
    return entry,receipt,arrays,dict(entry_path=str(entry_path.relative_to(ROOT)),entry_sha256=sha(entry_path),
        weights_sha256=sha(restore/'weights.npz'),CPU_restore_receipt_sha256=sha(restore/'receipt.json'),
        originating_training_run=receipt.get('originating_training_run'))


def run(args):
    identifier=args.identifier+'-routed'
    if not re.fullmatch('[a-z][a-z0-9-]{0,63}',identifier):raise ValueError('Safe routed model identifier required')
    project=BASE/(identifier+'-inference-01');output=BASE/(identifier+'-assembly-01')
    entry_path=BASE/'entries'/(identifier+'.json')
    if project.exists()or output.exists()or entry_path.exists():raise ValueError('Fresh inference package and assembly required')
    if not (DOCS/'routed-head-registration.json').is_file():raise ValueError('Register routing before assembly')
    original,original_receipt,large,large_origin=component(ROOT/'runs/pressure-methods-v4/entries/v4-student50-control.json')
    calibrated,calibrated_receipt,small,small_origin=component(BASE/'entries'/(args.identifier+'.json'))
    if not calibrated_receipt['fine_tensors_bitwise_unchanged']:raise ValueError('Calibrated fine CNN must be unchanged')
    if set(large)!=set(small):raise ValueError('Original and calibrated architectures differ')
    for name in large:
        if name!='coarse_logits'and not name.startswith('shape_head.')and not np.array_equal(large[name].view('u1'),small[name].view('u1')):
            raise ValueError('Every shared fine tensor must match C107 bitwise')
    orig_restore=ROOT/original['restore'];cal_restore=ROOT/calibrated['restore']
    if sha(orig_restore/'normalization.json')!=sha(cal_restore/'normalization.json'):raise ValueError('Shared C53 normalization differs')
    values={name:value.copy()for name,value in large.items()};mapping={name:dict(component='C107',source_parameter=name)for name in values}
    for name,value in small.items():
        if name=='coarse_logits'or name.startswith('shape_head.'):
            target='small_'+name;values[target]=value.copy();mapping[target]=dict(component=args.identifier,source_parameter=name)
    if sum(value.size for value in values.values())!=94956:raise ValueError('Expected shared fine CNN plus two1198parameter heads')
    prior=ROOT/original['project'];old=read(prior/'source-witness.json');project.mkdir();output.mkdir()
    for name,digest in old['sources'].items():
        if sha(prior/name)!=digest:raise ValueError('Original C107 source changed')
        shutil.copyfile(prior/name,project/name)
    shutil.copytree(prior/'tuldok_extension_sdk',project/'tuldok_extension_sdk',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(prior/'model.py',project/'routed_base_model.py')
    shutil.copyfile(HERE/'routed_pressure_model.py',project/'model.py');shutil.copyfile(HERE/'routed_descriptor.py',project/'routed_descriptor.py')
    spec=dict(maximum_small_axis_cells=16,small_head_component=args.identifier,large_head_component='C107',
        original_fine_parameters=92560,original_large_head_parameters=1198,calibrated_small_head_parameters=1198,
        parameters=94956,parameter_increase_fraction=1198/93758,selection_input='Grid counts only; no labels or measured errors',
        candidate_numerical_pressure_solve=False,registration_sha256=sha(DOCS/'routed-head-registration.json'))
    write(project/'routing.json',spec)
    config=read(project/'config.json');config['id']=identifier;write(project/'config.json',config)
    physical=(project/'physical_residual.py').read_text()
    old_line='logits = network.coarse_logits + network.shape_head(shape_features)'
    if physical.count(old_line)!=1:raise ValueError('Original logits selection seam changed')
    physical=physical.replace(old_line,'logits = network.logits_for_shape(shape_features, tuple(base.shape[2:]))')
    (project/'physical_residual.py').write_text(physical)
    names=set(old['sources'])|{'routed_base_model.py','routed_descriptor.py','routing.json'}
    witness=dict(schema='rheon-routed-neural-pressure-source-v1',architecture=describe(old['architecture'],spec),
        sources={name:sha(project/name)for name in sorted(names)},base_C107_witness_sha256=sha(prior/'source-witness.json'),
        config_sha256=sha(project/'config.json'),inference_only_assembly=True,no_training_recipe_provided=True)
    write(project/'source-witness.json',witness)
    np.savez(output/'weights.npz',**values);shutil.copyfile(orig_restore/'normalization.json',output/'normalization.json')
    inventory={name:dict(shape=list(value.shape),dtype=value.dtype.str,bytes=value.nbytes,sha256=hashlib.sha256(value.tobytes()).hexdigest())for name,value in values.items()}
    receipt=dict(status='completed',mode='local_weight_assembly',optimizer_updates=0,managed_training_run_claimed=False,
        parameters=94956,architecture=witness['architecture'],source_witness_sha256=sha(project/'source-witness.json'),
        parent_components=[large_origin,small_origin],parameter_origins=mapping,inventory=inventory,
        original_fine_and_large_head_tensors_bitwise_preserved=True,calibrated_small_head_tensors_bitwise_preserved=True,
        outputs={name:dict(sha256=sha(output/name),bytes=(output/name).stat().st_size)for name in ('weights.npz','normalization.json')},
        source_sha256=sha(__file__),assembled_unix=time.time(),candidate_numerical_pressure_solve=False,
        exact_fresh_CPU_reload_qualification_required=True)
    write(output/'assembly-receipt.json',receipt)
    files=[project/'source-witness.json',project/'config.json',project/'routing.json',output/'assembly-receipt.json',output/'weights.npz',output/'normalization.json']
    write(entry_path,dict(id=identifier,loader='routed-pressure-assembly-v1',project=str(project.relative_to(ROOT)),restore=str(output.relative_to(ROOT)),
        weights_sha256=sha(output/'weights.npz'),identities={str(path.relative_to(ROOT)):sha(path)for path in files},
        parent_components=[large_origin,small_origin],task='Local inference assembly of two managed learned heads; no new fitting'))
    print('Assembled pure-neural routed heads',identifier,'parameters94956; zero optimizer updates',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--identifier',required=True);run(parser.parse_args())
