"""Existing strict typed-input and CPU-restore custody helpers, without model imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch


def digest(raw):return hashlib.sha256(raw).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def inventory(network):
    return {name:dict(shape=list(value.shape),dtype='<f4',sha256=digest(value.detach().cpu().numpy().astype('<f4',copy=False).tobytes()))
            for name,value in network.state_dict().items()}

def save_weights(network,path):
    np.savez(path,**{name:value.detach().cpu().numpy().astype('<f4',copy=False)for name,value in network.state_dict().items()})

def load_weights(network,path,expected):
    with np.load(path,allow_pickle=False)as source:
        if set(source.files)!=set(network.state_dict()):raise ValueError('Complete parameter inventory required')
        values={}
        for name,current in network.state_dict().items():
            value=source[name]
            if value.dtype.str!='<f4'or tuple(value.shape)!=tuple(current.shape):raise ValueError('Parameter shape/dtype changed')
            if digest(value.tobytes())!=expected[name]['sha256']:raise ValueError('Parameter bytes changed')
            values[name]=torch.from_numpy(value.copy())
    network.load_state_dict(values,strict=True)
    if inventory(network)!=expected:raise ValueError('Loaded parameter inventory differs')

def validate_inputs(args):
    evidence=read(args.input_evidence)
    static={'manifest':args.manifest,'normalization':args.normalization,'runtime_manifest':args.runtime_manifest,
            **{f'part_{index:02d}':path for index,path in enumerate(args.shards)}}
    owners=set();identities={}
    expected=set(static)|({'trained_weights','trained_initial','trained_history','trained_probe','trained_receipt'}if args.mode=='restore'else set())
    if set(evidence)!=expected:raise ValueError('Unexpected or missing typed fitting/restore inputs')
    for slot,path in static.items():
        row=evidence[slot];raw=Path(path).read_bytes();measured=digest(raw)
        if row['reference_id']!=slot or row['measured_sha256']!=measured:raise ValueError('Typed static input identity failed')
        owners.add((row['project_id'],row['manifest_sha256']));identities[slot]=dict(bytes=len(raw),sha256=measured)
    if len(owners)!=1:raise ValueError('Mixed Project inputs rejected')
    return evidence,identities

def restore_custody(evidence,args):
    path=Path(__file__).with_name('custody_validator.py')
    spec=importlib.util.spec_from_file_location('temporal_custody_validator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    owner=None
    expected={'trained_weights':('weights.npz','checkpoint'),'trained_initial':('initial-weights.npz','checkpoint'),
              'trained_history':('history.json','training_diagnostics'),'trained_probe':('probe.npy','predictions'),
              'trained_receipt':('receipt.json','training_result')}
    for slot in ('trained_weights','trained_initial','trained_history','trained_probe','trained_receipt'):
        raw=Path(getattr(args,slot)).read_bytes();row=evidence[slot]
        if row['kind']!='run_artifact':raise ValueError('CPU restore requires managed Run artifacts')
        observed=module.validate_artifact_source(row['source'],digest(raw),len(raw))
        if (observed['artifact_name'],observed['artifact_role'])!=expected[slot]:raise ValueError('Restore artifact role differs')
        key=tuple(observed[k]for k in ('catalog_id','run_id','attempt_id','request_sha256','material_sha256'))
        if owner is not None and owner!=key:raise ValueError('Mixed training Run artifacts')
        owner=key
    return dict(zip(('catalog_id','run_id','attempt_id','request_sha256','material_sha256'),owner))
