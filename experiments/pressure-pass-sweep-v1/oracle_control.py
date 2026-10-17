"""No-learning trajectory control using the retained analytic Neumann solver.

The existing rollout owner handles IPC, deadlines and native child reaping.
The network monitor has no parameters and is never called; no weights are read.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
from types import SimpleNamespace
from material import read,sha


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result


def load(path):
    spec=read(path)
    if spec['schema']!='rheon-analytic-pressure-control-v1':raise ValueError('Explicit analytic control required')
    material=read(spec['material']);root=Path(material['root']);lab=root/material['config']['source_lab']
    import numpy as np
    import torch
    torch.set_num_threads(1)
    validator=module('analytic_request_validator',lab/'experiments/dependency-migration-v1/model_adapter.py')
    sys.path.insert(0,str(lab/'experiments/water-residual-learning-v1'))
    import audit
    audit.TORCH=torch
    oracle=module('analytic_independent_action',lab/'experiments/global-pressure-v1/verify_capture.py')
    return SimpleNamespace(np=np,torch=torch,model=None,network=torch.nn.Identity().eval(),normalization={},
        policy={'max_network_calls':0,'stopping':None},manifest=material['support'],
        validator=validator,audit=audit,oracle=oracle)


def predict(request,state,trace):
    sample=state.validator.sample_from_request(request,state);torch=state.torch
    with torch.no_grad():
        residual=sample['native_input_residual']
        delta=state.audit.spectral_correction(residual,sample)
        action=state.oracle.full_action(delta.flatten().tolist(),list(delta.shape[2:][::-1]),
            [float(sample['cell_size_m'][0])]*3,float(sample['rho'][0]))
        action=torch.tensor(action,dtype=torch.float64).reshape_as(residual)
        target=residual-residual.mean()
        defect=float(torch.linalg.vector_norm(action-target))/max(float(torch.linalg.vector_norm(target)),1e-30)
        if defect>=1e-9:raise ValueError('Analytic control failed independent operator check')
        pressure=sample['base']+delta
    if not bool(torch.isfinite(pressure).all())or float(pressure.flatten()[0])!=0:
        raise ValueError('Invalid analytic full pressure')
    trace.append(dict(proposals=0,kind='analytic-f64-spectral-control-no-network',independent_relative_defect=defect))
    return dict(kind='candidate',id=request['id'],role='full_pressure',pressure=pressure.flatten().tolist(),device='cpu',optimizer_updates=0)


def failure(error,request_id=None):
    return dict(kind='error',id=request_id,code='invalid',message=str(error)[:2000])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,required=True);parser.add_argument('--fixture',action='store_true')
    parser.add_argument('--native-control',action='store_true')
    args=parser.parse_args();material=read(args.material);root=Path(material['root']);lab=root/material['config']['source_lab']
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,{args.cpu})
    def stop(*_):raise InterruptedError('Analytic control cancelled or timed out')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(180)
    sys.path.insert(0,str(lab/'experiments/water-residual-learning-v1'));import rollout
    scene={**material['config']['scene']}
    if args.fixture:scene['steps']=scene['warmup_steps']+1
    manifest=None
    if not args.native_control:
        shutil.copyfile(__file__,output/'model_adapter.py');manifest=output/'manifest.json'
        manifest.write_text(json.dumps(dict(schema='rheon-analytic-pressure-control-v1',material=str(args.material.resolve()))))
    start=time.perf_counter()
    try:
        receipt=rollout.run(root/material['config']['binary'],scene,output/'native',adapter_manifest=manifest,
            control=args.native_control,timeout=170,threads=1)
    finally:signal.alarm(0)
    frames=[json.loads(line)for line in(output/'native/frames.jsonl').read_text().splitlines()]
    ai=[f for f in frames if f.get('ai_quality')]
    callbacks_path=output/'native/callbacks.jsonl'
    callbacks=[json.loads(line)for line in callbacks_path.read_text().splitlines()]if callbacks_path.exists()else[]
    if (receipt['result']['hard_stop']or len(ai)!=scene['steps']-scene['warmup_steps']
        or any(c['network_passes']!=0 for c in callbacks)or not receipt['native_process']['reaped']):
        raise ValueError('No-learning control did not qualify and drain')
    receipt.update(kind='native-cg-control-no-network'if args.native_control else'analytic-f64-spectral-control-no-network',
        neural_network_calls=0,weights_loaded=False,source_sha256=sha(__file__),cpu=args.cpu,
        total_elapsed_s=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        material_sha256=sha(args.material),frames_source_sha256=sha(output/'native/frames.jsonl'))
    (output/'control-receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',kind=receipt['kind'],steps=len(ai),
        worst_pressure_drift=max(f['relative_pressure_difference']for f in ai),elapsed_s=receipt['total_elapsed_s'])),flush=True)


if __name__=='__main__':main()
