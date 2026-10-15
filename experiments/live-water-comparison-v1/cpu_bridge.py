"""Bounded CPU-only inference protocol for Rheon's experimental water comparison."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

LIMIT=8*1024**2

class ProtocolError(ValueError):
    def __init__(self,code,message):self.code=code;super().__init__(message)

def failure(error,request_id=None):
    return {'kind':'error','code':getattr(error,'code','unavailable' if isinstance(error,(OSError,ImportError)) else 'invalid'),'id':request_id,'message':str(error)[:2000]}


def strict(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('Duplicate JSON key')
            result[k]=v
        return result
    def constant(value):raise ValueError('Nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)

def bounded(path,cap):
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=cap:raise ValueError('Invalid deployment file: '+str(path))
    return path.read_bytes()

def digest_file(path):
    total=0;h=hashlib.sha256()
    with path.open('rb') as stream:
        while raw:=stream.read(1024**2):total+=len(raw);h.update(raw)
    return h.hexdigest(),total

def send(value):
    raw=json.dumps(value,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>LIMIT:raise ValueError('Reply exceeds protocol bound')
    sys.stdout.buffer.write(raw+b'\n');sys.stdout.buffer.flush()

def load(path):
    root=path.parent.resolve();raw=bounded(path,1024**2);descriptor=strict(raw)
    if descriptor['schema']!='rheon-pressure-cpu-deployment-v2':raise ValueError('Unsupported deployment')
    policy=descriptor['context_policy']
    if set(policy)!={'mode','minimum_native_steps','training_snapshot_indices'} or policy['mode']!='native_prefix' or type(policy['minimum_native_steps']) is not int or not 0<=policy['minimum_native_steps']<=512 or not policy['training_snapshot_indices'] or min(policy['training_snapshot_indices'])!=policy['minimum_native_steps']:
        raise ValueError('Qualified checkpoint context policy required')
    for relative,expected in descriptor['files'].items():
        p=root/relative
        if p.resolve()!=p or not p.is_relative_to(root):raise ValueError('Unsafe deployment path')
        digest,size=digest_file(p)
        if digest!=expected['sha256'] or size!=expected['bytes']:raise ValueError('Deployment drift: '+relative)
    if digest_file(root/'weights.npz')[0]!=descriptor['checkpoint_sha256'] or digest_file(root/'normalization.json')[0]!=descriptor['normalization_sha256']:
        raise ValueError('Reported checkpoint/normalization identity mismatch')
    runtime_raw=bounded(root/'installed-runtime.json',16*1024**2);runtime=strict(runtime_raw)
    summary=strict(bounded(root/'sources/runtime-manifest.json',1024**2))
    if hashlib.sha256(runtime_raw).hexdigest()!=summary['manifest_sha256']:raise ValueError('Runtime selection mismatch')
    if Path(sys.executable).resolve()!=Path(runtime['python']).resolve() or Path(sys.executable).resolve()!=Path(descriptor['python']).resolve() or digest_file(Path(sys.executable))[0]!=descriptor['python_sha256']:raise ValueError('Interpreter mismatch')
    if runtime['profile']!='managed_torch_python_v1' or len(runtime['files'])>32768 or runtime['total_bytes']>6*1024**3:raise ValueError('Runtime inventory bounds')
    send({'kind':'loading','phase':'Checking installed CPU runtime assets'})
    total=0
    for asset in runtime['files']:
        digest,size=digest_file(Path(asset['path']));total+=size
        if size!=asset['bytes'] or digest!=asset['sha256']:raise ValueError('Installed runtime drift')
    if total!=runtime['total_bytes']:raise ValueError('Runtime aggregate mismatch')
    sys.path[:0]=[str(root/'sources'),runtime['site_packages']]
    import numpy as np
    import torch
    import model,data
    if torch.__version__!=runtime['packages']['torch'] or np.__version__!=runtime['packages']['numpy']:raise ValueError('Runtime package mismatch')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('TULDOK_TORCH_DEVICE')!='cpu':raise ValueError('Explicit CPU environment required')
    model.configure('cpu');network=model.PressureCorrectionNet()
    history=strict(bounded(root/'history.json',1024**2));final=f'checkpoint-{descriptor["training_updates"]:03d}.npz'
    data.helpers().load_weights(network,root/'weights.npz',history['checkpoints'][final]['inventory']);network.eval()
    normalization=strict(bounded(root/'normalization.json',1024**2))
    if torch.cuda.is_initialized() or any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in network.parameters()):raise ValueError('CPU float32 deployment required')
    send({'kind':'ready','checkpoint_sha256':descriptor['checkpoint_sha256'],'training_run_id':descriptor['training_run_id'],
        'device':'cpu','model_dtype':'float32','channels':7,'optimizer_updates':0,'cuda_initialized':False,'minimum_native_steps':policy['minimum_native_steps']})
    return np,torch,model,network,normalization,policy

def predict(value,state):
    np,torch,model,network,normalization,policy=state
    if set(value)!={'id','counts_xyz','cell_size_m','density_kg_m3','dt_s','base_pressure','native_input_residual'}:raise ValueError('Closed prediction request required')
    if type(value['id']) is not int or not 0<=value['id']<2**53:raise ValueError('Bounded integer request id required')
    if value['id']<=policy['minimum_native_steps']:
        raise ProtocolError('unsupported_context',f"Testing setup lacks native warm-up: need {policy['minimum_native_steps']} completed native steps before AI inference")
    counts=value['counts_xyz']
    if type(counts) is not list or len(counts)!=3 or any(type(n) is not int or not 5<=n<=128 for n in counts) or math.prod(counts)>65536:raise ValueError('Grid budget/shape')
    h=value['cell_size_m'];rho=value['density_kg_m3'];dt=value['dt_s']
    if type(h) not in (int,float) or not math.isfinite(h) or not 1/128<=h<=1 or rho!=1000.0 or dt!=1/60:raise ValueError('Cubic water metadata required')
    sample={}
    for field,source in [('base','base_pressure'),('native_input_residual','native_input_residual')]:
        array=np.asarray(value[source],dtype='<f8')
        if array.shape!=(math.prod(counts),) or not np.isfinite(array).all():raise ValueError('Finite full-grid fields required')
        sample[field]=torch.from_numpy(array.copy()).reshape(1,1,*counts[::-1])
    if float(sample['base'][0,0,0,0,0])!=0:raise ValueError('Gauge must be zero')
    sample.update(cell_size_m=torch.tensor([h],dtype=torch.float64),rho=torch.tensor([rho],dtype=torch.float64),dt=torch.tensor([dt],dtype=torch.float64))
    with torch.no_grad():pressure=model.predict_pressure(network,sample,normalization).reshape(-1).numpy()
    if not np.isfinite(pressure).all() or pressure[0]!=0 or torch.cuda.is_initialized():raise ValueError('Invalid CPU pressure output')
    return {'kind':'pressure','id':value['id'],'pressure':pressure.tolist(),'device':'cpu','optimizer_updates':0}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deployment',type=Path,required=True);args=parser.parse_args();state=load(args.deployment.absolute())
    while raw:=sys.stdin.buffer.readline(LIMIT+1):
        if len(raw)>LIMIT or not raw.endswith(b'\n'):raise ValueError('Incomplete or oversized protocol record')
        value=strict(raw);request_id=value.get('id') if type(value) is dict and type(value.get('id')) is int and 0<=value['id']<2**53 else None
        try:send(predict(value,state))
        except BaseException as error:send(failure(error,request_id));return 2
    return 0

if __name__=='__main__':
    try:code=main()
    except BaseException as error:send(failure(error));code=2
    sys.exit(code)
