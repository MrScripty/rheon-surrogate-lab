"""DEBUG regression: replay actual own history0-46 and assert equation target."""
import argparse,hashlib,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT,read,write,digest
from complete_cuda_graph_v1 import CompleteCudaGraphGenerator
from cuda_native_transfer_v1 import upload_state,download_outputs
from diagnose_causal_composition import inventory,measure
FIELDS=('pressure','vx','vy','vz','tracer')
def run(args):
    if args.output.exists():raise ValueError('Fresh red/green feedback evidence required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('GPU-only learned replay required')
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen replay source/fixture changed: '+path)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=reg['case'];scene=dict(read(ROOT/case['receipt'])['scene'],reference_substeps=case['reference_substeps'],forcing_control_substeps=1)
    spec=next(s for s in case['states'] if s['step']==0);path=ROOT/spec['path']
    if digest(path)!=spec['sha256']:raise ValueError('Actual initial native state changed')
    with np.load(path,allow_pickle=False) as a:initial={name:torch.from_numpy(a[name].copy()) for name in FIELDS}
    receipt=read(ROOT/reg['fixture_receipt']);truth_path=ROOT/receipt['reference_path']
    if digest(truth_path)!=receipt['reference_sha256']:raise ValueError('Exact native failed endpoint fixture changed')
    with np.load(truth_path,allow_pickle=False) as a:truth={name:a[name].copy() for name in (*FIELDS,'rhs')}
    observed=read(ROOT/reg['observation']);expected=observed['first_failing_K1']['output_sha256']
    args.output.mkdir(parents=True);model=CompleteCudaGraphGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']);before=inventory(model);began=time.monotonic()
    try:
        current=upload_state(initial);model.prepare_geometry(tuple(scene['counts']));model.prepare_block(current,scene,0,1,case['reference_substeps'])
        with torch.inference_mode():
            for start in range(reg['endpoint_step']):current=model.predict(current,scene,start,1,case['reference_substeps'])[0]
            emitted=download_outputs([current])[0]
            actual={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in emitted.items()}
            if actual!=expected:raise ValueError('Minimized replay does not match original actual failed own-history endpoint bytes')
            metrics=measure(emitted,truth,scene,read(ROOT/reg['normalization']))
            passed=all(value<=reg['maximum_error'] for value in metrics['relative'].values())
        if inventory(model)!=before:raise ValueError('Feedback loop changed weights')
        write(args.output/'results.json',dict(status='passed_target' if passed else 'reproduced_actual_equation_failure',metrics=metrics,
            endpoint_step=reg['endpoint_step'],actual_endpoint_hashes=actual,original_failure_endpoint_bitwise_equal=True,
            elapsed_diagnostic_seconds=time.monotonic()-began,latency_claim=False,cpu_model_forward=False,optimizer_updates=0,
            protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))
        print('RED-CAPABLE own-history endpoint',reg['endpoint_step'],'equation',metrics['relative']['equation'],'target',reg['maximum_error'],flush=True)
        if not passed:raise AssertionError('Actual above100axis own-history equation target exceeded')
    finally:model.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
