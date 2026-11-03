"""DEBUG: reconstruct one actual failed-reference endpoint, then checksum it."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT,read,write,digest
from continuous_reference_client import ContinuousReferenceClient
FIELDS=('pressure','vx','vy','vz','tracer')
def run(args):
    if args.output.exists():raise ValueError('Fresh captured diagnostic fixture required')
    reg=read(args.protocol)
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen diagnostic fixture dependency changed: '+path)
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    obs=read(ROOT/reg['observation']);case=reg['case'];receipt=read(ROOT/case['receipt']);scene=dict(receipt['scene'],reference_substeps=case['reference_substeps'],forcing_control_substeps=1)
    spec=next(s for s in case['states'] if s['step']==reg['restore_step']);path=ROOT/spec['path']
    if digest(path)!=spec['sha256']:raise ValueError('Actual old diagnostic restart state changed')
    with np.load(path,allow_pickle=False) as a:current={name:a[name].copy() for name in FIELDS}
    args.output.mkdir(parents=True);active=ContinuousReferenceClient(scene,args.output/'native',io_timeout_s=120.);result=dict(status='running',candidate_inference_executed=False,reference_restore_for_diagnostic_only=True)
    try:
        active.restore(reg['restore_step'],current)
        for step in range(reg['restore_step']+1,reg['endpoint_step']+1):
            frame=active.advance()
            if not frame['frame']['advanced']:raise ValueError('Diagnostic native reference rejected advance')
        packet,truth=active.inspect();truth.update(rhs=active.last_request['rhs'],native_input_residual=active.last_request['native_input_residual'],pressure_base=active.last_request['base'])
        actual={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in truth.items()}
        expected=obs['reference46']['actual_field_sha256']
        if packet['step']!=reg['endpoint_step'] or actual!=expected:raise ValueError('Diagnostic restarted teacher does not byte-match original uninterrupted failed endpoint')
        artifact=args.output/'reference46.npz';np.savez_compressed(artifact,**truth)
        if artifact.stat().st_size>reg['maximum_fixture_bytes']:raise ValueError('Finite debug fixture budget reached')
        custody=active.close();active=None
        if not custody['reaped'] or custody['returncode']!=0 or not custody['graceful_protocol_complete']:raise ValueError('Diagnostic native custody failed')
        result.update(status='captured_exact_uninterrupted_reference_endpoint_via_verified_restart',reference_path=str(artifact.relative_to(ROOT)),reference_sha256=digest(artifact),field_sha256=actual,
            original_uninterrupted_hashes_bitwise_equal=True,native_process=custody,protocol_sha256=digest(args.protocol),cpu_model_forward=False,optimizer_updates=0)
    finally:
        if active is not None:result['native_process']=active.close()
        write(args.output/'receipt.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
