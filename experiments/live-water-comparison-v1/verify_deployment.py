"""Real checkpoint/CPU pressure parity through the desktop subprocess protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def verify(deployment,packet,inference,output):
    descriptor=json.loads(deployment.read_text());lab=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(lab/'experiments/sealed-water-v1'));import data
    receipt=json.loads((inference/'receipt.json').read_text());case=receipt['cases'][0]
    reservation,parents=data.load((packet/'dataset.npz').read_bytes(),json.loads((packet/'splits.json').read_text()))
    parent=parents[case['parent']];slot=case['frame'];base=parent['arrays']['base'][slot]
    request={'id':case['native_frame']+1,'counts_xyz':parent['counts_xyz'],'cell_size_m':parent['cell_size_m'],'density_kg_m3':parent['density_kg_m3'],'dt_s':parent['entry']['dt_s'],
        'base_pressure':base.reshape(-1).tolist(),'native_input_residual':parent['arrays']['native_input_residual'][slot].reshape(-1).tolist()}
    zero={**request,'id':case['native_frame']+2,'native_input_residual':[0.0]*base.size}
    environment={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','TULDOK_TORCH_DEVICE':'cpu','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    result=subprocess.run([descriptor['python'],'-I','-S','-B',str(deployment.parent/descriptor['bridge']),'--deployment',str(deployment)],
        input=b''.join(json.dumps(r,allow_nan=False).encode()+b'\n' for r in (request,zero,{**request,'id':1})),env=environment,capture_output=True,timeout=120)
    replies=[json.loads(line) for line in result.stdout.splitlines()]
    errors=[r for r in replies if r['kind']=='error']
    if result.returncode!=2 or len(errors)!=1 or errors[0]['code']!='unsupported_context':raise ValueError(str(replies)+' '+result.stderr.decode()[:2000])
    ready=next(r for r in replies if r['kind']=='ready');predictions=[r for r in replies if r['kind']=='pressure']
    if ready['device']!='cpu' or ready['cuda_initialized'] or ready['optimizer_updates']!=0:raise ValueError('Invalid CPU startup')
    expected=data.helpers().read_npz((inference/'predictions.npz').read_bytes(),receipt['prediction_inventory'],2*1024**2)['case_000'].reshape(-1)
    actual=np.asarray(predictions[0]['pressure'],dtype='<f8');bypass=np.asarray(predictions[1]['pressure'],dtype='<f8')
    if actual.tobytes()!=expected.tobytes() or bypass.tobytes()!=base.tobytes():raise ValueError('CPU prediction/bypass parity failed')
    report={'status':'passed','checkpoint_sha256':ready['checkpoint_sha256'],'device':'cpu','cuda_initialized':False,'optimizer_updates':0,
        'exact_prediction_bytes':True,'zero_residual_exact':True,'unsupported_cold_start_rejected_before_inference':True,'counts_xyz':parent['counts_xyz'],'native_frame':case['native_frame'],
        'pressure_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'stderr_bytes':len(result.stderr)}
    output.write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('deployment','packet','inference','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();print(json.dumps(verify(a.deployment.absolute(),a.packet.absolute(),a.inference.absolute(),a.output.absolute())))
