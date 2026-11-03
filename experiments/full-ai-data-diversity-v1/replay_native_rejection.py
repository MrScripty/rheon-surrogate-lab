"""Reproduce the actual native physical rejection from its latest good saved state."""
from support import ROOT, BASE, read, write, digest
from native_forcing_client import NativeForcingClient
from native_client import FIELDS
import numpy as np
import torch
import os


if __name__=='__main__':
    os.sched_setaffinity(0,{23});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=ROOT/'docs/plans/full-ai-data-diversity-v1/train-capture-protocol-01.json'
    reg=read(protocol)
    row=next(r for r in reg['rows']if r['identifier']=='diversity-train-g001-cpm50')
    seed=BASE/'train-capture-01'/row['identifier']/'state-097.npz'
    out=BASE/'native-rejection-replay-01'
    if out.exists():raise ValueError('Fresh actual rejection replay required')
    out.mkdir()
    with np.load(seed,allow_pickle=False)as archive:state={k:archive[k].copy()for k in FIELDS}
    client=NativeForcingClient(row['scene'],out)
    result=dict(status='unexpectedly_reached_endpoint',source_state_sha256=digest(seed),
        source_protocol_sha256=digest(protocol),initial_actual_step=97,
        numerical_reference_only=True,optimizer_updates=0)
    try:
        client.restore(97,state)
        for step in range(98,241):
            frame=client.advance()
            if not frame['frame']['advanced'] or frame['frame']['step']!=step:
                result.update(status='reproduced_native_reference_rejection',expected_step=step,frame=frame)
                print('RED native rejection at expected step',step,frame['frame'],flush=True)
                break
    finally:
        result['native_process']=client.close()
        write(out/'result.json',result)
    raise SystemExit(1 if result['status']=='reproduced_native_reference_rejection'else 0)
