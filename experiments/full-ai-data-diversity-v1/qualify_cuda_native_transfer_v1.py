"""Actual native-byte and ownership qualification of the CUDA transfer adapter."""
import argparse
import hashlib
import os
from pathlib import Path
import numpy as np
import torch
from support import ROOT,read,write,digest
from cuda_native_transfer_v1 import FIELDS,upload_state,download_outputs

def hashes(row):return {name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in row.items()}
def run(args):
    if args.output.exists():raise ValueError('Fresh actual transfer qualification required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('CUDA-only adapter qualification required')
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen transfer source changed: '+path)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True);rows=[]
    with torch.inference_mode():
        for case in reg['cases']:
            path=ROOT/case['path']
            if digest(path)!=case['sha256']:raise ValueError('Native fixture changed')
            with np.load(path,allow_pickle=False) as a:golden={name:torch.from_numpy(a[name].copy()) for name in FIELDS}
            identity=hashes(golden);retained=[]
            for k in reg['horizons']:
                for repeat in range(reg['repetitions']):
                    payload={name:v.clone() for name,v in golden.items()};resident=upload_state(payload)
                    # Mutating pageable caller inputs must not change queued H2D.
                    for value in payload.values():value.zero_()
                    copied=download_outputs([resident]*k)
                    if any(hashes(row)!=identity for row in copied):raise ValueError('Actual native transfer bytes changed')
                    for row in copied:
                        if any(v.data_ptr()%v.element_size()!=0 for v in row.values()):raise ValueError('Misaligned native field view')
                    retained.append((copied,identity))
                    for value in resident.values():value.zero_()
                    zeros=download_outputs([resident]*k)
                    if any(bool(v.any()) for row in zeros for v in row.values()):raise ValueError('Later GPU mutation not observed')
                    if any(hashes(row)!=expected for previous,expected in retained for row in previous):raise ValueError('Later call overwrote retained physical output')
                    # Distinct intermediate physical states must not overlap.
                    if k>1:
                        copied[0]['vx'].zero_()
                        if hashes(copied[1])!=identity:raise ValueError('Intermediate output aliases a different state')
                        copied[0]['vx'].copy_(golden['vx'])
                    rows.append(dict(identifier=case['identifier'],K=k,repeat=repeat,native_field_bytes_equal=True,
                        caller_input_mutation_safe=True,retained_outputs_unchanged=True,native_alignment_verified=True,intermediate_ownership_verified=True))
            print('CUDA native transfer ownership',case['identifier'],'allK passed',flush=True)
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Transfer dependency changed: '+path)
    write(args.output/'results.json',dict(status='passed_actual_native_transfer_bytes_and_ownership',rows=rows,cases=len(reg['cases']),
        optimizer_updates=0,cpu_model_forward=False,inference_device='cuda',gpu_model=torch.cuda.get_device_name(),
        protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],expanded_reserved52_references_opened=0,latency_claim=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
