"""Actual asynchronous force-packet and complete-output graph ownership proof."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from support import ROOT,read,write,digest
from complete_cuda_graph_v1 import CompleteCudaGraphGenerator
from cuda_native_transfer_v1 import upload_state,download_outputs
from expanded_cuda_generator_v1 import FIELDS
from diagnose_causal_composition import inventory

def hashes(rows):return [{name:hashlib.sha256(v.numpy().tobytes()).hexdigest() for name,v in row.items()} for row in rows]
def run(args):
    if args.output.exists():raise ValueError('Fresh complete graph qualification required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('GPU-only qualification required')
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen complete graph source changed: '+path)
    if not read(ROOT/reg['control_qualification'])['all_accuracy_pass']:raise ValueError('Dynamic controls must qualify first')
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=CompleteCudaGraphGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']);before=inventory(model)
    baseline=read(ROOT/reg['baseline_outputs']);expected={(r['identifier'],r['K'],r['start'],r['lead']):r['output_sha256'] for r in baseline['rows']}
    bank=read(ROOT/reg['bank']);windows={}
    for spec in reg['window_sources']:
        for row in read(ROOT/spec['controller'])['completed']:windows[row['identifier']]=row
    args.output.mkdir(parents=True);rows=[]
    try:
        with torch.inference_mode():
            for selected in bank['trajectories']:
                if digest(ROOT/selected['receipt'])!=selected['receipt_sha256']:raise ValueError('Native scene changed')
                scene=read(ROOT/selected['receipt'])['scene'];n=selected['reference_substeps'];current={}
                for w in windows[selected['identifier']]['windows']:
                    if w['start'] not in reg['starts']:continue
                    path=ROOT/w['current_state_path']
                    if digest(path)!=w['current_state_sha256']:raise ValueError('Actual current native fixture changed')
                    with np.load(path,allow_pickle=False) as a:host={name:torch.from_numpy(a[name].copy()) for name in FIELDS}
                    current[w['start']]=upload_state(host)
                model.prepare_geometry(tuple(scene['counts']))
                for k in reg['horizons']:
                    model.prepare_block(current[0],scene,0,k,n)
                    submitted=[]
                    for start in reg['starts']:
                        emitted=model.predict(current[start],scene,start,k,n)
                        submitted.append((start,emitted))
                    # No output download/synchronization between these changing
                    # input/time packets. Each call must retain its true inputs.
                    for start,emitted in submitted:
                        identity=hashes(download_outputs(emitted));wanted=[expected[(selected['identifier'],k,start,lead)] for lead in range(1,k+1)]
                        if identity!=wanted:raise ValueError('Queued graph reused stale time/state packet or overwritten output')
                        rows.append(dict(identifier=selected['identifier'],K=k,start=start,all_complete_output_hashes_match=True,queued_before_any_download=True))
                    # Reuse the same graph key for a different known scene. The
                    # eager unchanged learned core provides its counterfactual.
                    altered=copy.deepcopy(scene);altered['seed']+=1
                    if 'forcing_schedule' in altered:
                        altered['forcing_schedule'][0]['center']=[.2,.6,.8]
                        altered['forcing_schedule'][0]['pulse']=.75
                    external=model.controls.generate(altered,0,k,n)
                    eager=model.model.predict(current[0],altered,0,k,external,n)
                    actual=model.predict(current[0],altered,0,k,n)
                    if hashes(download_outputs(actual))!=hashes(download_outputs(eager)):
                        raise ValueError('Captured scene controls were baked into graph')
                    # Old GPU outputs must still retain independent storage.
                    for start,emitted in submitted:
                        if hashes(download_outputs(emitted))!=[expected[(selected['identifier'],k,start,lead)] for lead in range(1,k+1)]:
                            raise ValueError('Later graph call overwrote retained output fields')
                model.release_graphs();print('Complete dynamic graph ownership',selected['identifier'],'passed',flush=True)
            if inventory(model)!=before:raise ValueError('Qualification changed learned tensors')
        for path,want in reg['source_sha256'].items():
            if digest(ROOT/path)!=want:raise ValueError('Qualification source changed: '+path)
        write(args.output/'results.json',dict(status='passed_dynamic_complete_graph_packets_and_owned_outputs',rows=rows,
            complete_baseline_output_states=sum(r['K'] for r in rows),all_actual_output_hashes_match=True,
            cross_scene_controls_dynamic=True,retained_gpu_output_ownership_verified=True,optimizer_updates=0,cpu_model_forward=False,
            inference_device='cuda',gpu_model=torch.cuda.get_device_name(),protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],
            expanded_reserved52_references_opened=0,latency_claim=False,model_identity=model.asset_identity))
    finally:model.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
