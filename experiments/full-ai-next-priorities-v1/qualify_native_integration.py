"""Qualify real C107 dispatch and event invalidation through native research IPC."""
import argparse
import os
from pathlib import Path
import torch
from study_support import ROOT,HERE,BASE,read,write,sha,load_checkpoint
from native_client import NativeClient
from temporal_events import reference,episode,STEPS
from evaluate_temporal import load_model

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh native integration qualification required')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{23});torch.set_num_threads(1)
    bank=read(BASE/'temporal-development-240-01/bank.json');row=bank['trajectories'][0]
    receipt=read(ROOT/row['receipt']);scene=dict(receipt['scene'],steps=STEPS)
    state=load_checkpoint('C107');client=NativeClient(scene,args.output/'C107-native-dispatch')
    try:
        frame=client.advance(method='C107',predict=lambda sample:state.model.predict_pressure(state.network,sample,state.normalization))
        custody=client.close()
    finally:
        if client.child.poll()is None:client.close()
    truth,changed=reference(scene,args.output/'event-reference');rows=[]
    for k in (1,2):
        network,norm,identity=load_model(BASE/f'b{k}-pair-02')
        result=episode(network,norm,scene,changed,truth,args.output/f'b{k}-event')
        if result['physical_steps']!=48 or result['event_discarded_states']!=k-1 or not result['event_invalidated_and_replanned']:
            raise ValueError('Real event interval or stale-output accounting failed')
        write(args.output/f'b{k}-event'/'receipt.json',dict(result,model=identity))
        rows.append(dict(horizons=k,physical_steps=48,event_discarded_states=result['event_discarded_states'],
            native_final_state_bitwise_verified=True,native_processes=result['native_processes']))
    write(args.output/'qualification.json',dict(status='passed',C107_native_frame=frame,C107_process=custody,rows=rows,
        source_sha256=sha(__file__),event_source_sha256=sha(HERE/'temporal_events.py'),
        candidate_numerical_pressure_solve=False,optimizer_updates=0,application_promotion=False,
        performance_claim=False,scope='Native structural restart and event accounting; physical acceptance remains independently evaluated'))
    print('Qualified C107 dispatch, exact native state transport and event invalidation for K1/K2',flush=True)
