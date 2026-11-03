"""Unexpected forcing changes invalidate unconsumed states; measure discarded work."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha
from native_client import NativeClient,FIELDS
from evaluate_temporal import load_model
from temporal_metrics import metrics,physical_arrays

EVENT_STEP=13
STEPS=48


def reference(scene,directory):
    directory.mkdir();client=NativeClient(scene,directory/'before-event');states=[];changed=dict(scene,acceleration=1.5*scene['acceleration'])
    custody=[]
    try:
        _,initial=client.inspect();states.append(initial)
        for step in range(1,STEPS+1):
            frame=client.advance()
            if not frame['frame']['advanced']:raise ValueError('Event reference rejected')
            packet,values=client.inspect();values['rhs']=client.last_request['rhs'];states.append(values)
            np.savez_compressed(directory/f'state-{step:03d}.npz',**values)
            if step==EVENT_STEP:
                custody.append(client.close());client=NativeClient(changed,directory/'after-event')
                client.restore(step,{name:values[name]for name in FIELDS})
                packet,restarted=client.inspect()
                if packet['step']!=step or any(not np.array_equal(restarted[name].view('u1'),values[name].view('u1'))for name in FIELDS):
                    raise ValueError('Forcing event restart changed physical state')
        custody.append(client.close())
    finally:
        if client.child.poll()is None:client.close()
    write(directory/'receipt.json',dict(status='completed',original_scene=scene,changed_scene=changed,event_step=EVENT_STEP,
        states={p.name:sha(p)for p in directory.glob('state-*.npz')},native_processes=custody,source_sha256=sha(__file__)))
    return states,changed


def episode(network,norm,scene,changed,truth,directory):
    initial={name:truth[0][name]for name in FIELDS};current=initial;step=0;revision=0
    generated=0;calls=0;event_discard=0;endpoint_discard=0;rows=[];block_seconds=[];custody=[]
    directory.mkdir();tick=time.perf_counter();client=NativeClient(scene,directory/'before-event');setup=time.perf_counter()-tick
    initial_mass=float(initial['tracer'].astype('f8').sum()/scene['cells_per_metre']**3)
    integration_seconds=0.;checks_seconds=0.;event_seconds=0.;first_invalid=None
    started=time.perf_counter()
    try:
        with torch.inference_mode():
            while step<STEPS:
                active_scene=scene if revision==0 else changed
                tick=time.perf_counter();outputs=network.predict(current,active_scene,step,norm);block_seconds.append(time.perf_counter()-tick)
                generated+=len(outputs);calls+=1
                for index,predicted in enumerate(outputs):
                    next_step=step+1
                    if revision!=int(next_step>EVENT_STEP):raise ValueError('Stale pre-event prediction was consumed')
                    tick=time.perf_counter()
                    if not all(bool(torch.isfinite(v).all())for v in predicted.values()):first_invalid=next_step;break
                    values=physical_arrays(predicted);checks_seconds+=time.perf_counter()-tick
                    tick=time.perf_counter();client.restore(next_step,values);integration_seconds+=time.perf_counter()-tick
                    step=next_step;current=predicted
                    rows.append(dict(step=step,forcing_revision=revision,metrics=metrics(predicted,truth[step],truth[step]['rhs'],active_scene,norm,initial_mass)))
                    if step==EVENT_STEP:
                        event_discard+=len(outputs)-index-1
                        tick=time.perf_counter();custody.append(client.close());client=NativeClient(changed,directory/'after-event')
                        client.restore(step,values);revision=1;event_seconds+=time.perf_counter()-tick
                        break
                    if step==STEPS:
                        endpoint_discard+=len(outputs)-index-1;break
                if first_invalid is not None:break
        elapsed=time.perf_counter()-started
        packet,final=client.inspect()
        expected=physical_arrays(current)
        if packet['step']!=step or any(not np.array_equal(final[name].view('u1'),expected[name].view('u1'))for name in FIELDS):
            raise ValueError('Native research state consumer differs from generated state')
        custody.append(client.close())
    finally:
        if client.child.poll()is None:client.close()
    if first_invalid is None and (generated!=step+event_discard+endpoint_discard or event_discard!=network.horizons-1):
        raise ValueError('Discard accounting failed for the unannounced step13 event')
    return dict(status='completed',horizons=network.horizons,physical_steps=step,requested_steps=STEPS,network_calls=calls,
        generated_states=generated,event_discarded_states=event_discard,endpoint_discarded_states=endpoint_discard,
        first_nonfinite_step=first_invalid,event_step=EVENT_STEP,event_acceleration_multiplier=1.5,
        full_block_seconds=block_seconds,prediction_seconds=sum(block_seconds),native_state_upload_seconds=integration_seconds,
        finite_check_seconds=checks_seconds,event_reinitialization_seconds=event_seconds,native_setup_seconds=setup,
        total_research_episode_seconds=elapsed,offline_metric_cost_in_total=True,native_processes=custody,rows=rows,
        native_state_setter_is_structural_only=True,application_physical_acceptance_claimed=False,fallback=False,
        event_invalidated_and_replanned=True,all_intermediate_states_exposed=True)


def run(args):
    if args.output.exists():raise ValueError('Fresh event episodes required')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{23});torch.set_num_threads(1)
    models={k:load_model(BASE/f'b{k}-pair-02')for k in (1,2,3,4)}
    bank=read(args.bank);rows=[]
    for row in bank['trajectories']:
        if row['split']!='development':raise ValueError('Consumed development event scenes only')
        receipt_path=ROOT/row['receipt']
        if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Reference scene changed')
        receipt=read(receipt_path);scene=dict(receipt['scene'],steps=STEPS)
        truth,changed=reference(scene,args.output/(row['identifier']+'-reference'))
        for k,(network,norm,identity)in models.items():
            result=episode(network,norm,scene,changed,truth,args.output/(row['identifier']+f'-b{k}'))
            write(args.output/(row['identifier']+f'-b{k}')/'receipt.json',dict(result,model=identity,source_sha256=sha(__file__)))
            rows.append(dict(identifier=row['identifier'],counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],
                **{key:value for key,value in result.items()if key not in ('rows','native_processes','full_block_seconds')}))
            print('Event episode K',k,row['identifier'],'discarded',result['event_discarded_states'],'stale states',flush=True)
    write(args.output/'summary.json',dict(status='completed',rows=rows,source_sha256=sha(__file__),bank_sha256=sha(args.bank),
        model_identities={k:v[2]for k,v in models.items()},cpu=23,threads=1,candidate_numerical_pressure_solve=False,
        caveat='Native research restart allocates workspaces and checks structural validity only. Episode total also includes independent offline metrics; timing components separate. No application promotion.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
