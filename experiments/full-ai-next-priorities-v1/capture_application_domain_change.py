"""Capture a supported physical-domain rescale at unchanged logical cell counts."""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import numpy as np
import torch
from bounded_native_client import BoundedNativeClient
from native_client import FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


class GeometryCheckedReference(BoundedNativeClient):
    def __init__(self,*args,geometry_checks,**kwargs):
        self.geometry_checks=geometry_checks
        super().__init__(*args,**kwargs)

    def read_json(self):
        packet=super().read_json()
        if packet['kind']=='predict':
            h=1/self.scene['cells_per_metre']
            if packet['cell_size_m']!=h or packet['rho']!=1000. or packet['dt']!=1/60:
                raise ValueError('Actual native request geometry/material/timestep differs from the declared external rescale')
            self.geometry_checks.append(dict(step=len(self.geometry_checks)+1,cell_size_m=packet['cell_size_m']))
        return packet


def run(args):
    if args.output.exists():raise ValueError('Fresh application reference capture required')
    protocol_path=DOCS/'application-domain-change-reference-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered reference source changed')
    for relative,digest in protocol['execution_assets_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Qualified native source/binary changed')
    bank_path=BASE/'temporal-development-240-01/bank.json'
    if sha(bank_path)!=protocol['bank_sha256']:raise ValueError('Fixed consumed development population required')
    args.output=args.output.resolve();args.output.mkdir(parents=True)
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    def stop(*_):raise InterruptedError('Event capture interrupted; reaping native reference child')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    rows=[];total_bytes=0
    for row in read(bank_path)['trajectories']:
        if row['split']!='development'or row['identifier']not in protocol['scene_identifiers']:
            raise ValueError('Only registered consumed development scenes may be captured')
        parent_path=ROOT/row['receipt']
        if sha(parent_path)!=row['receipt_sha256']:raise ValueError('Original native reference receipt changed')
        parent=read(parent_path);scene=dict(parent['scene'],steps=48)
        changed=dict(scene,cells_per_metre=24-scene['cells_per_metre']);directory=args.output/row['identifier'];directory.mkdir()
        geometry_checks=[]
        client=GeometryCheckedReference(scene,directory/'before-event',geometry_checks=geometry_checks);states=[];frames=[];custody=[];event_effect=False
        try:
            for step in range(49):
                if step:
                    frame=client.advance(method='spectral');frames.append(frame)
                    if not frame['frame']['advanced']:raise ValueError('Unchanged native reference gates rejected an event step')
                packet,values=client.inspect()
                if packet['step']!=step or packet['time_s']!=step/60:raise ValueError('Native event clock differs')
                rhs=np.zeros_like(values['pressure'])if step==0 else client.last_request['rhs'].copy()
                arrays=dict(values,rhs=rhs)
                if step<=14:
                    spec=parent['states'][step];source=parent_path.parent/spec['path']
                    if sha(source)!=spec['sha256']:raise ValueError('Original steady reference changed')
                    with np.load(source,allow_pickle=False)as old:
                        if step<=13:
                            if any(not np.array_equal(arrays[n].view('u1'),old[n].view('u1'))for n in (*FIELDS,'rhs')):
                                raise ValueError('Unannounced event altered its pre-event reference prefix')
                        else:event_effect=any(not np.array_equal(arrays[n].view('u1'),old[n].view('u1'))for n in ('pressure','vx','vy','vz','rhs'))
                path=directory/f'state-{step:03d}.npz';np.savez_compressed(path,**arrays);total_bytes+=path.stat().st_size
                states.append(dict(step=step,time_s=step/60,path=path.name,bytes=path.stat().st_size,sha256=sha(path),
                    initial_state=step==0,initial_rhs_placeholder=step==0,geometry_revision=int(step>13),
                    arrays={n:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest())for n,v in arrays.items()}))
                if step==13:
                    custody.append(client.close());client=GeometryCheckedReference(changed,directory/'after-event',geometry_checks=geometry_checks)
                    ack=client.restore(step,values)
                    if ack['numerical_steps_executed']!=0:raise ValueError('Event restart must not advance the reference')
                    restarted,restored=client.inspect()
                    if restarted['step']!=13 or any(not np.array_equal(restored[n].view('u1'),values[n].view('u1'))for n in FIELDS):
                        raise ValueError('Native event restart changed physical bytes')
            custody.append(client.close())
        finally:
            if client.last_custody is None:client.close()
        if not event_effect or not all(c['reaped']and c['returncode']==0 and c['graceful_protocol_complete']for c in custody):
            raise ValueError('Event must change next-step physical fields or equation request and every native child must complete and reap')
        if len(geometry_checks)!=48 or any(r['cell_size_m']!=1/(scene['cells_per_metre']if r['step']<=13 else changed['cells_per_metre'])for r in geometry_checks):
            raise ValueError('Every actual native pressure request must use the correct piecewise physical geometry')
        receipt_path=directory/'receipt.json'
        write(receipt_path,dict(status='completed',identifier=row['identifier']+'-domain-change13',split='development',
            original_identifier=row['identifier'],scene=scene,post_event_scene=changed,event_step=13,
            event_cells_per_metre_before=scene['cells_per_metre'],event_cells_per_metre_after=changed['cells_per_metre'],states=states,frames=frames,native_processes=custody,
            native_pressure_request_geometry_checks=geometry_checks,
            original_reference=str(parent_path.relative_to(ROOT)),original_reference_sha256=sha(parent_path),
            first14states_and_RHS_bitwise_preserved=True,event_changes_next_step_physics=True,
            initial_plus_consecutive_states=True,source_sha256=sha(__file__),protocol_sha256=sha(protocol_path),
            reference_backend='spectral numerical control',candidate_inference_component=False,
            future_reference_inputs=False,reserved_scenes_consumed=False))
        rows.append(dict(identifier=row['identifier']+'-domain-change13',original_identifier=row['identifier'],split='development',
            receipt=str(receipt_path.relative_to(ROOT)),receipt_sha256=sha(receipt_path),counts_xyz=scene['counts'],
            cells_per_metre=scene['cells_per_metre'],steps=48,event_step=13))
        write(args.output/'progress.json',dict(status='running',trajectories=rows,compressed_state_bytes=total_bytes))
        print('Captured declared application event:',row['identifier'],'pre-event prefix exact; physical box rescale effect verified',flush=True)
    if len(rows)!=8 or {r['original_identifier']for r in rows}!=set(protocol['scene_identifiers']):
        raise ValueError('All eight registered consumed scenes required')
    write(args.output/'bank.json',dict(status='completed',trajectories=rows,states_per_scene=49,compressed_state_bytes=total_bytes,
        original_development_bank_sha256=sha(bank_path),source_sha256=sha(__file__),protocol_sha256=sha(protocol_path),
        training_material=False,optimizer_updates=0,performance_claim=False,
        reference_states_must_not_enter_forecast_episode=True,reserved_scenes_consumed=False))
    print('Captured all8supported domain-change references; compressed state bytes:',total_bytes,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
