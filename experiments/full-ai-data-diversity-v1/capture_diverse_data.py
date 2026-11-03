"""Stream actual native trajectories; retain selected complete training pairs."""
import argparse
from pathlib import Path
import os
import shutil
import signal
import time
from support import ROOT, BASE, read, write, digest
from native_forcing_client import NativeForcingClient
import numpy as np
import torch


def run(args):
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh diverse capture directory required')
    reg=read(args.protocol)
    if reg['split'] not in ('pilot','train','development'):
        raise ValueError('Reserved references require a future concrete frozen selection')
    for name,expected in reg['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Registered capture source changed: '+name)
    qualification=read(ROOT/reg['qualification'])
    if qualification['status']!='passed_native_forcing_and_capture_contract':
        raise ValueError('Actual forcing/capture qualification required')
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    record=dict(status='running',split=reg['split'],trajectories=[],active=None,
        protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],
        optimizer_updates=0,reserved_references_captured=False,
        reference_method='Actual native single-track integration with offline full-spectral pressure',
        fields_are_training_reference_labels_only=True,native_reference_is_not_candidate_inference=True)
    selected=set(reg['sample_source_steps'])
    active=None;stored_bytes=0
    def save():write(args.output/'bank.json',record)
    def stop(*_):raise InterruptedError('Stopping capture and reaping native child')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save()
    try:
        for row in reg['rows']:
            if row['split'] != reg['split']:raise ValueError('Registered split mismatch')
            if shutil.disk_usage(args.output).free<reg['minimum_free_disk_bytes']:
                raise RuntimeError('Predeclared free-disk floor reached; preserve completed scenes')
            scene=row['scene'];steps=scene['steps']
            source_steps=sorted(s for s in selected if s<steps)
            retained={0,steps,*source_steps,*(s+1 for s in source_steps)}
            directory=args.output/row['identifier'];directory.mkdir()
            record['active']=dict(identifier=row['identifier'],step=0)
            save();started=time.perf_counter();active=NativeForcingClient(scene,directory)
            states=[];frames=[]
            try:
                for step in range(steps+1):
                    if step:
                        frame=active.advance()
                        if not frame['frame']['advanced'] or frame['frame']['step']!=step:
                            raise ValueError('Native reference rejected an actual physical advance')
                        frames.append(frame)
                    if step in retained:
                        packet,values=active.inspect()
                        if packet['step']!=step or packet['time_s']!=step/60:
                            raise ValueError('Actual state timestamp mismatch')
                        for key in ('rhs','native_input_residual','pressure_base'):
                            source={'rhs':'rhs','native_input_residual':'native_input_residual','pressure_base':'base'}[key]
                            values[key]=np.zeros_like(values['pressure']) if step==0 else active.last_request[source].copy()
                        path=directory/f'state-{step:03d}.npz'
                        np.savez_compressed(path,**values)
                        stored_bytes+=path.stat().st_size
                        if stored_bytes>reg['maximum_capture_bytes']:
                            raise RuntimeError('Predeclared stored-data budget exceeded')
                        states.append(dict(step=step,time_s=step/60,path=path.name,
                            sha256=digest(path),bytes=path.stat().st_size,
                            arrays={k:dict(dtype=v.dtype.str,shape=list(v.shape))for k,v in values.items()}))
                custody=active.close();active=None
                if not custody['reaped'] or custody['returncode']!=0 or not custody['graceful_protocol_complete']:
                    raise ValueError('Native child must complete and reap')
                receipt=dict(status='completed',identifier=row['identifier'],split=row['split'],
                    group=row.get('group'),scene=scene,profile=row['profile'],motion=row['motion'],
                    physical_extent_m=[n*(1/scene['cells_per_metre'])for n in scene['counts']],
                    voxel_count=__import__('math').prod(scene['counts']),actual_steps=steps,
                    training_source_steps=source_steps,states=states,frames=frames,
                    native_process=custody,elapsed_s=time.perf_counter()-started,
                    unretained_steps_were_integrated=True,stored_bytes=sum(s['bytes']for s in states),
                    future_physical_labels_are_not_inference_features=True,
                    protocol_sha256=digest(args.protocol))
                write(directory/'receipt.json',receipt)
                record['trajectories'].append(dict(identifier=row['identifier'],
                    split=row['split'],group=row.get('group'),cells_per_metre=scene['cells_per_metre'],
                    counts_xyz=scene['counts'],profile=row['profile'],motion=row['motion'],
                    receipt=str((directory/'receipt.json').relative_to(ROOT)),
                    receipt_sha256=digest(directory/'receipt.json'),actual_steps=steps,
                    stored_states=len(states),elapsed_s=receipt['elapsed_s'],stored_bytes=receipt['stored_bytes']))
                record.update(active=None,stored_bytes=stored_bytes);save()
                print('Captured',row['identifier'],scene['cells_per_metre'],scene['counts'],
                    steps,'actual steps',len(states),'stored states',f"{receipt['elapsed_s']:.2f}s",flush=True)
            finally:
                if active is not None:active.close();active=None
        record.update(status='completed',stored_bytes=stored_bytes)
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        if active is not None:active.close()
        record['active']=None;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
