"""Stream actual native trajectories; retain selected complete training pairs."""
import argparse
from pathlib import Path
import os
import shutil
import signal
import time
from support import ROOT, BASE, read, write, digest
from continuous_reference_client import ContinuousReferenceClient
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
    if qualification['status']!='passed_fine_time_external_control_qualification':
        raise ValueError('Actual forcing/capture qualification required')
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    record=dict(status='running',split=reg['split'],trajectories=[],active=None,
        protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],
        optimizer_updates=0,reserved_references_captured=False,
        reference_method='Actual native single-track integration with offline full-spectral pressure',
        fields_are_training_reference_labels_only=True,native_reference_is_not_candidate_inference=True)
    if args.identifier:
        matches=[r for r in reg['rows']if r['identifier']==args.identifier]
        if len(matches)!=1:raise ValueError('Exact registered scene selection required')
        reg['rows']=matches
    selected=set(reg['sample_source_steps'])
    active=None;stored_bytes=0;job_started=time.monotonic()
    def save():write(args.output/'bank.json',record)
    def stop(*_):raise InterruptedError('Stopping capture and reaping native child')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save()
    try:
        for row in reg['rows']:
            if row['split'] != reg['split']:raise ValueError('Registered split mismatch')
            if shutil.disk_usage(args.output).free<reg['minimum_free_disk_bytes']:
                raise RuntimeError('Predeclared free-disk floor reached; preserve completed scenes')
            scene=dict(row['scene'],reference_substeps=reg['reference_substeps'],forcing_control_substeps=1);steps=scene['steps']
            source_steps=sorted(s for s in selected if s<steps)
            retained={0,steps,*source_steps,*(s+1 for s in source_steps)}
            directory=args.output/row['identifier'];directory.mkdir()
            record['active']=dict(identifier=row['identifier'],step=0)
            save();started=time.perf_counter();active=ContinuousReferenceClient(scene,directory)
            states=[];frames=[]
            try:
                for step in range(steps+1):
                    if time.monotonic()-job_started>reg['maximum_wall_seconds']:
                        raise TimeoutError('Predeclared capture wall budget reached; reaping native child')
                    if step:
                        frame=active.advance()
                        if not frame['frame']['advanced'] or frame['frame']['step']!=step or not frame['frame']['quality']['qualified']:
                            write(directory/'reference-rejection.json',dict(identifier=row['identifier'],
                                expected_step=step,frame=frame,reference_only=True,
                                protocol_sha256=digest(args.protocol)))
                            raise ValueError('Native reference rejected an actual physical advance')
                        frames.append(frame)
                    if step%16==0:
                        record['active']['step']=step;save()
                    if step in retained:
                        packet,values=active.inspect()
                        if packet['step']!=step or packet['native_step']!=step*reg['reference_substeps'] or packet['time_s']!=step/60:
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
                            native_step=step*reg['reference_substeps'],reference_dt_s=1/(60*reg['reference_substeps']),
                            pressure_label_force_time_s=(step*reg['reference_substeps']-1)/(60*reg['reference_substeps']) if step else None,
                            sha256=digest(path),bytes=path.stat().st_size,
                            arrays={k:dict(dtype=v.dtype.str,shape=list(v.shape))for k,v in values.items()}))
                custody=active.close();active=None
                if not custody['reaped'] or custody['returncode']!=0 or not custody['graceful_protocol_complete']:
                    raise ValueError('Native child must complete and reap')
                receipt=dict(status='completed',identifier=row['identifier'],split=row['split'],
                    group=row.get('group'),scene=scene,profile=row['profile'],motion=row['motion'],
                    physical_extent_m=[n*(1/scene['cells_per_metre'])for n in scene['counts']],
                    voxel_count=__import__('math').prod(scene['counts']),actual_steps=steps,
                    actual_native_steps=sum(len(f['fine_frames']) for f in frames),
                    reference_substeps=reg['reference_substeps'],reference_dt_s=1/(60*reg['reference_substeps']),
                    output_dt_s=1/60,forcing_control_semantics='Original sampled macro controls held within each output interval',
                    reference_scheme_id='sampled-controls-native-substeps-'+str(reg['reference_substeps']),
                    temporal_convergence_qualified=False,
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
                    reference_substeps=reg['reference_substeps'],actual_native_steps=receipt['actual_native_steps'],
                    stored_states=len(states),elapsed_s=receipt['elapsed_s'],stored_bytes=receipt['stored_bytes']))
                record.update(active=None,stored_bytes=stored_bytes);save()
                print('Captured',row['identifier'],scene['cells_per_metre'],scene['counts'],
                    steps,'actual steps',len(states),'stored states',f"{receipt['elapsed_s']:.2f}s",flush=True)
            finally:
                if active is not None:
                    custody=active.close();active=None
                    write(directory/'failed-native-custody.json',custody)
        record.update(status='completed',stored_bytes=stored_bytes)
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        if active is not None:active.close()
        record['active']=None;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--identifier');run(p.parse_args())
