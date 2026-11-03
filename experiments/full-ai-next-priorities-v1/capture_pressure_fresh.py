"""Capture actual fresh native pressure requests after freezing Track A selection."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from study_support import ROOT,HERE,DOCS,read,write,sha
from native_client import NativeClient


def arrays_spec(arrays):
    return {name:dict(shape=list(value.shape),dtype=value.dtype.str,sha256=hashlib.sha256(value.tobytes()).hexdigest())for name,value in arrays.items()}


def run(args):
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh pressure capture directory required')
    selection=read(args.selection);reservation_path=DOCS/'pressure-scene-reservation.json'
    if selection['status']!='frozen_before_reserved_capture'or selection['scene_reservation_sha256']!=sha(reservation_path):
        raise ValueError('Final Track A weights and execution must freeze before capture')
    for model in selection['models']:
        if sha(ROOT/model['entry_path'])!=model['entry_sha256']:raise ValueError('Frozen pressure model entry changed')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    reservation=read(reservation_path);bank=dict(status='running',trajectories=[],active=None,
        source_sha256=sha(__file__),selection_sha256=sha(args.selection),reservation_sha256=sha(reservation_path),
        temporal_reserved_scenes_consumed=False,reference_method='Spectral solves on actual native requests; numerical reference only')
    try:
        for row in reservation['rows']:
            scene={k:v for k,v in row.items()if k not in ('identifier','split')};directory=args.output/row['identifier'];directory.mkdir()
            bank['active']=row['identifier'];write(args.output/'bank.json',bank)
            tick=time.perf_counter();client=NativeClient(scene,directory);cases=[];states=[];frames=[]
            def snapshot(step,physical):
                path=directory/f'state-{step:03d}.npz';np.savez_compressed(path,**physical)
                states.append(dict(step=step,time_s=step/60,path=path.name,sha256=sha(path),arrays=arrays_spec(physical)))
            try:
                packet,physical=client.inspect()
                if packet['step']!=0:raise ValueError('Fresh native scene did not start at step0')
                snapshot(0,physical)
                for step in range(1,65):
                    frame=client.advance()
                    if not frame['frame']['advanced']:raise ValueError('Fresh numerical reference rejected')
                    packet,physical=client.inspect()
                    if packet['step']!=step or packet['time_s']!=step/60:raise ValueError('Fresh native timestamp differs')
                    arrays=dict(base=client.last_request['base'],native_input_residual=client.last_request['native_input_residual'],
                        rhs=client.last_request['rhs'],reference=physical['pressure'])
                    path=directory/f'case-{step:03d}.npz';np.savez_compressed(path,**arrays)
                    cases.append(dict(step=step,time_s=step/60,path=path.name,sha256=sha(path),arrays=arrays_spec(arrays)))
                    if step in (12,48,64):snapshot(step,physical)
                    frames.append(frame)
                custody=client.close()
            finally:
                if client.child.poll()is None:client.close()
            receipt=dict(status='completed',identifier=row['identifier'],scene=scene,cases=cases,states=states,frames=frames,
                native_process=custody,source_sha256=sha(__file__),elapsed_s=time.perf_counter()-tick,
                pressure_requests='Exact supplied base and initial residual; reference RHS and pressure from the corresponding native transition')
            write(directory/'receipt.json',receipt)
            bank['trajectories'].append(dict(identifier=row['identifier'],counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],
                cases=64,receipt=str((directory/'receipt.json').relative_to(ROOT)),receipt_sha256=sha(directory/'receipt.json')))
            bank['active']=None;write(args.output/'bank.json',bank)
            print('Captured fresh pressure requests',row['identifier'],64,flush=True)
        bank['status']='completed'
    except BaseException as error:
        bank.update(status='failed',error=repr(error));raise
    finally:write(args.output/'bank.json',bank)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('selection','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
