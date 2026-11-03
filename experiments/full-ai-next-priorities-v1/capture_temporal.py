"""Finite full-state reference captures with pre-reserved whole-scene splits."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha
from native_client import NativeClient,FIELDS

def capture(args):
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh capture bank directory required')
    qualification=BASE/'temporal-data-qualification-01/qualification.json'
    if read(qualification)['status']!='passed':raise ValueError('Complete state contract must qualify first')
    reservation=read(DOCS/'temporal-scene-reservation.json')
    if not set(args.split)<= {'train','development','reserved'}:raise ValueError('Known scene splits required')
    if 'reserved'in args.split and not args.frozen_selection:
        raise ValueError('Reserved references require a frozen selection before capture')
    if args.frozen_selection and not args.frozen_selection.is_file():raise ValueError('Frozen selection file missing')
    args.output.mkdir(parents=True)
    shutil.copyfile(DOCS/'temporal-scene-reservation.json',args.output/'scene-reservation.json')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    record=dict(status='running',split=args.split,reference_method='full-spectral-on-actual-native-requests',
        state_contract_sha256=sha(qualification),reservation_sha256=sha(DOCS/'temporal-scene-reservation.json'),
        capture_source_sha256=sha(__file__),native_client_sha256=sha(HERE/'native_client.py'),
        numerical_reference_sha256=sha(ROOT/'experiments/pressure-methods-v1/numerics.py'),
        frozen_selection_sha256=sha(args.frozen_selection)if args.frozen_selection else None,
        trajectories=[],active=None,optimizer_updates=0)
    active=None
    try:
        for row in reservation['rows']:
            if row['split']not in args.split:continue
            scene={k:v for k,v in row.items()if k not in ('identifier','split')}
            directory=args.output/row['identifier'];directory.mkdir()
            record['active']=dict(identifier=row['identifier'],directory=str(directory.relative_to(ROOT)))
            write(args.output/'bank.json',record)
            started=time.perf_counter();active=NativeClient(scene,directory)
            steps=[];frames=[]
            try:
                for index in range(scene['steps']+1):
                    if index:
                        frame=active.advance()
                        if not frame['frame']['advanced']:raise ValueError('Reference physical rejection')
                        frames.append(frame)
                    packet,arrays=active.inspect()
                    if packet['step']!=index or packet['time_s']!=index/60:
                        raise ValueError('Consecutive native state timestamp mismatch')
                    rhs=np.zeros_like(arrays['pressure'])if index==0 else active.last_request['rhs']
                    path=directory/f'state-{index:03d}.npz'
                    np.savez_compressed(path,**arrays,rhs=rhs)
                    specs={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest())for k,v in {**arrays,'rhs':rhs}.items()}
                    steps.append(dict(step=index,time_s=packet['time_s'],path=path.name,bytes=path.stat().st_size,
                        sha256=sha(path),arrays=specs,initial_state=index==0))
                custody=active.close();active=None
                result=dict(status='completed',identifier=row['identifier'],split=row['split'],scene=scene,states=steps,
                    frames=frames,native_process=custody,elapsed_s=time.perf_counter()-started,
                    initial_plus_consecutive_states=True,source_sha256=sha(__file__),
                    request_reference='RHS and pressure on the same native transition; RHS is a label, never an inference feature',
                    state_contract_sha256=sha(qualification),future_reference_inputs=False)
                write(directory/'receipt.json',result)
                record['trajectories'].append(dict(identifier=row['identifier'],split=row['split'],
                    counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],states=len(steps),
                    receipt=str((directory/'receipt.json').relative_to(ROOT)),receipt_sha256=sha(directory/'receipt.json')))
                record['active']=None;write(args.output/'bank.json',record)
                print('Captured',row['identifier'],len(steps),'consecutive full states',flush=True)
            finally:
                if active is not None:active.close();active=None
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        if active is not None:active.close()
        record['active']=None;write(args.output/'bank.json',record)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-selection',type=Path)
    capture(parser.parse_args())
