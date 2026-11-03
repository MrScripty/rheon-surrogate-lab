"""Extend consumed development references from qualified full-state restarts."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha
from native_client import NativeClient,FIELDS
import hashlib


def run(output):
    output=output.resolve()
    if output.exists():raise ValueError('Fresh development extension directory required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);output.mkdir(parents=True)
    bank=read(BASE/'temporal-data-01/bank.json')
    record=dict(status='running',source_sha256=sha(__file__),source_bank_sha256=sha(BASE/'temporal-data-01/bank.json'),
        split='development',reserved_scenes_consumed=False,trajectories=[],active=None)
    try:
        for row in bank['trajectories']:
            if row['split']!='development':continue
            receipt_path=ROOT/row['receipt']
            if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Original development receipt changed')
            original=read(receipt_path);scene=dict(original['scene'],steps=240)
            directory=output/row['identifier'];directory.mkdir()
            record['active']=row['identifier'];write(output/'bank.json',record)
            steps=list(original['states']);frames=list(original['frames'])
            for state in steps:
                source=receipt_path.parent/state['path']
                if sha(source)!=state['sha256']:raise ValueError('Original full state changed')
                os.link(source,directory/state['path'])
            with np.load(directory/steps[-1]['path'],allow_pickle=False)as saved:
                current={name:saved[name].copy()for name in FIELDS}
            started=time.perf_counter();client=NativeClient(scene,directory)
            try:
                client.restore(128,current)
                packet,restored=client.inspect()
                if packet['step']!=128 or any(not np.array_equal(current[k].view('u1'),restored[k].view('u1'))for k in FIELDS):
                    raise ValueError('Extension restart bytes differ')
                for index in range(129,241):
                    frame=client.advance()
                    if not frame['frame']['advanced']:raise ValueError('Development reference rejected')
                    packet,arrays=client.inspect()
                    if packet['step']!=index or packet['time_s']!=index/60:raise ValueError('Extension timestamp differs')
                    arrays['rhs']=client.last_request['rhs'];path=directory/f'state-{index:03d}.npz'
                    np.savez_compressed(path,**arrays)
                    specs={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest())for k,v in arrays.items()}
                    steps.append(dict(step=index,time_s=index/60,path=path.name,bytes=path.stat().st_size,
                        sha256=sha(path),arrays=specs,initial_state=False))
                    frames.append(frame)
                custody=client.close()
            finally:
                if client.child.poll()is None:client.close()
            result=dict(original,scene=scene,states=steps,frames=frames,native_process=custody,
                source_sha256=sha(__file__),extended_from_original_sha256=sha(receipt_path),
                first_129_states_bitwise_preserved=True,qualified_restart_at_step=128,extension_seconds=time.perf_counter()-started)
            write(directory/'receipt.json',result)
            record['trajectories'].append(dict(row,states=len(steps),receipt=str((directory/'receipt.json').relative_to(ROOT)),
                receipt_sha256=sha(directory/'receipt.json')))
            record['active']=None;write(output/'bank.json',record)
            print('Extended development scene',row['identifier'],'to240steps',flush=True)
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:write(output/'bank.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
