"""Offline native control: does pressure history carry physical restart information?"""
import argparse
import os
from pathlib import Path
import numpy as np
from study_support import ROOT,HERE,BASE,read,write,sha
from native_client import NativeClient,FIELDS


def run(args):
    if args.output.exists():
        raise ValueError('Fresh native pressure-history control required')
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0,{23})
    bank=read(BASE/'temporal-data-01/bank.json')
    rows=[]
    for counts in([16,16,16],[9,11,11]):
        row=next(row for row in bank['trajectories']if row['split']=='train'and row['counts_xyz']==counts)
        path=ROOT/row['receipt']
        if sha(path)!=row['receipt_sha256']:
            raise ValueError('Training receipt changed')
        receipt=read(path)
        record=receipt['states'][8]
        source=path.parent/record['path']
        if sha(source)!=record['sha256']:
            raise ValueError('Restart state changed')
        with np.load(source,allow_pickle=False)as arrays:
            initial={name:arrays[name].copy()for name in FIELDS}
        sequences=[]
        for variant in('original_pressure','zero_pressure'):
            client=NativeClient(receipt['scene'],args.output/(row['identifier']+'-'+variant))
            current=dict(initial)
            if variant=='zero_pressure':
                current['pressure']=np.zeros_like(current['pressure'])
            frames=[]
            states=[]
            try:
                client.restore(8,current)
                for step in range(9,17):
                    frame=client.advance(method='spectral')
                    if not frame['frame']['advanced']:
                        raise ValueError('Offline native reference rejected')
                    packet,physical=client.inspect()
                    if packet['step']!=step:
                        raise ValueError('Reference clock changed')
                    states.append(physical)
                    frames.append(frame)
                custody=client.close()
            finally:
                if client.child.poll()is None:client.close()
            sequences.append(states)
            write(client.directory/'receipt.json',dict(status='completed',variant=variant,step_start=8,steps=8,
                native_process=custody,frames=frames,numerical_reference_only=True,candidate_inference=False))
        for offset,(original,zeroed)in enumerate(zip(*sequences,strict=True),9):
            differences={}
            for name in FIELDS:
                difference=float(np.max(np.abs(original[name].astype('f8')-zeroed[name].astype('f8'))))
                scale=max(1.,float(np.max(np.abs(original[name]))))
                tolerance=1e-8*scale if name=='pressure'else 1e-7
                if difference>tolerance:
                    raise ValueError('Pressure history affects physical restart beyond declared reference tolerance')
                differences[name]=dict(max_abs=difference,tolerance=tolerance,
                    bitwise_equal=np.array_equal(original[name].view('u1'),zeroed[name].view('u1')))
            rows.append(dict(identifier=row['identifier'],counts_xyz=counts,step=offset,fields=differences))
    write(args.output/'qualification.json',dict(status='passed',rows=rows,source_sha256=sha(__file__),
        scope='Closed-grid constant-density native spectral reference only; no general claim for other integrators or boundary conditions',
        pressure_history_role='Numerical pressure initialization; qualified physical trajectories match after resetting only current pressure',
        candidate_numerical_pressure_solve=False,offline_numerical_controls=True,optimizer_updates=0,
        reserved_scenes_consumed=False,original_state_contract_unchanged=True))
    print('Qualified native pressure-history reset on two8-step training-scene controls',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
