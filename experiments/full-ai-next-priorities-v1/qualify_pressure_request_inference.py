"""Compare native-file causal encoding against the managed four-grid CPU probe."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha
from pressure_request_inference import RequestPrediction


def run(args):
    if args.output.exists():raise ValueError('Fresh causal inference qualification required')
    os.sched_setaffinity(0,{23});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    component=RequestPrediction(args.entry)
    bank=read(BASE/'temporal-data-01/bank.json');pieces=[]
    with torch.inference_mode():
        for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
            parent=next(row for row in bank['trajectories']if row['split']=='train'and row['counts_xyz']==counts)
            path=ROOT/parent['receipt']
            if sha(path)!=parent['receipt_sha256']:raise ValueError('Training native receipt changed')
            receipt=read(path)
            for step in (0,8,120):
                spec=receipt['states'][step];path_field=path.parent/spec['path']
                if sha(path_field)!=spec['sha256']:raise ValueError('Current native field changed')
                with np.load(path_field,allow_pickle=False)as arrays:
                    fields={name:arrays[name].copy()for name in ('vx','vy','vz')}
                pieces.append(component.predict_normalized(fields,receipt['scene'],step).numpy().reshape(-1).copy())
    actual=np.concatenate(pieces).astype('<f4',copy=False)
    reference=np.load(ROOT/component.entry['restore']/'probe.npy',allow_pickle=False)
    difference=float(np.linalg.norm(actual.astype('f8')-reference.astype('f8')))/max(float(np.linalg.norm(reference)),1e-30)
    if difference>1e-6:raise ValueError('Causal native-file encoding differs from managed CPU probe')
    if 'pressure_request_objective'in sys.modules:raise ValueError('Analytical teacher imported into inference')
    write(args.output,dict(status='passed',component_entry_sha256=sha(args.entry),source_sha256=sha(__file__),
        inference_source_sha256=sha(HERE/'pressure_request_inference.py'),native_bank_sha256=sha(BASE/'temporal-data-01/bank.json'),
        elements=actual.size,pressure_request_relative_difference=difference,
        bitwise_equal=bool(np.array_equal(actual.view('<u4'),reference.view('<u4'))),
        all_four_grids=True,steps=[0,8,120],future_request_input=False,current_or_future_pressure_input=False,
        analytical_teacher_imported=False,optimizer_updates=0,cuda_initialized=torch.cuda.is_initialized()))
    print('Qualified native-file causal encoding against managed CPU probe; relative difference',difference,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
