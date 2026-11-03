"""Verified consumed/development pressure requests; blind scenes excluded by default."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha

def load(state,include_old_large=True):
    torch=state.torch;np=state.np
    original=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    inputs=ROOT/original['config']['validation_inputs']
    for name,expected in original['validation_sources'].items():
        if sha(inputs/name)!=expected:raise ValueError('Consumed evaluation input changed')
    _,parents=state.data.load(tuple((inputs/name).read_bytes() for name in
        ['validation_dataset.npz','validation_dataset_extra.npz']),read(inputs/'validation_splits.json'))
    result=[(dict(population='consumed128',trajectory_id=p['entry']['trajectory_id'],frame=f['index'],
        counts_xyz=p['counts_xyz'],cells_per_metre=round(1/p['cell_size_m'])),state.data.sample(p,index))
        for p in parents for index,f in enumerate(p['frames'])]
    if len(result)!=128:raise ValueError('Exact original 128 development requests required')
    records=[]
    if include_old_large:
        spec=importlib.util.spec_from_file_location('old_native_fields',ROOT/'experiments/pressure-methods-v1/pack_training.py')
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        reservation=read(ROOT/'runs/pressure-methods-v1/native-training-02/reservation.json')
        for row in reservation['scenes']:
            if row['campaign_group']=='domain-development':
                directory,receipt=native.captured(row)
                population='old-cube32' if row['counts_xyz']==[32]*3 else 'old-cube64' if row['counts_xyz']==[64]*3 else 'old-rectangular'
                records.append((row,directory,receipt,population))
    reservation=read(ROOT/'docs/plans/pressure-methods-v2/reservation.json')
    for row in reservation['scenes']:
        if row['population']!='development':continue
        directory=ROOT/'runs/pressure-methods-v2/native-data/capture'/row['trajectory_id']
        receipt=read(directory/'receipt.json')
        if receipt['row']!=row:raise ValueError('Scene reservation differs from retained capture')
        records.append((row,directory,receipt,'new-small' if max(row['counts_xyz'])<=16 else 'new-rectangular'))
    for row,directory,receipt,population in records:
        if receipt['status']!='qualified_pilot' or not receipt['native_process']['reaped']:
            raise ValueError('Only qualified, reaped independent native requests allowed')
        for frame in receipt['frames']:
            sample={}
            for field in ['base','rhs','reference','native_input_residual']:
                descriptor=frame['arrays']['pressure' if field=='reference' else field]
                path=directory/descriptor['path']
                if sha(path)!=descriptor['sha256']:raise ValueError('Retained native field changed')
                sample[field]=torch.from_numpy(np.fromfile(path,dtype='<f8').reshape(1,1,*row['counts_xyz'][::-1]))
            sample.update(cell_size_m=torch.tensor([1/row['cells_per_metre']],dtype=torch.float64),
                rho=torch.tensor([1000.],dtype=torch.float64),dt=torch.tensor([1/60],dtype=torch.float64))
            result.append((dict(population=population,trajectory_id=row['trajectory_id'],frame=frame['index'],
                counts_xyz=row['counts_xyz'],cells_per_metre=row['cells_per_metre']),sample))
    return result

def screen(cases):
    # Prespecified coverage: one original request per grid/spacing (up to eight),
    # one snapshot from each new development trajectory. No error-based selection.
    seen=set();result=[];old=0
    for identity,sample in cases:
        if identity['population']=='consumed128':
            key=(tuple(identity['counts_xyz']),identity['cells_per_metre'])
            if key in seen or old>=8:continue
            seen.add(key);old+=1;result.append((identity,sample))
        elif identity['population'].startswith('new-') and identity['trajectory_id'] not in seen:
            seen.add(identity['trajectory_id']);result.append((identity,sample))
    return result
