"""Freeze qualified current-policy pairs in the existing direction-bank contract."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import sys


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x')as stream:json.dump(value,stream,sort_keys=True,separators=(',',':'),allow_nan=False);stream.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--preparation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve(strict=True);a.output=a.output.resolve();preparation=a.preparation.resolve(strict=True);receipt=json.loads((preparation/'receipt.json').read_text())
    if receipt['status']!='passed'or receipt['fixture']or len(receipt['cases'])!=144 or not receipt['all_workers_reaped']or a.output.exists()or not a.output.is_relative_to(root):raise ValueError('full144 qualified states and fresh bank required')
    project=root/receipt['configuration']['project'];sys.path.insert(0,str(project));import data
    witness=json.loads((project/'source-witness.json').read_text())
    if digest(project/'data.py')!=witness['sources']['data.py']:raise ValueError('captured dataset inventory owner changed')
    arrays={'early':{},'late':{}};cases=[];identities=set();train_ids=set()
    for record in receipt['cases']:
        if not record['paired_bank_eligible']:continue
        key=json.dumps(record['identity'],sort_keys=True)
        if key in identities:raise ValueError('duplicate paired state')
        identities.add(key);path=(preparation/record['pair_path']).resolve(strict=True)
        if not path.is_relative_to(preparation)or digest(path)!=record['pair_sha256']:raise ValueError('qualified pair payload differs')
        late=next(r for r in record['prefixes']if r['prefix']==8)
        if record['identity']['kind']=='real':train_ids.add(record['identity']['trajectory_id'])
        case={k:record[k]for k in ('identity','counts_xyz','rho','dt','cell_size_m')};case['stages']={}
        with np.load(path,allow_pickle=False)as fields:
            if set(fields.files)!={'early_rhs','early_reference','late_rhs','late_reference'}:raise ValueError('exact qualified pair inventory required')
            for stage,measured in [('early',record['early']),('late',late)]:
                for field in ('rhs','reference'):arrays[stage][f'case_{len(cases):03d}_{field}']=fields[f'{stage}_{field}'].copy()
                case['stages'][stage]={'gain':1.,'unscaled_correction_rms_pa':measured['target_rms_pa'],
                    'scaled_correction_rms_pa':measured['target_rms_pa'],'operator_relative_error':measured['independent_operator_relative_defect']}
        cases.append(case)
    if len(cases)!=144 or len(train_ids)!=16:raise ValueError('all declared paired groups/parents required')
    a.output.mkdir(parents=True);stages={}
    for stage,values in arrays.items():
        path=a.output/(stage+'.npz')
        with path.open('xb')as stream:np.savez(stream,**values)
        if path.stat().st_size>32*1024**2:raise ValueError('captured direction slot budget exceeded')
        inventory=data.inventory(values)
        stages[stage]={'sha256':digest(path),'bytes':path.stat().st_size,'arrays':inventory}
    amplitude=receipt['cases'][0]['early']['target_rms_pa']*receipt['cases'][0]['early']['rescale_gain_to_train_rms']
    manifest={'schema':'rheon-pressure-direction-bank-v1','source':{'project':receipt['configuration']['project'],'artifacts':receipt['configuration']['artifacts'],
        'weights_sha256':receipt['weights_sha256'],'normalization_sha256':receipt['normalization_sha256'],'source_witness_sha256':receipt['source_witness_sha256']},
        'packet_splits_sha256':receipt['packet_splits_sha256'],'train_trajectory_ids':sorted(train_ids),'amplitude_rms_pa':amplitude,
        'source_corrections':8,'optimizer_updates':0,'cases':cases,'excluded':[],'stages':stages,
        'preparation':{'path':str(preparation),'sha256':digest(preparation/'receipt.json')},'source_sha256':digest(Path(__file__)),
        'amplitude_policy':'Native per-state amplitudes, gain1; amplitude_rms_pa references original training correction RMS, not a rescaling. Original preprocessing remains unchanged.'}
    write(a.output/'manifest.json',manifest)
    print(json.dumps({'cases':len(cases),'manifest_sha256':digest(a.output/'manifest.json'),'stages':{s:v['bytes']for s,v in stages.items()}}))

if __name__=='__main__':main()
