"""Prove small predictions match calibration and larger predictions retain C107."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
from study_support import ROOT,HERE,BASE,read,write,sha,load_checkpoint,historical_cases,equation_error

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--identifier',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh routed CPU qualification required')
    os.sched_setaffinity(0,{23});state=load_checkpoint(args.identifier);torch=state.torch
    parent=read(ROOT/'runs/pressure-methods-v4/entries/v4-student50-control.json');prior=ROOT/parent['project']
    expected=read(prior/'source-witness.json')['sources']['physical_residual.py']
    if sha(prior/'physical_residual.py')!=expected:raise ValueError('Original pressure function changed')
    spec=importlib.util.spec_from_file_location('unchanged_physical_reference_for_routing',prior/'physical_residual.py')
    physical=importlib.util.module_from_spec(spec);spec.loader.exec_module(physical)
    class HeadView:
        def __init__(self,network,small):
            self.network=network;self.coarse_logits=network.small_coarse_logits if small else network.coarse_logits
            self.shape_head=network.small_shape_head if small else network.shape_head
        def __call__(self,values):return self.network(values)
    data_project=BASE/'a-combined-project-01';slots={r['id']:ROOT/r['path']for r in read(data_project/'project.json')['data']}
    for slot,row in read(data_project/'source-witness.json')['inputs'].items():
        if sha(slots[slot])!=row['sha256']:raise ValueError('Consumed regression data changed')
    import train
    _,parents=state.data.load(tuple(slots[k].read_bytes()for k in ['dataset','dataset_extra',*train.VIEW_SLOTS]),read(slots['splits']))
    cases=[(dict(trajectory_id=p['trajectory_id'],native_frame=p['frames'][f]['index'],counts_xyz=p['counts_xyz'],split='original_physical_train'),state.data.sample(p,f))
           for p in parents for f in range(len(p['frames']))]
    cases.extend((dict(identity,split='consumed_historical_development'),sample)for identity,sample in historical_cases(state))
    before=state.data.helpers().state_inventory(state.network);rows=[]
    with torch.inference_mode():
        for identity,sample in cases:
            small=max(identity['counts_xyz'])<=16
            expected=physical.predict_pressure(HeadView(state.network,small),sample,state.normalization,state.model.owner,state.model.CORRECTION)
            actual=state.model.predict_pressure(state.network,sample,state.normalization)
            if not torch.equal(actual,expected):raise ValueError('Routed result differs from selected parent head and unchanged fine CNN')
            if not bool(torch.isfinite(actual).all())or actual[0,0,0,0,0]!=0:raise ValueError('Routed physical output contract failed')
            rows.append(dict(identity,selected_head='calibrated'if small else'C107',bitwise_parent_equivalence=True,
                equation_relative_residual=equation_error(state,actual,sample)))
    if state.data.helpers().state_inventory(state.network)!=before:raise ValueError('Zero-update routing qualification changed weights')
    write(args.output,dict(status='passed',identifier=args.identifier,rows=rows,parameters=94956,optimizer_updates=0,
        local_assembly_receipt_sha256=sha(ROOT/state.entry['restore']/'assembly-receipt.json'),entry_sha256=sha(BASE/'entries'/(args.identifier+'.json')),
        shared_fine_and_large_head_unchanged=True,small_head_matches_calibrated_parent=True,
        tested_requests=len(rows),larger_requests=sum(r['selected_head']=='C107'for r in rows),
        source_sha256=sha(__file__),cuda_initialized=torch.cuda.is_initialized(),candidate_numerical_pressure_solve=False,
        qualification_kind='Fresh CPU reload and bitwise component equivalence; no managed assembly Run claimed'))
    print('Qualified routed model on196requests; larger grids retain C107 bitwise',args.identifier,flush=True)
