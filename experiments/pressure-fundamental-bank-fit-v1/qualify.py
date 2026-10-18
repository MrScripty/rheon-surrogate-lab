"""Qualify every fitted fundamental without optimizer updates or CUDA admission."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,source_inputs,write
from qualify import qualify


def check(project,output):
    started=time.perf_counter();base_output=output.with_suffix('.base.json')
    if output.exists() or base_output.exists():
        raise ValueError('Fresh composite qualification required')
    qualify(project,base_output)
    import numpy as np
    import torch
    import model
    import data
    import fine_tuning
    import fit_bank
    inputs=source_inputs(project)
    _,parents=data.load(tuple(inputs[k].read_bytes()for k in ['dataset','dataset_extra']),read(inputs['splits']))
    if any(p['entry']['split']!='train'for p in parents):
        raise ValueError('Original train-only calibration required')
    cache={(i,j):data.sample(p,j)for i,p in enumerate(parents)for j in range(len(p['frames']))}
    normalization=model.fit_normalization(parents,data.fit_normalization)
    config=read(project/'config.json');updates=config['training']['updates']
    initial=ROOT/next(r['path']for r in read(project/'project.json')['data']if r['id']=='initial_weights')
    network=model.PressureCorrectionNet()
    fine_tuning.initialize(network,initial,data.helpers())
    fit_bank.bind(cache,updates)
    stats=fine_tuning.STATS['fundamental_bank'];count=len(fit_bank.SAMPLES)
    assert updates==count*200 and stats['expected_exposures']==[200]*count
    assert all(sorted(cycle.tolist())==list(range(count))for cycle in fit_bank.ORDER.reshape(-1,count))
    assert np.array_equal(fit_bank.ORDER,fit_bank.sampling(count,updates,fine_tuning.SPEC['fitting_bank']['sampling_seed']))
    assert stats['resident_tensor_bytes']==fit_bank.validate(fine_tuning.SPEC['fitting_bank'])
    before=data.helpers().state_inventory(network);teacher_calls=fine_tuning.STATS['teacher_forwards']
    cases=[]
    for index,(case,sample)in enumerate(zip(stats['cases'],fit_bank.SAMPLES,strict=True)):
        assert list(sample['base'].shape[2:])==list(reversed(case['counts_xyz']))
        assert sample['cell_size_m'].numel()==1 and float(sample['cell_size_m'])==1/case['cells_per_metre']
        defect=float(torch.linalg.vector_norm(model.full_operator(sample['reference'],sample['cell_size_m'],sample['rho'])-sample['rhs'])/torch.linalg.vector_norm(sample['rhs']))
        assert defect<1e-10
        network.zero_grad(set_to_none=True)
        value=fit_bank.loss(network,sample,normalization)
        assert value.requires_grad and bool(torch.isfinite(value))
        value.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in network.parameters())
        assert any(bool((p.grad!=0).any())for p in network.parameters())
        cases.append(dict(case=index,**case,independent_operator_defect=defect,initial_loss=float(value.detach()),finite_nonzero_gradient=True))
    assert teacher_calls==fine_tuning.STATS['teacher_forwards']
    assert before==data.helpers().state_inventory(network)
    assert stats['optimizer_examples']==0 and stats['optimizer_exposures']==[0]*count
    fit_bank.observe(network,normalization,0)
    assert len(stats['trace'][0]['measurements'])==4*count
    trainer=(project/'train.py').read_text()
    assert 'fit_bank.one_update('in trainer and "fit_bank.bind(cache,config['training']['updates'])"in trainer
    assert not torch.cuda.is_initialized()
    result=read(base_output)
    result.update(fundamental_bank=dict(cases=cases,resident_tensor_bytes=stats['resident_tensor_bytes'],
        amplitude_rms_pa=stats['amplitude_rms_pa'],sampling_order_sha256=stats['sampling_order_sha256'],
        expected_exposures=stats['expected_exposures'],no_teacher_loss=True,original_train_only_scalar_calibration=True,
        baseline_trace=stats['trace'][0]),supplemental_qualification_source_sha256=sha(__file__),
        total_elapsed_s=time.perf_counter()-started)
    write(output,result)
    print('Qualified',count,'fundamental cases',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();check(args.project.resolve(),args.output.resolve())
