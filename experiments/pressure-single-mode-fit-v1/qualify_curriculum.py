"""Qualify spectral exposure while retaining the historical matched control."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,source_inputs,write
from qualify import qualify


def check(project,control,output):
    started=time.perf_counter()
    base_output=output.with_suffix('.base.json')
    if output.exists() or base_output.exists():
        raise ValueError('Fresh composite qualification required')
    qualify(project,base_output)
    import torch
    import model
    import data
    import analytical
    import original_analytical
    import fine_tuning
    config=read(project/'config.json');previous=read(control/'config.json')
    assert {k:v for k,v in config.items()if k!='id'}=={k:v for k,v in previous.items()if k!='id'}
    objective=read(project/'fine-tuning.json');old_objective=read(control/'fine-tuning.json')
    assert {k:v for k,v in objective.items()if k!='arm'}=={k:v for k,v in old_objective.items()if k!='arm'}
    witness=read(project/'source-witness.json');old_witness=read(control/'source-witness.json')
    assert witness['inputs']==old_witness['inputs']
    assert witness['normalization_sha256']==old_witness['normalization_sha256']
    for name,identity in old_witness['sources'].items():
        if name not in ['analytical.py','fine-tuning.json','config.json']:
            assert witness['sources'][name]==identity,name
    assert sha(project/'original_analytical.py')==sha(control/'analytical.py')
    assert analytical.bank is original_analytical.bank
    inputs=source_inputs(project)
    _,parents=data.load(tuple(inputs[k].read_bytes()for k in ['dataset','dataset_extra']),read(inputs['splits']))
    actual=[data.sample(p,j)for p in parents for j in range(len(p['frames']))]
    amplitude=analytical.training_correction_rms(actual)
    primary_spec=config['analytical_feedback'];spec=config['supplemental_feedback'];updates=config['training']['updates']
    primary=analytical.bank(actual[0],primary_spec,updates,amplitude_pa=amplitude)
    samples,order,evidence=analytical.supplemental_bank(actual[0],spec,updates,primary,primary_spec,amplitude)
    control_samples,old_order,old_evidence=original_analytical.supplemental_bank(actual[0],spec,updates,primary,primary_spec,amplitude)
    assert bool((order==old_order).all())
    assert evidence['resident_tensor_bytes']==old_evidence['resident_tensor_bytes']
    assert evidence['grid_exposures']==old_evidence['grid_exposures']
    assert evidence['combined_resident_tensor_bytes']==old_evidence['combined_resident_tensor_bytes']
    defect=0.
    for sample in samples:
        ratio=float(torch.linalg.vector_norm(model.full_operator(sample['reference'],sample['cell_size_m'],sample['rho'])-sample['rhs'])/torch.linalg.vector_norm(sample['rhs']))
        defect=max(defect,ratio)
        assert ratio<1e-10
    initial=ROOT/next(r['path']for r in read(project/'project.json')['data']if r['id']=='initial_weights')
    network=model.PressureCorrectionNet()
    fine_tuning.initialize(network,initial,data.helpers())
    normalization=model.fit_normalization(parents,data.fit_normalization)
    sample=samples[0]
    value=model.training_loss(network,sample,normalization,
        criterion=lambda p:analytical.loss(p,sample,model.full_operator)).mean()
    value.backward()
    assert bool(torch.isfinite(value))
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in network.parameters())
    assert any(bool((p.grad!=0).any())for p in network.parameters())
    assert data.helpers().state_inventory(network)==objective['initial_inventory']
    result=read(base_output)
    result.update(matched_control=dict(project=str(control.relative_to(ROOT)),source_witness_sha256=sha(control/'source-witness.json'),
        config_equal_except_id=True,objective_equal_except_arm=True,original_sources_exact_except_curriculum=True,
        inputs_and_normalization_exact=True,primary_bank_owner_unchanged=True,supplemental_draw_order_exact=True,
        resident_tensor_bytes_equal=True),
        hard_bank=dict(samples=len(samples),bytes=evidence['resident_tensor_bytes'],
            combined_bytes=evidence['combined_resident_tensor_bytes'],grid_exposures=evidence['grid_exposures'],
            order_sha256=evidence['sampling_order_sha256'],description_sha256=evidence['bank_description_sha256'],
            independent_operator_worst_defect=defect,actual_mode_category_exposures=evidence['actual_mode_category_exposures'],
            amplitude_rms_pa=amplitude),curriculum_finite_nonzero_gradient=True,
        supplemental_qualification_source_sha256=sha(__file__),total_elapsed_s=time.perf_counter()-started)
    write(output,result)
    print('Qualified matched frequency curriculum',evidence['actual_mode_category_exposures'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['project','control','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    check(args.project.resolve(),args.control.resolve(),args.output.resolve())
