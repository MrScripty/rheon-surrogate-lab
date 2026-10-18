"""Verify the exact fitted case on CPU without any optimizer updates."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, source_inputs, write
from qualify import qualify


def check(project, output):
    started = time.perf_counter()
    base_output=output.with_suffix('.base.json')
    if output.exists() or base_output.exists():
        raise ValueError('Fresh composite qualification required')
    qualify(project, base_output)
    import torch
    import model
    import data
    import fine_tuning
    import single_case
    inputs = source_inputs(project)
    _, parents = data.load(tuple(inputs[k].read_bytes() for k in ['dataset','dataset_extra']), read(inputs['splits']))
    cache = {(i,j):data.sample(p,j) for i,p in enumerate(parents) for j in range(len(p['frames']))}
    normalization = model.fit_normalization(parents,data.fit_normalization)
    initial = ROOT/next(r['path'] for r in read(project/'project.json')['data'] if r['id']=='initial_weights')
    network = model.PressureCorrectionNet()
    fine_tuning.initialize(network,initial,data.helpers())
    single_case.bind(cache)
    sample = single_case.SAMPLE
    assert tuple(sample['reference'].shape) == (1,1,16,15,13)
    assert sample['cell_size_m'].numel() == 1 and float(sample['cell_size_m']) == 1/16
    defect = float(torch.linalg.vector_norm(model.full_operator(sample['reference'],sample['cell_size_m'],sample['rho'])-sample['rhs'])/torch.linalg.vector_norm(sample['rhs']))
    assert defect < 1e-10
    before = data.helpers().state_inventory(network)
    teacher_calls = fine_tuning.STATS['teacher_forwards']
    value = single_case.loss(network,normalization)
    assert value.requires_grad and bool(torch.isfinite(value))
    value.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in network.parameters())
    assert any(bool((p.grad != 0).any()) for p in network.parameters())
    assert teacher_calls == fine_tuning.STATS['teacher_forwards']
    assert before == data.helpers().state_inventory(network)
    assert fine_tuning.STATS['single_case']['optimizer_examples'] == 0
    single_case.observe(network,normalization,0)
    trace = fine_tuning.STATS['single_case']['trace'][0]
    assert [r['passes'] for r in trace['measurements']] == [1,2,4,8]
    assert not torch.cuda.is_initialized()
    result = read(base_output)
    result.update(single_case=dict(shape_zyx=list(sample['base'].shape[2:]),cell_size_m=float(sample['cell_size_m']),
        amplitude_rms_pa=fine_tuning.STATS['single_case']['amplitude_rms_pa'],
        independent_operator_defect=defect,finite_nonzero_gradient=True,no_teacher_loss=True,
        original_train_only_scalar_calibration=True,initial_loss=float(value.detach()),baseline_trace=trace),
        supplemental_qualification_source_sha256=sha(__file__),total_elapsed_s=time.perf_counter()-started)
    write(output,result)
    print('Qualified single manufactured fitting case',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    check(args.project.resolve(),args.output.resolve())
