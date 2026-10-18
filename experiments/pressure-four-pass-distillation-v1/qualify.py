"""Qualify initialization, recurrence, autograd and frozen-teacher custody."""
import argparse
from pathlib import Path
import sys
import time

from common import ROOT, read, sha, source_inputs, write


def qualify(project, output):
    started = time.perf_counter()
    sys.path.insert(0, str(project))
    import torch
    import model
    import data
    import fine_tuning
    model.configure('cpu')
    torch.set_num_threads(1)
    config = read(project/'config.json')
    model.select(config)
    witness = read(project/'source-witness.json')
    for name, identity in witness['sources'].items():
        if sha(project/name) != identity:
            raise ValueError('Captured source changed: ' + name)
    inputs = source_inputs(project)
    _, parents = data.load(tuple(inputs[k].read_bytes() for k in ['dataset','dataset_extra']), read(inputs['splits']))
    if any(p['entry']['split'] != 'train' for p in parents):
        raise ValueError('Qualification cannot consume validation fields')
    normalization = model.fit_normalization(parents, data.fit_normalization)
    from policy import canonical, sha as byte_sha
    assert byte_sha(canonical(normalization)) == witness['normalization_sha256']
    document = read(project/'project.json')
    initial = ROOT/next(r['path'] for r in document['data'] if r['id']=='initial_weights')
    network = model.PressureCorrectionNet().eval()
    fine_tuning.initialize(network, initial, data.helpers())
    assert data.helpers().state_inventory(network) == fine_tuning.SPEC['initial_inventory']
    teacher_before = {k:v.detach().clone() for k,v in fine_tuning.TEACHER.state_dict().items()}
    sample = data.sample(parents[0], 0)
    state = sample['base']
    recurrence = []
    import physical_residual
    with torch.no_grad():
        for passes in range(1,9):
            remaining = sample['native_input_residual']-model.full_operator(state-sample['base'],sample['cell_size_m'],sample['rho'])
            state = physical_residual.predict_pressure(network,{**sample,'base':state,'native_input_residual':remaining},normalization,model.owner,{'activation':'linear','steps':1})
            if passes in (1,2,4,8):
                direct = physical_residual.predict_pressure(network,sample,normalization,model.owner,{'activation':'linear','steps':passes})
                assert torch.equal(direct,state)
                recurrence.append(dict(passes=passes,exact=True))
        assert torch.equal(model.predict_pressure(network,sample,normalization),
                           physical_residual.predict_pressure(network,sample,normalization,model.owner,{'activation':'linear','steps':2}))
    network.train()
    loss = model.training_loss(network,sample,normalization).mean()
    assert bool(torch.isfinite(loss)) and loss.requires_grad
    loss.backward()
    gradients = [p.grad for p in network.parameters()]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
    assert any(bool((g!=0).any()) for g in gradients)
    assert all(p.grad is None and not p.requires_grad for p in fine_tuning.TEACHER.parameters())
    assert all(torch.equal(teacher_before[k],v) for k,v in fine_tuning.TEACHER.state_dict().items())
    assert data.helpers().state_inventory(network) == fine_tuning.SPEC['initial_inventory'], 'No optimizer may run in qualification'
    with torch.no_grad():
        expected = fine_tuning.teacher_pressure(model.baseline,sample,normalization)
        again = fine_tuning.teacher_pressure(model.baseline,sample,normalization)
        assert torch.equal(expected,again) and fine_tuning.STATS['teacher_cache_hits'] >= 1
        assert fine_tuning.CACHE_BYTES <= fine_tuning.SPEC['teacher_cache_bytes']
        empty = {**sample,'base':torch.zeros_like(sample['base']),'reference':torch.zeros_like(sample['reference']),
                 'rhs':torch.zeros_like(sample['rhs']),'native_input_residual':torch.zeros_like(sample['native_input_residual'])}
        count = fine_tuning.STATS['teacher_forwards']
        zero = model.training_loss(network,empty,normalization)
        assert float(zero.mean()) == 0 and count == fine_tuning.STATS['teacher_forwards']
    assert not torch.cuda.is_initialized()
    write(output,dict(status='passed',project=str(project.relative_to(ROOT)),project_sha256=sha(project/'project.json'),
                      source_witness_sha256=sha(project/'source-witness.json'),initial_weights_sha256=sha(initial),
                      exact_initial_inventory=True,normalization_unchanged=True,recurrence=recurrence,
                      unchanged_default_deployment_passes=2,finite_nonzero_gradient=True,frozen_teacher_unchanged=True,
                      zero_residual_bypasses_teacher=True,teacher_cache_bounded=True,optimizer_updates=0,
                      cpu_autograd_verification_only=True,cuda_initialized=False,initial_loss=float(loss),
                      elapsed_s=time.perf_counter()-started,source_sha256=sha(__file__)))
    print('Qualified',project.name,float(loss),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    qualify(args.project.resolve(),args.output.resolve())
