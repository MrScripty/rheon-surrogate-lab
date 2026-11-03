"""Qualify actual hybrid recurrence, label isolation and differentiable objective."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha, source_inputs


def qualify(project,output):
    started = time.perf_counter()
    sys.path.insert(0,str(project))
    import torch
    import model
    import data
    import fine_tuning
    import physical_residual
    model.configure('cpu')
    torch.set_num_threads(1)
    config = read(project/'config.json')
    model.select(config)
    witness = read(project/'source-witness.json')
    for name,expected in witness['sources'].items():
        if sha(project/name) != expected:
            raise ValueError('Captured hybrid source changed')
    paths = source_inputs(project)
    _,parents = data.load(tuple(paths[k].read_bytes() for k in ['dataset','dataset_extra']),read(paths['splits']))
    if any(p['entry']['split'] != 'train' for p in parents):
        raise ValueError('Training-only qualification required')
    normalization = model.fit_normalization(parents,data.fit_normalization)
    network = model.PressureCorrectionNet().eval()
    initial = ROOT/next(row['path'] for row in read(project/'project.json')['data'] if row['id']=='initial_weights')
    fine_tuning.initialize(network,initial,data.helpers())
    inventory = data.helpers().state_inventory(network)
    teacher = {k:v.detach().clone() for k,v in fine_tuning.TEACHER.state_dict().items()}
    sample = data.sample(parents[0],0)
    trace = []
    with torch.no_grad():
        state = sample['base']
        for passes in range(1,9):
            remaining = sample['native_input_residual']-model.full_operator(state-sample['base'],sample['cell_size_m'],sample['rho'])
            state = physical_residual.predict_pressure(network,{**sample,'base':state,'native_input_residual':remaining},
                normalization,model.owner,{'activation':'linear','steps':1})
            if passes in [1,2,4,8]:
                direct = physical_residual.predict_pressure(network,sample,normalization,model.owner,{'activation':'linear','steps':passes})
                error = float(torch.linalg.vector_norm(direct-state))/max(float(torch.linalg.vector_norm(direct)),1e-30)
                if error > 2e-12:
                    raise ValueError('Hybrid pass recurrence differs beyond float64 accumulation tolerance')
                trace.append(dict(passes=passes,relative_l2_difference=error,tolerance=2e-12,bitwise_equal=bool(torch.equal(direct,state))))
        ordinary = model.predict_pressure(network,sample,normalization)
        direct = physical_residual.predict_pressure(network,sample,normalization,model.owner,{'activation':'linear','steps':2})
        if not torch.equal(ordinary,direct):
            raise ValueError('Deployment must use the trained hybrid method')
        changed_labels = {**sample,'rhs':sample['rhs']+100,'reference':sample['reference']-123}
        if not torch.equal(ordinary,model.predict_pressure(network,changed_labels,normalization)):
            raise ValueError('Numerical/neural inference accessed labels')
    network.train()
    loss = model.training_loss(network,sample,normalization).mean()
    if not bool(torch.isfinite(loss)) or not loss.requires_grad:
        raise ValueError('Finite differentiable hybrid loss required')
    loss.backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in network.parameters()):
        raise ValueError('Hybrid gradients failed')
    if not any(bool((p.grad != 0).any()) for p in network.parameters()):
        raise ValueError('All hybrid gradients vanished')
    if (inventory != data.helpers().state_inventory(network) or inventory != fine_tuning.SPEC['initial_inventory']
            or any(p.grad is not None or p.requires_grad for p in fine_tuning.TEACHER.parameters())
            or any(not torch.equal(teacher[k],v) for k,v in fine_tuning.TEACHER.state_dict().items())):
        raise ValueError('Qualification changed initial weights or frozen teacher')
    with torch.no_grad():
        empty = {k:torch.zeros_like(v) if k in ['base','rhs','reference','native_input_residual'] else v for k,v in sample.items()}
        if float(model.training_loss(network,empty,normalization).mean()) != 0:
            raise ValueError('Zero pressure equation must have zero loss')
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only qualification required')
    write(output,dict(status='passed',project=str(project.relative_to(ROOT)),project_sha256=sha(project/'project.json'),
        source_witness_sha256=sha(project/'source-witness.json'),initial_weights_sha256=sha(initial),
        exact_initial_inventory=True,normalization_unchanged=True,recurrence=trace,
        unchanged_default_deployment_passes=2,finite_nonzero_gradient=True,frozen_teacher_unchanged=True,
        zero_residual_bypasses_teacher=True,inference_label_independent=True,optimizer_updates=0,
        cpu_autograd_verification_only=True,cuda_initialized=False,initial_loss=float(loss.detach()),
        elapsed_s=time.perf_counter()-started,source_sha256=sha(__file__)))
