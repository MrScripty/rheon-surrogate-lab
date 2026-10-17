"""Qualify repaired clipping through original consumed-field optimizer updates."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-project','training-source','artifact-root','analysis','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    sys.path[:0] = [str(args.baseline_project.resolve()), str(args.training_source.resolve())]
    import torch
    import numpy as np
    import gradient_clipping
    from stability_screen import StabilityProbe
    from gradient_screen import DiagnosticLimit
    baseline = load('baseline_trainer', args.baseline_project/'train.py')
    trainer = load('repaired_trainer', args.training_source/'train.py')
    config = json.loads((args.baseline_project/'config.json').read_text())
    baseline.model.select(config)
    baseline.model.configure('cuda')
    document = json.loads((args.baseline_project/'project.json').read_text())
    inputs = {row['id']:(args.artifact_root/row['path']).read_bytes() for row in document['data']}
    _, parents = baseline.data.load((inputs['dataset'],inputs['dataset_extra']),json.loads(inputs['splits']))
    normalization = baseline.model.fit_normalization(parents,baseline.data.fit_normalization)
    parent = min((i for i,p in enumerate(parents) if p['entry']['split']=='train'),
                 key=lambda i:np.prod(parents[i]['counts_xyz']))
    sample = baseline.data.sample(parents[parent],0,device='cuda')
    indices = np.asarray([[parent,0]],dtype='<i8')
    def construct():
        torch.manual_seed(config['training']['seed'])
        network = baseline.model.PressureCorrectionNet().cuda()
        optimizer = torch.optim.Adam(network.parameters(),lr=config['training']['learning_rate'],
                                     betas=(.9,.999),eps=1e-8,weight_decay=0)
        return network,optimizer
    def step(owner, network, optimizer, extra=True):
        kwargs = dict(cache={(parent,0):sample},strategy=config['execution'])
        if extra:
            for label in ('analytical','supplemental','direction','current'):
                kwargs[label+'_sample'] = sample
                kwargs[label+'_weight'] = .1
        return owner.one_update(network,optimizer,parents,indices,normalization,**kwargs)
    original, original_optimizer = construct()
    expected = [step(baseline,original,original_optimizer) for _ in range(2)]
    cpu_rng, cuda_rng = torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
    repaired, optimizer = construct()
    reference=dict(initial_inventory={name:dict(shape=list(value.shape),sha256=hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest())for name,value in repaired.state_dict().items()}, analytical_seeds={}, direction_seeds={}, sampling_order_sha256={})
    real_order=np.tile(indices,(5000,1,1));order=np.zeros(5000,dtype='<i8')
    for label in ('real','analytical','supplemental','direction','current'):
        reference['sampling_order_sha256'][label]=hashlib.sha256((real_order if label=='real'else order).tobytes()).hexdigest()
    events=[];probe=StabilityProbe(trainer,2,reference,events.append)
    for label in reference['sampling_order_sha256']:probe.capture_order(label,real_order if label=='real'else order,{})
    probe.install()
    try:
        actual=[]
        for _ in range(2):
            actual.append(step(trainer,repaired,optimizer))
            trainer.verify_cuda_update(repaired,optimizer,combined=True)
        try:step(trainer,repaired,optimizer)
        except DiagnosticLimit:pass
        else:raise AssertionError('Integrated screen did not cap before third update')
        assert probe.completed==probe.cuda_checks==2 and probe.repairs==0
        assert events[-1]['event']=='diagnostic_cap_reached' and events[-1]['successful_cuda_checks']==2
    finally:probe.close()
    assert actual == expected
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert all(torch.equal(value,original.state_dict()[key])for key,value in repaired.state_dict().items())
    for p,q in zip(repaired.parameters(),original.parameters()):
        assert torch.equal(p.grad,q.grad)
        for key in ('step','exp_avg','exp_avg_sq'):
            assert torch.equal(optimizer.state[p][key],original_optimizer.state[q][key])
    analysis = json.loads(args.analysis.read_text())
    assert analysis['classification']=='finite_components_norm_reduction_overflow'
    observed = analysis['failure_observation']['update']['gradient_snapshot']['parameters']
    pair = sorted((r['ordinary_float32_norm']for r in observed),reverse=True)[:2]
    # Hooks replace backward gradients at the actual one_update clipping seam.
    # These are explicitly scalar-derived fixtures, not captured model gradients.
    def inject(network, values):
        handles=[]
        for index,p in enumerate(network.parameters()):
            value=values[index] if index<len(values) else 0.
            def hook(gradient,value=value):
                result=torch.zeros_like(gradient)
                result.reshape(-1)[0]=value
                return result
            handles.append(p.register_hook(hook))
        return handles
    reports=[]
    maximum=torch.finfo(torch.float32).max
    for label,values in [('observed-norm-pair',pair),('nan',[float('nan')]),
                         ('infinity',[float('inf')]),('unrepresentable',[.9*maximum,.9*maximum])]:
        outcomes=[]
        for owner in (baseline,trainer):
            network,opt=construct()
            before={k:v.clone()for k,v in network.state_dict().items()}
            handles=inject(network,values)
            step_calls=[]
            guard=opt.register_step_pre_hook(lambda *unused:step_calls.append(True))
            repaired_events=[]
            native=trainer.clip_l2_
            if owner is trainer:
                trainer.clip_l2_=lambda parameters,limit: native(parameters,limit,on_repair=repaired_events.append)
            error=None
            try:
                result=step(owner,network,opt,extra=False)
            except RuntimeError as caught:
                error=str(caught)
            finally:
                trainer.clip_l2_=native
                guard.remove()
                for handle in handles:handle.remove()
            if label=='observed-norm-pair' and owner is trainer:
                assert error is None and len(step_calls)==1 and len(repaired_events)==1
                trainer.verify_cuda_update(network,opt,combined=True)
                clipped=[float(p.grad.reshape(-1)[0])for p in network.parameters()]
                independent=math.hypot(*values)
                assert abs(result[1]/independent-1)<2e-6
                assert abs(math.hypot(*clipped)-1)<2e-6
                assert all(abs(a-b/independent)<2e-6 for a,b in zip(clipped,values))
                outcomes.append(dict(owner='integrated',optimizer_steps=1,repaired=True,
                    independent_expected_norm=independent,norm=result[1],clipped_norm=math.hypot(*clipped),events=repaired_events))
            else:
                assert error and error.startswith('The total norm of order 2.0 for gradients from')
                assert not step_calls and not opt.state
                assert all(torch.equal(v,before[k])for k,v in network.state_dict().items())
                for index,p in enumerate(network.parameters()):
                    wanted=values[index]if index<len(values)else 0.
                    target=torch.zeros_like(p.grad);target.reshape(-1)[0]=wanted
                    torch.testing.assert_close(p.grad,target,rtol=0,atol=0,equal_nan=True)
                assert not repaired_events
                outcomes.append(dict(owner='original'if owner is baseline else'integrated',
                    rejected=True,optimizer_steps=0,weights_and_gradients_unchanged=True,message=error))
        if label!='observed-norm-pair':assert outcomes[0]['message']==outcomes[1]['message']
        reports.append(dict(case=label,outcomes=outcomes))
    # Prove an unrelated native RuntimeError is never converted to a repair.
    native=gradient_clipping.ORIGINAL_CLIP
    sentinel=RuntimeError('unrelated native execution failure')
    def unrelated(*args,**kwargs):raise sentinel
    gradient_clipping.ORIGINAL_CLIP=unrelated
    try:
        ps=[torch.nn.Parameter(torch.zeros(1,device='cuda'))for _ in pair]
        for p,value in zip(ps,pair):p.grad=torch.tensor([value],device='cuda')
        try:gradient_clipping.clip_l2_(ps)
        except RuntimeError as caught:assert caught is sentinel
        else:raise AssertionError('Unrelated RuntimeError was masked')
        assert all(float(p.grad)==float(torch.tensor(v,dtype=torch.float32))for p,v in zip(ps,pair))
    finally:gradient_clipping.ORIGINAL_CLIP=native
    result=dict(format='pressure-coarse-clipping-qualification-v1',status='passed',elapsed_s=time.perf_counter()-started,
        device='cuda',gpu=torch.cuda.get_device_name(0),exact_original_update_results=True,
        exact_weights_gradients_adam_rng=True, integrated_observer_and_cuda_cap_qualified=True,
        observer_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath('pressure-coarse-gradient-v1/gradient_screen.py').read_bytes()).hexdigest(),ordinary_optimizer_steps=4,repaired_fixture_optimizer_steps=1,
        rejected_fixture_optimizer_steps=0,fitting_sequence_updates=0,simulations_advanced=0,
        observed_pair=pair,integrated_cases=reports,unrelated_error_identity_preserved=True,
        source_sha256=hashlib.sha256(Path(__file__).with_name('stability_screen.py').read_bytes()).hexdigest(),
        trainer_sha256=hashlib.sha256((args.training_source/'train.py').read_bytes()).hexdigest(),
        clipping_sha256=hashlib.sha256((args.training_source/'gradient_clipping.py').read_bytes()).hexdigest(),
        qualifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_trainer_sha256=hashlib.sha256((args.baseline_project/'train.py').read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
        scope='Actual CUDA one_update on consumed smallest real frame: ordinary five-stream exact differential, then backward hooks inject scalar-derived minimized norm and negative fixtures at the clipping seam. No original1398 gradient tensors or repaired fitting trajectory/accuracy proof.')
    with args.output.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k]for k in ('status','elapsed_s','exact_weights_gradients_adam_rng','repaired_fixture_optimizer_steps')}))


if __name__=='__main__':main()
