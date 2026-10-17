"""Actual CUDA updates preserve the qualified observer; finish count is explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifact-root','repair-qualification','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    sys.path.insert(0,str(args.project.resolve()))
    import numpy as np
    import torch
    import train
    from stability_screen import StabilityProbe
    from full_fit import FullFitProbe
    prior=json.loads(args.repair_qualification.read_text())
    assert prior['status']=='passed'and prior['exact_weights_gradients_adam_rng']
    assert digest(args.project/'train.py')==prior['trainer_sha256']
    assert digest(args.project/'gradient_clipping.py')==prior['clipping_sha256']
    assert digest(args.project/'stability_screen.py')==prior['source_sha256']
    config=json.loads((args.project/'config.json').read_text());train.model.select(config);train.model.configure('cuda')
    document=json.loads((args.project/'project.json').read_text())
    inputs={r['id']:(args.artifact_root/r['path']).read_bytes()for r in document['data']}
    _,parents=train.data.load((inputs['dataset'],inputs['dataset_extra']),json.loads(inputs['splits']))
    normalization=train.model.fit_normalization(parents,train.data.fit_normalization)
    parent=min((i for i,p in enumerate(parents)if p['entry']['split']=='train'),key=lambda i:np.prod(parents[i]['counts_xyz']))
    sample=train.data.sample(parents[parent],0,device='cuda');indices=np.asarray([[parent,0]],dtype='<i8')
    def construct():
        torch.manual_seed(config['training']['seed']);network=train.model.PressureCorrectionNet().cuda()
        opt=torch.optim.Adam(network.parameters(),lr=config['training']['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0)
        return network,opt
    def execute(kind):
        network,opt=construct();order=np.zeros(5000,dtype='<i8');real=np.tile(indices,(5000,1,1))
        reference=dict(initial_inventory={name:dict(shape=list(v.shape),sha256=hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest())for name,v in network.state_dict().items()},analytical_seeds={},direction_seeds={},sampling_order_sha256={})
        for label in ('real','analytical','supplemental','direction','current'):reference['sampling_order_sha256'][label]=hashlib.sha256((real if label=='real'else order).tobytes()).hexdigest()
        events=[];probe=kind(train,2,reference,events.append)
        for label in reference['sampling_order_sha256']:probe.capture_order(label,real if label=='real'else order,{})
        probe.install();results=[]
        try:
            if kind is FullFitProbe:
                try:probe.finish()
                except ValueError as error:assert str(error)=='Full fit requires every configured optimizer/CUDA check'
                else:raise AssertionError('Premature fit accepted')
            for _ in range(2):
                results.append(train.one_update(network,opt,parents,indices,normalization,cache={(parent,0):sample},strategy=config['execution'],
                    analytical_sample=sample,analytical_weight=.1,supplemental_sample=sample,supplemental_weight=.1,
                    direction_sample=sample,direction_weight=.1,current_sample=sample,current_weight=.1))
                train.verify_cuda_update(network,opt,combined=True)
            if kind is FullFitProbe:
                event=probe.finish();assert event['event']=='full_fit_completed'and event['successful_cuda_checks']==2
                probe.cuda_checks-=1
                try:probe.finish()
                except ValueError:pass
                else:raise AssertionError('Missing final CUDA check accepted')
                probe.cuda_checks+=1
            assert probe.completed==probe.cuda_checks==2 and probe.repairs==0
        finally:probe.close()
        return network,opt,results,torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
    original,old_opt,expected,cpu,cuda=execute(StabilityProbe)
    network,opt,actual,new_cpu,new_cuda=execute(FullFitProbe)
    assert actual==expected and torch.equal(cpu,new_cpu)and torch.equal(cuda,new_cuda)
    assert all(torch.equal(v,original.state_dict()[key])for key,v in network.state_dict().items())
    for p,q in zip(network.parameters(),original.parameters()):
        assert torch.equal(p.grad,q.grad)
        for key in ('step','exp_avg','exp_avg_sq'):assert torch.equal(opt.state[p][key],old_opt.state[q][key])
    result=dict(format='pressure-coarse-full-fit-entry-qualification-v1',status='passed',elapsed_s=time.perf_counter()-started,
        exact_predecessor_update_results=True,exact_weights_gradients_adam_rng=True,full_fit_finish_boundary_qualified=True,
        actual_cuda_optimizer_steps=4,fitting_sequence_updates=0,simulations_advanced=0,device='cuda',gpu=torch.cuda.get_device_name(0),
        source_sha256=digest(Path(__file__).with_name('full_fit.py')),qualifier_sha256=digest(Path(__file__)),
        sources={name:digest(args.project/name)for name in ('train.py','gradient_clipping.py','stability_screen.py','gradient_screen.py','capture_policy.py')},
        repair_qualification_sha256=digest(args.repair_qualification),
        scope='Two previous capped-observer vs two full-fit-observer CUDA updates on one consumed real field reused in five streams. Native-first repair qualification is reused by exact source identity; no5000 trajectory or new pressure accuracy proof.')
    with args.output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k]for k in ('status','elapsed_s','actual_cuda_optimizer_steps','full_fit_finish_boundary_qualified')}))


if __name__=='__main__':main()
