"""Measure one published pre-state's independent CPU counterfactual endpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,source_inputs,write
from steps import adam,interpolate


def measure(config_path,update,cpu,output):
    started=time.perf_counter();config=read(config_path)
    if output.exists() or update not in config['updates']:
        raise ValueError('Fresh output and declared update required')
    os.sched_setaffinity(0,{cpu})
    project,study=ROOT/config['project'],ROOT/config['study'];restore=study/'restore'
    receipt=read(study/'receipt.json');prior=read(ROOT/config['replay_results'])
    assert receipt['status']=='completed' and all(receipt['artifact_parity'].values())
    assert prior['final_weights_exact'] and prior['final_quality_exact']
    assert sha(restore/'weights.npz')==receipt['weights_sha256']==prior['replay_weights_sha256']
    for phase in ('training','restore'):
        terminal=read(study/('terminal-'+receipt[phase+'_run_id']+'.json'))
        assert terminal['state']=='completed' and terminal['custody_phase']=='drained'
    name='adam-trace-%03d.npz'%update;trace=restore/name
    assert sha(trace)==sha(study/'train'/name)==receipt['restore_receipt']['outputs'][name]['sha256']==receipt['training_receipt']['outputs'][name]['sha256']
    with zipfile.ZipFile(trace) as archive:
        assert archive.testzip() is None and sum(i.file_size for i in archive.infolist())<=12*1024**2
    sys.path.insert(0,str(project))
    import numpy as np
    import torch
    import model
    import data
    import fine_tuning
    import fit_bank
    import adam_trace
    model.configure('cpu');torch.set_num_threads(1);model.select(read(project/'config.json'))
    witness=read(project/'source-witness.json')
    for name,identity in witness['sources'].items():assert sha(project/name)==identity,name
    network=model.PressureCorrectionNet()
    initial=ROOT/next(r['path'] for r in read(project/'project.json')['data'] if r['id']=='initial_weights')
    fine_tuning.initialize(network,initial,data.helpers())
    normalization=read(restore/'normalization.json');inputs=source_inputs(project)
    _,parents=data.load(tuple(inputs[k].read_bytes() for k in ('dataset','dataset_extra')),read(inputs['splits']))
    assert all(p['entry']['split']=='train' for p in parents)
    cache={(i,j):data.sample(p,j) for i,p in enumerate(parents) for j in range(len(p['frames']))}
    fit_bank.bind(cache,200);cases=fine_tuning.STATS['fundamental_bank']['cases']
    assert cases==receipt['training_receipt']['fine_tuning_statistics']['fundamental_bank']['cases']
    params=list(network.parameters());count=sum(p.numel() for p in params)
    assert 80*count<=config['vector_workspace_limit_bytes']
    with np.load(trace,allow_pickle=False) as archive:
        metadata=json.loads(archive['metadata'].tobytes())
        assert metadata==prior['metadata']
        captured=next(r for r in metadata['records'] if r['update']==update)
        assert metadata['layout']==[dict(name=n,shape=list(p.shape),count=p.numel()) for n,p in network.named_parameters()]
        def vector(key):
            value=archive['u%03d_%s'%(update,key)]
            assert value.dtype==np.float32 and value.shape==(count,) and np.isfinite(value).all()
            return torch.from_numpy(value)
        before,after,gradient=vector('parameters_before'),vector('parameters_after'),vector('gradient_clipped')
        assert captured['steps_before']==[float(update-1)]*len(params)
        moments=(vector('exp_avg_before'),vector('exp_avg_sq_before')) if update>1 else None
        if update==1:
            assert captured['initialized_before']==[False]*len(params)
            assert not bool(vector('exp_avg_before').any()) and not bool(vector('exp_avg_sq_before').any())
        preserved=adam(before,gradient,captured['group'],update-1,moments)
        reset=adam(before,gradient,captured['group'],0)
    error=float((preserved-after).abs().max())
    assert error<=config['adam_parameter_absolute_tolerance']
    assert update!=1 or torch.equal(preserved,reset)
    del moments
    endpoints=dict(observed=after,preserved=preserved,reset=reset)
    evaluations={};candidates=[];evaluation_s=0.
    for policy in config['policies']:
        for fraction in config['fractions']:
            candidate=interpolate(before,endpoints[policy],fraction)
            digest=hashlib.sha256(candidate.numpy().tobytes()).hexdigest()
            if digest not in evaluations:
                cursor=0
                with torch.no_grad():
                    for parameter in params:
                        parameter.copy_(candidate[cursor:cursor+parameter.numel()].reshape(parameter.shape));cursor+=parameter.numel()
                assert cursor==count
                clock=time.perf_counter();metrics=adam_trace.metrics(network,normalization)
                elapsed=time.perf_counter()-clock;evaluation_s+=elapsed
                assert len(metrics['losses'])==8 and len(metrics['pressure'])==32
                assert all(np.isfinite(v) for v in metrics['losses'])
                assert all(np.isfinite(r['relative_equation_residual']) for r in metrics['pressure'])
                evaluations[digest]=dict(metrics=metrics,elapsed_s=elapsed)
            candidates.append(dict(policy=policy,fraction=fraction,parameter_sha256=digest,
                delta_l2=float(torch.linalg.vector_norm(candidate.double()-before.double())),
                delta_max_absolute=float((candidate-before).abs().max())))
    assert all(p.grad is None for p in params)
    assert not torch.cuda.is_initialized() and fine_tuning.STATS['teacher_forwards']==0
    assert fine_tuning.STATS['fundamental_bank']['optimizer_updates']==0
    assert torch.get_num_threads()==1
    result=dict(schema='rheon-adam-step-measurement-v1',status='completed',update=update,cases=cases,
        learning_rate=captured['group']['lr'],adam_reconstruction_max_absolute_error=error,
        initialized_reset_control_exact=update==1 and torch.equal(preserved,reset),
        candidates=candidates,evaluations=evaluations,cuda_observed_before=captured['before'],cuda_observed_after=captured['after'],
        parameter_count=count,vector_workspace_bound_bytes=80*count,parameter_dtype='float32',
        optimizer_updates=0,disposable_adam_steps=2,teacher_forwards=0,cuda_initialized=False,
        cpu=cpu,threads=1,elapsed_s=time.perf_counter()-started,evaluation_s=evaluation_s,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scope='Independent one-step endpoints using saved CUDA gradient; no retrained moment history or deployment claim.',
        sources={str(p.relative_to(ROOT)):sha(p) for p in [config_path,ROOT/config['replay_results'],study/'receipt.json',trace,project/'source-witness.json',project/'adam_trace.py',HERE/'measure.py',HERE/'steps.py']})
    write(output,result)
    print('Measured',update,len(evaluations),'unique states',result['elapsed_s'],'s',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('update','cpu'):p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args();measure(a.config.resolve(),a.update,a.cpu,a.output.resolve())
