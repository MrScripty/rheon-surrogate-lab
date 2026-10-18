"""Stream observed Adam states through bounded CPU reconstruction and geometry."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, source_inputs, write


def analyze(config_path, project, study, output):
    started = time.perf_counter()
    if output.exists():
        raise ValueError('Fresh CPU analysis receipt required')
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    config = read(config_path)
    receipt = read(study/'receipt.json')
    assert receipt['status']=='completed' and all(receipt['artifact_parity'].values())
    assert receipt['training_receipt']['execution']['device']=='cuda:0'
    assert receipt['restore_receipt']['optimizer_updates']==0
    assert not receipt['restore_receipt']['execution']['cuda_initialized']
    restored = study/'restore'
    trace_names=['adam-trace-%03d.npz'%u for u in config['capture_updates']]
    for name in trace_names:
        assert sha(study/'train'/name)==sha(restored/name)==receipt['training_receipt']['outputs'][name]['sha256']
        assert sha(restored/name)==receipt['restore_receipt']['outputs'][name]['sha256']
        with zipfile.ZipFile(restored/name) as archive:
            assert archive.testzip() is None and sum(i.file_size for i in archive.infolist()) <= 12*1024**2
    custody=[]
    for phase in ('training','restore'):
        terminal=read(study/('terminal-'+receipt[phase+'_run_id']+'.json'))
        assert terminal['state']=='completed' and terminal['custody_phase']=='drained'
        custody.append(dict(phase=phase,run_id=terminal['id'],state=terminal['state'],custody_phase=terminal['custody_phase']))
    sys.path.insert(0,str(project))
    import numpy as np
    import torch
    import model
    import data
    import fine_tuning
    import fit_bank
    model.configure('cpu');torch.set_num_threads(1);model.select(read(project/'config.json'))
    witness=read(project/'source-witness.json')
    for name,identity in witness['sources'].items():
        assert sha(project/name)==identity
    network=model.PressureCorrectionNet()
    initial=ROOT/next(r['path'] for r in read(project/'project.json')['data'] if r['id']=='initial_weights')
    fine_tuning.initialize(network,initial,data.helpers())
    normalization=read(restored/'normalization.json')
    inputs=source_inputs(project)
    _,parents=data.load(tuple(inputs[k].read_bytes() for k in ('dataset','dataset_extra')),read(inputs['splits']))
    cache={(i,j):data.sample(p,j) for i,p in enumerate(parents) for j in range(len(p['frames']))}
    fit_bank.bind(cache,200)
    cases=fine_tuning.STATS['fundamental_bank']['cases']
    params=list(network.parameters());count=sum(p.numel() for p in params)
    state=SimpleNamespace(torch=torch,network=network,normalization=normalization,model=model)
    spec=importlib.util.spec_from_file_location('component_owner',ROOT/'experiments/pressure-loss-components-v1/measure.py')
    component=importlib.util.module_from_spec(spec);spec.loader.exec_module(component)

    def flatten(values):
        return torch.cat([v.detach().reshape(-1) for v in values])

    def load(vector):
        cursor=0
        with torch.no_grad():
            for p in params:
                p.copy_(vector[cursor:cursor+p.numel()].reshape(p.shape));cursor+=p.numel()
        assert cursor==count

    def geometry(vector, direction):
        vector=vector.double();direction=direction.double()
        a,b=float(torch.linalg.vector_norm(vector)),float(torch.linalg.vector_norm(direction))
        dot=float(vector@direction)
        return dict(dot=dot,cosine=dot/a/b if a and b else None,gradient_l2=a,direction_l2=b)

    results=[]
    with ExitStack() as stack:
        archives={u:stack.enter_context(np.load(restored/('adam-trace-%03d.npz'%u),allow_pickle=False)) for u in config['capture_updates']}
        metadata=json.loads(archives[1]['metadata'].tobytes())
        assert all(metadata==json.loads(a['metadata'].tobytes()) for a in archives.values())
        assert metadata['updates']==config['capture_updates']
        assert metadata['parameter_count']==count==370016
        assert metadata['layout']==[dict(name=n,shape=list(p.shape),count=p.numel()) for n,p in network.named_parameters()]
        assert set().union(*(set(a.files) for a in archives.values()))=={'metadata',*('u%03d_%s'%(u,k) for u in config['capture_updates'] for k in metadata['vector_names'])}
        for captured in metadata['records']:
            update=captured['update'];archive=archives[update]
            def vector(key):
                value=archive['u%03d_%s'%(update,key)]
                assert value.dtype==np.float32 and value.shape==(count,) and np.isfinite(value).all()
                return torch.from_numpy(value)
            before,after=vector('parameters_before'),vector('parameters_after')
            raw,clipped=vector('gradient_raw'),vector('gradient_clipped')
            delta=after.double()-before.double()
            group=captured['group']
            assert group['betas']==[.9,.999] and group['eps']==1e-8 and group['weight_decay']==0 and not group['amsgrad']
            assert captured['steps_before']==[float(update-1)]*len(params) and captured['steps_after']==[float(update)]*len(params)
            parameter=torch.nn.Parameter(before.clone())
            optimizer=torch.optim.Adam([parameter],lr=group['lr'],betas=tuple(group['betas']),eps=group['eps'],weight_decay=0)
            if update>1:
                optimizer.state[parameter]=dict(step=torch.tensor(float(update-1)),exp_avg=vector('exp_avg_before').clone(),exp_avg_sq=vector('exp_avg_sq_before').clone())
            else:
                assert captured['initialized_before']==[False]*len(params)
                assert not bool(vector('exp_avg_before').any()) and not bool(vector('exp_avg_sq_before').any())
            parameter.grad=clipped.clone();optimizer.step()
            reconstruction=parameter.detach().double()-after.double()
            adam_error=float(reconstruction.abs().max())
            adam_relative=float(torch.linalg.vector_norm(reconstruction)/torch.linalg.vector_norm(delta))
            moments=optimizer.state[parameter]
            ideal=-group['lr']*(moments['exp_avg'].double()/(1-group['betas'][0]**update))/(torch.sqrt(moments['exp_avg_sq'].double()/(1-group['betas'][1]**update))+group['eps'])
            eligible=ideal!=0
            quantization=dict(scope='Float64 formula using CPU-reconstructed float32 post-Adam moments; approximate diagnostic, not a higher-precision training run.',
                nonzero_intended_updates=int(eligible.sum()), unchanged_despite_nonzero_intent=int(((before==after)&eligible).sum()),
                fraction_nonzero_intent_unchanged=float(((before==after)&eligible).sum()/eligible.sum()) if bool(eligible.any()) else None,
                ideal_delta_l2=float(torch.linalg.vector_norm(ideal)))
            del optimizer,parameter,reconstruction,ideal,eligible,moments
            load(before);network.train()
            mean=torch.zeros(count,dtype=torch.float64);capped=torch.zeros_like(mean);rows=[];weak=None
            for index,sample in enumerate(fit_bank.SAMPLES):
                loss=fit_bank.loss(network,sample,normalization)
                gradient=flatten(torch.autograd.grad(loss,params));value=float(loss.detach());del loss
                mean.add_(gradient,alpha=1/8)
                norm=float(torch.linalg.vector_norm(gradient.double()))
                capped.add_(gradient,alpha=min(1.,1/(norm+1e-6))/8)
                projection=geometry(gradient,delta)
                actual=captured['after']['losses'][index]-captured['before']['losses'][index]
                rows.append(dict(case=index,**cases[index],cpu_loss_before=value,
                    gpu_loss_before=captured['before']['losses'][index],gpu_loss_after=captured['after']['losses'][index],
                    actual_loss_change=actual,predicted_first_order_loss_change=projection['dot'],
                    full_gradient_vs_delta=projection,full_gradient_vs_raw=geometry(gradient,raw),
                    nonlinear_remainder=actual-projection['dot']))
                if index==7:weak=gradient.clone()
                del gradient
            mean_error=float(torch.linalg.vector_norm(mean-raw.double())/torch.linalg.vector_norm(raw.double()))
            weak_terms,joint=component.components(state,fit_bank.SAMPLES[7])
            joint_gradient=flatten(torch.autograd.grad(joint,params,retain_graph=True))
            assert torch.equal(joint_gradient,weak), 'Independent component reconstruction must match full CPU gradient'
            del joint_gradient
            term_rows={}
            for key in ('canonical','final','early'):
                gradient=flatten(torch.autograd.grad(weak_terms[key],params,retain_graph=key!='early'))
                term_rows[key]=dict(actual_delta=geometry(gradient,delta),raw_mean=geometry(gradient,raw))
                del gradient
            del weak_terms,joint
            cap_scale=min(1.,1/(float(torch.linalg.vector_norm(capped))+1e-6))
            capped.mul_(-cap_scale)
            results.append(dict(update=update,learning_rate=group['lr'],
                delta_l2=float(torch.linalg.vector_norm(delta)),delta_max_absolute=float(delta.abs().max()),
                fraction_parameters_unchanged=float((before==after).double().mean()),
                parameter_update_quantization=quantization,
                raw_gradient_l2=float(torch.linalg.vector_norm(raw.double())),clipped_gradient_l2=float(torch.linalg.vector_norm(clipped.double())),
                actual_delta_vs_negative_clipped=geometry(-clipped,delta),
                adam_reconstruction_max_abs=adam_error,adam_reconstruction_relative_delta_l2=adam_relative,
                adam_reconstruction_passed=adam_error<=config['adam_reconstruction_absolute_tolerance'],
                cpu_mean_vs_captured_cuda_gradient_relative_l2=mean_error,
                gradient_precision_check_passed=mean_error<=config['gradient_reconstruction_relative_tolerance'],
                weak_terms=term_rows,weak_vs_negative_capped_mean=geometry(weak,capped),
                actual_delta_vs_negative_capped_mean=geometry(delta,capped),
                cases=rows,observed_before=captured['before'],observed_after=captured['after']))
            del before,after,raw,clipped,delta,mean,capped,weak
    final=read(restored/'history.json')['checkpoints']['checkpoint-200.npz']['inventory']
    data.helpers().load_weights(network,restored/'weights.npz',final)
    measurements=fit_bank.measure(network,normalization,fit_bank.SAMPLES)
    reference=ROOT/config['control_restore']
    with np.load(reference/'weights.npz',allow_pickle=False) as a,np.load(restored/'weights.npz',allow_pickle=False) as b:
        assert a.files==b.files
        weights_error=max(float(np.max(np.abs(a[k].astype(np.float64)-b[k].astype(np.float64)))) for k in a.files)
        weights_exact=all(np.array_equal(a[k],b[k]) for k in a.files)
    historical=read(ROOT/'runs/pressure-balanced-shapes-v1/balanced-shapes-cpu-01/C69-fitted.json')['measurements']
    fields=('case','passes','relative_equation_residual')
    quality_exact=all({k:a[k] for k in fields}=={k:b[k] for k in fields} for a,b in zip(historical,measurements,strict=True))
    assert cases==read(ROOT/'runs/pressure-balanced-shapes-v1/balanced-shapes-cpu-01/C69-fitted.json')['cases']
    differences=[dict(case=a['case'],passes=a['passes'],reference=a['relative_equation_residual'],replay=b['relative_equation_residual']) for a,b in zip(historical,measurements,strict=True)]
    assert all(a['case']==b['case'] and a['passes']==b['passes'] for a,b in zip(historical,measurements,strict=True))
    assert not torch.cuda.is_initialized() and torch.get_num_threads()==1
    assert fine_tuning.STATS['teacher_forwards']==0
    result=dict(schema='rheon-adam-replay-analysis-v1',status='completed',model_promoted=False,baseline_retained='C53',
        custody=custody,artifact_parity={**receipt['artifact_parity'],**{n:sha(study/'train'/n)==sha(restored/n) for n in trace_names}},
        reference_weights_sha256=sha(reference/'weights.npz'),replay_weights_sha256=sha(restored/'weights.npz'),
        final_weights_exact=weights_exact,final_weights_max_absolute_error=weights_error,final_quality_exact=quality_exact,
        final_pressure_comparison=differences,observations=results,metadata=metadata,
        adam_absolute_tolerance=config['adam_reconstruction_absolute_tolerance'],gradient_relative_tolerance=config['gradient_reconstruction_relative_tolerance'],
        all_adam_reconstructions_passed=all(r['adam_reconstruction_passed'] for r in results),
        all_gradient_precision_checks_passed=all(r['gradient_precision_check_passed'] for r in results),
        cpu=max(os.sched_getaffinity(0)),threads=1,cuda_initialized=False,optimizer_updates=0,
        disposable_adam_reconstruction_steps=len(results),elapsed_s=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        vector_workspace_bound_bytes=64*1024**2,
        scope='Observed CUDA before/after losses and deltas are direct evidence. CPU projections are local approximations, restricted by recorded CPU/CUDA reconstruction error; capped direction is counterfactual Euclidean geometry, not a new Adam trajectory.',
        sources={str(p.relative_to(ROOT)):sha(p) for p in [config_path,project/'source-witness.json',study/'receipt.json',reference/'weights.npz',restored/'weights.npz',*(restored/n for n in trace_names),Path(component.__file__),HERE/'analyze.py']})
    write(output,result)
    print('Replay weights exact:',weights_exact,'quality exact:',quality_exact,'CPU Adam:',result['all_adam_reconstructions_passed'],flush=True)
    print([(r['update'],r['cases'][7]['predicted_first_order_loss_change'],r['cases'][7]['actual_loss_change']) for r in results],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','project','study','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    analyze(args.config.resolve(),args.project.resolve(),args.study.resolve(),args.output.resolve())
