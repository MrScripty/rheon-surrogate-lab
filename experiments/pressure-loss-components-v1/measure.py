"""Diagnose captured loss gradients without updating a model or optimizer."""
import argparse
import math
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,source_inputs,write

TERMS=['canonical','final','early','full','without_early']


def geometry(paths):
    """Stream float32 vectors; accumulate dot products in float64."""
    import numpy as np
    import torch
    dots=torch.zeros((len(paths),len(paths)),dtype=torch.float64)
    for i,left in enumerate(paths):
        a=torch.from_numpy(np.load(left,allow_pickle=False)).to(torch.float64)
        for j in range(i+1):
            b=torch.from_numpy(np.load(paths[j],allow_pickle=False)).to(torch.float64)
            dots[i,j]=dots[j,i]=a@b;del b
        del a
    norms=dots.diag().sqrt().tolist();dots=dots.tolist()
    cosines=[[dots[i][j]/norms[i]/norms[j]if norms[i]>0 and norms[j]>0 else None for j in range(len(paths))]for i in range(len(paths))]
    if not all(math.isfinite(v)for row in dots for v in row):raise ValueError('Finite gradient geometry required')
    return dict(norms=norms,dot_matrix=dots,cosine_matrix=cosines,
        negative_pairs=[dict(first=i,second=j,cosine=cosines[i][j])for i in range(len(paths))for j in range(i+1,len(paths))if cosines[i][j]is not None and cosines[i][j]<0])


def components(state,sample):
    import analytical
    import fine_tuning
    import physical_residual
    torch=state.torch;pressures=[];spec=fine_tuning.SPEC
    prediction=physical_residual.predict_pressure(state.network,sample,state.normalization,state.model.owner,
        {'activation':'linear','steps':spec['training_passes']},pressures)
    if len(pressures)!=4 or spec['teacher_weight']!=0 or spec['early_weight']!=.05:
        raise ValueError('Captured four-pass zero-teacher early-weight objective required')
    canonical=analytical.loss(prediction,sample,state.model.full_operator)
    initial=sample['native_input_residual'];base=sample['base']
    equations=[torch.log1p(fine_tuning.relative_square(initial-state.model.full_operator(p-base,sample['cell_size_m'],sample['rho']),initial)/spec['relative_equation_target']**2)for p in pressures]
    partial=canonical+equations[-1]
    early=spec['early_weight']*torch.stack(equations[:-1]).mean(0)
    joint=partial+early
    return dict(canonical=canonical,final=equations[-1],early=early),joint.mean()


def measure(config_path,identifier,cpu,output,vectors):
    started=time.perf_counter();config=read(config_path)
    if output.exists()or vectors.exists()or identifier not in config['models']:raise ValueError('Fresh diagnostic outputs and explicit model required')
    os.sched_setaffinity(0,{cpu});sys.path.insert(0,str(ROOT/'experiments/pressure-pass-sweep-v1'))
    import inference
    material_path=ROOT/config['material'];material=read(material_path)
    state=inference.load(material,config['objective_models'][identifier]);torch=state.torch
    import fine_tuning
    import fit_bank
    import analytical
    from policy import canonical,sha as byte_sha
    project=ROOT/state.entry['project'];weights=ROOT/state.entry['restore']/'weights.npz'
    if identifier=='C53':
        weights=ROOT/next(r['path']for r in read(project/'project.json')['data']if r['id']=='initial_weights')
        if sha(weights)!=next(r['weights_sha256']for r in material['models']if r['id']=='C53'):raise ValueError('C53 identity differs')
        fine_tuning.initialize(state.network,weights,state.data.helpers())
    control=ROOT/next(r['project']for r in material['models']if r['id']=='C66')
    objective=read(project/'fine-tuning.json');previous=read(control/'fine-tuning.json');objective.pop('arm');previous.pop('arm')
    witness=read(project/'source-witness.json');old=read(control/'source-witness.json')
    numerical=lambda w:{k:v for k,v in w['sources'].items()if k not in ['config.json','fine-tuning.json']}
    if objective!=previous or numerical(witness)!=numerical(old)or witness['normalization_sha256']!=old['normalization_sha256']:
        raise ValueError('Common captured numerical objective and normalization required')
    inputs=source_inputs(project)
    _,parents=state.data.load(tuple(inputs[k].read_bytes()for k in ['dataset','dataset_extra']),read(inputs['splits']))
    if any(p['entry']['split']!='train'for p in parents):raise ValueError('Train-only scalar calibration required')
    actual=[state.data.sample(p,j)for p in parents for j in range(len(p['frames']))]
    amplitude=analytical.training_correction_rms(actual);samples,cases,residency=fit_bank.build(actual[0],fine_tuning.SPEC['fitting_bank'],amplitude)
    params=list(state.network.named_parameters());parameters=[p for _,p in params];count=sum(p.numel()for p in parameters)
    # Accumulator, full vector, difference and bounded conversion/flatten temporaries.
    workspace=32*count;disk=len(TERMS)*len(samples)*4*count
    if workspace>config['vector_workspace_max_bytes']or disk>config['vector_disk_max_bytes']:raise MemoryError('Bounded live vector workspace and disk required')
    vectors.mkdir(parents=True);before=state.data.helpers().state_inventory(state.network);calls=fine_tuning.STATS['teacher_forwards'];rows=[]
    state.network.train()
    def gradient(loss,retain=False):
        values=torch.autograd.grad(loss,parameters,retain_graph=retain)
        if any(not bool(torch.isfinite(v).all())for v in values):raise ValueError('Finite component gradients required')
        return torch.cat([v.detach().reshape(-1)for v in values])
    def path(index,term):return vectors/(str(index)+'-'+term+'.npy')
    for index,(sample,case)in enumerate(zip(samples,cases,strict=True)):
        defect=float(torch.linalg.vector_norm(state.model.full_operator(sample['reference'],sample['cell_size_m'],sample['rho'])-sample['rhs'])/torch.linalg.vector_norm(sample['rhs']))
        if defect>1e-10:raise ValueError('Independent operator labels differ')
        original=fit_bank.loss(state.network,sample,state.normalization);full_value=float(original.detach())
        full=gradient(original);state.np.save(path(index,'full'),full.numpy());del full,original
        terms,joint_loss=components(state,sample);values={k:float(v.detach())for k,v in terms.items()}
        if not all(math.isfinite(v)for v in values.values())or not math.isclose(sum(values.values()),full_value,rel_tol=config['value_relative_tolerance'],abs_tol=1e-14):raise ValueError('Component values do not sum to captured full objective')
        full=torch.from_numpy(state.np.load(path(index,'full'),allow_pickle=False))
        joint=gradient(joint_loss,retain=True)
        if not torch.equal(joint,full):raise ValueError('Reconstructed joint gradient differs from captured full gradient')
        del full,joint
        without=gradient((terms['canonical']+terms['final']).mean(),retain=True)
        state.np.save(path(index,'without_early'),without.numpy());del without
        accumulator=torch.zeros(count,dtype=torch.float64)
        for term in ['canonical','final','early']:
            vector=gradient(terms[term],retain=term!='early');state.np.save(path(index,term),vector.numpy());accumulator.add_(vector);del vector
        del terms,joint_loss
        full=torch.from_numpy(state.np.load(path(index,'full'),allow_pickle=False));difference=accumulator-full
        full_norm=float(torch.linalg.vector_norm(full.to(torch.float64)));error=float(torch.linalg.vector_norm(difference))/max(full_norm,1e-300)
        if not math.isfinite(error):raise ValueError('Finite separate-gradient additivity error required')
        del accumulator,full,difference
        pair=geometry([path(index,t)for t in TERMS]);denom=pair['norms'][3]**2
        rows.append(dict(case=index,**case,values=values,full_value=full_value,value_sum_error=sum(values.values())-full_value,
            joint_gradient_exact=True,separate_gradient_precision_check_passed=error<=config['gradient_relative_tolerance'],
            gradient_sum_relative_l2=error,geometry=pair,projection_on_full={t:pair['dot_matrix'][j][3]/denom if denom>0 else None for j,t in enumerate(TERMS)},operator_defect=defect))
    across={term:geometry([path(i,term)for i in range(len(samples))])for term in TERMS}
    assert before==state.data.helpers().state_inventory(state.network)
    assert calls==fine_tuning.STATS['teacher_forwards']==0 and not torch.cuda.is_initialized()and torch.get_num_threads()==1
    prior_path=ROOT/config['prior_gradients']/(identifier+'-gradients.json');replay=None
    if prior_path.exists():
        prior=read(prior_path);replay=dict(source=str(prior_path.relative_to(ROOT)),sha256=sha(prior_path))
        if prior['weights_sha256']!=sha(weights)or prior['parameter_order']!=[dict(name=n,shape=list(p.shape),elements=p.numel())for n,p in params]:raise ValueError('Prior full-gradient identity differs')
        if any(not math.isclose(r['full_value'],p['loss'],rel_tol=1e-12,abs_tol=1e-14)for r,p in zip(rows,prior['cases'],strict=True)):raise ValueError('Prior full-loss replay differs')
        if across['full']['dot_matrix']!=prior['dot_matrix']:raise ValueError('Prior full-gradient dot matrix replay differs')
        replay['exact_full_loss_and_dot_matrix']=True
    selected=next(r for r in material['models']if r['id']==identifier)
    selected_config=read(ROOT/selected['project']/'config.json')
    checkpoint=ROOT/selected['restore']/f"checkpoint-{selected_config['training']['updates']:03d}.npz"
    with zipfile.ZipFile(checkpoint)as archive:inventory=archive.namelist()
    allowed={n+'.npy'for n in state.data.helpers().state_inventory(state.network)}
    # Match the entire archive against the network inventory, not just Adam-like names.
    network_only=set(inventory)==allowed
    if not network_only:raise ValueError('Checkpoint inventory needs review before classifying optimizer state')
    sources=[config_path,material_path,weights,project/'config.json',project/'fine-tuning.json',project/'source-witness.json',checkpoint,Path(__file__).resolve()]
    result=dict(status='passed',model=identifier,terms=TERMS,cases=rows,across=across,optimizer_updates=0,
        weights_sha256=sha(weights),inventory_unchanged=True,inventory_sha256=byte_sha(canonical(before)),parameter_count=count,
        amplitude_rms_pa=amplitude,fitting_tensor_bytes=residency,vector_workspace_bound_bytes=workspace,vector_disk_bytes=disk,
        vectors={str(p.relative_to(ROOT)):sha(p)for p in sorted(vectors.glob('*.npy'))},prior_full_gradient_replay=replay,
        optimizer_direction=dict(status='unavailable',reason='Retained checkpoint archive contains network arrays only; exact Adam moments are absent.',checkpoint_arrays=inventory,checkpoint_sha256=sha(checkpoint)),
        cpu=cpu,threads=torch.get_num_threads(),cuda_initialized=False,teacher_forwards=calls,elapsed_s=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        interpretation='Component additivity establishes agreement with the captured objective. Geometry is local Euclidean alignment, not Adam dynamics or proof of training causality.',
        sources={str(p.relative_to(ROOT)):sha(p)for p in sources})
    write(output,result);print(identifier,{t:len(v['negative_pairs'])for t,v in across.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['config','output','vectors']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--cpu',type=int,required=True)
    a=p.parse_args();measure(a.config.resolve(),a.model,a.cpu,a.output.resolve(),a.vectors.resolve())
