"""Measure C53/C66 gradients under one captured C66 objective; never optimize."""
import argparse
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,source_inputs,write


def geometry(vectors):
    import torch
    dots=torch.zeros((len(vectors),len(vectors)),dtype=torch.float64)
    for i,vector in enumerate(vectors):
        left=vector.to(torch.float64)
        for j in range(i+1):
            value=left@vectors[j].to(torch.float64)
            dots[i,j]=dots[j,i]=value
    norms=dots.diag().sqrt()
    if not bool(torch.isfinite(dots).all())or bool((norms<=0).any()):
        raise ValueError('Finite nonzero gradient vectors required')
    cosines=dots/(norms[:,None]*norms[None,:])
    if not torch.allclose(cosines.diag(),torch.ones_like(norms),atol=1e-12,rtol=0):
        raise ValueError('Gradient self-alignment differs from one')
    return norms,dots,cosines


def measure(config_path,identifier,cpu,output):
    started=time.perf_counter();config=read(config_path)
    if output.exists()or identifier not in config['gradient_models']:
        raise ValueError('Fresh gradient result and explicit checkpoint required')
    os.sched_setaffinity(0,{cpu})
    sys.path.insert(0,str(ROOT/'experiments/pressure-pass-sweep-v1'))
    import inference
    material_path=ROOT/config['baseline_material'];material=read(material_path)
    state=inference.load(material,'C66');torch=state.torch
    import fine_tuning
    import fit_bank
    import analytical
    from policy import canonical,sha as byte_sha
    project=ROOT/state.entry['project']
    if fine_tuning.SPEC['training_passes']!=4 or fine_tuning.SPEC['teacher_weight']!=0:
        raise ValueError('Same captured four-pass zero-teacher objective required')
    if identifier=='C53':
        initial=ROOT/next(r['path']for r in read(project/'project.json')['data']if r['id']=='initial_weights')
        entry=next(m for m in material['models']if m['id']=='C53')
        if sha(initial)!=entry['weights_sha256']:
            raise ValueError('C53 initialization differs from verified baseline')
        fine_tuning.initialize(state.network,initial,state.data.helpers())
        weights_sha256=sha(initial)
    else:weights_sha256=state.entry['weights_sha256']
    inputs=source_inputs(project)
    _,parents=state.data.load(tuple(inputs[k].read_bytes()for k in ['dataset','dataset_extra']),read(inputs['splits']))
    if any(p['entry']['split']!='train'for p in parents):
        raise ValueError('Train-only scalar calibration required')
    actual=[state.data.sample(p,j)for p in parents for j in range(len(p['frames']))]
    amplitude=analytical.training_correction_rms(actual)
    samples,cases,residency=fit_bank.build(actual[0],fine_tuning.SPEC['fitting_bank'],amplitude)
    parameters=list(state.network.named_parameters());count=sum(p.numel()for _,p in parameters)
    vector_bytes=len(samples)*count*4
    if vector_bytes>config['gradient_vector_max_bytes']:
        raise MemoryError('Bounded gradient-vector bank required')
    before=state.data.helpers().state_inventory(state.network)
    teacher_calls=fine_tuning.STATS['teacher_forwards'];vectors=[];rows=[]
    state.network.train()
    for index,(sample,case)in enumerate(zip(samples,cases,strict=True)):
        state.network.zero_grad(set_to_none=True)
        defect=float(torch.linalg.vector_norm(state.model.full_operator(sample['reference'],sample['cell_size_m'],sample['rho'])-sample['rhs'])/torch.linalg.vector_norm(sample['rhs']))
        if defect>1e-10:raise ValueError('Independent operator labels differ')
        loss=fit_bank.loss(state.network,sample,state.normalization)
        if not loss.requires_grad or not bool(torch.isfinite(loss)):
            raise ValueError('Finite differentiable loss required')
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all())for _,p in parameters):
            raise ValueError('Finite gradient for every parameter required')
        vector=torch.cat([p.grad.detach().reshape(-1)for _,p in parameters])
        norm=float(torch.linalg.vector_norm(vector.to(torch.float64)))
        if not math.isfinite(norm)or norm<=0:raise ValueError('Nonzero gradient required')
        vectors.append(vector)
        rows.append(dict(case=index,**case,loss=float(loss.detach()),gradient_l2=norm,
            clip_scale=min(1.,1/norm),clipped_gradient_l2=min(1.,norm),operator_defect=defect))
    norms,dots,cosines=geometry(vectors)
    assert before==state.data.helpers().state_inventory(state.network)
    assert teacher_calls==fine_tuning.STATS['teacher_forwards']
    assert not torch.cuda.is_initialized()and torch.get_num_threads()==1
    negatives=[dict(first=i,second=j,cosine=float(cosines[i,j]))for i in range(len(samples))for j in range(i+1,len(samples))if float(cosines[i,j])<0]
    result=dict(status='passed',model=identifier,objective_checkpoint='C66',optimizer_updates=0,
        weights_sha256=weights_sha256,inventory_unchanged=True,inventory_sha256=byte_sha(canonical(before)),
        cases=rows,dot_matrix=dots.tolist(),cosine_matrix=cosines.tolist(),negative_pairs=negatives,
        first_case_cosines=cosines[0].tolist(),amplitude_rms_pa=amplitude,parameter_count=count,
        parameter_order=[dict(name=n,shape=list(p.shape),elements=p.numel())for n,p in parameters],
        gradient_accumulation_dtype='float64 from float32 parameter gradients',gradient_vector_bytes=vector_bytes,
        fitting_tensor_bytes=residency,teacher_forwards=teacher_calls,cuda_initialized=False,
        cpu=cpu,threads=torch.get_num_threads(),elapsed_s=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        interpretation='Negative cosine is local Euclidean loss-gradient conflict. It does not measure Adam moments or establish the cause of training outcomes.',
        sources={str(p.relative_to(ROOT)):sha(p)for p in [config_path,material_path,project/'config.json',project/'fine-tuning.json',project/'source-witness.json',Path(__file__).resolve()]})
    write(output,result);print(identifier,'negative pairs',len(negatives),'of',len(samples)*(len(samples)-1)//2,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--cpu',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();measure(args.config.resolve(),args.model,args.cpu,args.output.resolve())
