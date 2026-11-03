"""All-field short-block and autonomous rollout evaluation of the trained prototype."""
import argparse
import hashlib
import os
from pathlib import Path
from profile_frozen_components import ROOT,read,write,digest
from restored_recurrent_sampler import load_model
from diagnose_causal_composition import inventory,measure,FIELDS
from recurrent_sampler_native import dimensions
import numpy as np
import torch


def identity(values):
    return {name:dict(dtype=value.detach().numpy().dtype.str,shape=list(value.shape),sha256=hashlib.sha256(value.detach().numpy().tobytes()).hexdigest())for name,value in values.items()}


def run(args):
    if args.output.exists():raise ValueError('Fresh complete trained prototype evaluation required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered trained prototype/evaluation source changed: '+path)
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    norm=read(ROOT/reg['normalization']);bank=read(ROOT/reg['bank']);args.output.mkdir(parents=True)
    model=load_model(reg['pair'],reg['library_path']);before=inventory(model);rows=[];sources={};artifacts=[]
    def native(path,receipt,step):
        spec=receipt['states'][step];state=path.parent/spec['path']
        if digest(state)!=spec['sha256']or spec['step']!=step:raise ValueError('Native complete-state/timestamp evidence changed')
        sources[str(state.relative_to(ROOT))]=digest(state)
        with np.load(state,allow_pickle=False)as archive:return {k:archive[k].copy()for k in (*FIELDS,'rhs')}
    def score(output,truth,scene):
        if set(output)!=set(FIELDS):raise ValueError('Every pressure/carrier field required')
        sizes=dict(pressure=int(np.prod(scene['counts'])),**{name:int(np.prod(shape))for name,shape in zip(('vx','vy','vz','tracer'),dimensions(scene['counts']))})
        for name,value in output.items():
            if value.dtype!=(torch.float64 if name=='pressure'else torch.float32)or list(value.shape)!=[sizes[name]]or not bool(torch.isfinite(value).all()):
                raise ValueError('Complete typed finite physical outputs required')
        result=measure(output,truth,scene,norm)
        result['criterion_pass']={name:value<=reg['maximum_error']for name,value in result['relative'].items()}
        result['all_accuracy_criteria_pass']=all(result['criterion_pass'].values())
        return result
    try:
        with torch.inference_mode():
            for trajectory in bank['trajectories']:
                path=ROOT/trajectory['receipt']
                if digest(path)!=trajectory['receipt_sha256']:raise ValueError('Registered reference receipt changed')
                receipt=read(path);scene=receipt['scene'];sources[str(path.relative_to(ROOT))]=digest(path)
                for horizons in reg['horizons']:
                    starts=reg['starts']if args.mode=='blocks'else[0]
                    for start in starts:
                        initial=native(path,receipt,start);current={k:initial[k]for k in FIELDS};saved={}
                        stop=start+horizons if args.mode=='blocks'else reg['rollout_steps'];step=start
                        while step<stop:
                            outputs=model.predict(current,scene,step,horizons)
                            if len(outputs)!=horizons:raise ValueError('Every intermediate state required')
                            for lead,output in enumerate(outputs,1):
                                actual=step+lead
                                if actual>stop:raise ValueError('Registered endpoint must align with block size')
                                # The candidate never receives the future label.
                                truth=native(path,receipt,actual);metrics=score(output,truth,scene)
                                rows.append(dict(identifier=trajectory['identifier'],counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],
                                    horizons=horizons,start=start,step=actual,lead=lead,metrics=metrics,output_identity=identity(output)))
                                if args.mode=='blocks'or actual in reg['snapshots']:
                                    for key,value in output.items():saved[f'step{actual}_{key}']=value.numpy().copy()
                            current=outputs[-1];step+=horizons
                        artifact=args.output/(trajectory['identifier']+f'-k{horizons}-start{start}.npz')
                        np.savez_compressed(artifact,**saved);artifacts.append(dict(path=artifact.name,sha256=digest(artifact),bytes=artifact.stat().st_size))
                    print(args.mode,trajectory['identifier'],'K',horizons,'complete typed states scored',flush=True)
            if inventory(model)!=before or torch.cuda.is_initialized():raise ValueError('Evaluation changed parameters/device')
        summaries=[]
        for horizons in reg['horizons']:
            for regime in ('small','32cubed'):
                chosen=[row for row in rows if row['horizons']==horizons and(max(row['counts_xyz'])<=16)==(regime=='small')]
                summaries.append(dict(horizons=horizons,regime=regime,complete_states=len(chosen),
                    worst={key:max(row['metrics']['relative'][key]for row in chosen)for key in ('equation','pressure','velocity','tracer')},
                    all_fields_pass=all(row['metrics']['all_accuracy_criteria_pass']for row in chosen)))
        write(args.output/'results.json',dict(status='completed_trained_full_state_'+args.mode,summaries=summaries,rows=rows,
            source_evidence=sources,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),model_identity=model.asset_identity,
            learned_inventory=before,physical_snapshots=artifacts,every_intermediate_state_scored=True,
            every_output_typed_finite_and_hashed=True,truth_used_only_after_causal_prediction=True,
            own_history_without_teacher_resets=args.mode=='rollout',rollout_steps=reg['rollout_steps']if args.mode=='rollout'else None,
            optimizer_updates=0,cuda_initialized=False,all_plan_owners_closed=True,
            fresh_holdout_consumed=reg['fresh_holdout'],latency_claim=False,full_generator_target_pass=False))
    finally:model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mode',choices=('blocks','rollout'),required=True)
    run(parser.parse_args())
