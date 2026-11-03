"""Serial complete-application comparison with deferred physical evidence with actual native controls and consumer."""
import argparse
import hashlib
import os
from pathlib import Path
import random
import statistics
from profile_frozen_components import ROOT,read,write,digest
from restored_recurrent_sampler import load_model
from diagnose_causal_composition import inventory,FIELDS
from forecast_application_episode import episode as forecast_episode
from native_application_episode import episode as native_episode
import numpy as np
import torch


class FixedHorizon:
    def __init__(self,model,horizons):self.model=model;self.horizons=horizons
    def predict(self,fields,scene,step,normalization):return self.model.predict(fields,scene,step,self.horizons)


def run(args):
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh matched application comparison required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered application source/evidence changed: '+path)
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True);model=load_model(reg['pair'],reg['library_path']);before=inventory(model)
    cases=[];rows=[];retained={};rng=random.Random(reg['seed'])
    for trajectory in read(ROOT/reg['bank'])['trajectories']:
        if trajectory['identifier']not in reg['identifiers']:continue
        path=ROOT/trajectory['receipt'];receipt=read(path)
        if digest(path)!=trajectory['receipt_sha256']:raise ValueError('Matched application scene changed')
        spec=receipt['states'][0];state=path.parent/spec['path']
        if digest(state)!=spec['sha256']:raise ValueError('Matched application initial state changed')
        with np.load(state,allow_pickle=False)as source:initial={k:source[k].copy()for k in FIELDS}
        cases.append(dict(identifier=trajectory['identifier'],scene=receipt['scene'],initial=initial))
    if len(cases)!=len(reg['identifiers']):raise ValueError('Exact matched application population required')
    try:
        for case in cases:
            for scenario in reg['scenarios']:
                for repeat in range(-1,reg['repeats']):
                    methods=list(reg['methods']);rng.shuffle(methods)
                    for method in methods:
                        steps=reg['warmup_steps']if repeat<0 else reg['steps']
                        directory=args.output/(case['identifier']+'-'+scenario+'-'+str(repeat)+'-'+method)
                        if method.startswith('sampler-k'):
                            horizons=int(method[-1]);result,outputs=forecast_episode(FixedHorizon(model,horizons),{},case['scene'],case['initial'],directory,steps=steps,scenario=scenario)
                        else:result,outputs=native_episode(method,None,case['scene'],case['initial'],directory,steps=steps,scenario=scenario)
                        physical=None;repeat_equal=None
                        if repeat>=0:
                            measured=[dict(step=entry['step'],fields={name:dict(dtype=np.asarray(value.detach().cpu().numpy()if isinstance(value,torch.Tensor)else value).dtype.str,
                                sha256=hashlib.sha256(np.asarray(value.detach().cpu().numpy()if isinstance(value,torch.Tensor)else value).tobytes()).hexdigest())for name,value in entry['fields'].items()})for entry in outputs]
                            key=(case['identifier'],scenario,method)
                            if key not in retained or retained[key]['identity']!=measured:
                                physical_path=directory/'physical-outputs.npz'
                                np.savez_compressed(physical_path,**{f"step{entry['step']}_{name}":np.asarray(value.detach().cpu().numpy()if isinstance(value,torch.Tensor)else value).copy()for entry in outputs for name,value in entry['fields'].items()})
                                physical=dict(path=str(physical_path.relative_to(ROOT)),sha256=digest(physical_path),steps=[entry['step']for entry in outputs],identity=measured)
                                repeat_equal=None if key not in retained else False
                                retained[key]=physical
                            else:physical=retained[key];repeat_equal=True
                        del outputs
                        row=dict(identifier=case['identifier'],method=method,scenario=scenario,repeat=repeat,warmup=repeat<0,result=result,physical_output_evidence=physical,outputs_bitwise_equal_to_retained_repeat=repeat_equal)
                        rows.append(row);write(directory/'result.json',row)
                    write(args.output/'progress.json',dict(status='running',rows=len(rows),last_case=case['identifier'],scenario=scenario,repeat=repeat))
                    print(case['identifier'],scenario,'repeat',repeat,'all six application methods complete',flush=True)
        if inventory(model)!=before or torch.cuda.is_initialized():raise ValueError('Application comparison changed model/device')
        summaries=[]
        for scenario in reg['scenarios']:
            for method in reg['methods']:
                chosen=[r for r in rows if not r['warmup']and r['method']==method and r['scenario']==scenario]
                complete=[r for r in chosen if r['result']['completed_physical_interval']]
                times=[r['result']['application_seconds']*1000/r['result']['physical_steps']for r in complete]
                summaries.append(dict(method=method,scenario=scenario,episodes=len(chosen),completed=len(complete),
                    complete_population=len(complete)==len(chosen),median_application_ms_per_accepted_step=statistics.median(times)if times else None,
                    maximum_episode_mean_ms_per_accepted_step=max(times)if times else None,
                    maximum_complete_model_block_ms=max((1000*t for r in complete for t in r['result'].get('full_block_seconds',[])),default=None)))
        write(args.output/'results.json',dict(status='completed_matched_application_comparison',rows=rows,summaries=summaries,
            source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),cpu=reg['cpu'],threads=1,batch_size=1,
            optimizer_updates=0,model_identity=model.asset_identity,all_native_processes_reaped=True,
            candidate_numerical_advances=0,native_control_acceptance_gates_unchanged=True,
            native_consumer_is_structural_not_physical_acceptance_certification=True,
            full_physical_accuracy_evaluated_separately=True,fresh_holdout_consumed=False))
    finally:model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
