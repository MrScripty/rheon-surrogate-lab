"""Repeat a captured learned corrector with freshly computed residuals on CPU.

This inference policy adds no native solver/refinement. It preserves the supplied
previous pressure, fixed cubic operator and checkpoint. Count every proposal;
each proposal already contains the checkpoint's configured recurrent passes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys

LAB=Path(__file__).resolve().parents[2]
MAX_NETWORK_EVALUATIONS=32


def validate_budget(proposals,width):
    if type(width) is not int or not 1<=width<=4 or type(proposals) is not int or not 1<=proposals or proposals*width>MAX_NETWORK_EVALUATIONS:
        raise ValueError('one to eight proposals for four-pass networks; at most 32 network evaluations')


def stopping_policy(value):
    expected={'schema','relative_residual','absolute_residual','predicted_divergence'}
    if type(value) is not dict or set(value)!=expected or value['schema']!='rheon-learned-stopping-policy-v1':
        raise ValueError('closed learned stopping policy required')
    for name,maximum in [('relative_residual',1e-6),('absolute_residual',1e-12),('predicted_divergence',1e-7)]:
        x=value[name]
        if type(x) not in (int,float) or not math.isfinite(x) or not 0<x<=maximum:
            raise ValueError('stopping policy cannot relax existing physical limits')
    return value


def equation_quality(model,pressure,sample,rhs,policy):
    import torch
    error=rhs-model.full_operator(pressure,sample['cell_size_m'],sample['rho'])
    norm=float(torch.linalg.vector_norm(error));rhs_norm=float(torch.linalg.vector_norm(rhs))
    predicted=float(error.abs().amax()*sample['dt'][0]/sample['cell_size_m'][0]**3)
    finite=all(math.isfinite(x) for x in (norm,rhs_norm,predicted))
    return {'residual_l2':norm,'rhs_l2':rhs_norm,'relative_residual':norm/max(rhs_norm,1e-30),
        'predicted_divergence':predicted,'equation_passed':finite
        and norm<=max(policy['absolute_residual'],policy['relative_residual']*rhs_norm)
        and predicted<=policy['predicted_divergence']}


def correct(model,network,normalization,sample,budget,stopping=None,trace=None):
    validate_budget(budget,model.CORRECTION['steps'])
    initial=sample['base'];pressure=initial
    if stopping is not None:
        stopping_policy(stopping)
        if sample['base'].shape[0]!=1:raise ValueError('adaptive stopping requires one physical sample')
        rhs=sample['native_input_residual']+model.full_operator(initial,sample['cell_size_m'],sample['rho'])
        quality=equation_quality(model,pressure,sample,rhs,stopping)
        if quality['equation_passed']:
            if trace is not None:trace.append({'proposals':0,**quality})
            return pressure
    for index in range(budget):
        residual=sample['native_input_residual'] if index==0 else sample['native_input_residual']-model.full_operator(pressure-initial,sample['cell_size_m'],sample['rho'])
        pressure=model.predict_pressure(network,{**sample,'base':pressure,'native_input_residual':residual},normalization)
        if stopping is not None:
            quality=equation_quality(model,pressure,sample,rhs,stopping)
            if trace is not None:trace.append({'proposals':index+1,**quality})
            if quality['equation_passed']:break
        elif trace is not None:trace.append({'proposals':index+1})
    return pressure


def predict(task):
    import audit
    parent,slot,case,budget,stopping=task
    sample=audit.DATA.sample(parent,slot,labels=False)
    with audit.TORCH.no_grad():
        trace=[];pressure=correct(audit.MODEL,audit.NETWORK,audit.NORM,sample,budget,stopping,trace)
    return case,pressure[0,0].numpy().copy(),os.getpid(),sorted(os.sched_getaffinity(0)),audit.TORCH.cuda.is_initialized(),trace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifacts','packet','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--population',choices=('train','validation'),required=True)
    parser.add_argument('--proposals',type=int,choices=range(1,33),required=True)
    parser.add_argument('--adaptive-profile',type=Path)
    args=parser.parse_args()
    stopping=stopping_policy(json.loads(args.adaptive_profile.read_text())) if args.adaptive_profile else None
    sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
    import audit
    sys.path.insert(0,str(args.project.absolute()))
    import data
    config=json.loads((args.project/'config.json').read_text())
    validate_budget(args.proposals,config['correction']['steps'])
    import numpy as np
    reservation,parents=data.read_view(args.packet.absolute())
    cases=[{'parent':i,'frame':slot,'native_frame':frame['index'],'trajectory_id':parent['entry']['trajectory_id'],
        'split':args.population,'counts_xyz':parent['counts_xyz'],'cell_size_m':parent['cell_size_m']}
        for i,parent in enumerate(parents) if parent['entry']['split']==args.population for slot,frame in enumerate(parent['frames'])]
    if not cases:raise ValueError('nonempty selected population required')
    tasks=[(parents[c['parent']],c['frame'],c,args.proposals,stopping) for c in cases]
    cpus=sorted(os.sched_getaffinity(0));cpus=cpus[1:] or cpus
    args.output.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(tasks)),mp_context=multiprocessing.get_context('spawn'),
        initializer=audit.initialize,initargs=(args.project.absolute(),args.artifacts.absolute(),cpus)) as pool:
        records=list(pool.map(predict,tasks))
    arrays={f'case_{i:03d}':r[1] for i,r in enumerate(records)}
    with (args.output/'predictions.npz').open('xb') as stream:np.savez(stream,**arrays)
    source=json.loads((args.artifacts/'receipt.json').read_text())
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    receipt={'schema':'rheon-learned-iteration-diagnostic-v1','optimizer_updates':0,
        'scope':'local CPU diagnostic from a managed restore; not a managed inference Run',
        'source_cpu_restore':str(args.artifacts.absolute()),'source_cpu_restore_sha256':digest(args.artifacts/'receipt.json'),
        'source_witness_sha256':digest(args.project/'source-witness.json'),
        'weights_sha256':source['outputs']['weights.npz']['sha256'],
        'normalization_sha256':source['outputs']['normalization.json']['sha256'],
        'packet_experiment_sha256':hashlib.sha256(data.canonical(reservation)).hexdigest(),
        'cases':cases,'prediction_inventory':data.inventory(arrays),'predictions_sha256':digest(args.output/'predictions.npz'),
        'runtime':{'device':'cpu','cuda_initialized':any(r[4] for r in records)},
        'inference_policy':{'name':'fixed-repeated-learned-correction','proposals':args.proposals,
            'nn_passes_per_pressure_solve':args.proposals*config['correction']['steps'],
            'native_pressure_refinement':False,'cpu_physical_arithmetic':'unchanged float64 operator/base accumulation'},
        'used_cpus':sorted({cpu for r in records for cpu in r[3]}),
        'source_sha256':{str(Path(__file__).relative_to(LAB)):digest(Path(__file__)),
            str(Path(audit.__file__).relative_to(LAB)):digest(Path(audit.__file__))}}
    if stopping is not None:
        receipt['inference_policy'].update(name='adaptive-learned-correction',stopping=stopping,
            scope='equation/predicted-divergence stopping only; actual native physical checks remain required')
        receipt['case_iterations']=[{'case':r[0],'trace':r[5],'proposals':r[5][-1]['proposals'],
            'network_passes':r[5][-1]['proposals']*config['correction']['steps']} for r in records]
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cases':len(cases),'policy':receipt['inference_policy'],'cuda_initialized':receipt['runtime']['cuda_initialized']}),flush=True)


if __name__=='__main__':main()
