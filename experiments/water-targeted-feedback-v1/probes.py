"""Bounded axis/mode and reconstruction probes on a minimized consumed solve."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

from diagnose import LAB,digest,read,write
sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
import audit
sys.path.insert(0,str(LAB/'experiments/water-stability-learning-v1'))
from learned_iteration import predict_proposal
sys.path.insert(0,str(LAB/'experiments/water-multigrid-learning-v1'))
from mode_probe import probe


def evaluate(task):
    torch=audit.TORCH
    if task['kind']=='mode':
        return {'kind':'mode',**probe(task['sample'])}
    import numpy as np
    with np.load(task['sample'],allow_pickle=False) as arrays:
        sample={k:torch.from_numpy(arrays[k].copy()) for k in arrays.files}
    permutation=task['permutation']
    for k in ('base','native_input_residual'):
        sample[k]=sample[k].permute(0,1,*[v+2 for v in permutation]).contiguous()
    sample['native_input_residual']*=task['gain']
    if task['zero_base']:sample['base'].zero_()
    initial=sample['native_input_residual'];base=sample['base'];pressure=base.clone()
    stages=[]
    with torch.no_grad():
        for step in range(1,33):
            residual=initial-audit.MODEL.full_operator(pressure-base,sample['cell_size_m'],sample['rho'])
            pressure=predict_proposal(audit.MODEL,audit.NETWORK,{**sample,'base':pressure,'native_input_residual':residual},audit.NORM,True)
            remaining=initial-audit.MODEL.full_operator(pressure-base,sample['cell_size_m'],sample['rho'])
            stages.append({'calls':step,'relative_to_initial':audit.vector_norm(remaining)/audit.vector_norm(initial)})
    return {**task,'stages':stages,'cuda_initialized':torch.cuda.is_initialized(),
        'affinity':sorted(os.sched_getaffinity(0))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('models','sample','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();started=time.perf_counter();models=read(args.models)
    tasks=[]
    for perm in ([0,1,2],[1,2,0],[2,0,1]):
        tasks.append({'kind':'actual','sample':str(args.sample.absolute()),'permutation':perm,'gain':1.,'zero_base':False})
    for gain,zero in ((1.,True),(-2.,False),(.25,False)):
        tasks.append({'kind':'actual','sample':str(args.sample.absolute()),'permutation':[0,1,2],'gain':gain,'zero_base':zero})
    for counts in ([16,23,25],[10,7,25]):
        for frequencies in ([1,0,0],[0,1,0],[0,0,1],[1,2,0],[0,1,2],[2,0,1],[3,2,1],[4,4,4]):
            tasks.append({'kind':'mode','sample':{'counts_xyz':counts,'cell_size_m':1/16,
                'components_xyz':[[frequencies,1.]],'amplitude_pa':2.8535734640782584,'population':'targeted'}})
    cpus=sorted(os.sched_getaffinity(0))[1:];rows=[]
    for model in models:
        with ProcessPoolExecutor(max_workers=min(16,len(cpus)),mp_context=multiprocessing.get_context('spawn'),
            initializer=audit.initialize,initargs=(LAB/model['project'],LAB/model['artifacts'],cpus)) as pool:
            records=list(pool.map(evaluate,tasks))
        rows.append({'model':model['name'],'weights_sha256':digest(LAB/model['artifacts']/'weights.npz'),'records':records})
    write(args.output,{'models':rows,'sample_sha256':digest(args.sample),'source_sha256':digest(Path(__file__)),
        'elapsed_seconds':time.perf_counter()-started,'optimizer_updates':0,
        'scope':'consumed-case CPU diagnosis; finite mode probes, not full native qualification or global contraction proof'})
    print({'models':len(rows),'tasks_per_model':len(tasks),'elapsed_seconds':time.perf_counter()-started})


if __name__=='__main__':main()
