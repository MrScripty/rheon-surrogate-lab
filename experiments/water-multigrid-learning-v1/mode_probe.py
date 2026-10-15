"""Independently seeded Neumann-mixture probes at a matched 32-evaluation cap."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import sys

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
import audit
sys.path.insert(0,str(LAB/'experiments/water-stability-learning-v1'))
from learned_iteration import correct,stopping_policy


def probe(task):
    torch=audit.TORCH;counts=task['counts_xyz'];shape=counts[::-1]
    pressure=torch.zeros((1,1,*shape),dtype=torch.float64);laplacian=torch.zeros_like(pressure)
    for frequencies,weight in task['components_xyz']:
        factors=[torch.cos(torch.pi*k*(torch.arange(n,dtype=torch.float64)+.5)/n) for n,k in zip(shape,frequencies[::-1])]
        mode=torch.einsum('z,y,x->zyx',*factors)[None,None]
        eigenvalue=math.fsum(2-2*math.cos(math.pi*k/n) for n,k in zip(counts,frequencies))
        pressure+=weight*mode;laplacian+=(weight*eigenvalue)*mode
    pressure-=pressure[:,:,:1,:1,:1].clone()
    gain=task['amplitude_pa']/float(pressure.square().mean().sqrt());pressure*=gain;laplacian*=gain
    sample={'base':torch.zeros_like(pressure),'reference':pressure,
        'cell_size_m':torch.tensor([task['cell_size_m']],dtype=torch.float64),
        'rho':torch.tensor([1000.],dtype=torch.float64),'dt':torch.tensor([1/60],dtype=torch.float64)}
    sample.update(rhs=laplacian*(task['cell_size_m']/1000.))
    sample['native_input_residual']=sample['rhs'].clone()
    ideal=audit.ratio(pressure,sample)
    if not ideal<1e-11:raise ValueError('independent eigenvalue oracle disagrees with implemented operator')
    policy=stopping_policy({'schema':'rheon-learned-stopping-policy-v1','relative_residual':1e-6,'absolute_residual':1e-12,'predicted_divergence':1e-7})
    with torch.no_grad():
        trace=[];prediction=correct(audit.MODEL,audit.NETWORK,audit.NORM,sample,32,policy,trace,single_evaluation=True)
    return {**task,'trace':trace,'ideal_f64_relative_residual':ideal,
        'first_relative_residual':trace[0]['relative_residual'],'final_relative_residual':audit.ratio(prediction,sample),
        'network_evaluations':trace[-1]['proposals'],'equation_passed':trace[-1]['equation_passed'],
        'affinity':sorted(os.sched_getaffinity(0)),'cuda_initialized':torch.cuda.is_initialized()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifacts','packet','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,default=2026101517)
    args=p.parse_args();sys.path.insert(0,str(args.project.absolute()))
    import data
    _,parents=data.read_view(args.packet);parent=next(p for p in parents if p['entry']['split']=='train')
    delta=parent['arrays']['reference'][0]-parent['arrays']['base'][0];amplitude=float((delta**2).mean()**.5)
    rng=random.Random(args.seed);tasks=[]
    for counts in ([16]*3,[24]*3,[32]*3,[40]*3,[24,16,12]):
        for index in range(12):
            components=[]
            for component in range(3):
                # Low-frequency and arbitrary-frequency populations are separate.
                frequencies=[rng.randrange(min(n,4) if index<6 else n) for n in counts]
                if not any(frequencies):frequencies[0]=1
                components.append((frequencies,rng.uniform(-1,1)))
            for multiplier in (.5,-2.):tasks.append({'counts_xyz':counts,'cell_size_m':1/16,'population':'low' if index<6 else 'broad',
                'components_xyz':components,'amplitude_pa':amplitude*multiplier})
    cpus=sorted(os.sched_getaffinity(0));cpus=cpus[1:] or cpus;args.output.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(tasks)),mp_context=multiprocessing.get_context('spawn'),
        initializer=audit.initialize,initargs=(args.project.absolute(),args.artifacts.absolute(),cpus)) as pool:
        records=list(pool.map(probe,tasks))
    groups=[]
    for counts in ([16]*3,[24]*3,[32]*3,[40]*3,[24,16,12]):
        for population in ('low','broad'):
            rows=[r for r in records if r['counts_xyz']==counts and r['population']==population]
            groups.append({'counts_xyz':counts,'population':population,'cases':len(rows),'equation_passed':sum(r['equation_passed'] for r in rows),
                'mean_network_evaluations':statistics.mean(r['network_evaluations'] for r in rows),
                'worst_first_relative_residual':max(r['first_relative_residual'] for r in rows),
                'worst_final_relative_residual':max(r['final_relative_residual'] for r in rows)})
    result={'schema':'rheon-multigrid-mixed-mode-report-v1','seed':args.seed,'mixtures':60,'signed_cases':len(records),'cases':records,'aggregate':groups,
        'weights_sha256':hashlib.sha256((args.artifacts/'weights.npz').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cuda_initialized':any(r['cuda_initialized'] for r in records),
        'used_cpus':sorted({c for r in records for c in r['affinity']}),
        'scope':'60 analytical development mixtures at two signed amplitudes (120 cases), separate seed; not CFD simulations, no global convergence or native physical qualification claim'}
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
