"""CPU development probes of independently seeded mixed Neumann modes.

Uses captured model/normalization/checkpoint sources through the existing CPU
loader. Analytical eigenvalue labels are checked against the implemented
operator. This probes sampled error amplification, not global convergence or
full fluid simulation. No optimizer or CUDA context is constructed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import sys

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
import audit


def probe(task):
    torch=audit.TORCH;counts=task['counts_xyz'];shape=counts[::-1]
    reference=torch.zeros((1,1,*shape),dtype=torch.float64);laplacian=torch.zeros_like(reference)
    for frequencies,weight in task['components_xyz']:
        factors=[torch.cos(torch.pi*k*(torch.arange(n,dtype=torch.float64)+.5)/n)
                 for n,k in zip(shape,frequencies[::-1])]
        mode=torch.einsum('z,y,x->zyx',*factors)[None,None]
        eigenvalue=math.fsum(2-2*math.cos(math.pi*k/n) for n,k in zip(counts,frequencies))
        reference=reference+weight*mode;laplacian=laplacian+(weight*eigenvalue)*mode
    reference=reference-reference[:,:,:1,:1,:1].clone()
    gain=task['amplitude_pa']/float(reference.square().mean().sqrt())
    reference=reference*gain;laplacian=laplacian*gain
    sample={'base':torch.zeros_like(reference),'reference':reference,
        'cell_size_m':torch.tensor([task['cell_size_m']],dtype=torch.float64),
        'rho':torch.tensor([1000.],dtype=torch.float64),'dt':torch.tensor([1/60],dtype=torch.float64)}
    sample['rhs']=laplacian*(task['cell_size_m']/1000.)
    sample['native_input_residual']=sample['rhs'].clone()
    ideal=audit.ratio(reference,sample)
    assert ideal<1e-11,'analytical eigenvalues disagree with implemented sealed operator'
    scale=audit.MODEL.scales(audit.NORM)[audit.NORM.get('output_scale','pressure')]
    raw=(reference/scale).float()
    rounded=raw.double()*scale;rounded=rounded-rounded[:,:,:1,:1,:1].clone()
    with torch.no_grad():
        current=sample['base'];trace=[]
        for _ in range(8):
            residual=sample['rhs']-audit.MODEL.full_operator(current,sample['cell_size_m'],sample['rho'])
            current=audit.MODEL.predict_pressure(audit.NETWORK,{**sample,'base':current,'native_input_residual':residual},audit.NORM)
            trace.append(audit.ratio(current,sample))
    return {**task,'trace':trace,'first_relative_residual':trace[0],
        'eight_proposal_relative_residual':trace[-1],'ideal_f64_relative_residual':ideal,
        'rounded_f32_relative_residual':audit.ratio(rounded,sample),
        'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),
        'cuda_initialized':torch.cuda.is_initialized()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifacts','packet','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026101513)
    args=parser.parse_args();sys.path.insert(0,str(args.project.absolute()))
    import data
    _,parents=data.read_view(args.packet.absolute());parent=next(p for p in parents if p['entry']['split']=='train')
    delta=parent['arrays']['reference'][0]-parent['arrays']['base'][0]
    amplitude=float((delta**2).mean()**.5);rng=random.Random(args.seed);tasks=[]
    for counts,h in [([16,16,16],1/16),([24,16,12],1/16),([12,8,10],1/8)]:
        for index in range(12):
            components=[]
            for component in range(3):
                frequencies=[rng.randrange(n) for n in counts]
                if not any(frequencies):frequencies[0]=1
                if index>=6 and component==0:frequencies=[n-1 for n in counts]
                components.append((frequencies,rng.uniform(-1.,1.)))
            for multiplier in (.5,-2.):
                tasks.append({'counts_xyz':counts,'cell_size_m':h,'components_xyz':components,
                    'amplitude_pa':amplitude*multiplier})
    cpus=sorted(os.sched_getaffinity(0));cpus=cpus[1:] or cpus
    args.output.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(tasks)),mp_context=multiprocessing.get_context('spawn'),
        initializer=audit.initialize,initargs=(args.project.absolute(),args.artifacts.absolute(),cpus)) as pool:
        records=list(pool.map(probe,tasks))
    result={'schema':'rheon-mixed-mode-development-probe-v1','seed':args.seed,
        'scope':'72 sampled analytical mixtures; separate development seed; no convergence/CFD qualification claim',
        'source_project':str(args.project),'cpu_restore':str(args.artifacts),
        'source_witness_sha256':hashlib.sha256((args.project/'source-witness.json').read_bytes()).hexdigest(),
        'weights_sha256':hashlib.sha256((args.artifacts/'weights.npz').read_bytes()).hexdigest(),
        'normalization_sha256':hashlib.sha256((args.artifacts/'normalization.json').read_bytes()).hexdigest(),
        'source_sha256':{str(path.relative_to(LAB)):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (Path(__file__).resolve(),Path(audit.__file__).resolve())},
        'total':len(records),'first_proposal_reduces_error':sum(r['trace'][0]<1 for r in records),
        'worst_first_relative_residual':max(r['trace'][0] for r in records),
        'worst_eight_proposal_relative_residual':max(r['trace'][-1] for r in records),
        'cuda_initialized':any(r['cuda_initialized'] for r in records),'cases':records}
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
