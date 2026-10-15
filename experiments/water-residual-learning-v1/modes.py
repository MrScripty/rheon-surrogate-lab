"""Probe captured correctors on known Neumann modes; no simulation/optimizer.

Manufactured compatible RHS fields isolate spatial modes that a single captured
frame cannot identify. Their amplitude matches the fitted pressure correction.
These are algebraic diagnostics, not cold-start or native physical qualification.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import audit


def mode(task):
    counts,k,h,amplitude=task;torch=audit.TORCH
    factors=[torch.cos(torch.pi*frequency*(torch.arange(n,dtype=torch.float64)+.5)/n) for n,frequency in zip(counts[::-1],k[::-1])]
    reference=torch.einsum('z,y,x->zyx',*factors)[None,None]
    reference=reference-reference[:,:,:1,:1,:1]
    reference=reference*(amplitude/float(reference.square().mean().sqrt()))
    sample={'base':torch.zeros_like(reference),'reference':reference,'cell_size_m':torch.tensor([h],dtype=torch.float64),
        'rho':torch.tensor([1000.],dtype=torch.float64),'dt':torch.tensor([1/60],dtype=torch.float64)}
    sample['rhs']=audit.MODEL.full_operator(reference,sample['cell_size_m'],sample['rho'])
    sample['native_input_residual']=sample['rhs'].clone()
    with torch.no_grad():
        predicted=audit.MODEL.predict_pressure(audit.NETWORK,sample,audit.NORM)
        first=audit.ratio(predicted,sample)
        # Maximum learned correction for a zero input is zero by public contract.
        current=predicted;trace=[first]
        for _ in range(7):
            residual=sample['rhs']-audit.MODEL.full_operator(current,sample['cell_size_m'],sample['rho'])
            current=audit.MODEL.predict_pressure(audit.NETWORK,{**sample,'base':current,'native_input_residual':residual},audit.NORM)
            trace.append(audit.ratio(current,sample))
    return {'counts_xyz':counts,'mode_xyz':k,'cell_size_m':h,'target_pressure_rms_pa':amplitude,
        'first_relative_residual':first,'eight_proposal_relative_residual':trace[-1],'trace':trace,
        'first_proposal_reduces_equation_error':first<1,'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),
        'cuda_initialized':torch.cuda.is_initialized()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifacts','packet','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.project.absolute()))
    import data
    _,parents=data.read_view(args.packet.absolute());parent=next(p for p in parents if p['entry']['split']=='train')
    target=parent['arrays']['reference'][0]-parent['arrays']['base'][0]
    amplitude=float((target**2).mean()**.5)
    tasks=[]
    for counts,h in [([16,16,16],1/16),([24,16,12],1/16),([12,8,10],1/8)]:
        candidates=[(1,0,0),(0,1,0),(0,0,1),(2,1,0),(1,2,3),(4,4,4),(counts[0]-1,0,0),(counts[0]-1,counts[1]-1,counts[2]-1)]
        tasks.extend((counts,k,h,amplitude) for k in candidates)
    cpus=sorted(os.sched_getaffinity(0));cpus=cpus[1:] or cpus
    args.output.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(tasks)),mp_context=multiprocessing.get_context('spawn'),
        initializer=audit.initialize,initargs=(args.project.absolute(),args.artifacts.absolute(),cpus)) as pool:
        records=list(pool.map(mode,tasks))
    result={'schema':'rheon-residual-mode-probe-v1','scope':'manufactured compatible operator modes; no simulation, optimizer or physical qualification',
        'target_pressure_rms_pa':amplitude,'project':str(args.project),'cpu_restore':str(args.artifacts),
        'cases':records,'first_proposal_reduces_error':sum(r['first_proposal_reduces_equation_error'] for r in records),
        'total':len(records),'worst_first_relative_residual':max(r['first_relative_residual'] for r in records),
        'worst_eight_proposal_relative_residual':max(r['eight_proposal_relative_residual'] for r in records),
        'cuda_initialized':any(r['cuda_initialized'] for r in records)}
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
