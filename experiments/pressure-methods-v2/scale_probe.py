"""Bounded manufactured-pressure scaling; no native-data or coupled-flow claim."""
import argparse
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
             str(ROOT/'experiments/pressure-pass-sweep-v1'),str(ROOT/'experiments/pressure-methods-v1')]
from common import read,write,sha

def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024

def sample(n,cpm,torch):
    shape=(1,1,n,n,n);pressure=torch.zeros(shape,dtype=torch.float64);q=torch.zeros_like(pressure)
    components=[((1,0,0),1.),((0,2,0),.7),((0,0,3),.5),((2,1,1),.4),((n//3,n//4,n//5),.1)]
    for mode,weight in components:
        axes=[torch.cos(math.pi/n*k*(torch.arange(n,dtype=torch.float64)+.5)) for k in mode]
        field=axes[0][:,None,None]*axes[1][None,:,None]*axes[2][None,None,:]
        eigenvalue=sum(4*math.sin(math.pi*k/(2*n))**2 for k in mode)
        pressure+=weight*field;q+=weight*eigenvalue*field
    amplitude=read(ROOT/'runs/pressure-methods-v1/manufactured-bank/bank.json')['amplitude_rms_pa']
    multiplier=amplitude/float(pressure.square().mean().sqrt());pressure*=multiplier;q*=multiplier
    pressure-=pressure[:,:,:1,:1,:1].clone()
    h=torch.tensor([1/cpm],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
    rhs=q*(h/rho).reshape(1,1,1,1,1)
    return dict(base=torch.zeros_like(rhs),rhs=rhs,native_input_residual=rhs,
        reference=pressure,cell_size_m=h,rho=rho,dt=torch.tensor([1/60],dtype=torch.float64)),components

def probe(args):
    if args.output.exists():raise ValueError('Fresh scale probe required')
    os.sched_setaffinity(0,{args.cpu})
    if available()<12*1024**3+args.size**3*512:raise MemoryError('Host reserve insufficient for declared scale probe')
    import inference
    material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    if args.evaluation:
        evaluated=read(args.evaluation)
        if evaluated['status']!='completed':raise ValueError('Completed evaluated checkpoint required')
        entry=evaluated['entry'];material=dict(material,models=[entry]);name=entry['id']
    else:name='C53'
    tick=time.perf_counter();state=inference.load(material,name);loading=time.perf_counter()-tick
    import scalable,numerics,numerical
    torch=state.torch
    qualifier_start=time.perf_counter()
    wrapper,qualification=scalable.qualify(state)
    recurrence=scalable.qualify_recurrence(state,wrapper,args.kind,nodes=args.nodes,passes=args.passes) if args.passes else []
    qualifier_seconds=time.perf_counter()-qualifier_start
    args.output.mkdir(parents=True)
    work,components=sample(args.size,args.cells_per_metre,torch)
    norm=max(float(torch.linalg.vector_norm(work['rhs'])),1e-300)
    action=lambda pressure:numerics.fine_action(pressure,work['cell_size_m'],work['rho'])
    defect=float(torch.linalg.vector_norm(action(work['reference'])-work['rhs']))/norm
    if defect>2e-10:raise ValueError('Independent analytic label disagrees with fine operator')
    rows=[]
    with torch.no_grad():
        if not args.skip_controls:
            tick=time.perf_counter()
            reference=work['base']+numerics.correction(work['native_input_residual'],work['cell_size_m'],work['rho'],None)
            cold_s=time.perf_counter()-tick
            error=float(torch.linalg.vector_norm(work['rhs']-action(reference)))/norm
            tick=time.perf_counter()
            repeated=work['base']+numerics.correction(work['native_input_residual'],work['cell_size_m'],work['rho'],None)
            warm_s=time.perf_counter()-tick
            if not torch.equal(reference,repeated):raise ValueError('Warm spectral output changed')
            rows.append(dict(policy='spectral',relative_residual=error,cold_seconds=cold_s,warm_seconds=warm_s,
                qualified=error<=1e-6))
            del reference,repeated
            # Compare a numerical coarse/fine iteration on a common budget.
            tick=time.perf_counter();pressure,iterations=numerical.pcg(work,nodes=6,max_iterations=256)
            error=float(torch.linalg.vector_norm(work['rhs']-action(pressure)))/norm
            rows.append(dict(policy='numerical-two-level-pcg',relative_residual=error,iterations=iterations,
                seconds=time.perf_counter()-tick,qualified=error<=1e-6))
            del pressure
        if args.passes:
            if available()<12*1024**3+args.size**3*512:raise MemoryError('Host reserve changed before neural inference')
            tick=time.perf_counter()
            pressure,trace=scalable.predict(wrapper,work,state.normalization,passes=args.passes,kind=args.kind,nodes=args.nodes)
            predict_seconds=time.perf_counter()-tick
            centered=pressure-work['reference'];centered-=centered.mean()
            reference=work['reference']-work['reference'].mean()
            rows.append(dict(policy=args.kind,passes=args.passes,trace=trace,seconds=predict_seconds,
                relative_residual=trace[-1]['relative_residual'],qualified=trace[-1]['relative_residual']<=1e-6,
                mean_free_pressure_relative_l2=float(torch.linalg.vector_norm(centered))/max(float(torch.linalg.vector_norm(reference)),1e-300)))
    result=dict(status='completed',grid_dimensions=[args.size]*3,cells_per_metre=args.cells_per_metre,
        physical_extent_m=[args.size/args.cells_per_metre]*3,rows=rows,components_zyx=components,
        sample_scope='Manufactured pressure equations only, independent analytic labels; not native captures or coupled simulations',
        source_sha256=sha(__file__),scalable_source_sha256=sha(Path(__file__).with_name('scalable.py')),
        numerical_source_sha256=sha(Path(__file__).with_name('numerical.py')),entry=state.entry,
        wrapper_qualification=qualification,recurrence_qualification=recurrence,qualifier_seconds=qualifier_seconds,
        nodes=args.nodes,analytic_label_defect=defect,model_loading_seconds=loading,
        peak_worker_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,host_available_bytes_after=available(),
        old_adapter_ceiling_changed=False,optimizer_updates=0,threads=1,cpu=args.cpu,
        timing_scope='Prepared physical tensors; numerical cold/warm separated, neural stages include true residual evaluation; concurrent campaign work may affect timings')
    write(args.output/'results.json',result)
    print('Completed scale probe',args.size,args.cells_per_metre,[(r['policy'],r['qualified']) for r in rows],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,choices=[16,32,64,128,256],required=True)
    parser.add_argument('--cells-per-metre',type=int,choices=[8,16,32,64],required=True)
    parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--skip-controls',action='store_true')
    parser.add_argument('--nodes',type=int,choices=[4,6],default=6)
    parser.add_argument('--kind',choices=['fine','numerical-coarse','learned-coarse'],default='fine')
    parser.add_argument('--passes',type=int,choices=range(0,17),default=8)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=21)
    args=parser.parse_args()
    if args.evaluation:args.evaluation=args.evaluation.resolve()
    args.output=args.output.resolve();probe(args)
