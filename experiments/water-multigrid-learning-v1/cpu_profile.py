"""Bounded CPU-only mechanism profile; separate from whole-simulation timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
import audit
sys.path.insert(0,str(LAB/'experiments/water-stability-learning-v1'))
from learned_iteration import correct,stopping_policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifacts','packet','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--axis-count',type=int,choices=(16,24,32,40),required=True)
    p.add_argument('--threads',type=int,choices=(1,4,8),required=True)
    p.add_argument('--layout',choices=('contiguous','channels-last-3d'),default='contiguous')
    args=p.parse_args();cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<args.threads:raise ValueError('explicit affinity must include requested thread count')
    audit.initialize(args.project.absolute(),args.artifacts.absolute(),cpus)
    # The ordinary checkpoint loader deliberately pins one core. Mechanism
    # profiling borrows an explicitly supplied core partition for this process.
    os.sched_setaffinity(0,set(cpus));torch=audit.TORCH;torch.set_num_threads(args.threads)
    before={name:value.detach().contiguous().numpy().tobytes() for name,value in audit.NETWORK.state_dict().items()}
    layout_hook=None
    if args.layout=='channels-last-3d':
        audit.NETWORK.to(memory_format=torch.channels_last_3d)
        layout_hook=audit.NETWORK.register_forward_pre_hook(lambda _module,inputs:(inputs[0].to(memory_format=torch.channels_last_3d),))
    if any(value.detach().contiguous().numpy().tobytes()!=before[name] for name,value in audit.NETWORK.state_dict().items()):
        raise ValueError('layout changed learned tensor values')
    _,parents=audit.DATA.read_view(args.packet);parent=next(p for p in parents if p['entry']['split']=='train')
    template=audit.DATA.sample(parent,0);n=args.axis_count
    import analytical
    reference=torch.zeros((1,1,n,n,n),dtype=torch.float64)
    template={**template,'base':reference,'cell_size_m':torch.tensor([1/16],dtype=torch.float64)}
    amplitude=float((parent['arrays']['reference'][0]-parent['arrays']['base'][0]).std())
    sample=analytical.make_sample(template,[((1,2,3),1.),((n-1,n-1,n-1),.25)],amplitude)
    policy=stopping_policy(json.loads((LAB/'experiments/water-inference-budget-v1/configs/adaptive.json').read_text()))
    timings=[];network_times=[];operator_times=[];active={}
    def before(*_):active['network']=time.perf_counter()
    def after(*_):network_times.append(time.perf_counter()-active['network'])
    hooks=[audit.NETWORK.register_forward_pre_hook(before),audit.NETWORK.register_forward_hook(after)]
    original=audit.MODEL.full_operator
    def operator(*args,**kwargs):
        start=time.perf_counter()
        try:return original(*args,**kwargs)
        finally:operator_times.append(time.perf_counter()-start)
    audit.MODEL.full_operator=operator
    try:
        with torch.no_grad():
            correct(audit.MODEL,audit.NETWORK,audit.NORM,sample,32,policy,single_evaluation=True)
            network_times.clear();operator_times.clear()
            for _ in range(3):
                trace=[];start=time.perf_counter()
                pressure=correct(audit.MODEL,audit.NETWORK,audit.NORM,sample,32,policy,trace,single_evaluation=True)
                timings.append({'seconds':time.perf_counter()-start,'network_evaluations':trace[-1]['proposals'],
                    'relative_residual':trace[-1]['relative_residual'],'equation_passed':trace[-1]['equation_passed']})
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                correct(audit.MODEL,audit.NETWORK,audit.NORM,sample,32,policy,single_evaluation=True)
        events=profiler.key_averages()
    finally:
        audit.MODEL.full_operator=original
        for hook in hooks:hook.remove()
        if layout_hook is not None:layout_hook.remove()
    result={'schema':'rheon-pressure-cpu-mechanism-profile-v1','counts_xyz':[n]*3,'cell_size_m':1/16,
        'threads':args.threads,'layout':args.layout,'parameter_values_preserved':True,'affinity':cpus,'cuda_initialized':torch.cuda.is_initialized(),
        'weights_sha256':hashlib.sha256((args.artifacts/'weights.npz').read_bytes()).hexdigest(),
        'python':sys.executable,'torch':torch.__version__,'timings':timings,
        'median_seconds':statistics.median(r['seconds'] for r in timings),
        'network_forward_seconds_including_profile':sum(network_times),'operator_seconds_including_profile':sum(operator_times),
        'top_self_cpu':[{'name':e.key,'self_cpu_us':e.self_cpu_time_total,'calls':e.count} for e in sorted(events,key=lambda e:e.self_cpu_time_total,reverse=True)[:20]],
        'scope':'manufactured Neumann mixture; 1 warm-up, 3 unprofiled repeats, 1 instrumented pass; no IPC/native physics or CFD speedup claim'}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('counts_xyz','threads','median_seconds','timings')}),flush=True)


if __name__=='__main__':main()
