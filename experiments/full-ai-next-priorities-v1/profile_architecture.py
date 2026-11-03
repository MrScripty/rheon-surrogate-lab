"""Profile mapped candidates before fitting; these are feasibility measurements."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from study_support import ROOT,read,write,sha,historical_cases,equation_error


def run(args):
    if args.output.exists():raise ValueError('Fresh initial architecture profile required')
    os.sched_setaffinity(0,{23});project=args.project.resolve();sys.path.insert(0,str(project))
    import torch,model,data
    model.configure('cpu');torch.set_num_threads(1);config=read(project/'config.json');model.select(config)
    network=model.PressureCorrectionNet().eval();initial=read(project/'architecture-initialization.json')
    data.helpers().load_weights(network,project/'architecture-initial.npz',initial['inventory'])
    state=SimpleNamespace(torch=torch,model=model,data=data,network=network,normalization=read(project/'fixed-normalization.json'))
    cases=historical_cases(state);rows=[]
    with torch.inference_mode():
        for index in (0,126,127):
            identity,sample=cases[index]
            model.predict_pressure(network,sample,state.normalization)
            durations=[]
            for repeat in range(5):
                tick=time.perf_counter();pressure=model.predict_pressure(network,sample,state.normalization)
                durations.append(time.perf_counter()-tick)
            costs=defaultdict(lambda:dict(seconds=0.,calls=0));hooks=[]
            for name,module in network.named_modules():
                if isinstance(module,torch.nn.Conv3d)or module is network or name=='shape_head':
                    label='fine_network'if module is network else'global_shape_head'if name=='shape_head'else'convolutions'
                    def pre(mod,values):mod._architecture_profile_tick=time.perf_counter()
                    def post(mod,values,out,label=label):
                        costs[label]['seconds']+=time.perf_counter()-mod._architecture_profile_tick
                        costs[label]['calls']+=1
                    hooks.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
            try:model.predict_pressure(network,sample,state.normalization)
            finally:
                for hook in hooks:hook.remove()
            rows.append(dict(**identity,initial_equation_error=equation_error(state,pressure,sample),
                feasibility_repeat_ms=[v*1000 for v in durations],median_ms=statistics.median(durations)*1000,components=dict(costs)))
    write(args.output,dict(status='completed',project_sha256=sha(project/'project.json'),source_sha256=sha(__file__),
        mapped_initialization_sha256=sha(project/'architecture-initialization.json'),parameters=sum(p.numel()for p in network.parameters()),
        rows=rows,optimizer_updates=0,cpu=23,threads=1,
        caveat='Initial mapped weights, three diagnostic requests and potentially concurrent training on another core. Feasibility only; final performance requires the balanced full-population benchmark.'))
    print('Profiled initial architecture',project.name,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
