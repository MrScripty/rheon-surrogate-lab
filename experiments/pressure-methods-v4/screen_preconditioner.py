"""Same-request GCR comparison before training the preconditioner objective."""
import argparse
import os
from pathlib import Path
import time
from v4_support import ROOT,load,native,write,sha,summary,pressure_error

def run(args):
    if args.output.exists():raise ValueError('Fresh preconditioner screen required')
    os.sched_setaffinity(0,{23});state,wrapper,parity=load()
    import torch,cases,preconditioned
    from frozen_matrix import run_policy
    branch={k:v.detach()for k,v in state.network.state_dict().items()if k=='coarse_logits'or k.startswith('shape_head.')}
    population=cases.screen(cases.load(state))+native(state)
    rows=[];started=time.perf_counter()
    with torch.inference_mode():
        for identity,sample in population:
            for kind in ['recurrence','learned','multilevel','multilevel-neural']:
                tick=time.perf_counter()
                if kind=='recurrence':pressure,work=run_policy(wrapper,sample,state.normalization,branch,'learned',max_passes=8)
                else:pressure,work=preconditioned.solve(wrapper,sample,state.normalization,branch,kind=kind)
                seconds=time.perf_counter()-tick
                rows.append(dict(**identity,method=kind,seconds=seconds,**work,pressure_relative_l2=pressure_error(pressure,sample)))
            print('GCR screen',identity['counts_xyz'],identity['frame'],flush=True)
    write(args.output,dict(status='completed',entry=state.entry,wrapper_parity=parity,rows=rows,
        summary=[dict(method=kind,**summary([r for r in rows if r['method']==kind]))for kind in ['recurrence','learned','multilevel','multilevel-neural']],
        source_sha256=sha(__file__),preconditioner_source_sha256=sha(Path(__file__).with_name('preconditioned.py')),
        multilevel_source_sha256=sha(Path(__file__).with_name('multilevel.py')),cpu=23,threads=1,optimizer_updates=0,elapsed_s=time.perf_counter()-started,
        scope='Consumed development only. One exploratory timing. Eight-iteration ceiling; all proposals and true residual recomputation count. No training or blind qualification.'))
    print('Completed frozen preconditioner screen',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
