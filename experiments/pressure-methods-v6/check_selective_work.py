"""Observe actual CNN input sizes and call counts on a consumed native64³ request."""
import os
from v6_support import ROOT,HERE,BASE,write,sha
from solver_v2 import load,Policy
def run():
    output=BASE/'selective-work-qualification-01.json'
    if output.exists():raise ValueError('Fresh selective-work observation required')
    os.sched_setaffinity(0,{22});state,network=load(ROOT/'runs/pressure-methods-v5/entries/v5-spectrum.json');t=state.torch;t.set_num_threads(1)
    from evaluate_v3 import native
    meta,sample=next((i,s)for i,s in native(state,'development')if i['counts_xyz']==[64]*3);rows=[]
    with t.inference_mode():
        for variant in ['all','first','first2','stagnation','coarse1']:
            policy=Policy(state,network,optimized=True,channels_last=True,variant=variant);sizes=[]
            hook=policy.wrapper.register_forward_pre_hook(lambda module,args:sizes.append(list(args[0].shape)))
            p,w=policy.solve(sample,('observed',variant));policy.commit(w['qualified']);hook.remove()
            if len(sizes)!=w['network_calls']:raise ValueError('Reported CNN calls differ from actual forwards')
            if variant=='first'and len(sizes)>1:raise ValueError('First-only call limit exceeded')
            if variant=='first2'and len(sizes)>2:raise ValueError('Two-call limit exceeded')
            if variant=='coarse1'and(any(s[2:]!=[32]*3 for s in sizes)or not sizes):raise ValueError('Coarse CNN was not evaluated on the actual first child grid')
            if variant!='coarse1'and any(s[2:]!=[64]*3 for s in sizes):raise ValueError('Unexpected local CNN grid')
            rows.append(dict(variant=variant,qualified=w['qualified'],actual_cnn_shapes=sizes,actual_forward_cells=sum(s[2]*s[3]*s[4]for s in sizes),work=w))
    write(output,dict(status='passed',input_identity=meta,rows=rows,source_sha256=sha(__file__),solver_sha256=sha(HERE/'solver_v2.py'),optimizer_updates=0))
    print('Observed actual selective call limits and32³ coarse CNN inputs on native64³',flush=True)
if __name__=='__main__':run()
