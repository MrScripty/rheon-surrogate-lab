"""The numerical control equals the original skip when learned weights are zero."""
import copy,os
from v6_support import ROOT,HERE,BASE,write,sha
from solver import load
from fixed_control import FixedJacobi
def run():
    output=BASE/'fixed-control-qualification-01.json'
    if output.exists():raise ValueError('Fresh fixed-control check required')
    os.sched_setaffinity(0,{22});state,network=load(ROOT/'runs/pressure-methods-v5/entries/v5-spectrum.json');t=state.torch;t.set_num_threads(1)
    zero=copy.deepcopy(network);fixed=FixedJacobi();rows=[];t.manual_seed(2026101919)
    with t.inference_mode():
        for p in zero.parameters():p.zero_()
        for shape in [(1,1,1),(1,5,7),(17,13,9),(64,64,64),(128,128,128),(256,256,256)]:
            q=t.randn((1,1,*shape),dtype=t.float32);a=zero(q);b=fixed(q)
            if not t.equal(a,b):raise ValueError('Fixed q/8 does not isolate the existing skip')
            rows.append(dict(counts_zyx=shape,exact_tensor_equality=True))
    if list(fixed.parameters()):raise ValueError('Fixed control unexpectedly has trainable parameters')
    write(output,dict(status='passed',rows=rows,trainable_parameters=0,optimizer_updates=0,source_sha256=sha(__file__),control_sha256=sha(HERE/'fixed_control.py'),
      interpretation='Same f32 input conversion and q/8 skip as the learned graph, with zero learned convolutions. No learned checkpoint is changed.'))
    print('Fixed control exactly isolates the existing skip through full256³',flush=True)
if __name__=='__main__':run()
