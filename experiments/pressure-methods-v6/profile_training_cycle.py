"""Reproduce excess backward work at the actual frozen-hierarchy seam."""
import argparse,os,time,statistics
from pathlib import Path
from v6_support import ROOT,HERE,BASE,write,sha
def run(a):
    if a.output.exists():raise ValueError('Fresh diagnostic required')
    os.sched_setaffinity(0,{22})
    import torch,multilevel
    torch.set_num_threads(1);torch.manual_seed(2026101916)
    module=multilevel
    if a.module!='original':module=__import__(a.module)
    h=torch.tensor([.0625],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
    r=torch.randn((1,1,a.size,a.size,a.size),dtype=torch.float64);r-=r.mean();r.requires_grad_(True)
    gradient=torch.randn_like(r);rows=[]
    for trial in range(4):
        r.grad=None;tick=time.perf_counter();d=module.correction(r,h,rho);forward=time.perf_counter()-tick
        tick=time.perf_counter();d.backward(gradient);backward=time.perf_counter()-tick
        if not bool(torch.isfinite(r.grad).all()):raise ValueError('Nonfinite frozen-hierarchy adjoint')
        if trial:rows.append(dict(forward_seconds=forward,backward_seconds=backward))
    ratio=statistics.median(x['backward_seconds']for x in rows)/statistics.median(x['forward_seconds']for x in rows)
    write(a.output,dict(status='passed'if ratio<=2 else'excess_backward_work',module=a.module,size=a.size,rows=rows,backward_forward_ratio=ratio,
      source_sha256=sha(__file__),operator_source_sha256=sha(module.__file__),cpu=22,threads=1,optimizer_updates=0,
      criterion='A frozen linear symmetric cycle has a one-cycle adjoint plus gauge transpose; allow twice forward time. This is a hardware-specific diagnostic, not final inference timing.'))
    print('Frozen-cycle backward/forward ratio:',round(ratio,3),'limit2',flush=True)
    if a.require and ratio>2:raise AssertionError('Frozen hierarchy backward exceeds two forward cycles')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--module',default='original');p.add_argument('--size',type=int,default=64);p.add_argument('--output',type=Path,required=True);p.add_argument('--require',action='store_true');run(p.parse_args())
