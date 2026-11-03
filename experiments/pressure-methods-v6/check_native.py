"""Check every boundary orientation, degenerate axes, large grids and ownership."""
import argparse,os,time
from pathlib import Path
from v6_support import ROOT,HERE,BASE,read,write,sha
from solver_v3 import load
from native_local import NativeLocal
def run(args):
    if args.output.exists():raise ValueError('Fresh check output required')
    os.sched_setaffinity(0,{22});state,network=load(ROOT/'runs/pressure-methods-v5/entries/v5-spectrum.json');t=state.torch;t.set_num_threads(1)
    native=NativeLocal(network,state.normalization,streaming=args.streaming);t.manual_seed(2026101911);rows=[]
    shapes=[(1,1,1),(1,3,5),(7,1,3),(3,5,1),(3,5,7),(8,8,8),(17,25,33),(64,64,64),(128,128,128)]
    if args.large:shapes.append((256,256,256))
    with t.inference_mode():
        for shape in shapes:
            r=t.randn((1,1,*shape),dtype=t.float64);r-=r.mean();h=t.tensor([1/16],dtype=t.float64);rho=t.tensor([1000.],dtype=t.float64)
            sample=dict(base=t.zeros_like(r),native_input_residual=r,cell_size_m=h,rho=rho,dt=t.tensor([1/60],dtype=t.float64))
            # Independent captured graph and explicit feature conversion; this
            # large-grid reference intentionally bypasses its old field budget.
            scale=native.scale;q=((r*(rho/h).reshape(1,1,1,1,1))/scale).to(t.float32)
            tick=time.perf_counter();raw=network(q);expected=raw.to(t.float64)*scale;expected=expected-expected[:,:,:1,:1,:1].clone();torch_s=time.perf_counter()-tick
            tick=time.perf_counter();actual=native(r,h,rho);native_s=time.perf_counter()-tick
            error=float(t.linalg.vector_norm(actual-expected))/max(float(t.linalg.vector_norm(expected)),1e-300)
            if error>2e-6 or not bool(t.isfinite(actual).all())or float(actual[0,0,0,0,0])!=0:raise ValueError('Native full-boundary parity failed: '+str((shape,error)))
            saved=actual.clone();native(r*.5,h,rho)
            if not t.equal(saved,actual):raise ValueError('Native output aliases mutable workspace')
            rows.append(dict(counts_zyx=shape,relative_l2_difference=error,torch_seconds=torch_s,native_seconds=native_s,output_ownership=True))
    bad=t.zeros((1,1,3,3,3),dtype=t.float64);bad[0,0,0,0,0]=float('nan')
    try:native(bad,h,rho)
    except ValueError:pass
    else:raise ValueError('Nonfinite native input accepted')
    native.close();write(args.output,dict(status='passed',rows=rows,large=args.large,streaming=args.streaming,optimizer_updates=0,
        source_sha256=sha(__file__),dependencies={str(p.relative_to(ROOT)):sha(p)for p in [HERE/('local_native_stream.cpp'if args.streaming else'local_native.cpp'),HERE/'native_local.py',BASE/('local-native-stream-build-01/build.json'if args.streaming else'local-native-build-01/build.json')]},
        timing_scope='Exploratory one-shot parity profile; not a final repeated speed claim'))
    print('Qualified native smoother on',len(rows),'full boundary/grid cases',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'native-qualification-01.json');p.add_argument('--large',action='store_true');p.add_argument('--streaming',action='store_true');run(p.parse_args())
