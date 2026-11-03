"""Exact CUDA division bits and independent dyadic midpoint controls."""
import argparse
import ctypes as C
import math
import os
import resource
import time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
from support import ROOT, read, write, digest
from cuda_kernel_runtime_v1 import CudaKernels, check
from fused_cuda_generator_v1 import pointer


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only scalar arithmetic evidence required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:
            raise ValueError('Frozen quotient dependency changed: '+p)
    os.sched_setaffinity(0,{reg['host_cpu']})
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.empty(0,device='cuda');kernels=CudaKernels(reg['build'])
    fn=C.c_void_p();check(kernels.driver.cuModuleGetFunction(C.byref(fn),kernels.module,b'integer_quotient_primitive'),'quotient primitive')
    kernels.functions['integer_quotient_primitive']=fn
    args.output.mkdir(parents=True)
    result=dict(status='running',rows=[],rational_quotient_checks=0,rational_midpoint_comparisons=0,
        inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),
        cpu_model_forward=False,learned_model_forward=False,optimizer_updates=0,latency_claim=False,
        expanded_reserved52_references_opened=0,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol))
    def save():write(args.output/'results.json',result)
    def upload(a):return torch.from_numpy(np.ascontiguousarray(a)).to('cuda')
    def fail(label,indices,a,b,r,out,ref,modes,lo,hi,initial):
        ix=np.asarray(indices[:32],dtype=np.int64)
        np.savez_compressed(args.output/'counterexample.npz',label=np.asarray(label),a=a[ix],b=b[ix],reciprocal=r[ix],
                            output=out[ix],reference=ref[ix],modes=modes[ix],low_compare=lo[ix],high_compare=hi[ix],initial=initial[ix])
        raise AssertionError(label)
    rng=np.random.default_rng(reg['seed']);started=time.monotonic()
    try:
        with torch.inference_mode(),np.errstate(over='ignore',invalid='ignore',under='ignore',divide='ignore'):
            for pop in reg['populations']:
                tag,count=pop['name'],pop['count'];checks=0;midchecks=0
                for begin in range(0,count,reg['chunk_size']):
                    if time.monotonic()-started>reg['maximum_wall_seconds']:
                        raise TimeoutError('Finite quotient primitive deadline')
                    n=min(reg['chunk_size'],count-begin);index=np.arange(begin,begin+n)
                    if tag=='arbitrary_normals':
                        a=np.ldexp(rng.uniform(1.,2.,n),rng.integers(-399,400,n))
                        b=np.ldexp(rng.uniform(1.,2.,n),rng.integers(-19,20,n));r=1./b
                    elif tag=='all_six_spacings':
                        b=1./np.asarray(reg['cells_per_metre'],dtype=np.float64)[index%6]
                        a=rng.uniform(0.,128.,n)*b
                        a[index%3==0]=np.nextafter(a[index%3==0],np.inf)
                        a[index%3==1]=np.nextafter(a[index%3==1],-np.inf)
                        r=1./b
                    elif tag=='broad_bits':
                        a=rng.integers(0,2**64,size=n,dtype=np.uint64).view(np.float64)
                        b=rng.integers(0,2**64,size=n,dtype=np.uint64).view(np.float64);r=1./b
                    elif tag=='guards_and_exponents':
                        ae=np.array([0.,-0.,np.nextafter(0.,1.),-np.nextafter(0.,1.),np.finfo(np.float64).tiny,
                            -1.,2.**-400,np.nextafter(2.**-400,0.),np.nextafter(2.**-400,np.inf),
                            2.**400,np.nextafter(2.**400,0.),np.nextafter(2.**400,np.inf),np.finfo(np.float64).max,np.inf,-np.inf,np.nan])
                        be=np.array([0.,-0.,2.**-20,np.nextafter(2.**-20,0.),np.nextafter(2.**-20,np.inf),
                            2.**20,np.nextafter(2.**20,0.),np.nextafter(2.**20,np.inf),-1.,np.finfo(np.float64).tiny,np.inf,-np.inf,np.nan])
                        a=ae[index%len(ae)];b=be[(index//len(ae))%len(be)];r=1./b
                    elif tag=='midpoint_and_binade_neighbors':
                        # Exact Fraction constructs the scaled midpoint before
                        # conversion; nearest and adjacent numerators straddle it.
                        base_count=reg['exact_midpoint_bases_per_chunk']
                        q=np.ldexp(np.ones(base_count),rng.integers(-300,301,base_count))
                        q[1::3]=np.nextafter(q[1::3],0.);q[2::3]=np.nextafter(q[2::3],np.inf)
                        bs=1./np.resize(np.asarray(reg['cells_per_metre'],dtype=np.float64),base_count)
                        aa=np.empty(base_count,dtype=np.float64)
                        for j,(qq,bb) in enumerate(zip(q,bs)):
                            neighbor=np.nextafter(qq,0. if j%2 else np.inf)
                            aa[j]=float((F.from_float(float(qq))+F.from_float(float(neighbor)))*F.from_float(float(bb))/2)
                        a=np.resize(aa,n);b=np.resize(bs,n)
                        a[index%3==0]=np.nextafter(a[index%3==0],0.)
                        a[index%3==2]=np.nextafter(a[index%3==2],np.inf);r=1./b
                    elif tag=='bad_reciprocals':
                        a=np.ldexp(rng.uniform(1.,2.,n),rng.integers(-100,101,n))
                        b=np.ldexp(rng.uniform(1.,2.,n),rng.integers(-10,11,n));r=1./b
                        variants=index%8
                        r[variants==0]*=1.01;r[variants==1]*=.99;r[variants==2]=0.;r[variants==3]=-r[variants==3]
                        r[variants==4]=np.inf;r[variants==5]=np.nan
                        r[variants==6]=np.nextafter(r[variants==6],0.)
                        r[variants==7]=np.nextafter(r[variants==7],np.inf)
                    else:raise ValueError('Unregistered quotient population')
                    inp=[upload(v) for v in (a,b,r)]
                    gpu=[torch.empty(n,dtype=t,device='cuda') for t in
                         (torch.float64,torch.float64,torch.int32,torch.int32,torch.int32,torch.float64)]
                    kernels.launch('integer_quotient_primitive',n,[C.c_longlong(n),*(pointer(v) for v in inp+gpu)])
                    out,ref,modes,lo,hi,initial=[v.cpu().numpy() for v in gpu]
                    bad=np.flatnonzero(out.view(np.uint64)!=ref.view(np.uint64))
                    if len(bad):fail('Original CUDA division bits changed',bad,a,b,r,out,ref,modes,lo,hi,initial)
                    if np.any((modes<0)|(modes>3)):
                        raise AssertionError('Unrecognized quotient decision')
                    remaining=reg['rational_quotient_checks_per_population']-checks
                    valid=np.flatnonzero(np.isfinite(a)&np.isfinite(b)&(a!=0.)&(b!=0.)&np.isfinite(ref))
                    quota=min(remaining,math.ceil(reg['rational_quotient_checks_per_population']/(count/reg['chunk_size'])),len(valid))
                    chosen=valid[np.linspace(0,len(valid)-1,quota,dtype=np.int64)] if quota else []
                    for i in chosen:
                        exact=float(F.from_float(float(a[i]))/F.from_float(float(b[i])))
                        if np.asarray(exact).view(np.uint64)!=out.view(np.uint64)[i]:
                            fail('Independent exact rational quotient mismatch',[i],a,b,r,out,ref,modes,lo,hi,initial)
                    checks+=quota;quotient_take=quota
                    remaining=reg['maximum_midpoint_checks_per_population']-midchecks
                    valid=np.flatnonzero((lo!=9)&(hi!=9))
                    quota=min(remaining,math.ceil(reg['maximum_midpoint_checks_per_population']/(count/reg['chunk_size'])),len(valid))
                    chosen=valid[np.linspace(0,len(valid)-1,quota,dtype=np.int64)] if quota else []
                    for i in chosen:
                        aa,bb,qq=map(F.from_float,map(float,(a[i],b[i],initial[i])))
                        for upper,observed in [(False,lo[i]),(True,hi[i])]:
                            neighbor=F.from_float(float(np.nextafter(initial[i],np.inf if upper else 0.)))
                            boundary=(qq+neighbor)*bb/2
                            exact=int(aa>boundary)-int(aa<boundary)
                            if int(observed)!=exact:
                                fail('Exact dyadic midpoint comparator mismatch',[i],a,b,r,out,ref,modes,lo,hi,initial)
                    midchecks+=quota
                    result['rows'].append(dict(population=tag,begin=begin,samples=n,all_bits_exact=True,
                        initial=int(np.count_nonzero(modes==0)),adjacent=int(np.count_nonzero(modes==1)),zero=int(np.count_nonzero(modes==2)),fallback=int(np.count_nonzero(modes==3)),
                        rational_quotient_checks=quotient_take,
                        midpoint_comparisons=2*len(chosen)))
                    result['rational_quotient_checks']=sum(r['rational_quotient_checks'] for r in result['rows'])
                    result['rational_midpoint_comparisons']=sum(r['midpoint_comparisons'] for r in result['rows'])
                    save()
                if checks!=reg['rational_quotient_checks_per_population']:
                    raise ValueError('Complete independent rational quotient population required')
                result.setdefault('population_checks',[]).append(dict(population=tag,rational_quotient_checks=checks,midpoint_comparisons=2*midchecks))
                print('Integer quotient population',tag,count,'bits and exact rational controls passed',flush=True)
            result['rational_quotient_checks']=sum(r['rational_quotient_checks'] for r in result['population_checks'])
            result['rational_midpoint_comparisons']=sum(r['midpoint_comparisons'] for r in result['population_checks'])
        total=sum(r['samples'] for r in result['rows'])
        if total!=reg['expected_scalar_pairs'] or result['rational_quotient_checks']!=reg['expected_rational_quotient_checks'] or result['rational_midpoint_comparisons']<reg['minimum_rational_midpoint_comparisons']:
            raise ValueError('Exact finite primitive coverage required')
        if any(sum(r[name] for r in result['rows'])==0 for name in ('initial','adjacent','zero','fallback')):
            raise ValueError('Every fast/correction/zero/fallback decision must be exercised')
        if torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']:
            raise ValueError('Primitive memory ceiling exceeded')
        for p,h in reg['source_sha256'].items():
            if digest(ROOT/p)!=h:raise ValueError('Quotient dependency changed during execution: '+p)
        result.update(status='passed_integer_midpoint_quotient_GPU_bits_and_rational_controls',scalar_pairs=total,
            all_F64_bits_equal=True,independent_rational_controls_pass=True,
            decisions={name:sum(r[name] for r in result['rows']) for name in ('initial','adjacent','zero','fallback')},
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_diagnostic_seconds=time.monotonic()-started)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:kernels.close();save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
