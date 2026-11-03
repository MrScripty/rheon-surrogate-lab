"""Bitwise scalar division controls independent of any learned forward pass."""
import argparse,ctypes as C,hashlib,math,os,struct
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
from support import ROOT,read,write,digest
from cuda_kernel_runtime_v1 import CudaKernels,check


def run(args):
    if args.output.exists():raise ValueError('Fresh primitive evidence required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('GPU-only primitive controls required')
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen primitive source changed: '+p)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.empty(0,device='cuda');kernels=CudaKernels(reg['build']);entry=C.c_void_p()
    check(kernels.driver.cuModuleGetFunction(C.byref(entry),kernels.module,b'qualify_shared_reciprocal'),'get primitive entry')
    kernels.functions['qualify_shared_reciprocal']=entry;generator=np.random.default_rng(reg['seed'])
    args.output.mkdir(parents=True);rows=[];rational_count=0;rational_fail=0;reference_rational_fail=0
    def normal(n,lo,hi,signed=True):
        exponent=generator.integers(lo+1023,hi+1024,size=n,dtype=np.uint64)
        mantissa=generator.integers(0,1<<52,size=n,dtype=np.uint64)
        sign=generator.integers(0,2,size=n,dtype=np.uint64)<<np.uint64(63) if signed else np.zeros(n,dtype=np.uint64)
        return (sign|(exponent<<np.uint64(52))|mantissa).view(np.float64)
    def verify(label,aa,bb):
        nonlocal rational_count,rational_fail,reference_rational_fail
        aa=np.asarray(aa,dtype=np.float64).reshape(-1);bb=np.broadcast_to(np.asarray(bb,dtype=np.float64),aa.shape).copy()
        for begin in range(0,len(aa),reg['maximum_chunk_elements']):
            a=aa[begin:begin+reg['maximum_chunk_elements']].copy();b=bb[begin:begin+reg['maximum_chunk_elements']].copy()
            with np.errstate(all='ignore'):r=np.divide(np.float64(1.),b)
            tensors=[torch.from_numpy(v).to('cuda') for v in [a,b,r]]
            output=[torch.empty(len(a),dtype=torch.float64,device='cuda') for _ in range(4)]
            guard=torch.empty(len(a),dtype=torch.int32,device='cuda');ptr=lambda v:C.c_void_p(v.data_ptr())
            params=[C.c_longlong(len(a)),*(ptr(v) for v in tensors),*(ptr(v) for v in output),ptr(guard)]
            kernels.launch('qualify_shared_reciprocal',len(a),params)
            ref,got,first,rref=[v.cpu().numpy() for v in output];mask=guard.cpu().numpy().astype(bool)
            different=got.view(np.uint64)!=ref.view(np.uint64)
            rbad=(r.view(np.uint64)!=rref.view(np.uint64))&mask
            indices=np.flatnonzero(mask)
            if len(indices):
                selected=indices[np.linspace(0,len(indices)-1,min(reg['rational_samples_per_chunk'],len(indices)),dtype=np.int64)]
                for i in selected:
                    exact=Fraction(float(a[i]))/Fraction(float(b[i]));numerator=abs(exact.numerator);denominator=exact.denominator
                    e=numerator.bit_length()-denominator.bit_length()
                    if (numerator<denominator*(1<<e)) if e>=0 else (numerator*(1<<(-e))<denominator):e-=1
                    ulp=Fraction(2)**(e-52)
                    rational_count+=1
                    rational_fail+=int(abs(Fraction(float(first[i]))-exact)>ulp)
                    reference_rational_fail+=int(struct.pack('<d',float(exact))!=struct.pack('<d',float(ref[i])))
            mismatch=np.flatnonzero(different)[:16]
            rows.append(dict(label=label,offset=begin,elements=len(a),guarded_elements=int(mask.sum()),fallback_elements=int((~mask).sum()),
                differing_output_bits=int(different.sum()),provided_reciprocal_different_bits_on_fast_domain=int(rbad.sum()),
                input_sha256=hashlib.sha256(a.tobytes()+b.tobytes()+r.tobytes()).hexdigest(),
                mismatches=[dict(a=float(a[i]).hex(),b=float(b[i]).hex(),r=float(r[i]).hex(),reference_bits=hex(int(ref.view(np.uint64)[i])),actual_bits=hex(int(got.view(np.uint64)[i]))) for i in mismatch]))
            print('Scalar precision',label,begin,'elements',len(a),'different',int(different.sum()),flush=True)
    try:
        with torch.inference_mode():
            for density in reg['cells_per_metre']:
                h=1./density
                verify('physical_uniform_cpm'+str(density),generator.uniform(-1024*h,1024*h,reg['random_per_density']),h)
                positions=(np.arange(0,1025,dtype=np.float64)[:,None]+np.array([0.,.5]))*h
                positions=np.concatenate([positions.reshape(-1),np.nextafter(positions,-np.inf).reshape(-1),np.nextafter(positions,np.inf).reshape(-1)])
                verify('physical_grid_neighbors_cpm'+str(density),np.concatenate([positions,-positions]),h)
            verify('guarded_arbitrary_normals',normal(reg['arbitrary_normals'],-399,399),normal(reg['arbitrary_normals'],-19,19,False))
            verify('extreme_arbitrary_fallbacks',normal(reg['extreme_normals'],-1000,1000),normal(reg['extreme_normals'],-1000,1000))
            avalues=np.array([0.,-0.,np.inf,-np.inf,np.nan,float.fromhex('0x1p-1074'),-float.fromhex('0x1p-1074'),float.fromhex('0x1p-1022'),-float.fromhex('0x1p-1022'),np.finfo(np.float64).max,-np.finfo(np.float64).max,*[float.fromhex('0x1p'+str(e)) for e in [-401,-400,-399,0,399,400,401]]],dtype=np.float64)
            avalues=np.concatenate([avalues,np.nextafter(avalues,-np.inf),np.nextafter(avalues,np.inf)])
            bvalues=np.array([0.,-0.,1.,-1.,.01,.02,.04,.0625,.1,.125,np.inf,-np.inf,np.nan,*[float.fromhex('0x1p'+str(e)) for e in [-1000,-21,-20,-19,19,20,21,1000]]])
            verify('special_and_guard_boundaries',np.repeat(avalues,len(bvalues)),np.tile(bvalues,len(avalues)))
        passed=all(r['differing_output_bits']==0 and r['provided_reciprocal_different_bits_on_fast_domain']==0 for r in rows) and rational_fail==0 and reference_rational_fail==0
        write(args.output/'results.json',dict(status='passed_shared_reciprocal_scalar_bit_precision' if passed else 'failed_shared_reciprocal_scalar_precision',rows=rows,
            tested_scalar_pairs=sum(r['elements'] for r in rows),differing_output_bits=sum(r['differing_output_bits'] for r in rows),
            exact_rational_prerequisite_samples=rational_count,exact_rational_prerequisite_failures=rational_fail,independent_GPU_division_vs_exact_rational_failures=reference_rational_fail,
            model_forward_executed=False,cpu_model_forward=False,optimizer_updates=0,latency_claim=False,expanded_reserved52_references_opened=0,
            inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))
        if not passed:raise AssertionError('Scalar RN bits or final-correction prerequisites failed')
    finally:kernels.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
