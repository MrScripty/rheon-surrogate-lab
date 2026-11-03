"""Compiled learned parameter bytes, ordered learned dots and GPU transport."""
import argparse
import ctypes as C
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from nested_F64_pressure_graph_v1 import NestedF64PressureGraph
from bound_sampler_pressure_graph_v1 import BoundSamplerPressureGraph
from cuda_kernel_runtime_v1 import check
from fused_cuda_generator_v1 import pointer
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory


def bits_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape: return False
    integer = np.uint64 if a.dtype == np.float64 else np.uint32
    return bool(((a.view(integer)==b.view(integer)) | (np.isnan(a)&np.isnan(b))).all())


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda': raise ValueError('Fresh GPU qualification required')
    reg = read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h: raise ValueError('Frozen bound-sampler dependency changed: '+p)
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True);started=time.monotonic();rng=np.random.default_rng(reg['seed'])
    models=[NestedF64PressureGraph(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']),
            BoundSamplerPressureGraph(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build'])]
    before=[inventory(m) for m in models]
    if before[0]!=before[1]:raise ValueError('Identical original learned inventories required')
    core=models[1].model;kernels=models[1].nested_kernels
    for name in ('bound_parameter_bits','bound_point_primitive','bound_mix_gate_primitive'):
        f=C.c_void_p();check(kernels.driver.cuModuleGetFunction(C.byref(f),kernels.module,name.encode()),name);kernels.functions[name]=f
    result=dict(status='running',active=None,parameter_bits_exact=False,point_rows=[],mix_gate_rows=[],transport_rows=[],
                inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),cpu_model_forward=False,
                optimizer_updates=0,expanded_reserved52_references_opened=0,latency_claim=False,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol))
    def save():write(args.output/'results.json',result)
    def upload(a):return torch.from_numpy(np.ascontiguousarray(a)).to('cuda')
    def bounded():
        if time.monotonic()-started>reg['maximum_wall_seconds']:raise TimeoutError('Finite learned sampler control deadline')
        if torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']:raise ValueError('Bound sampler control resource ceiling exceeded')
    try:
        with torch.inference_mode(),np.errstate(over='ignore',under='ignore',invalid='ignore'):
            m=read(ROOT/reg['manifest'])['tensors'];p=m['point'];meta=np.array(p['rows']+p['channels']+[v for o in p['offsets'] for v in o],dtype='<i4')
            expected=np.array(p['coefficients']['bits_u64']+m['mid_gate']['bits_u64']+m['mix']['bits_u64'],dtype='<u8')
            actual=torch.empty(len(expected),device='cuda',dtype=torch.uint64);metadata=torch.empty(len(meta),device='cuda',dtype=torch.int32)
            kernels.launch('bound_parameter_bits',max(len(expected),len(meta)),[pointer(actual),pointer(metadata)])
            if not np.array_equal(actual.cpu().numpy(),expected) or not np.array_equal(metadata.cpu().numpy(),meta):raise ValueError('Compiled learned parameter or metadata bytes differ')
            result['parameter_bits_exact']=True
            for population in reg['scalar_populations']:
                count=population['count'];name=population['name'];counts=(5,6,7);sizes=dimensions(counts)[:3]
                fields=[rng.standard_normal(s).astype('<f4') for s in sizes]
                if name=='finite_bits':
                    for i,s in enumerate(sizes):
                        a=rng.integers(0,2**32,size=s,dtype=np.uint32);a[((a>>23)&255)==255]^=np.uint32(1<<23);fields[i]=a.view('<f4')
                elif name=='zero_nonfinite':
                    values=np.array([0.,-0.,np.inf,-np.inf,np.nan,2.**-149,-2.**-149,1.,-1.],dtype='<f4')
                    fields=[np.resize(values,math.prod(s)).reshape(s) for s in sizes]
                elif name!='normal_random':raise ValueError('Unregistered scalar population')
                queries=np.empty((count,4),dtype='<i4');queries[:,0]=np.arange(count)%12
                for axis,n in enumerate(counts):queries[:,axis+1]=rng.integers(-2,n+3,count)
                fg=[upload(a) for a in fields];qg=upload(queries);a=torch.empty(count,device='cuda',dtype=torch.float64);b=torch.empty_like(a)
                kernels.launch('bound_point_primitive',count,[C.c_longlong(count),*(pointer(v) for v in fg),*(C.c_int(n) for n in counts),pointer(qg),*(pointer(v) for v in core.kernel_sparse['point']),pointer(a),pointer(b)])
                aa,bb=a.cpu().numpy(),b.cpu().numpy()
                if not bits_equal(aa,bb):
                    bad=np.flatnonzero(~((aa.view(np.uint64)==bb.view(np.uint64))|(np.isnan(aa)&np.isnan(bb))));np.savez_compressed(args.output/'point-counterexample.npz',queries=queries[bad[:32]],candidate=aa[bad[:32]],reference=bb[bad[:32]])
                    raise ValueError('Compiled learned point dot changed ordered output bits')
                result['point_rows'].append(dict(population=name,samples=count,all_output_bits_or_NaN_classes_exact=True))
                values=rng.standard_normal((count,5)).astype('<f8');heads=(np.arange(count)%12).astype('<i4')
                if name=='finite_bits':values=np.ldexp(values,rng.integers(-1000,1001,size=(count,5)))
                if name=='zero_nonfinite':values[:]=np.resize(np.array([0.,-0.,np.inf,-np.inf,np.nan,1.,-1.,np.nextafter(0.,1.)]),count*5).reshape(count,5)
                vg,hg=upload(values),upload(heads);a=torch.empty(2*count,device='cuda',dtype=torch.float64);b=torch.empty_like(a)
                kernels.launch('bound_mix_gate_primitive',count,[C.c_longlong(count),pointer(vg),pointer(hg),pointer(core.mid_gate),pointer(core.mix),pointer(a),pointer(b)])
                if not bits_equal(a.cpu().numpy(),b.cpu().numpy()):raise ValueError('Compiled learned gate or mix changed ordered output bits/classes')
                result['mix_gate_rows'].append(dict(population=name,samples=2*count,all_output_bits_or_NaN_classes_exact=True))
                save();bounded();print('Boundlearnedscalarcontrols',name,count,'point/mix/gateexact',flush=True)
            for case in reg['cases']:
                scene=read(ROOT/case['receipt'])['scene'];spec=case['primitive_state'];counts=tuple(scene['counts']);sizes=dimensions(counts)
                if digest(ROOT/spec['path'])!=spec['sha256']:raise ValueError('Consumed transport field source changed')
                with np.load(ROOT/spec['path'],allow_pickle=False) as archive:
                    native=[archive[n].copy().reshape(s) for n,s in zip(FIELDS[1:],sizes)]
                for pattern in reg['transport_patterns']:
                    if pattern=='actual':fields=[upload(a) for a in native]
                    elif pattern=='random':fields=[upload(rng.normal(0.,.05,s).astype('<f4')) for s in sizes]
                    elif pattern=='zero_impulse':
                        arrays=[np.zeros(s,dtype='<f4') for s in sizes]
                        for i,array in enumerate(arrays):array.flat[len(array.flat)//2]=(-1.)**i*.1
                        fields=[upload(a) for a in arrays]
                    elif pattern=='nonfinite_boundary':
                        arrays=[a.copy() for a in native]
                        for array in arrays:array.flat[:8]=np.array([0.,-0.,np.nan,np.inf,-np.inf,2.**-149,-2.**-149,1.],dtype='<f4')
                        fields=[upload(a) for a in arrays]
                    else:raise ValueError('Unregistered transport pattern')
                    outputs=[]
                    for model in models:
                        model.model._nested_enabled=True;model.model.configure_transport(1./scene['cells_per_metre'],1./480.)
                        outputs.append(model.model.transport(fields,counts,(1./480.)/(1./scene['cells_per_metre'])))
                    for n,a,b in zip(FIELDS[1:],*outputs):
                        aa,bb=a.cpu().numpy(),b.cpu().numpy()
                        if not bits_equal(aa,bb):raise ValueError('Compiled complete learned transport changed output: '+str((case['identifier'],pattern,n)))
                    result['transport_rows'].append(dict(identifier=case['identifier'],counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],pattern=pattern,complete_carrier_arrays=4,all_output_bits_or_NaN_classes_exact=True))
                    save();bounded()
                print('Boundlearnedtransportcontrols',case['identifier'],'allpatterns exact',flush=True)
            if len(result['transport_rows'])!=reg['expected_transport_patterns'] or [inventory(m) for m in models]!=before:raise ValueError('All registered transport patterns and unchanged original weights required')
            for p,h in reg['source_sha256'].items():
                if digest(ROOT/p)!=h:raise ValueError('Bound sampler dependency changed during controls')
            result.update(status='passed_bound_sampler_parameter_dot_and_transport_controls',active=None,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_diagnostic_seconds=time.monotonic()-started,whole_model_accuracy_verified=False)
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:
        for m in models:m.close()
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
