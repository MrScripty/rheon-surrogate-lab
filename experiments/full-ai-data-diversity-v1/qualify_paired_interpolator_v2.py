"""Compensated interpolation: independent rational, parent, and grid controls."""
import argparse
import ctypes as C
import math
import os
from pathlib import Path
import resource
import time
from fractions import Fraction as F
import numpy as np
import torch
from support import ROOT, read, write, digest
from cuda_kernel_runtime_v1 import CudaKernels, check
from fused_cuda_generator_v1 import pointer
from cuda_native_transfer_v1 import FIELDS


def exact(d, f):
    ds = [F.from_float(float(v)) for v in d]; fs = [F.from_float(float(v)) for v in f]
    value = F(0)
    for bz in range(2):
        for by in range(2):
            for bx in range(2):
                value += ds[4*bz+2*by+bx] * (fs[0] if bx else 1-fs[0]) * (fs[1] if by else 1-fs[1]) * (fs[2] if bz else 1-fs[2])
    return value, max(map(abs, ds))


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda': raise ValueError('Fresh CUDA scalar controls required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen interpolation dependency changed: '+p)
    os.sched_setaffinity(0, {reg['host_cpu']}); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.empty(0, device='cuda')
    kernels = CudaKernels(reg['build'])
    parent = CudaKernels(reg['parent_build'])
    for name in ('nested_scalar_primitive', 'nested_grid_primitive'):
        function = C.c_void_p(); check(kernels.driver.cuModuleGetFunction(C.byref(function), kernels.module, name.encode()), name)
        kernels.functions[name] = function
        function = C.c_void_p(); check(parent.driver.cuModuleGetFunction(C.byref(function),parent.module,name.encode()),name)
        parent.functions[name] = function
    args.output.mkdir(parents=True); started = time.monotonic(); rng = np.random.default_rng(reg['seed'])
    result = dict(status='running', active=None, rows=[], grid_rows=[], scalar_samples=0, grid_samples=0, rational_checks=0,
                  inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                  cpu_model_forward=False, learned_model_forward_executed=False, optimizer_updates=0,
                  expanded_reserved52_references_opened=0, latency_claim=False,
                  protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
    def save(): write(args.output/'results.json', result)
    def upload(a): return torch.from_numpy(np.ascontiguousarray(a)).to('cuda')
    def bounded():
        if time.monotonic()-started > reg['maximum_wall_seconds']: raise TimeoutError('Finite interpolation control deadline')
        if (torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > reg['maximum_peak_rss_bytes']): raise ValueError('Interpolation resource ceiling exceeded')
    try:
        with torch.inference_mode(), np.errstate(over='ignore', invalid='ignore', under='ignore'):
            for population in reg['populations']:
                name, count = population['name'], population['count']
                d = rng.standard_normal((count, 8)).astype(np.float32)
                f = rng.random((count, 3), dtype=np.float64)
                clips = (np.arange(count)&1).astype(np.int32)
                if name == 'normal_random':
                    d *= np.ldexp(np.ones(count, dtype=np.float32), rng.integers(-55, 56, count))[:, None]
                elif name == 'finite_bits':
                    bits = rng.integers(0, 2**32, (count, 8), dtype=np.uint32)
                    exponent = (bits>>23)&255
                    bits[exponent == 255] ^= np.uint32(1<<23)
                    d = bits.view(np.float32)
                elif name == 'endpoints_cancellation':
                    d = np.tile(np.array([1., -1., 0., -0., 2.**-149, -2.**-149, 2.**100, -2.**100], dtype=np.float32), (count, 1))
                    t = np.array([0., 1., .5, np.nextafter(.5, 0.), np.nextafter(.5, 1.), np.nextafter(0., 1.), np.nextafter(1., 0.)])
                    for axis in range(3): f[:, axis] = t[(np.arange(count)//(len(t)**axis))%len(t)]
                elif name == 'pair_cancellation':
                    t = np.array([1.,-1.,np.nextafter(np.float32(1.),np.float32(2.)),-np.nextafter(np.float32(1.),np.float32(2.)),.5,-.5,0.,-0.],dtype=np.float32)
                    d = np.tile(t,(count,1))*np.ldexp(np.ones(count,dtype=np.float32),rng.integers(-45,46,count))[:,None]
                    t = np.array([0.,1.,.5,np.nextafter(.5,0.),np.nextafter(.5,1.),np.nextafter(0.,1.),np.nextafter(1.,0.)])
                    for axis in range(3):f[:,axis]=t[(np.arange(count)//(len(t)**axis))%len(t)]
                elif name == 'rounding_edges':
                    base=(1.+rng.random(count)).astype(np.float32)
                    high=np.nextafter(base,np.float32(np.inf))
                    d=np.stack([base,high,high,base,base,high,high,base],axis=1)
                    d*=np.ldexp(np.ones(count,dtype=np.float32),rng.integers(-45,46,count))[:,None]
                    t=np.array([0.,1.,.5,np.nextafter(.5,0.),np.nextafter(.5,1.)])
                    for axis in range(3):f[:,axis]=t[(np.arange(count)//(len(t)**axis))%len(t)]
                    d[:128]=np.tile(np.array([0.,-0.,-0.,0.,0.,-0.,-0.,0.],dtype=np.float32),(128,1))
                    for axis in range(3):f[:128,axis]=(np.arange(128)//(2**axis))%2
                elif name == 'nonfinite_fallback':
                    t = np.array([0., -0., np.finfo(np.float32).max, np.finfo(np.float32).tiny, np.inf, -np.inf, np.nan], dtype=np.float32)
                    d[:, 0] = t[np.arange(count)%len(t)]
                    t = np.array([0., 1., .5, -1e-12, 1.+1e-12, np.inf, -np.inf, np.nan])
                    f[:, 0] = t[(np.arange(count)//7)%len(t)]
                else: raise ValueError('Unregistered interpolation population')
                dg, fg, cg = upload(d), upload(f), upload(clips)
                out = torch.empty(count, device='cuda', dtype=torch.float64); ref = torch.empty_like(out)
                fallback = torch.empty(count, device='cuda', dtype=torch.int32)
                kernels.launch('nested_scalar_primitive', count, [C.c_longlong(count), pointer(dg), pointer(fg), pointer(cg), pointer(out), pointer(ref), pointer(fallback)])
                pa=torch.empty_like(out);pr=torch.empty_like(out);pf=torch.empty_like(fallback)
                parent.launch('nested_scalar_primitive',count,[C.c_longlong(count),pointer(dg),pointer(fg),pointer(cg),pointer(pa),pointer(pr),pointer(pf)])
                a, b, fb = out.cpu().numpy(), ref.cpu().numpy(), fallback.cpu().numpy()
                parent_values=pa.cpu().numpy()

                valid = np.isfinite(d).all(axis=1)&np.isfinite(f).all(axis=1)&(f>=0).all(axis=1)&(f<=1).all(axis=1)
                magnitude_each=np.abs(d.astype(np.float64))
                eligible=valid&((magnitude_each==0.)|((magnitude_each>=2.**-60)&(magnitude_each<=2.**60))).all(axis=1)
                if not np.array_equal(fb!=0,~eligible):raise ValueError('A priori pair guard classification changed')
                parent_same=(a.view(np.uint64)==parent_values.view(np.uint64))|(np.isnan(a)&np.isnan(parent_values))
                if not parent_same[~eligible].all():raise ValueError('Range/nonfinite guard must use exact parent output')
                endpoints=valid&((f==0.)|(f==1.)).all(axis=1)
                if not parent_same[endpoints].all():raise ValueError('Exact endpoints including signed zero changed')
                same = (a.view(np.uint64) == b.view(np.uint64)) | (np.isnan(a)&np.isnan(b))
                if not bool(same[~valid].all()): raise ValueError('Nonfinite/invalid fallback differs from original weighted program')
                magnitude = np.abs(d.astype(np.float64)).max(axis=1)
                error = np.abs(a-b)
                bound = (reg['candidate_bound_u']*2.**-48+reg['reference_bound_u']*2.**-53)*magnitude
                bad = np.flatnonzero(valid&(~np.isfinite(a)|~np.isfinite(b)|(error>bound)))
                if len(bad):
                    take = bad[:32]; np.savez_compressed(args.output/'counterexample.npz', donors=d[take], fractions=f[take], clips=clips[take], candidate=a[take], reference=b[take], bound=bound[take])
                    raise ValueError('Nested interpolation left conservative scalar error control')
                rational_eligible = np.flatnonzero(valid)
                chosen = rational_eligible[np.linspace(0, len(rational_eligible)-1, reg['rational_checks_per_population'], dtype=np.int64)]
                for i in chosen:
                    truth, m = exact(d[i], f[i])
                    if abs(F.from_float(float(a[i]))-truth)>F(reg['candidate_bound_u'],2**48)*m or abs(F.from_float(float(b[i]))-truth)>F(reg['reference_bound_u'],2**53)*m:
                        raise ValueError('Independent exact-rational scalar control failed')
                result['rational_checks'] += len(chosen); result['scalar_samples'] += count
                result['rows'].append(dict(population=name, samples=count, fallback=int(fb.sum()), pair_eligible=int(eligible.sum()), exact_parent_guard_outputs=True, exact_signed_zero_endpoints=True, changed_parent_F32_bits=int((a.astype(np.float32).view(np.uint32)!=parent_values.astype(np.float32).view(np.uint32)).sum()), scalar_error_controls_passed=True,
                                           rational_checks=len(chosen), changed_F64_bits=int((~same).sum()),
                                           changed_stored_F32_bits=int((a.astype(np.float32).view(np.uint32)!=b.astype(np.float32).view(np.uint32)).sum()),
                                           maximum_error_over_global_donor_magnitude=float(np.max(error[valid]/np.maximum(magnitude[valid], np.finfo(float).tiny)))))
                result['active'] = dict(population=name); save(); bounded()
                print('Compensatedpairgenericscalar', name, count, 'error/rational controls passed', flush=True)
            for case in reg['cases']:
                scene = read(ROOT/case['receipt'])['scene']; spec = case['primitive_state']
                if digest(ROOT/spec['path']) != spec['sha256']: raise ValueError('Consumed interpolation input changed')
                with np.load(ROOT/spec['path'], allow_pickle=False) as archive:
                    for kind, name in enumerate(FIELDS[1:]):
                        dims = [scene['counts'][axis]+(kind==axis) for axis in range(3)]
                        field = archive[name].copy(); count = reg['queries_per_actual_field']
                        queries = np.stack([rng.uniform(-2., n+1., count) for n in dims], axis=1)
                        for axis, n in enumerate(dims): queries[:32, axis] = np.resize(np.array([0., -0., n-1., .5, np.nextafter(1., 0.), np.nextafter(1., 2.), np.nan, np.inf, -np.inf]),32)
                        clips = (np.arange(count)&1).astype(np.int32)
                        xg, qg, cg = upload(field), upload(queries), upload(clips)
                        output = torch.empty(count, dtype=torch.float64, device='cuda'); reference = torch.empty_like(output)
                        kernels.launch('nested_grid_primitive', count, [C.c_longlong(count), pointer(xg), *(C.c_int(n) for n in dims), pointer(qg), pointer(cg), pointer(output), pointer(reference)])
                        a,b = output.cpu().numpy(), reference.cpu().numpy()
                        valid = ~np.isnan(queries).any(axis=1)
                        magnitude = float(np.abs(field.astype(np.float64)).max())
                        bound = (reg['candidate_bound_u']*2.**-48+reg['reference_bound_u']*2.**-53)*magnitude
                        if not np.isnan(a[~valid]).all() or not np.isnan(b[~valid]).all() or not np.isfinite(a[valid]).all() or not bool((np.abs(a[valid]-b[valid])<=bound).all()):
                            raise ValueError('Consumed-grid interpolation coordinate/error control failed')
                        result['grid_samples'] += count
                        result['grid_rows'].append(dict(identifier=case['identifier'], field=name, counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'], samples=count, all_error_controls_passed=True,
                                                        changed_F64_bits=int((a.view(np.uint64)!=b.view(np.uint64)).sum()), changed_stored_F32_bits=int((a.astype(np.float32).view(np.uint32)!=b.astype(np.float32).view(np.uint32)).sum())))
                save(); bounded()
            if result['scalar_samples']!=reg['expected_scalar_samples'] or result['grid_samples']!=reg['expected_grid_samples'] or result['rational_checks']!=reg['expected_rational_checks']:
                raise ValueError('Full finite primitive population required')
            for p,h in reg['source_sha256'].items():
                if digest(ROOT/p)!=h: raise ValueError('Interpolation dependency changed during controls')
            result.update(status='passed_compensated_pair_error_rational_parent_guard_and_grid_controls', active=None,
                          peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                          elapsed_diagnostic_seconds=time.monotonic()-started, model_accuracy_verified=False)
    except BaseException as e:
        result.update(status='failed',error=repr(e)); raise
    finally:
        kernels.close(); parent.close(); save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
