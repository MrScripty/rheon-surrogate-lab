"""GPU interval arithmetic gates; no learned forward or timing claim."""
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


def tri(d, q):
    return sum(d[4*z+2*y+x] * (q[0] if x else 1-q[0]) * (q[1] if y else 1-q[1])
               * (q[2] if z else 1-q[2]) for z in range(2) for y in range(2) for x in range(2))


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only arithmetic qualification required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT / p) != h:
            raise ValueError('Frozen arithmetic dependency changed: ' + p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.empty(0, device='cuda')
    kernels = CudaKernels(reg['build'])
    for name in ('middle_sample_primitive', 'middle_interval_primitive', 'middle_propagation_primitive'):
        fn = C.c_void_p()
        check(kernels.driver.cuModuleGetFunction(C.byref(fn), kernels.module, name.encode()), name)
        kernels.functions[name] = fn
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(reg['seed'])
    result = dict(status='running', rows=[], actual_grid_rows=[], rational_checks=0,
                  inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                  cpu_model_forward=False, learned_model_forward_executed=False, optimizer_updates=0,
                  expanded_reserved52_references_opened=0, latency_claim=False,
                  protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
    started = time.monotonic()
    def save():
        write(args.output / 'results.json', result)
    def upload(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to('cuda')
    def outputs(size, types):
        return [torch.empty(size, dtype=t, device='cuda') for t in types]
    def deadline():
        if time.monotonic()-started > reg['maximum_wall_seconds']:
            raise TimeoutError('Finite arithmetic population deadline')
    def fail(label, **arrays):
        np.savez_compressed(args.output / 'counterexample.npz', label=np.asarray(label), **arrays)
        raise AssertionError(label)
    try:
        with torch.inference_mode(), np.errstate(over='ignore', invalid='ignore', under='ignore'):
            for pop in reg['populations']:
                deadline()
                tag, size = pop['name'], pop['count']
                low = rng.random((size, 3))
                high = np.minimum(1., low + np.power(10., rng.uniform(-15., -5., (size, 3))))
                actual = np.minimum(high, np.maximum(low, low + (high-low)*rng.random((size, 3))))
                clips = (np.arange(size) & 1).astype(np.int32)
                if tag == 'normal_intervals':
                    donors = rng.standard_normal((size, 8)).astype(np.float32)
                    donors *= np.ldexp(np.ones(size, dtype=np.float32), rng.integers(-55, 56, size))[:, None]
                elif tag == 'broad_bits':
                    donors = rng.integers(0, 2**32, size=(size, 8), dtype=np.uint32).view(np.float32)
                elif tag == 'midpoints_and_cancellation':
                    base = np.ldexp(rng.uniform(.5, 1., size).astype(np.float32), rng.integers(-40, 41, size))
                    donors = np.repeat(base[:, None], 8, axis=1).astype(np.float32)
                    donors[:, 1] = np.nextafter(base, np.float32(np.inf))
                    low[:] = high[:] = actual[:] = [0.5, 0., 0.]
                    odd = np.arange(size) % 2 == 1
                    donors[odd] = np.tile(np.array([1., -1.], dtype=np.float32), (int(odd.sum()), 4))
                    low[odd] = high[odd] = actual[odd] = .5
                elif tag == 'guards_and_boundaries':
                    edge = np.array([0., -0., 2.**-60, np.nextafter(np.float32(2.**-60), np.float32(0.)),
                        2.**60, np.nextafter(np.float32(2.**60), np.float32(np.inf)),
                        np.nextafter(np.float32(0.), np.float32(1.)), np.finfo(np.float32).max,
                        np.inf, -np.inf, np.nan, 1., -1.], dtype=np.float32)
                    t = np.array([0., -0., .5, 1., np.nextafter(0., 1.), np.nextafter(1., 0.),
                                  np.nextafter(.5, 0.), np.nextafter(.5, 1.), np.nan, np.inf, -1.])
                    index = np.arange(size)
                    donors = np.repeat(edge[index % len(edge), None], 8, axis=1)
                    donors[:, 0] = np.nextafter(donors[:, 0], np.float32(np.inf))
                    low[:, 0] = high[:, 0] = actual[:, 0] = t[(index // len(edge)) % len(t)]
                else:
                    raise ValueError('Unregistered scalar population')
                inputs = [upload(a) for a in (donors, low, high, actual, clips)]
                out = outputs(size, (torch.float32, torch.float32, torch.int32, torch.float64, torch.float64, torch.float64))
                kernels.launch('middle_interval_primitive', size, [C.c_longlong(size), *(pointer(v) for v in inputs+out)])
                aa, bb, accepted, approx, radius, raw = [v.cpu().numpy() for v in out]
                bad = np.flatnonzero(aa.view(np.uint32) != bb.view(np.uint32))
                if len(bad):
                    ix=bad[:32];fail('Terminal stored bits changed', donors=donors[ix], low=low[ix], high=high[ix], actual=actual[ix], output=aa[ix], reference=bb[ix])
                valid = np.flatnonzero(radius > 0.)
                if np.any(np.abs(approx[valid]-raw[valid]) > radius[valid]):
                    ix=valid[np.abs(approx[valid]-raw[valid]) > radius[valid]][:32]
                    fail('Terminal reference escaped interval', donors=donors[ix], low=low[ix], high=high[ix], actual=actual[ix], approximation=approx[ix], radius=radius[ix], raw=raw[ix])
                count = reg['rational_checks_per_population']
                if len(valid) < count:
                    raise ValueError('Insufficient valid rational-check population')
                for i in valid[np.linspace(0, len(valid)-1, count, dtype=np.int64)]:
                    d = [F.from_float(float(v)) for v in donors[i]]
                    lo, hi, q = [[F.from_float(float(v)) for v in a[i]] for a in (low, high, actual)]
                    a, r, reference = [F.from_float(float(v)) for v in (approx[i], radius[i], raw[i])]
                    magnitude=max(map(abs,d)); exact_lo,exact_q=tri(d,lo),tri(d,q)
                    if (abs(a-exact_lo)>5413*F(1,2**48)*magnitude or abs(reference-exact_q)>512*F(1,2**53)*magnitude
                            or abs(exact_q-exact_lo)>2*magnitude*sum(b-a for a,b in zip(lo,hi))
                            or r < magnitude/F(2**32)+2*magnitude*sum(b-a for a,b in zip(lo,hi)) or abs(a-reference)>r):
                        fail('Independent rational interval proof failed', donors=donors[i:i+1], low=low[i:i+1], high=high[i:i+1], actual=actual[i:i+1])
                if tag == 'midpoints_and_cancellation' and bool(accepted.any()):
                    raise AssertionError('Uncertain midpoint or signed-zero case accepted')
                result['rational_checks'] += count
                result['rows'].append(dict(population=tag, samples=size, accepted_fast=int(np.count_nonzero(accepted)),
                    fallback=int(size-np.count_nonzero(accepted)), all_bits_exact=True, rational_checks=count))
                save()
                print('Middle terminal interval', tag, size, 'bits/enclosures passed', flush=True)
            deadline()
            size = reg['propagation_samples']
            actual = rng.standard_normal((size, 3))*np.exp2(rng.integers(-50, 51, (size, 3)))
            radius = np.abs(actual)*rng.uniform(0., 1e-7, (size, 3))
            low, high = np.nextafter(actual-radius, -np.inf), np.nextafter(actual+radius, np.inf)
            mix = rng.standard_normal((size, 9))*np.exp2(rng.integers(-20, 21, (size, 9)))
            h = np.exp2(rng.uniform(-10., 0., size));dt=np.exp2(rng.uniform(-10., -4., size))
            upper = rng.integers(1, 129, (size, 3))*h[:, None]
            position = rng.random((size, 3))*upper;phase=(rng.integers(0,2,(size,3))*.5).astype(np.float64)
            for j in range(32):
                index=size-32+j
                if j%4==0:dt[index]=0.
                elif j%4==1:h[index]=-1.
                elif j%4==2:mix[index,0]=np.nan
                else:low[index,0]=high[index,0]+1.
            inp=[upload(a) for a in (actual, low, high, mix, position, upper, phase, dt, h)]
            out=[torch.empty((size,3),dtype=torch.float64,device='cuda') for _ in range(3)]+[torch.empty(size,dtype=torch.int32,device='cuda')]
            kernels.launch('middle_propagation_primitive',size,[C.c_longlong(size),*(pointer(v) for v in inp+out)])
            qlow,qhigh,ref,valid=[v.cpu().numpy() for v in out];mask=valid.astype(bool)
            if bool(valid[-32:].any()) or not bool(valid[:-32].all()):
                raise AssertionError('Propagation guards did not match finite/invalid populations')
            if not bool(((qlow[mask]<=ref[mask]) & (ref[mask]<=qhigh[mask])).all()):
                ix=np.flatnonzero(mask & np.any((qlow>ref)|(ref>qhigh),axis=1))[:32]
                fail('Original RN64 coordinate escaped propagated interval', actual=actual[ix], low=low[ix], high=high[ix], mix=mix[ix], qlow=qlow[ix], qhigh=qhigh[ix], reference=ref[ix])
            result['propagation']=dict(samples=size, valid=int(valid.sum()), explicitly_invalid=32, all_reference_coordinates_enclosed=True)
            save()
            for case in reg['cases']:
                deadline();scene=read(ROOT/case['receipt'])['scene'];spec=case['primitive_state']
                if digest(ROOT/spec['path']) != spec['sha256']:
                    raise ValueError('Actual consumed initial fields changed')
                with np.load(ROOT/spec['path'],allow_pickle=False) as archive:
                    for kind,name in enumerate(('vx','vy','vz')):
                        nx,ny,nz=[v+int(axis==kind) for axis,v in enumerate(scene['counts'])]
                        size=reg['queries_per_actual_field'];q=np.empty((size,3),dtype=np.float64)
                        for axis,width in enumerate((nx,ny,nz)):
                            q[:,axis]=rng.uniform(-2.,width+1.,size)
                            q[:16,axis]=np.resize(np.array([0.,-0.,width-1.,.5,1.,np.nextafter(1.,0.),np.nextafter(1.,2.),np.nan,np.inf,-np.inf]),16)
                        field,query=upload(archive[name].copy()),upload(q)
                        out=outputs(size,(torch.float64,torch.float64,torch.float64,torch.int32))
                        kernels.launch('middle_sample_primitive',size,[C.c_longlong(size),pointer(field),C.c_int(nx),C.c_int(ny),C.c_int(nz),pointer(query),*(pointer(v) for v in out)])
                        lo,hi,ref,valid=[v.cpu().numpy() for v in out];mask=valid.astype(bool)
                        if not bool(((lo[mask]<=ref[mask]) & (ref[mask]<=hi[mask])).all()):
                            ix=np.flatnonzero(mask & ((lo>ref)|(ref>hi)))[:32]
                            fail('Actual middle reference escaped enclosure', coordinates=q[ix], low=lo[ix], high=hi[ix], reference=ref[ix])
                        if not bool(valid[np.isfinite(q).all(axis=1)].all()):
                            raise AssertionError('Finite actual middle sample unexpectedly invalid')
                        result['actual_grid_rows'].append(dict(identifier=case['identifier'],field=name,counts_xyz=[nx,ny,nz],
                            cells_per_metre=scene['cells_per_metre'],samples=size,enclosed=int(valid.sum()),all_valid_enclosures_pass=True))
                save()
                print('Middle interval actual grid',case['identifier'],'all velocity components passed',flush=True)
        if (result['rational_checks'] != reg['expected_rational_checks'] or len(result['actual_grid_rows']) != reg['expected_actual_fields']
                or sum(r['samples'] for r in result['rows']) != reg['expected_scalar_samples']):
            raise ValueError('Exact registered population required')
        if torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']:
            raise ValueError('Finite resource caps exceeded')
        for p,h in reg['source_sha256'].items():
            if digest(ROOT/p)!=h:
                raise ValueError('Arithmetic dependency changed during execution: '+p)
        result.update(status='passed_guarded_middle_intervals_and_all26_actual_grids',all_F32_bits_equal=True,
            independent_rational_bounds_pass=True,all_reference_intervals_pass=True,
            scalar_samples=sum(r['samples'] for r in result['rows']),actual_grid_samples=sum(r['samples'] for r in result['actual_grid_rows']),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_diagnostic_seconds=time.monotonic()-started)
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        kernels.close();save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
