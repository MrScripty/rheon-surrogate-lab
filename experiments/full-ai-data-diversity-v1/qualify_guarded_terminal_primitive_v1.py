"""GPU terminal interpolation bits and independent exact-rational enclosures."""
import argparse
import ctypes as C
import hashlib
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
        raise ValueError('Fresh GPU-only arithmetic evidence required')
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
    for name in ('terminal_pair_primitive', 'terminal_grid_primitive'):
        function = C.c_void_p()
        check(kernels.driver.cuModuleGetFunction(C.byref(function), kernels.module, name.encode()), name)
        kernels.functions[name] = function
    rng = np.random.default_rng(reg['seed'])
    args.output.mkdir(parents=True)
    result = dict(status='running', rows=[], actual_grid_rows=[], rational_checks=0, active=None,
                  inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                  learned_model_forward_executed=False, cpu_model_forward=False, optimizer_updates=0,
                  expanded_reserved52_references_opened=0, latency_claim=False,
                  source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    started = time.monotonic()
    alpha, u64 = F(1, 2**48), F(1, 2**53)

    def upload(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to('cuda')

    def save():
        write(args.output / 'results.json', result)

    def fail(label, donors, fractions, out, ref, fast, approx, radius, raw, indices):
        take = np.asarray(indices[:32], dtype=np.int64)
        np.savez_compressed(args.output / 'counterexample.npz', label=np.asarray(label), donors=donors[take],
                            fractions=fractions[take], output=out[take], reference=ref[take], accepted=fast[take],
                            approximation=approx[take], radius=radius[take], raw_reference=raw[take])
        raise AssertionError(label + ': retained concrete arithmetic counterexamples')

    try:
        with torch.inference_mode(), np.errstate(over='ignore', invalid='ignore', under='ignore'):
            for population in reg['populations']:
                tag, count = population['name'], population['count']
                chunks = math.ceil(count / reg['chunk_size'])
                quota = math.ceil(reg['rational_checks_per_population'] / chunks)
                rational_for_population = 0
                for begin in range(0, count, reg['chunk_size']):
                    if time.monotonic() - started > reg['maximum_wall_seconds']:
                        raise TimeoutError('Finite terminal arithmetic deadline')
                    size = min(reg['chunk_size'], count - begin)
                    fractions = rng.random((size, 3), dtype=np.float64)
                    clips = ((np.arange(begin, begin + size) & 1) != 0).astype(np.int32)
                    if tag == 'normal_random':
                        donors = rng.standard_normal((size, 8)).astype(np.float32)
                        donors *= np.ldexp(np.ones(size, dtype=np.float32), rng.integers(-55, 56, size=size))[:, None]
                    elif tag == 'broad_bits':
                        donors = rng.integers(0, 2**32, size=(size, 8), dtype=np.uint32).view(np.float32)
                    elif tag == 'midpoint_and_cancellation':
                        base = np.ldexp(rng.uniform(.5, 1., size).astype(np.float32), rng.integers(-40, 41, size))
                        donors = np.repeat(base[:, None], 8, axis=1).astype(np.float32)
                        donors[:, 1] = np.nextafter(base, np.float32(np.inf))
                        fractions[:] = [0.5, 0., 0.]
                        selected = (np.arange(begin, begin + size) & 1) != 0
                        donors[selected] = np.tile(np.array([1., -1.], dtype=np.float32), (int(selected.sum()), 4))
                        fractions[selected] = .5
                    elif tag == 'guards_and_boundaries':
                        v = np.array([0., -0., 2.**-60, np.nextafter(np.float32(2.**-60), np.float32(0.)),
                                      np.nextafter(np.float32(2.**-60), np.float32(np.inf)), 2.**60,
                                      np.nextafter(np.float32(2.**60), np.float32(np.inf)),
                                      np.nextafter(np.float32(2.**60), np.float32(0.)),
                                      np.nextafter(np.float32(0.), np.float32(1.)), np.finfo(np.float32).tiny,
                                      np.finfo(np.float32).max, np.inf, -np.inf, np.nan], dtype=np.float32)
                        t = np.array([0., -0., .5, 1., np.nextafter(0., 1.), np.nextafter(1., 0.),
                                      np.nextafter(.5, 0.), np.nextafter(.5, 1.), -1e-12, 1.+1e-12,
                                      np.nan, np.inf, -np.inf], dtype=np.float64)
                        index = np.arange(begin, begin + size)
                        donors = np.repeat(v[index % len(v), None], 8, axis=1)
                        donors[:, 0] = np.nextafter(donors[:, 0], np.float32(np.inf))
                        fractions[:, 0] = t[(index // len(v)) % len(t)]
                        fractions[:, 1] = t[(index // (len(v) * len(t))) % len(t)]
                    else:
                        raise ValueError('Unregistered scalar population')
                    donors = np.ascontiguousarray(donors)
                    dg, fg, cg = upload(donors), upload(fractions), upload(clips)
                    outputs = [torch.empty(size, dtype=dtype, device='cuda') for dtype in
                               (torch.float32, torch.float32, torch.int32, torch.float64, torch.float64, torch.float64)]
                    kernels.launch('terminal_pair_primitive', size,
                                   [C.c_longlong(size), pointer(dg), pointer(fg), pointer(cg), *(pointer(v) for v in outputs)])
                    out, ref, fast, approx, radius, raw = [v.cpu().numpy() for v in outputs]
                    bad = np.flatnonzero(out.view(np.uint32) != ref.view(np.uint32))
                    if len(bad):
                        fail('F32 bits changed', donors, fractions, out, ref, fast, approx, radius, raw, bad)
                    valid = np.flatnonzero(radius > 0.)
                    bad = valid[np.logical_not(np.abs(approx[valid] - raw[valid]) <= radius[valid])]
                    if len(bad):
                        fail('Reference left conservative enclosure', donors, fractions, out, ref, fast, approx, radius, raw, bad)
                    take = min(quota, reg['rational_checks_per_population'] - rational_for_population, len(valid))
                    chosen = valid[np.linspace(0, len(valid) - 1, take, dtype=np.int64)] if take else []
                    for i in chosen:
                        ds = [F.from_float(float(v)) for v in donors[i]]
                        fx, fy, fz = [F.from_float(float(v)) for v in fractions[i]]
                        exact = F(0)
                        for bz in range(2):
                            for by in range(2):
                                for bx in range(2):
                                    exact += ds[4*bz+2*by+bx] * (fx if bx else 1-fx) * (fy if by else 1-fy) * (fz if bz else 1-fz)
                        magnitude = max(abs(v) for v in ds)
                        aa, rr, bb = map(F.from_float, map(float, (approx[i], raw[i], radius[i])))
                        if (abs(aa-exact) > 5413*alpha*magnitude or abs(rr-exact) > 512*u64*magnitude
                                or bb != magnitude/F(2**32) or abs(aa-rr) > bb):
                            fail('Independent rational bound failed', donors, fractions, out, ref, fast, approx, radius, raw, [i])
                    rational_for_population += take
                    result['rational_checks'] += take
                    if tag == 'midpoint_and_cancellation' and bool(fast.any()):
                        fail('Uncertain midpoint/cancellation accepted', donors, fractions, out, ref, fast, approx, radius, raw, np.flatnonzero(fast))
                    result['rows'].append(dict(population=tag, begin=begin, samples=size,
                        accepted_fast=int(np.count_nonzero(fast)), fallback=int(size-np.count_nonzero(fast)),
                        every_F32_bit_equal=True, reference_enclosure_checks=len(valid), rational_checks=take))
                    result['active'] = dict(population=tag, checked_through=begin+size)
                    save()
                if rational_for_population != reg['rational_checks_per_population']:
                    raise ValueError('Full independently checked rational population required')
                print('Guarded terminal scalar population', tag, count, 'bits and rational bounds passed', flush=True)
            for case in reg['cases']:
                scene = read(ROOT / case['receipt'])['scene']
                spec = case['primitive_state']
                if digest(ROOT / spec['path']) != spec['sha256']:
                    raise ValueError('Actual consumed interpolation input changed')
                with np.load(ROOT / spec['path'], allow_pickle=False) as archive:
                    for kind, name in enumerate(FIELDS[1:]):
                        nx, ny, nz = (scene['counts'][0]+(kind==0), scene['counts'][1]+(kind==1), scene['counts'][2]+(kind==2))
                        size = reg['queries_per_actual_field']
                        q = rng.uniform(-2., max(nx, ny, nz)+2., size=(size, 3)).astype(np.float64)
                        for axis, width in enumerate((nx, ny, nz)):
                            q[:, axis] = rng.uniform(-2., width+1., size)
                            q[:32, axis] = np.resize(np.array([0., -0., width-1., .5, 1., np.nextafter(1., 0.),
                                                            np.nextafter(1., 2.), np.nan, np.inf, -np.inf]), 32)
                        clips = (np.arange(size) & 1).astype(np.int32)
                        field, query, clip = upload(archive[name].copy()), upload(q), upload(clips)
                        out = torch.empty(size, dtype=torch.float32, device='cuda')
                        ref = torch.empty_like(out)
                        kernels.launch('terminal_grid_primitive', size, [C.c_longlong(size), pointer(field),
                            C.c_int(nx), C.c_int(ny), C.c_int(nz), pointer(query), pointer(clip), pointer(out), pointer(ref)])
                        aa, bb = out.cpu().numpy(), ref.cpu().numpy()
                        if np.any(aa.view(np.uint32) != bb.view(np.uint32)):
                            np.savez_compressed(args.output/'actual-grid-counterexample.npz', identifier=np.asarray(case['identifier']),
                                                field=np.asarray(name), coordinates=q, output=aa, reference=bb)
                            raise AssertionError('Actual generic CUDA sampling bits changed')
                        result['actual_grid_rows'].append(dict(identifier=case['identifier'], field=name,
                            counts_xyz=[nx, ny, nz], cells_per_metre=scene['cells_per_metre'], samples=size,
                            all_bits_exact=True))
                if time.monotonic()-started > reg['maximum_wall_seconds']:
                    raise TimeoutError('Actual interpolation corpus deadline')
                print('Guarded terminal actual grid', case['identifier'], 'all carriers exact', flush=True)
        if (torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > reg['maximum_peak_rss_bytes']):
            raise ValueError('Primitive resource ceiling')
        for p, h in reg['source_sha256'].items():
            if digest(ROOT/p) != h:
                raise ValueError('Frozen primitive source changed during execution: '+p)
        result.update(status='passed_guarded_terminal_scalar_enclosures_and_all26_actual_grids', active=None,
            scalar_samples=sum(r['samples'] for r in result['rows']), actual_grid_samples=sum(r['samples'] for r in result['actual_grid_rows']),
            accepted_fast=sum(r['accepted_fast'] for r in result['rows']), fallback=sum(r['fallback'] for r in result['rows']),
            all_F32_bits_equal=True, independent_rational_bounds_pass=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_diagnostic_seconds=time.monotonic()-started)
    except BaseException as error:
        result.update(status='failed', active=None, error=repr(error))
        raise
    finally:
        kernels.close()
        save()


FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
