"""Bounded native-generated trajectories with a spectral pressure reference."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'experiments/pressure-four-pass-distillation-v1'), str(ROOT/'experiments/pressure-methods-v1'), str(ROOT/'experiments/pressure-methods-v2')]
from common import read, write, sha
from large_native import read_field, available

def run(args):
    if args.output.exists(): raise ValueError('Fresh capture required')
    os.sched_setaffinity(0, {args.cpu})
    scene = read(args.scene); cells = math.prod(scene['counts'])
    if not 1 <= scene['steps'] <= 32 or cells > 128**3 or scene['warmup_steps'] != 0:
        raise ValueError('Bounded native spectral trajectory required, with no hidden CG warmup')
    if available() < 12*1024**3+cells*512: raise MemoryError('Host reserve unavailable')
    build = read(args.build/'build-receipt.json'); binary = args.build/'rheon-large-single-track'
    if build['status'] != 'qualified_build' or sha(binary) != build['binary_sha256']:
        raise ValueError('Qualified build required')
    for path, expected in build['sources'].items():
        if sha(ROOT/path) != expected: raise ValueError('Native source changed')
    import numpy as np
    import torch
    import numerics
    torch.set_num_threads(1)
    h = torch.tensor([1/scene['cells_per_metre']], dtype=torch.float64)
    rho = torch.tensor([1000.], dtype=torch.float64)
    sys.path.insert(0, str(ROOT/'runs/dependency-migration-v1/lab/experiments/global-pressure-v1'))
    spec = importlib.util.spec_from_file_location('v3_capture_owner', ROOT/'runs/dependency-migration-v1/lab/experiments/global-pressure-v1/capture.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    args.output.mkdir(parents=True)
    frames = []; snapshots = []; callbacks = []; child = None; complete = False; pending = None
    tick = time.perf_counter()
    with (args.output/'stderr.log').open('xb') as stderr:
        with owner.bounded_process([str(binary)], args.timeout, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr, env={**os.environ,'RAYON_NUM_THREADS':'1'}) as child:
            child.stdin.write(json.dumps(scene).encode()+b'\n'); child.stdin.flush()
            while True:
                raw = child.stdout.readline(65537)
                if not raw or len(raw)>65536: raise ValueError('Bounded complete header required')
                packet = json.loads(raw)
                if packet['kind'] == 'ready': ready = packet
                elif packet['kind'] == 'predict':
                    begin = time.perf_counter()
                    if (packet['cells'] != cells or packet['counts_xyz'] != scene['counts'] or packet['dtype'] != '<f8'
                            or packet['fields'] != ['base','native_input_residual'] or packet['cell_size_m'] != float(h[0])
                            or packet['rho'] != 1000. or packet['dt'] != 1/60):
                        raise ValueError('Native equation metadata differs')
                    shape = (1,1,*scene['counts'][::-1])
                    base = torch.from_numpy(np.frombuffer(read_field(child.stdout,cells),dtype='<f8').reshape(shape))
                    residual = torch.from_numpy(np.frombuffer(read_field(child.stdout,cells),dtype='<f8').reshape(shape))
                    method_start = time.perf_counter()
                    with torch.no_grad():
                        rhs = residual+numerics.fine_action(base,h,rho)
                        pressure = base+numerics.correction(residual,h,rho,None)
                        pressure = pressure-pressure[:,:,:1,:1,:1].clone()
                        relative = float(torch.linalg.vector_norm(rhs-numerics.fine_action(pressure,h,rho)))/max(float(torch.linalg.vector_norm(rhs)),1e-300)
                    method_seconds = time.perf_counter()-method_start
                    child.stdin.write(memoryview(np.ascontiguousarray(pressure.numpy(),dtype='<f8')).cast('B')); child.stdin.flush()
                    callbacks.append(dict(relative_residual=relative,method_seconds=method_seconds,seconds=time.perf_counter()-begin))
                    pending = dict(base=base,rhs=rhs,reference=pressure,native_input_residual=residual)
                elif packet['kind'] == 'frame':
                    frame = packet['frame']; index = len(frames)
                    frames.append(dict(**frame,preparation_profile=packet['preparation_profile']))
                    if frame['advanced'] and args.save_fields and index in {0,scene['steps']-1}:
                        if pending is None: raise ValueError('Reference fields missing')
                        arrays = {}
                        for name,value in pending.items():
                            path = args.output/f'frame-{index:03d}-{name}.f64'
                            value.numpy().astype('<f8',copy=False).tofile(path)
                            arrays[name] = dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size,
                                shape=list(value.shape),dtype='<f8')
                        snapshots.append(dict(index=index,time_s=(index+1)/60,arrays=arrays))
                    pending = None
                elif packet['kind'] == 'complete': complete = True; break
                else: raise ValueError('Unknown packet')
            child.stdin.close()
            if child.stdout.read(1) != b'' or child.wait(timeout=10) != 0:
                raise ValueError('Native worker failed to drain')
    completed = complete and len(frames)==scene['steps'] and all(f['advanced'] for f in frames)
    result = dict(status='completed' if completed else 'pressure_rejection',scene=scene,
        frames=frames,snapshots=snapshots,callbacks=callbacks,setup=ready,wall_seconds=time.perf_counter()-tick,
        native_process=dict(pid=child.pid,returncode=child.returncode,reaped=child.poll() is not None),
        source_sha256=sha(__file__),scene_sha256=sha(args.scene),build_receipt_sha256=sha(args.build/'build-receipt.json'),
        cpu=args.cpu,threads=1,optimizer_updates=0,peak_worker_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_native_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        teacher='Native Rheon advection/forcing/viscosity/projection with exact sealed-grid spectral pressure; original physical gates. Not original-CG trajectory bytes.',
        grid_dimensions=scene['counts'],cells_per_metre=scene['cells_per_metre'],physical_extent_m=[n/scene['cells_per_metre'] for n in scene['counts']])
    write(args.output/'receipt.json',result)
    if not completed: raise ValueError('Native spectral teacher rejected')
    print('Captured',scene['counts'],scene['cells_per_metre'],len(frames),'frames',flush=True)
    if frames:
        print('Median preparation ms', {k:round(1000*statistics.median(f['preparation_profile'][k] for f in frames),3) for k in frames[0]['preparation_profile']},flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=23)
    parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('--save-fields',action='store_true')
    run(parser.parse_args())
