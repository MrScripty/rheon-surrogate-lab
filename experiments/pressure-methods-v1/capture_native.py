"""Independently qualify versioned larger-grid and substep native pilot captures."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LAB = ROOT/'runs/dependency-migration-v1/lab'
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


ORACLE_PATH = LAB/'experiments/global-pressure-v1/verify_capture.py'
STREAM_PATH = LAB/'experiments/global-pressure-v1/capture.py'
oracle = module('verify_capture', ORACLE_PATH)
stream_owner = module('pressure_methods_capture_stream', STREAM_PATH)
FIELDS = dict(base='<f8', rhs='<f8', pressure='<f8', native_input_residual='<f8', tracer='<f4',
              **{prefix+'_'+axis:'<f4' for prefix in ('pre', 'corrected') for axis in 'uvw'})


def shape(name, counts):
    dims = list(counts)
    if name.startswith(('pre_', 'corrected_')):
        dims['uvw'.index(name[-1])] += 1
    return dims


def check_frame(directory, row, frame, dt):
    counts = row['counts_xyz']
    h = [1/row['cells_per_metre']]*3
    rho = 1000.
    values = {}
    if set(frame['arrays']) != set(FIELDS):
        raise ValueError('Complete closed frame fields required')
    for name, declaration in frame['arrays'].items():
        path = directory/declaration['path']
        expected = math.prod(shape(name, counts))*np.dtype(FIELDS[name]).itemsize
        if (expected > 4*1024**2 or path.stat().st_size != expected or sha(path) != declaration['sha256']
                or declaration['shape_xyz'] != shape(name, counts) or declaration['dtype'] != FIELDS[name]):
            raise ValueError('Raw field identity/shape/size failed: '+name)
        array = np.fromfile(path, dtype=FIELDS[name])
        if not bool(np.isfinite(array).all()):
            raise ValueError('Nonfinite retained field')
        values[name] = array.tolist()
    if values['base'][0] != 0. or values['pressure'][0] != 0.:
        raise ValueError('Changed pressure gauge')
    pre = [values['pre_'+a] for a in 'uvw']
    corrected = [values['corrected_'+a] for a in 'uvw']
    rebuilt = oracle.reconstruct_rhs(pre, counts, h, dt)
    if not np.allclose(rebuilt, values['rhs'], rtol=2e-14, atol=1e-14):
        raise ValueError('Independent RHS failed')
    product = oracle.full_action(values['pressure'], counts, h, rho)
    residual = np.asarray(values['rhs'])-product
    residual_norm = math.sqrt(math.fsum(float(v*v) for v in residual))
    rhs_norm = math.sqrt(math.fsum(v*v for v in values['rhs']))
    if residual_norm > max(1e-12, 1e-8*rhs_norm):
        raise ValueError('Reference residual gate failed')
    predicted_divergence = dt*float(np.abs(residual).max())/math.prod(h)
    if predicted_divergence > 1e-7:
        raise ValueError('Predicted divergence gate failed')
    base_action = oracle.full_action(values['base'], counts, h, rho)
    if not np.allclose(np.asarray(values['rhs'])-base_action, values['native_input_residual'], rtol=2e-12, atol=2e-14):
        raise ValueError('Independent initial residual failed')
    divergence = oracle.divergence(corrected, counts, h)
    actual = max(map(abs, divergence))
    if actual > 1e-5:
        raise ValueError('Actual float32 divergence gate failed')
    net = math.fsum(divergence)*math.prod(h)
    absolute = math.fsum(map(abs, divergence))*math.prod(h)
    if abs(net) > max(1e-12, 1e-8*absolute):
        raise ValueError('Sealed net flux gate failed')
    for d, axis in enumerate('uvw'):
        dims = shape('corrected_'+axis, counts)
        array = np.asarray(values['corrected_'+axis]).reshape(dims[::-1])
        if not bool((np.take(array, [0, dims[d]-1], axis=2-d) == 0).all()):
            raise ValueError('Wall flux gate failed')
    courant = dt*math.fsum(max(map(abs, corrected[d]))/h[d] for d in range(3))
    if courant > .5:
        raise ValueError('Courant gate failed')
    viscosity = frame['viscosity']
    if (viscosity['dynamic_viscosity_pa_s'] != .001
            or viscosity['kinematic_viscosity_m2_s'] != .001/rho
            or viscosity['kinetic_after'] > viscosity['kinetic_before']*(1+1e-12)+1e-15):
        raise ValueError('Water viscosity/dissipation gate failed')
    return dict(reference_relative_residual=residual_norm/rhs_norm if rhs_norm else 0.,
        predicted_divergence_max_s_inv=predicted_divergence, actual_divergence_max_s_inv=actual,
        net_flux_m3_s=net, courant=courant)


def capture(build, row, output, fixture):
    receipt = read(build/'build-receipt.json')
    reservation = read(build/'reservation.json')
    binary = build/'rheon-pressure-methods-capture'
    if receipt['status'] != 'qualified_build' or sha(binary) != receipt['binary_sha256']:
        raise ValueError('Qualified native binary required')
    if read(build/'binding.json') != receipt['identity']['binding']:
        raise ValueError('Changed build binding')
    for path, expected in receipt['identity']['binding']['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Changed bound native source '+path)
    if sha(build/'reservation.json') != receipt['identity']['binding']['reservation_sha256']:
        raise ValueError('Changed reservation')
    if row not in reservation['scenes'] or row['split'] != 'train':
        raise ValueError('Only exactly reserved training pilots allowed')
    substeps = row['teacher_substeps']
    config = reservation['configuration']
    dt = config['dt_s']/substeps
    selected = [0] if fixture else [(n+1)*substeps-1 for n in config['snapshot_steps']]
    steps = 1 if fixture else config['steps']*substeps
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    state = dict(status='running', row=row, fixture=fixture, model_dt_s=config['dt_s'], native_dt_s=dt,
        reservation_sha256=sha(build/'reservation.json'), binary_sha256=sha(binary),
        source_sha256=sha(__file__), oracle_sha256=sha(ORACLE_PATH), stream_owner_sha256=sha(STREAM_PATH),
        pressure_label_scope='Native fine-step pressure, not a substituted label for a coarse-step request')
    write(output/'receipt.json', state)
    tick = time.perf_counter()
    child = None
    try:
        with (output/'native-stderr.log').open('xb') as stderr:
            with stream_owner.bounded_process([str(binary), '--produce'], 600,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                    env={**os.environ, 'RAYON_NUM_THREADS':'1'}) as child:
                child.stdin.write(json.dumps(dict(trajectory_id=row['trajectory_id'], fixture=fixture)).encode())
                child.stdin.close()
                payload = 0
                for step in selected:
                    size = struct.unpack('<I', stream_owner.read_exact(child.stdout, 4))[0]
                    if not 0 < size <= 65536:
                        raise ValueError('Snapshot header bound')
                    frame = json.loads(stream_owner.read_exact(child.stdout, size))
                    if frame['kind'] != 'frame' or frame['index'] != step or frame['time_s'] != (step+1)*dt:
                        raise ValueError('Physical snapshot time/index mismatch')
                    inventory = {}
                    for descriptor in frame.pop('arrays'):
                        name = descriptor.pop('name')
                        if name not in FIELDS or name in inventory:
                            raise ValueError('Unexpected/duplicate field')
                        expected = math.prod(shape(name, row['counts_xyz']))*np.dtype(FIELDS[name]).itemsize
                        if descriptor['bytes'] != expected or not 0 < expected <= 4*1024**2:
                            raise ValueError('Array byte bound')
                        raw = stream_owner.read_exact(child.stdout, expected)
                        payload += len(raw)
                        if payload > 64*1024**2:
                            raise ValueError('Per-trajectory payload budget exceeded')
                        path = output/f'frame-{step:03d}-{name}.bin'
                        path.write_bytes(raw)
                        inventory[name] = dict(**descriptor, path=path.name, sha256=sha(path))
                    frame['arrays'] = inventory
                    frame['independent'] = check_frame(output, row, frame, dt)
                    frames.append(frame)
                size = struct.unpack('<I', stream_owner.read_exact(child.stdout, 4))[0]
                if not 0 < size <= 65536:
                    raise ValueError('Completion header bound')
                complete = json.loads(stream_owner.read_exact(child.stdout, size))
                if complete != dict(kind='complete', steps=steps, snapshots=len(selected), native_steps_qualified=steps):
                    raise ValueError('Incomplete native physical qualification')
                if child.stdout.read(1) != b'' or child.wait(timeout=10) != 0:
                    raise ValueError('Native producer did not drain')
        state.update(status='qualified_fixture' if fixture else 'qualified_pilot', frames=frames,
            native_steps_qualified=steps, payload_bytes=payload)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', frames=frames)
        raise
    finally:
        state.update(elapsed_s=time.perf_counter()-tick,
            native_process=None if child is None else dict(pid=child.pid, returncode=child.returncode, reaped=child.poll() is not None))
        write(output/'receipt.json', state)
    print(row['trajectory_id'], state['status'], state['elapsed_s'], flush=True)
    return state


if __name__ == '__main__':
    def stop(*_):
        raise InterruptedError('Capture interrupted; stream owner reaps native producer')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trajectory', required=True)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    reservation = read(args.build/'reservation.json')
    row = next(r for r in reservation['scenes'] if r['trajectory_id'] == args.trajectory)
    capture(args.build.resolve(), row, args.output.resolve(), args.fixture)
