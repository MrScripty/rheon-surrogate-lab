"""DEBUG: native-only boundary fixtures verified against uninterrupted hashes."""
import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT, read, write, digest
from continuous_reference_client import ContinuousReferenceClient

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')


def run(args):
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh diagnostic fixtures required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen fixture dependency changed: ' + path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    observed = read(ROOT / reg['observation'])
    references = {r['step']: r['actual_field_sha256'] for r in observed['reference_rows']}
    case = reg['case']
    scene = dict(read(ROOT / case['receipt'])['scene'],
                 reference_substeps=case['reference_substeps'], forcing_control_substeps=1)
    spec = next(s for s in case['states'] if s['step'] == reg['restore_step'])
    if digest(ROOT / spec['path']) != spec['sha256']:
        raise ValueError('Diagnostic native restart changed')
    with np.load(ROOT / spec['path'], allow_pickle=False) as archive:
        initial = {n: archive[n].copy() for n in FIELDS}
    args.output.mkdir(parents=True)
    native = ContinuousReferenceClient(scene, args.output / 'native', io_timeout_s=120.)
    result = dict(status='running', fixtures=[], cpu_model_forward=False,
                  native_restore_for_diagnosis_only=True, optimizer_updates=0)
    try:
        native.restore(reg['restore_step'], initial)
        for step in range(reg['restore_step'] + 1, max(reg['capture_steps']) + 1):
            frame = native.advance()
            if not frame['frame']['advanced']:
                raise ValueError('Native reference rejected diagnostic advance')
            if step not in reg['capture_steps']:
                continue
            packet, truth = native.inspect()
            truth.update(rhs=native.last_request['rhs'],
                         native_input_residual=native.last_request['native_input_residual'],
                         pressure_base=native.last_request['base'])
            hashes = {n: hashlib.sha256(v.tobytes()).hexdigest() for n, v in truth.items()}
            if packet['step'] != step or hashes != references[step]:
                raise ValueError('Restarted fixture differs from actual uninterrupted reference')
            path = args.output / f'reference{step}.npz'
            np.savez_compressed(path, **truth)
            if sum(p.stat().st_size for p in args.output.glob('*.npz')) > reg['maximum_fixture_bytes']:
                raise ValueError('Diagnostic fixture byte budget exceeded')
            result['fixtures'].append(dict(step=step, path=str(path.relative_to(ROOT)),
                                           sha256=digest(path), field_sha256=hashes,
                                           uninterrupted_reference_bitwise_equal=True))
            print('Verified actual native boundary fixture', step, flush=True)
        result['status'] = 'captured_verified_actual_native_boundary'
    finally:
        custody = native.close()
        result['native_process'] = custody
        result.update(protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
        if not custody['reaped'] or custody['returncode'] != 0 or not custody['graceful_protocol_complete']:
            result['status'] = 'native_custody_failed'
        write(args.output / 'receipt.json', result)
    if result['status'] != 'captured_verified_actual_native_boundary':
        raise ValueError('Native fixture did not complete cleanly')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
