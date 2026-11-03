"""Offline complete-state windows, verified against uninterrupted native replay.

Training labels only. Every restart and intermediate field is checked against
the same native lineage; this is not a learned candidate or timing benchmark.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import signal
import time
from support import ROOT, read, write, digest
from native_forcing_client import NativeForcingClient
from continuous_reference_client import ContinuousReferenceClient
import numpy as np
import torch

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')
LABELS = ('rhs', 'native_input_residual', 'pressure_base')


def identities(values):
    return {k: dict(dtype=v.dtype.str, shape=list(v.shape),
                    sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in values.items()}


def checked_frame(client, step):
    frame = client.advance()
    value = frame['frame']
    if not value['advanced'] or value['step'] != step or not value['quality']['qualified'] or value['quality']['hard_stop']:
        raise ValueError('Native quality gate rejected a reference advance')
    return frame


def inspect_labels(client, step, substeps):
    packet, values = client.inspect()
    if packet['step'] != step or packet['time_s'] != step/60:
        raise ValueError('Actual reference output clock changed')
    if substeps != 1 and packet['native_step'] != step*substeps:
        raise ValueError('Actual native work changed')
    for target, source in zip(LABELS, ('rhs', 'native_input_residual', 'base')):
        values[target] = client.last_request[source].copy()
    return values


def run(args):
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh verified window output required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Registered reference source changed: '+path)
    if reg['maximum_lead'] != 4 or not reg['uninterrupted_verification']:
        raise ValueError('All four leads and an uninterrupted control are required')
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    client = None
    rows = []
    custody = []
    frames_executed = 0
    reference_steps_executed = 0
    stored_bytes = 0

    def stop(*unused):
        raise InterruptedError('Stop and reap the owned native reference')

    def bounded():
        if time.monotonic()-started > reg['maximum_wall_seconds']:
            raise TimeoutError('Finite reference window budget reached')
        if shutil.disk_usage(args.output).free < reg['minimum_free_disk_bytes']:
            raise ValueError('Reference window disk floor reached')

    def close():
        nonlocal client
        if client is None:
            return
        owner = client.close()
        client = None
        custody.append(owner)
        if not owner['reaped'] or owner['returncode'] != 0 or not owner['graceful_protocol_complete']:
            raise ValueError('Native reference must complete gracefully and reap')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    result = dict(status='running', rows=rows, native_processes=custody,
                  protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'],
                  reference_only=True, reserved_references_consumed=False,
                  optimizer_updates=0, latency_claim=False, full_generator_target_pass=False)
    write(args.output/'results.json', result)
    try:
        for selected in reg['trajectories']:
            path = ROOT/selected['receipt']
            if digest(path) != selected['receipt_sha256']:
                raise ValueError('Training receipt changed')
            receipt = read(path)
            if receipt['split'] != 'train' or receipt['status'] != 'completed':
                raise ValueError('Completed training scenes only')
            scene = receipt['scene']
            substeps = receipt.get('reference_substeps', 1)
            factory = ContinuousReferenceClient if substeps != 1 else NativeForcingClient
            specs = {s['step']: s for s in receipt['states']}
            desired = {s+lead for s in reg['starts'] for lead in range(1, 5)}
            if any(s not in specs or s+4 > scene['steps'] for s in reg['starts']):
                raise ValueError('Registered current states and full four-step intervals required')
            directory = args.output/selected['identifier']
            control = {}
            control_frames = {}
            client = factory(scene, directory/'uninterrupted-control')
            # This independent replay advances from the actual native initial state.
            for step in range(1, max(desired)+1):
                bounded()
                frame = checked_frame(client, step)
                frames_executed += 1
                reference_steps_executed += substeps
                if step in desired:
                    values = inspect_labels(client, step, substeps)
                    control[step] = identities(values)
                    control_frames[step] = frame
                    if step in specs:
                        source = path.parent/specs[step]['path']
                        if digest(source) != specs[step]['sha256']:
                            raise ValueError('Existing reference bytes changed')
                        with np.load(source, allow_pickle=False) as archive:
                            expected = {k: archive[k].copy() for k in (*FIELDS, *LABELS)}
                        if identities(expected) != control[step]:
                            raise ValueError('Uninterrupted replay differs from original capture')
            close()
            for start in reg['starts']:
                bounded()
                source = path.parent/specs[start]['path']
                if digest(source) != specs[start]['sha256']:
                    raise ValueError('Saved current-state bytes changed')
                with np.load(source, allow_pickle=False) as archive:
                    initial = {k: archive[k].copy() for k in FIELDS}
                block = directory/f'start-{start:03d}'
                client = factory(scene, block)
                client.restore(start, initial)
                packet, restored = client.inspect()
                if packet['step'] != start or identities(restored) != identities(initial):
                    raise ValueError('Native restore changed the current complete state')
                outputs = []
                for lead in range(1, 5):
                    bounded()
                    step = start+lead
                    frame = checked_frame(client, step)
                    frames_executed += 1
                    reference_steps_executed += substeps
                    values = inspect_labels(client, step, substeps)
                    actual = identities(values)
                    if actual != control[step]:
                        raise ValueError('Restarted window differs from uninterrupted complete output')
                    output = block/f'state-{step:03d}.npz'
                    np.savez_compressed(output, **values)
                    stored_bytes += output.stat().st_size
                    if stored_bytes > reg['maximum_capture_bytes']:
                        raise ValueError('Finite window storage cap reached')
                    outputs.append(dict(lead=lead, step=step, time_s=step/60,
                        native_step=step*substeps, path=str(output.relative_to(args.output)),
                        sha256=digest(output), arrays=actual, frame=frame,
                        uninterrupted_frame=control_frames[step], bitwise_equal_uninterrupted=True))
                close()
                rows.append(dict(identifier=selected['identifier'], split='train', group=receipt['group'],
                    cells_per_metre=scene['cells_per_metre'], counts_xyz=scene['counts'], start=start,
                    source_current_state=str(source.relative_to(ROOT)), source_sha256=digest(source),
                    reference_substeps=substeps, reference_dt_s=1/(60*substeps), output_dt_s=1/60,
                    forcing_control_semantics='Macro-held sampled controls; original source lineage',
                    source_receipt=selected['receipt'], source_receipt_sha256=selected['receipt_sha256'],
                    outputs=outputs))
                write(args.output/'results.json', result)
                print('Verified native window', selected['identifier'], start, 'all4complete labels', flush=True)
        expected = len(reg['trajectories'])*len(reg['starts'])
        if len(rows) != expected:
            raise ValueError('Full registered pilot population required')
        for path, expected_hash in reg['source_sha256'].items():
            if digest(ROOT/path) != expected_hash:
                raise ValueError('Reference source changed during capture: '+path)
        result.update(status='passed_complete_multistep_reference_windows',
            complete_output_states=len(rows)*4, physical_advances_executed=frames_executed,
            native_fine_advances_executed=reference_steps_executed, stored_bytes=stored_bytes,
            every_output_byte_compared_with_uninterrupted_native_control=True,
            temporal_convergence_claim=False)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        try:
            close()
        finally:
            write(args.output/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
