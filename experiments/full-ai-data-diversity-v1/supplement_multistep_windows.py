"""Add actual native K1-4 labels to completed expanded training scenes.

Reuses existing state files and writes missing leads only. Restart semantics
are qualified separately; each available prior state is rechecked bytewise.
"""
import argparse
import os
from pathlib import Path
import shutil
import signal
import time
from support import ROOT, read, write, digest
from capture_multistep_windows import FIELDS, LABELS, identities, checked_frame, inspect_labels
from native_forcing_client import NativeForcingClient
from continuous_reference_client import ContinuousReferenceClient
import numpy as np
import torch


def run(args):
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh multistep supplement directory required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Registered supplement source changed: '+path)
    qualification = read(ROOT/reg['qualification'])
    if (qualification['status'] != 'passed_complete_multistep_reference_windows'
            or qualification['complete_output_states'] != 32
            or not qualification['every_output_byte_compared_with_uninterrupted_native_control']
            or not all(c['reaped'] and c['returncode'] == 0 and c['graceful_protocol_complete']
                       for c in qualification['native_processes'])):
        raise ValueError('Actual uninterrupted/restore qualification required')
    if reg['maximum_lead'] != 4:
        raise ValueError('All four intermediate outputs required')
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    child = None
    record = dict(status='running', completed=[], active=None, native_processes=[],
        protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'],
        expected_scene_count=len(reg['trajectories']), all_expanded_train_scenes_required=reg['all_expanded_train_scenes_required'],
        pending_train_scenes_outside_this_snapshot=reg['pending_train_scenes_outside_this_snapshot'],
        generated_new_states=0, reused_existing_states=0, complete_output_labels=0,
        output_advances_executed=0, native_fine_advances_executed=0, stored_bytes=0,
        reference_only=True, reserved_references_consumed=False, optimizer_updates=0,
        model_targets_verified=False, latency_claim=False, temporal_convergence_claim=False)

    def save():
        record['updated_unix'] = time.time()
        write(args.output/'controller.json', record)

    def bounded(scene_started):
        if time.monotonic()-started > reg['maximum_controller_wall_seconds']:
            raise TimeoutError('Finite supplement controller budget reached')
        if time.monotonic()-scene_started > reg['maximum_scene_wall_seconds']:
            raise TimeoutError('Finite scene supplement budget reached')
        if shutil.disk_usage(args.output).free < reg['minimum_free_disk_bytes']:
            raise ValueError('Supplement free-disk floor reached')

    def close():
        nonlocal child
        if child is not None:
            owner = child.close()
            child = None
            record['native_processes'].append(owner)
            if not owner['reaped'] or owner['returncode'] != 0 or not owner['graceful_protocol_complete']:
                raise ValueError('Owned reference must complete gracefully and reap')

    def stop(*unused):
        raise InterruptedError('Stop supplementation and reap reference child')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        for selected in reg['trajectories']:
            scene_started = time.monotonic()
            path = ROOT/selected['receipt']
            if digest(path) != selected['receipt_sha256']:
                raise ValueError('Completed source receipt changed')
            receipt = read(path)
            if receipt['status'] != 'completed' or receipt['split'] != 'train':
                raise ValueError('Completed train-only scenes required')
            specs = {s['step']: s for s in receipt['states']}
            scene = receipt['scene']
            substeps = receipt.get('reference_substeps', 1)
            if substeps not in (1, 8):
                raise ValueError('Only separately qualified reference lineages allowed')
            factory = NativeForcingClient if substeps == 1 else ContinuousReferenceClient
            windows = []
            for start in reg['starts']:
                bounded(scene_started)
                if start not in specs or start+1 not in specs or start+4 > scene['steps']:
                    raise ValueError('Current state and immediate independent label required')
                source = path.parent/specs[start]['path']
                if digest(source) != specs[start]['sha256']:
                    raise ValueError('Current native state bytes changed')
                with np.load(source, allow_pickle=False) as archive:
                    initial = {k: archive[k].copy() for k in FIELDS}
                directory = args.output/selected['identifier']/f'start-{start:03d}'
                child = factory(scene, directory)
                record['active'] = dict(identifier=selected['identifier'], start=start,
                    pid=child.child.pid, output=str(directory.relative_to(ROOT)))
                save()
                child.restore(start, initial)
                packet, restored = child.inspect()
                if packet['step'] != start or identities(restored) != identities(initial):
                    raise ValueError('Complete native restore byte mismatch')
                outputs = []
                for lead in range(1, 5):
                    bounded(scene_started)
                    step = start+lead
                    frame = checked_frame(child, step)
                    values = inspect_labels(child, step, substeps)
                    actual = identities(values)
                    record['output_advances_executed'] += 1
                    record['native_fine_advances_executed'] += substeps
                    existing = step in specs
                    if existing:
                        target = path.parent/specs[step]['path']
                        if digest(target) != specs[step]['sha256']:
                            raise ValueError('Existing comparison label changed')
                        with np.load(target, allow_pickle=False) as archive:
                            expected = {k: archive[k].copy() for k in (*FIELDS, *LABELS)}
                        if actual != identities(expected):
                            raise ValueError('Restored continuation differs from independent existing native label')
                        record['reused_existing_states'] += 1
                    else:
                        target = directory/f'state-{step:03d}.npz'
                        np.savez_compressed(target, **values)
                        record['stored_bytes'] += target.stat().st_size
                        record['generated_new_states'] += 1
                        if record['stored_bytes'] > reg['maximum_total_capture_bytes']:
                            raise ValueError('Finite supplement storage budget reached')
                    outputs.append(dict(lead=lead, step=step, time_s=step/60,
                        native_step=step*substeps, path=str(target.relative_to(ROOT)),
                        sha256=digest(target), arrays=actual, frame=frame,
                        existing_reference_reused=existing,
                        existing_native_reference_bitwise_equal=True if existing else None,
                        full_uninterrupted_per_scene_comparison=False))
                    record['complete_output_labels'] += 1
                close()
                windows.append(dict(start=start, current_state_path=str(source.relative_to(ROOT)),
                    current_state_sha256=digest(source), outputs=outputs))
                record['active'] = None
                save()
            record['completed'].append(dict(identifier=selected['identifier'], split='train',
                group=receipt['group'], counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                profile=receipt['profile'], motion=receipt['motion'], source_receipt=selected['receipt'],
                source_receipt_sha256=selected['receipt_sha256'], reference_substeps=substeps,
                output_dt_s=1/60, reference_dt_s=1/(60*substeps),
                control_semantics='Original sampled macro controls held within output intervals', windows=windows))
            save()
            print('Supplemented', selected['identifier'], 'four mature-ready K1-4 windows', flush=True)
        for path, expected in reg['source_sha256'].items():
            if digest(ROOT/path) != expected:
                raise ValueError('Supplement source changed during capture: '+path)
        if record['complete_output_labels'] != len(reg['trajectories'])*len(reg['starts'])*4:
            raise ValueError('Every registered intermediate label required')
        record['status'] = 'completed_registered_training_window_supplements'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        try:
            close()
        finally:
            record['active'] = None
            save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
