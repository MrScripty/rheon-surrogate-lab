"""Capture240/s reference windows with separately registered archive storage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from capture_memory_telemetry_v1 import process_memory
from fine_reference_client_v2 import FineReferenceClient
from long_reference_scene_v1 import initial_state
from reserve_long_scenes_v1 import instantiate

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')
OUTPUT_OFFSETS = (0, 2, 4, 6, 8, 12, 16, 24, 32)


def selected_windows(row, scene, reg):
    starts = set(reg['base_fine_starts'])
    if row['profile'] in ('regular_pulse', 'irregular_pulse'):
        pulses = [frame['pulse'] for frame in scene['forcing_schedule']]
        edges = [i for i in range(1, len(pulses)) if pulses[i] != pulses[i - 1]]
        for target in reg['event_selection_centres']:
            if edges:
                edge = min(edges, key=lambda i: abs(i - target))
                starts.add(max(0, min(edge - 8, 4800 - 32)))
    if any(not 0 <= n <= 4800 - 32 for n in starts):
        raise ValueError('Full K4/30 interval must fit within each master trajectory')
    retained = {start + offset for start in starts for offset in OUTPUT_OFFSETS}
    retained.update(start + offset for start in reg['fine_correction_starts'] for offset in range(33))
    return sorted(starts), retained


def run(args):
    args.output = args.output.resolve()
    args.protocol = args.protocol.resolve()
    if args.output.exists():
        raise ValueError('Fresh long-reference capture directory required')
    if not args.output.is_relative_to(ROOT):
        raise ValueError('Receipt and native-log root must remain in the project')
    reg = read(args.protocol)
    archive_root = Path(reg['state_archive_root']).resolve()
    if archive_root.exists():
        raise ValueError('Fresh independently owned archive root required')
    parent = archive_root.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < reg['required_archive_admission_bytes']:
        raise ValueError('Declared archive admission exceeds currently free storage')
    for name, h in reg['source_sha256'].items():
        if digest(ROOT / name) != h:
            raise ValueError('Registered capture source changed: ' + name)
    qualified = read(ROOT / reg['qualification'])
    if qualified['status'] != 'passed_long_reference_fine_clock_forces_restart_and_duration':
        raise ValueError('Actual fine-clock and complete-state restart qualification required')
    candidate_qualification = read(ROOT / reg['finite_f32_qualification'])
    if candidate_qualification['status'] != 'passed_finite_f32_local_original_replay_and_rejected_prefix_recovery':
        raise ValueError('Explicit finite-F32 reference regression qualification required')
    reserved = read(ROOT / reg['reservation'])
    rows = [row for row in reserved['rows'] if row['identifier'] in reg['identifiers']]
    if len(rows) != len(reg['identifiers']) or any(row['split'] not in ('train', 'development') for row in rows):
        raise ValueError('Exact nonreserved registered scenario population required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    archive_root.mkdir(parents=True)
    began = time.monotonic()
    client = None
    stored = 0
    bank = dict(status='running', active=None, completed=[], reference_rejections=[],
                controller_pid=os.getpid(), source_sha256=reg['source_sha256'],
                protocol_sha256=digest(args.protocol), reference_rate_per_s=240,
                reference_dt_s=1 / 240, duration_per_master_s=20,
                numerical_reference_only=True, new_parameter_fits=0, learned_forwards=0,
                cpu_model_forward=False, reserved52_references_opened=0,
                full_generator_target_pass=False,
                timing_status='operational_reference_capture_not_model_benchmark',
                state_archive_root=str(archive_root), external_archives=True,
                reference_temporal_convergence_qualified=False,
                reference_numerical_contract='experimental finite IEEE F32',
                native_build=reg['native_build'],
                training_status_at_capture='candidate; full capture and model compatibility qualification pending')

    def save():
        bank.update(updated_unix=time.time(), stored_state_bytes=stored)
        write(args.output / 'bank.json', bank)

    def bounded():
        if time.monotonic() - began > reg['maximum_wall_seconds']:
            raise TimeoutError('Registered finite long-reference capture wall budget')
        if stored > reg['maximum_capture_bytes']:
            raise ValueError('Registered retained-state capture budget exceeded')
        if shutil.disk_usage(args.output).free < reg['minimum_free_disk_bytes']:
            raise ValueError('Registered free-disk floor reached')
        if shutil.disk_usage(archive_root).free < reg['minimum_free_disk_bytes']:
            raise ValueError('Registered archive-volume free-disk floor reached')
        if process_memory()['peak_since_exec_bytes'] > reg['maximum_peak_rss_bytes']:
            raise ValueError('Registered reference process-memory budget exceeded')

    def stop(*unused):
        raise InterruptedError('Stop long reference capture and reap owned native child')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        for row in rows:
            bounded()
            scene = instantiate(row)
            starts, retained = selected_windows(row, scene, reg)
            directory = args.output / row['identifier']
            directory.mkdir()
            archive_directory = archive_root / row['identifier']
            archive_directory.mkdir()
            tick = time.monotonic()
            client = FineReferenceClient(scene, directory / 'native', build=ROOT / reg['native_build'])
            client.restore(0, initial_state(scene, row['initial_state_recipe']))
            bank['active'] = dict(identifier=row['identifier'], native_pid=client.child.pid,
                                  fine_step=0, physical_time_s=0.)
            save()
            states, frames = [], []
            rejected = None
            maxima = {n: 0. for n in ('courant', 'actual_divergence', 'predicted_divergence', 'residual_l2')}
            try:
                for step in range(4801):
                    if step:
                        try:
                            packet = client.advance()
                        except EOFError as advance_error:
                            stderr = (directory / 'native/native-stderr.log').read_text()
                            if 'Newtonian box viscosity rejected operation: ArithmeticFailure' not in stderr:
                                raise
                            close_error = None
                            try:
                                client.close()
                            except (OSError, EOFError) as error:
                                close_error = repr(error)
                            custody = client.last_custody
                            if custody is None or not custody['reaped'] or custody['returncode'] != 1:
                                raise ValueError('Native arithmetic rejection must retain exact terminal custody') from advance_error
                            rejected = dict(identifier=row['identifier'], expected_fine_step=step,
                                physical_time_s=step / 240, rate_per_s=240,
                                kind='native_viscosity_arithmetic_rejection',
                                original_advance_error=repr(advance_error), close_error=close_error,
                                native_stderr=stderr, native_process=custody,
                                successful_fine_updates=len(frames),
                                no_thresholds_or_controls_changed=True)
                            write(directory / 'reference-rejection.json', rejected)
                            write(directory / 'failed-native-custody.json', custody)
                            client = None
                            break
                        frames.append(packet)
                        frame = packet['frame']
                        if (not frame['advanced'] or frame['step'] != step
                                or not frame['quality']['qualified'] or frame['quality']['hard_stop']):
                            rejected = dict(identifier=row['identifier'], expected_fine_step=step,
                                            physical_time_s=step / 240, frame=packet,
                                            rate_per_s=240, no_thresholds_or_controls_changed=True)
                            write(directory / 'reference-rejection.json', rejected)
                            break
                        for n in maxima:
                            maxima[n] = max(maxima[n], frame['quality'][n])
                    if step in retained:
                        packet, values = client.inspect()
                        if packet['step'] != step or packet['time_s'] != step / 240:
                            raise ValueError('Captured fine-state timestamp differs')
                        for target, source in [('rhs', 'rhs'), ('native_input_residual', 'native_input_residual'), ('pressure_base', 'base')]:
                            values[target] = (np.zeros_like(values['pressure']) if step == 0
                                              else client.last_request[source].copy())
                        if any(not np.isfinite(value).all() for value in values.values()):
                            raise ValueError('Nonfinite complete reference label')
                        path = archive_directory / f'fine-{step:05d}.npz'
                        np.savez_compressed(path, **values)
                        stored += path.stat().st_size
                        states.append(dict(fine_step=step, physical_time_s=step / 240,
                            path=str(path), bytes=path.stat().st_size, sha256=digest(path),
                            reference_rhs_available=step != 0,
                            arrays={n: dict(dtype=v.dtype.str, shape=list(v.shape),
                                            sha256=hashlib.sha256(v.tobytes()).hexdigest())
                                    for n, v in values.items()}))
                    if step % 16 == 0:
                        bounded()
                    if step % 240 == 0:
                        bank['active'].update(fine_step=step, physical_time_s=step / 240)
                        save()
                        print('Long reference', row['identifier'], step, '/4800', flush=True)
                if client is not None:
                    custody = client.close()
                    client = None
                if not custody['reaped']:
                    raise ValueError('Owned native reference must reap')
                if rejected is None and (custody['returncode'] or not custody['graceful_protocol_complete']):
                    raise ValueError('Successful native reference must complete and drain')
                successful_updates = sum(bool(f['frame']['advanced']) for f in frames)
                native_error_without_frame = rejected is not None and rejected.get('kind') == 'native_viscosity_arithmetic_rejection'
                windows = []
                stored_steps = {r['fine_step'] for r in states}
                if rejected is None:
                    if len(frames) != 4800 or any(n not in stored_steps for n in retained):
                        raise ValueError('Incomplete actual reference trajectory or labels')
                    for start in starts:
                        for rate in (30, 60, 120):
                            stride = 240 // rate
                            for k in (1, 2, 3, 4):
                                targets = [start + j * stride for j in range(1, k + 1)]
                                if any(n not in stored_steps for n in [start, *targets]):
                                    raise ValueError('Missing complete all12 temporal-block labels')
                                windows.append(dict(start_fine_step=start, output_steps_per_s=rate,
                                    output_dt_s=1 / rate, K=k, block_duration_s=k / rate,
                                    target_fine_steps=targets))
                receipt = dict(status='completed' if rejected is None else 'reference_rejected',
                    reference_numerical_contract='experimental finite IEEE F32',
                    native_build=reg['native_build'],
                    training_status_at_capture='candidate; full capture and model compatibility qualification pending',
                    identifier=row['identifier'], physical_scenario_group_id=row['physical_scenario_group_id'],
                    split=row['split'], scene=scene, initial_state_recipe=row['initial_state_recipe'],
                    reference_rate_per_s=240, reference_dt_s=1 / 240,
                    actual_reference_updates=successful_updates,
                    attempted_reference_updates=len(frames) + int(native_error_without_frame), physical_duration_s=successful_updates / 240,
                    states=states, windows=windows, fine_correction_starts=reg['fine_correction_starts'],
                    fine_correction_all32_future_states_retained=True if rejected is None else False,
                    frames=frames, quality_maxima=maxima, native_process=custody,
                    elapsed_seconds=time.monotonic() - tick,
                    retained_state_bytes=sum(r['bytes'] for r in states),
                    all_unretained_steps_actually_integrated=True,
                    fine_controls=row['fine_controls'], source_sha256=reg['source_sha256'],
                    protocol_sha256=digest(args.protocol), future_reference_fields_are_targets_only=True,
                    temporal_convergence_qualified=False, model_target_claim=False,
                    state_archive_root=str(archive_directory),
                    archive_path_contract='Absolute registered NPZ paths; exact native arrays unchanged',
                    pressure_label_time_semantics='Native projection pressure from the final fine interval, whose force sample is at endpoint_time-reference_dt; not a separate endpoint-time pressure solve')
                write(directory / 'receipt.json', receipt)
                record = dict(identifier=row['identifier'], split=row['split'],
                    cells_per_metre=scene['cells_per_metre'], counts_xyz=scene['counts'],
                    receipt=str((directory / 'receipt.json').relative_to(ROOT)),
                    receipt_sha256=digest(directory / 'receipt.json'),
                    fine_updates=receipt['actual_reference_updates'], stored_states=len(states),
                    retained_state_bytes=receipt['retained_state_bytes'],
                    elapsed_seconds=receipt['elapsed_seconds'], quality_maxima=maxima)
                bank['completed' if rejected is None else 'reference_rejections'].append(record)
                bank['active'] = None
                save()
            finally:
                if client is not None:
                    try:
                        client.close()
                    finally:
                        if client.last_custody is not None:
                            write(directory / 'failed-native-custody.json', client.last_custody)
                        client = None
        if torch.cuda.is_initialized():
            raise ValueError('Numerical reference capture unexpectedly initializedCUDA')
        bank.update(status='completed' if not bank['reference_rejections'] else 'completed_with_reference_rejections',
                    elapsed_seconds=time.monotonic() - began, memory_counters=process_memory())
    except BaseException as error:
        bank.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic() - began)
        raise
    finally:
        if client is not None:
            try:
                client.close()
            finally:
                if client.last_custody is not None:
                    write(directory / 'failed-native-custody.json', client.last_custody)
        bank['active'] = None
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
