"""Matched240vs480 local-horizon reference diagnostics; no learned inference.

Both clocks start from the same audited240 state. This measures local temporal
discrepancy; it does not establish global20-second integration convergence.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from fine_reference_client_v2 import FineReferenceClient
from continuous_force_program import evaluate

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')
OFFSETS = (2, 4, 6, 8, 12, 16, 24, 32)


def discrepancy(actual, reference):
    delta = np.concatenate([(actual[n].astype(np.float64) - reference[n]).ravel()
                            for n in actual])
    ref = np.concatenate([reference[n].astype(np.float64).ravel() for n in actual])
    absolute, norm = float(np.linalg.norm(delta)), float(np.linalg.norm(ref))
    return dict(relative_l2=absolute / max(norm, 1e-30), absolute_l2=absolute,
                reference_l2=norm, denominator_floor=1e-30)


def run(args):
    if args.output.exists():
        raise ValueError('Fresh matched-reference comparison output required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen reference comparison dependency changed: ' + name)
    qualification = read(ROOT / reg['finer_qualification'])
    if qualification['status'] != 'passed480_long_reference_fine_clock_forces_restart_and_duration':
        raise ValueError('New fine-state client must be explicitly qualified at480/s')
    audit = read(ROOT / reg['audit'])
    if audit['status'] != 'passed_completed_reference_labels_windows_and_late_replay':
        raise ValueError('Completed coarse master must first pass numerical audit')
    path = ROOT / reg['receipt']
    receipt = read(path)
    if receipt['identifier'] not in {c['identifier'] for c in audit['cases']}:
        raise ValueError('Coarse receipt is absent from supplied audit')
    coarse = receipt['scene']
    finer = deepcopy(coarse)
    finer.update(reference_substeps=8, forcing_control_substeps=8)
    finer['forcing_program']['dt'] = 1 / 480
    finer['forcing_schedule'] = [evaluate(finer['forcing_program'], i * (1 / 480))
                                for i in range(finer['steps'] * 8)]
    if finer['forcing_schedule'][::2] != coarse['forcing_schedule']:
        raise ValueError('Matched clocks changed the prescribed physical force program')
    states = {s['fine_step']: s for s in receipt['states']}

    def load(step):
        spec = states[step]
        source = path.parent / spec['path']
        if digest(source) != spec['sha256']:
            raise ValueError('Audited coarse state changed')
        with np.load(source, allow_pickle=False) as archive:
            return {n: archive[n].copy() for n in FIELDS}

    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', cases=[], native_processes=[], fine_updates=0,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        comparison='Matched initial state;240 master endpoints versus two480 fine advances per240 interval.',
        local_comparison_only=True, global20second_temporal_convergence_qualified=False,
        selected_training_reference_rate_per_s=240, new_parameter_fits=0,
        learned_forwards=0, cpu_model_forward=False, reserved52_references_opened=0,
        full_generator_target_pass=False,
        timing_status='resource_contention_operational_diagnostic_only')
    try:
        for start in reg['start_fine_steps240']:
            client = FineReferenceClient(finer, args.output / f'start-{start}')
            try:
                client.restore(2 * start, load(start))
                for fine in range(1, 65):
                    if time.monotonic() - started > reg['maximum_wall_seconds']:
                        raise TimeoutError('Registered local comparison wall budget')
                    packet = client.advance()
                    result['fine_updates'] += 1
                    if not packet['frame']['advanced'] or not packet['frame']['quality']['qualified']:
                        raise ValueError('Finer local reference failed native quality')
                    if fine % 2 == 0 and fine // 2 in OFFSETS:
                        offset = fine // 2
                        packet, value = client.inspect()
                        if packet['time_s'] != (start + offset) / 240:
                            raise ValueError('Clocks did not reach identical physical time')
                        coarse_value = load(start + offset)
                        errors = {}
                        for label, names in [('pressure', ('pressure',)),
                                             ('velocity', ('vx', 'vy', 'vz')),
                                             ('tracer', ('tracer',))]:
                            errors[label] = discrepancy({n: coarse_value[n] for n in names},
                                                        {n: value[n] for n in names})
                        result['cases'].append(dict(start_fine_step240=start,
                            start_time_s=start / 240, future_fine_offset240=offset,
                            elapsed_s=offset / 240, endpoint_time_s=(start + offset) / 240,
                            counts_xyz=coarse['counts'], cells_per_metre=coarse['cells_per_metre'],
                            force_profile=coarse['forcing_program']['profile'],
                            force_motion=coarse['forcing_program']['motion'], errors=errors))
            finally:
                try:
                    custody = client.close()
                finally:
                    if client.last_custody is not None:
                        result['native_processes'].append(client.last_custody)
                if custody['returncode'] or not custody['reaped'] or not custody['graceful_protocol_complete']:
                    raise ValueError('Finer reference child did not complete and drain')
            write(args.output / 'results.json', result)
        if torch.cuda.is_initialized():
            raise ValueError('Offline reference diagnostic initializedCUDA')
        maxima = {n: max(c['errors'][n]['relative_l2'] for c in result['cases'])
                  for n in ('pressure', 'velocity', 'tracer')}
        result.update(status='completed_local240vs480_reference_diagnostic',
            worst_relative_l2=maxima,
            all_local_discrepancies_within_1e_minus5=all(v <= 1e-5 for v in maxima.values()),
            elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
