"""Verify registered completed development windows, including sealed complete subsets."""
import argparse
import hashlib
import os
from pathlib import Path
import time
from support import ROOT, read, write, digest
import numpy as np
import torch


def run(args):
    if args.output.exists():
        raise ValueError('Fresh supplement integrity result required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Registered supplement audit source changed: '+path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if reg['split'] != 'development':
        raise ValueError('Development-only archive verification required; reserved access forbidden')
    supplemental = read(ROOT/reg['supplement'])
    baseline = read(ROOT/reg['baseline_audit'])
    if supplemental['status'] != reg['expected_controller_status'] or supplemental['active'] is not None:
        raise ValueError('Registered terminal native supplement required')
    if supplemental['status'] == 'failed' and supplemental.get('error') != reg['expected_terminal_error']:
        raise ValueError('Only the explicitly recorded storage-stop subset may be verified')
    if baseline['status'] != 'completed_expanded_archive_integrity_audit':
        raise ValueError('Actual baseline archive integrity evidence required')
    previous = {r['path']: r for r in baseline['files']}
    if len(supplemental['native_processes']) < 4*len(supplemental['completed']) or not all(
        r['reaped'] and r['returncode'] == 0 and r['graceful_protocol_complete']
        for r in supplemental['native_processes']):
        raise ValueError('All native window owners must be reaped')
    result = dict(status='running', active=None, rows=[], decoded_new_states=0,
        source_controller_status=supplemental['status'],
        extra_closed_children_from_uncommitted_partial_scene=len(supplemental['native_processes'])-4*len(supplemental['completed']),
        scope='Only complete registered scene rows; uncommitted partial scene outputs/counters are excluded.',
        verified_reused_states=0, bytes_hashed=0, native_processes_reaped=len(supplemental['native_processes']),
        reserved_references_opened=0, field_arrays_decoded=0, optimizer_updates=0,
        latency_claim=False, temporal_convergence_claim=False,
        protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
    write(args.output, result)
    began = time.monotonic()
    try:
        for row in supplemental['completed']:
            if row['split'] != reg['split'] or [w['start'] for w in row['windows']] != reg['starts']:
                raise ValueError('Full registered development window population required')
            result['active'] = dict(identifier=row['identifier'])
            for window in row['windows']:
                source = previous[window['current_state_path']]
                if source['sha256'] != window['current_state_sha256'] or digest(ROOT/source['path']) != source['sha256']:
                    raise ValueError('Saved source state differs from verified baseline')
                if [v['lead'] for v in window['outputs']] != [1, 2, 3, 4]:
                    raise ValueError('Every intermediate lead required')
                for output in window['outputs']:
                    if time.monotonic()-began > reg['maximum_wall_seconds']:
                        raise TimeoutError('Finite supplement integrity budget reached')
                    path = ROOT/output['path']
                    if digest(path) != output['sha256']:
                        raise ValueError('Saved complete output archive changed')
                    result['bytes_hashed'] += path.stat().st_size
                    if result['bytes_hashed'] > reg['maximum_archive_bytes']:
                        raise ValueError('Finite supplement archive budget reached')
                    if (output['step'] != window['start']+output['lead']
                            or output['time_s'] != output['step']/60
                            or output['native_step'] != output['step']*row['reference_substeps']
                            or not output['frame']['frame']['advanced']
                            or not output['frame']['frame']['quality']['qualified']):
                        raise ValueError('Complete actual reference clock/quality proof required')
                    if output['existing_reference_reused']:
                        old = previous[output['path']]
                        declared = {k: {a: v[a] for a in ('dtype', 'shape', 'sha256')}
                                    for k, v in old['fields'].items()}
                        if old['sha256'] != output['sha256'] or declared != output['arrays']:
                            raise ValueError('Reused state differs from decoded baseline evidence')
                        result['verified_reused_states'] += 1
                    else:
                        with np.load(path, allow_pickle=False) as archive:
                            if set(archive.files) != set(output['arrays']):
                                raise ValueError('All saved physical and request fields required')
                            for name, expected in output['arrays'].items():
                                value = archive[name]
                                actual = dict(dtype=value.dtype.str, shape=list(value.shape),
                                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
                                if actual != expected or not np.isfinite(value).all():
                                    raise ValueError('Decoded complete output bytes/dtype/finite check failed')
                                result['field_arrays_decoded'] += 1
                        if digest(path) != output['sha256']:
                            raise ValueError('Archive changed during decoding')
                        result['decoded_new_states'] += 1
            result['rows'].append(dict(identifier=row['identifier'], counts_xyz=row['counts_xyz'],
                cells_per_metre=row['cells_per_metre'], reference_substeps=row['reference_substeps'],
                windows=4, complete_output_labels=16, all_intermediate_labels_verified=True))
            result['active'] = None
            write(args.output, result)
        if (result['decoded_new_states'] != reg['expected_new_states']
                or result['verified_reused_states'] != reg['expected_reused_states']
                or len(result['rows']) != reg['expected_scenes']):
            raise ValueError('Full registered supplement population required')
        for path, expected in reg['source_sha256'].items():
            if digest(ROOT/path) != expected:
                raise ValueError('Supplement audit source changed during verification: '+path)
        result.update(status='completed_development_window_archive_integrity_audit',
            every_new_saved_field_decoded_and_hash_verified=True,
            every_reused_archive_hash_and_decoded_baseline_provenance_verified=True)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['active'] = None
        write(args.output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
