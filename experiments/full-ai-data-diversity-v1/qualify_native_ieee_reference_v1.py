"""Compare a new finite-F32 reference contract with audited original snapshots.

Original accepted trajectories remain original evidence. This qualification
checks local late replays and short recovery/restart on the rejected scenes;
it does not claim full20second recovery or physical temporal convergence.
"""
import argparse
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index, LongTemporalDataset, FIELDS
from fine_reference_client_v2 import FineReferenceClient
from capture_memory_telemetry_v1 import process_memory


def hashes(values):
    return {n: hashlib.sha256(v.tobytes()).hexdigest() for n, v in values.items()}


def run(args):
    if args.output.exists():
        raise ValueError('Fresh reference policy qualification required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen qualification source changed: ' + name)
    build = ROOT / reg['candidate_build']
    receipt = read(build / 'build-receipt.json')
    if receipt['status'] != 'qualified_build' or not receipt['finite_f32_subnormals_supported']:
        raise ValueError('Explicit tested finite-F32 candidate required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', original_case_replays=[], recovered_case_prefixes=[],
        native_processes=[], actual_native_advances=0, original_complete_state_comparisons=0,
        new_parameter_fits=0, learned_forwards=0, cpu_model_forward=False,
        reserved52_references_opened=0, full_generator_target_pass=False,
        candidate_build=reg['candidate_build'], reference_cpu_policy=reg['reference_cpu_policy'],
        numerical_contract='finite IEEE F32 including subnormals; new experimental reference epoch',
        full20second_failed_case_recovery_proven=False, temporal_convergence_proven=False,
        adopted_for_training=False, source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))

    def bounded():
        if (time.monotonic() - started > reg['maximum_wall_seconds']
                or process_memory()['peak_since_exec_bytes'] > reg['maximum_peak_rss_bytes']):
            raise ValueError('Registered reference qualification budget exceeded')

    def close(client):
        try:
            custody = client.close()
        finally:
            if client.last_custody is not None:
                result['native_processes'].append(client.last_custody)
        if not custody['reaped'] or custody['returncode'] or not custody['graceful_protocol_complete']:
            raise ValueError('Candidate reference child failed to drain')

    def advance(client):
        bounded()
        frame = client.advance()
        result['actual_native_advances'] += 1
        if not frame['frame']['advanced'] or not frame['frame']['quality']['qualified']:
            raise ValueError('Original physical quality gates rejected candidate reference')

    def compare(client, expected):
        packet, actual = client.inspect()
        if any(actual[n].tobytes() != expected[n].tobytes() for n in FIELDS):
            raise ValueError('Finite-F32 candidate changed original accepted snapshot bytes')
        for n in ('rhs', 'base', 'native_input_residual'):
            key = 'pressure_base' if n == 'base' else n
            if client.last_request[n].tobytes() != expected[key].tobytes():
                raise ValueError('Candidate changed original pressure supervision bytes')
        return packet, actual

    try:
        index = build_index([ROOT / p for p in reg['audits']])
        if len(index['masters']) != reg['expected_original_masters']:
            raise ValueError('Exact original audited comparison population required')
        dataset = LongTemporalDataset(index)
        for identifier, (_, original, states) in dataset.receipts.items():
            for start in reg['replay_starts']:
                client = FineReferenceClient(original['scene'], args.output / identifier / f'start-{start}', build=build)
                try:
                    initial = dataset._load(identifier, start)
                    client.restore(start, {n: initial[n] for n in FIELDS})
                    compared = 0
                    for offset in range(1, 33):
                        advance(client)
                        if start + offset in states:
                            packet, _ = compare(client, dataset._load(identifier, start + offset))
                            if packet['time_s'] != (start + offset) / 240:
                                raise ValueError('Candidate changed physical replay clock')
                            compared += 1
                    result['original_case_replays'].append(dict(identifier=identifier,
                        split=original['split'], start_fine_step=start,
                        native_advances=32, complete_snapshots_and_pressure_labels_bitwise_equal=compared))
                    result['original_complete_state_comparisons'] += compared
                finally:
                    close(client)
            write(args.output / 'results.json', result)
            print('Finite-F32 original replay exact:', identifier, flush=True)
        for name in reg['rejected_receipts']:
            original_path = ROOT / name
            original = read(original_path)
            if original['status'] != 'reference_rejected' or original['actual_reference_updates'] != 1:
                raise ValueError('Exact original rejected prefix required')
            saved = {s['fine_step']: s for s in original['states']}

            def load(step):
                spec = saved[step]
                path = original_path.parent / spec['path']
                if digest(path) != spec['sha256']:
                    raise ValueError('Original failed-prefix state changed')
                with np.load(path, allow_pickle=False) as a:
                    return {n: a[n].copy() for n in a.files}

            identifier = original['identifier']
            client = FineReferenceClient(original['scene'], args.output / identifier / 'recovery-prefix', build=build)
            try:
                initial = load(0)
                client.restore(0, {n: initial[n] for n in FIELDS})
                for step in range(1, 65):
                    advance(client)
                    if step == 1:
                        compare(client, load(1))
                    if step == 32:
                        _, checkpoint = client.inspect()
                packet, final = client.inspect()
                final_labels = {n: client.last_request[n].copy() for n in ('rhs', 'base', 'native_input_residual')}
                if packet['step'] != 64 or packet['time_s'] != 64 / 240:
                    raise ValueError('Recovered prefix did not reach correct physical time')
            finally:
                close(client)
            replay = FineReferenceClient(original['scene'], args.output / identifier / 'recovery-restart32', build=build)
            try:
                replay.restore(32, checkpoint)
                for _ in range(32):
                    advance(replay)
                _, restored = replay.inspect()
                if hashes(restored) != hashes(final) or any(replay.last_request[n].tobytes() != a.tobytes() for n, a in final_labels.items()):
                    raise ValueError('Recovered candidate restart changed complete fields or labels')
            finally:
                close(replay)
            result['recovered_case_prefixes'].append(dict(identifier=identifier, uninterrupted_fine_steps=64,
                restart_at=32, compared_restarted_steps=32, original_first_step_bytes_exact=True,
                native_physical_quality_all_pass=True, restored_final_state_and_labels_bitwise_exact=True))
            write(args.output / 'results.json', result)
            print('Finite-F32 recovered rejected prefix and restart:', identifier, flush=True)
        if torch.cuda.is_initialized():
            raise ValueError('Offline reference qualification initialized CUDA')
        result.update(status='passed_finite_f32_local_original_replay_and_rejected_prefix_recovery',
            all_original_compared_fields_and_pressure_labels_bitwise_exact=True,
            all_recovered_prefixes_and_restart_checks_pass=True,
            elapsed_seconds=time.monotonic() - started, memory_counters=process_memory())
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
