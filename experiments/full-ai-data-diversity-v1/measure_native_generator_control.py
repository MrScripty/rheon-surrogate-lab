"""Matched full-state native Rheon control; never a learned candidate path."""
import argparse
import collections
import os
from pathlib import Path
import statistics
import time
from support import ROOT, read, write, digest
from native_forcing_client import NativeForcingClient
from diagnose_causal_composition import FIELDS, measure
import numpy as np
import torch


def run(args):
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh native generator control required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Registered native control source changed: '+path)
    if reg['mode'] not in ('qualification', 'benchmark') or len(reg['cores']) != 1:
        raise ValueError('Explicit single-core native control mode required')
    timed = reg['mode'] == 'benchmark'
    if timed:
        if not reg['reference_sessions_revalidated_terminal_before_registration']:
            raise ValueError('Actual terminal reference tool custody required')
        for path, allowed in reg['required_terminal_reference_controllers'].items():
            controller = read(ROOT/path)
            if controller['status'] not in allowed or controller.get('active') is not None:
                raise ValueError('CPU must be clear of reference workers: '+path)
    for task in Path('/proc/self/task').iterdir():
        os.sched_setaffinity(int(task.name), set(reg['cores']))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    norm = read(ROOT/reg['normalization'])
    args.output.mkdir(parents=True)
    rows, owners = [], []
    client = None

    def state(path, receipt, step):
        spec = receipt['states'][step]
        source = path.parent/spec['path']
        if spec['step'] != step or digest(source) != spec['sha256']:
            raise ValueError('Complete reference bytes/clock changed')
        with np.load(source, allow_pickle=False) as archive:
            return {k: archive[k].copy() for k in (*FIELDS, 'rhs')}

    try:
        for trajectory in read(ROOT/reg['bank'])['trajectories']:
            path = ROOT/trajectory['receipt']
            if digest(path) != trajectory['receipt_sha256']:
                raise ValueError('Matched current-state receipt changed')
            receipt = read(path)
            scene = receipt['scene']
            # NativeClient pins Rayon to one worker; inherited affinity is also one CPU.
            tick = time.perf_counter() if timed else None
            client = NativeForcingClient(scene, args.output/trajectory['identifier'])
            setup_ms = (time.perf_counter()-tick)*1000 if timed else None
            for start in reg['starts']:
                initial = state(path, receipt, start)
                for k in reg['horizons']:
                    # One separately recorded warm call; no best-of-repeat selection.
                    for repeat in range(-1, reg['repeats']):
                        client.restore(start, initial)
                        outputs, frames = [], []
                        tick = time.perf_counter() if timed else None
                        for lead in range(1, k+1):
                            frame = client.advance(method='native-pcg')
                            value = frame['frame']
                            if (not value['advanced'] or value['step'] != start+lead
                                    or not value['quality']['qualified'] or value['quality']['hard_stop']):
                                raise ValueError('Native control rejected a physical advance')
                            packet, output = client.inspect()
                            if packet['step'] != start+lead or packet['time_s'] != (start+lead)/60:
                                raise ValueError('Native control output clock differs')
                            outputs.append(output)
                            frames.append(frame)
                        block_ms = (time.perf_counter()-tick)*1000 if timed else None
                        metrics = []
                        for lead, output in enumerate(outputs, 1):
                            predicted = {name: torch.from_numpy(value) for name, value in output.items()}
                            truth = state(path, receipt, start+lead)
                            scored = measure(predicted, truth, scene, norm)
                            scored['all_accuracy_criteria_pass'] = all(
                                value <= reg['maximum_error'] for value in scored['relative'].values())
                            metrics.append(scored)
                        rows.append(dict(identifier=trajectory['identifier'], counts_xyz=scene['counts'],
                            cells_per_metre=scene['cells_per_metre'], start=start, horizons=k,
                            repeat=repeat, warm_call=repeat == -1, setup_ms=setup_ms,
                            complete_block_with_state_transfer_ms=block_ms,
                            ms_per_generated_step=block_ms/k if timed else None,
                            native_compute_block_ms=sum(f['frame']['frame_seconds'] for f in frames)*1000 if timed else None,
                            all_outputs_scored=True, metrics=metrics,
                            all_accuracy_pass=all(m['all_accuracy_criteria_pass'] for m in metrics)))
            owner = client.close()
            client = None
            owners.append(owner)
            if not owner['reaped'] or owner['returncode'] != 0 or not owner['graceful_protocol_complete']:
                raise ValueError('Native control owner must complete and reap')
            print('Native full-state control', trajectory['identifier'], reg['mode'], flush=True)
    finally:
        if client is not None:
            owners.append(client.close())
        write(args.output/'progress.json', dict(rows=rows, native_processes=owners))
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Native control source changed during run: '+path)
    summaries = []
    if timed:
        groups = collections.defaultdict(list)
        for row in rows:
            if not row['warm_call']:
                groups[(row['horizons'], row['cells_per_metre'], tuple(row['counts_xyz']))].append(row)
        for (k, density, grid), values in sorted(groups.items()):
            latencies = sorted(r['ms_per_generated_step'] for r in values)
            summaries.append(dict(horizons=k, cells_per_metre=density, counts_xyz=list(grid),
                observations=len(values), median_ms_per_step=statistics.median(latencies),
                p95_ms_per_step=latencies[min(len(latencies)-1, int(.95*len(latencies)))],
                maximum_ms_per_step=max(latencies), maximum_complete_block_ms=k*max(latencies),
                median_native_compute_ms_per_step=statistics.median(r['native_compute_block_ms']/k for r in values),
                all_accuracy_pass=all(r['all_accuracy_pass'] for r in values)))
    if torch.cuda.is_initialized():
        raise ValueError('Native CPU control initialized CUDA')
    write(args.output/'results.json', dict(status='completed_native_generator_control_'+reg['mode'],
        rows=rows, summaries=summaries, native_processes=owners,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        cores=reg['cores'], every_intermediate_output_scored=True,
        all_accuracy_pass=all(r['all_accuracy_pass'] for r in rows), optimizer_updates=0,
        candidate_inference=False, numerical_solver_is_explicit_reference_control=True,
        timing_claim=timed, clocks_read=timed, full_generator_target_pass=False,
        timing_scope='Resident native advance plus complete field inspection/transfer for every lead; restore and truth scoring outside ROI',
        native_compute_scope='Separately report compiled native frame time without field transfer',
        setup_and_warm_calls_separately_reported=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
