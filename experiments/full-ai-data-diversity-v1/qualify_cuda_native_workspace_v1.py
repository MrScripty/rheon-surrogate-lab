"""Native-byte, ownership, stream and error-recovery checks for reusable staging.

Only copies and known byte-pattern controls run on GPU. No learned model or
native reference solver runs here; elapsed times are not inference benchmarks.
"""
import argparse
import hashlib
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from support import ROOT, read, write, digest
from cuda_native_transfer_v1 import FIELDS
from cuda_native_workspace_v1 import NativeStateWorkspace


def hashes(fields):
    return {n: hashlib.sha256(v.numpy().tobytes()).hexdigest() for n, v in fields.items()}


def run(args):
    if args.output.exists():
        raise ValueError('Fresh transfer-workspace evidence required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('CUDA workspace qualification required')
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen workspace dependency changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    rows, setups, guards = [], [], []
    with torch.inference_mode():
        for case in reg['cases']:
            if digest(ROOT / case['path']) != case['sha256']:
                raise ValueError('Native transfer fixture changed')
            with np.load(ROOT / case['path'], allow_pickle=False) as archive:
                golden = {n: torch.from_numpy(archive[n].copy()) for n in FIELDS}
            negated = {n: -v for n, v in golden.items()}
            zeros = {n: torch.zeros_like(v) for n, v in golden.items()}
            workspace = NativeStateWorkspace(golden, maximum_horizons=4)
            staging_addresses = (workspace.pinned_input.data_ptr(), workspace.pinned_output.data_ptr(),
                                 workspace.device_input.data_ptr())
            retained = []
            setups.append(dict(identifier=case['identifier'], **workspace.setup))
            for k in reg['horizons']:
                for repeat in range(reg['repetitions']):
                    if time.monotonic() - started > reg['maximum_wall_seconds']:
                        raise TimeoutError('Bounded workspace qualification deadline reached')
                    reference = golden if repeat % 2 == 0 else negated
                    expected = hashes(reference)
                    payload = {n: v.clone() for n, v in reference.items()}
                    def emit(resident):
                        # This happens after packing and before asynchronous
                        # transfer completion is guaranteed to the caller.
                        for value in payload.values():
                            value.zero_()
                        return [resident] * k
                    copied = workspace.run(payload, emit)
                    if any(hashes(row) != expected for row in copied):
                        raise ValueError('Reusable staging changed actual native bytes')
                    if any(v.is_pinned() for row in copied for v in row.values()):
                        raise ValueError('Returned outputs must own ordinary CPU storage')
                    if any(v.data_ptr() % v.element_size() for row in copied for v in row.values()):
                        raise ValueError('Native output views must remain aligned')
                    if k > 1:
                        copied[0]['vx'].zero_()
                        if hashes(copied[1]) != expected:
                            raise ValueError('Distinct intermediate CPU outputs overlap')
                        copied[0]['vx'].copy_(reference['vx'])
                    retained.append((copied, expected))
                    later = workspace.run(zeros, lambda resident: [resident] * k)
                    if any(hashes(row) != hashes(zeros) for row in later):
                        raise ValueError('Later changing input did not replace staged native state')
                    if any(hashes(row) != identity for previous, identity in retained for row in previous):
                        raise ValueError('Later transaction overwrote retained outputs')
                    if staging_addresses != (workspace.pinned_input.data_ptr(), workspace.pinned_output.data_ptr(),
                                              workspace.device_input.data_ptr()):
                        raise ValueError('Prepared input/output staging was reallocated')
                    rows.append(dict(identifier=case['identifier'], K=k, repeat=repeat,
                        native_bytes_exact=True, caller_mutation_after_pack_safe=True,
                        retained_outputs_independent=True, intermediate_ownership_verified=True,
                        returned_storage_not_pinned=True, native_alignment_verified=True,
                        persistent_staging_addresses_unchanged=True))
            # A queued GPU operation followed by a callback exception must not
            # leave the transaction locked or its pinned input unsafe to reuse.
            def fail_after_enqueue(resident):
                for value in resident.values():
                    value.zero_()
                raise RuntimeError('intentional workspace callback failure')
            try:
                workspace.run(golden, fail_after_enqueue)
            except RuntimeError as error:
                if str(error) != 'intentional workspace callback failure':
                    raise
            else:
                raise AssertionError('Callback failure was swallowed')
            recovered = workspace.run(golden, lambda resident: [resident])
            if hashes(recovered[0]) != hashes(golden):
                raise ValueError('Workspace failed to recover after queued callback failure')
            try:
                workspace.run(golden, lambda resident: [resident] * 5)
            except ValueError:
                pass
            else:
                raise AssertionError('Unprepared output count accepted')
            reentrant_rejected = []
            def reentrant(resident):
                try:
                    workspace.run(golden, lambda value: [value])
                except RuntimeError:
                    reentrant_rejected.append(True)
                return [resident]
            recovered = workspace.run(golden, reentrant)
            if reentrant_rejected != [True] or hashes(recovered[0]) != hashes(golden):
                raise ValueError('Reentrant transaction was not safely rejected')
            with torch.cuda.stream(torch.cuda.Stream()):
                try:
                    workspace.run(golden, lambda resident: [resident])
                except ValueError:
                    pass
                else:
                    raise AssertionError('Workspace accepted a different CUDA stream')
            if any(hashes(row) != identity for previous, identity in retained for row in previous):
                raise ValueError('Error recovery changed retained physical output bytes')
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            if peak > reg['maximum_peak_rss_bytes'] or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']:
                raise ValueError('Workspace qualification memory ceiling exceeded')
            guards.append(dict(identifier=case['identifier'], callback_failure_drained_and_recovered=True,
                               excessive_output_count_rejected=True, reentrant_call_rejected=True,
                               wrong_stream_rejected=True, retained_outputs_survive_errors=True,
                               observed_peak_rss_bytes=peak))
            del workspace, retained, copied, later, recovered
            print('Reusable CUDA native staging', case['identifier'], 'allK/ownership/recovery passed', flush=True)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Workspace dependency changed during qualification: ' + name)
    write(args.output / 'results.json', dict(
        status='passed_native_workspace_bytes_ownership_stream_and_failure_recovery',
        rows=rows, cases=len(reg['cases']), setups=setups, guards=guards,
        inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
        learned_model_forward_executed=False, cpu_model_forward=False, optimizer_updates=0,
        expanded_reserved52_references_opened=0, latency_claim=False,
        elapsed_diagnostic_seconds=time.monotonic() - started,
        protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
