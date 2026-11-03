"""Actual/arbitrary complete native entry bytes, ownership and failure custody."""
import argparse
import hashlib
import os
import resource
import threading
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils import deterministic
from support import ROOT, read, write, digest
from complete_native_entry_v1 import CompleteNativeEntry
from debug.gather_fused_graph_v1 import GatherFusedGraph
from cuda_native_transfer_v1 import FIELDS, upload_state, download_outputs
from diagnose_causal_composition import inventory


def hashes(rows):
    return [{n: hashlib.sha256(v.numpy().tobytes()).hexdigest() for n, v in r.items()} for r in rows]


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only native entry evidence required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT / p) != h:
            raise ValueError('Frozen native entry dependency changed: ' + p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    deterministic.fill_uninitialized_memory = True
    torch.manual_seed(reg['seed'])
    args.output.mkdir(parents=True)
    entry = CompleteNativeEntry(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    baseline = GatherFusedGraph(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = [inventory(m) for m in (entry, baseline)]
    if before[0] != before[1]:
        raise ValueError('Identical restored learned tensors required')
    result = dict(status='running', active=None, rows=[], guards=[], inference_device='cuda', display_label='GPU (CUDA)',
        cpu_model_forward=False, optimizer_updates=0, latency_claim=False, expanded_reserved52_references_opened=0,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for case in reg['cases']:
                scene = read(ROOT / case['receipt'])['scene']
                spec, n = case['primitive_state'], case['reference_substeps']
                start = spec['step']
                if digest(ROOT / spec['path']) != spec['sha256']:
                    raise ValueError('Native primitive state changed')
                with np.load(ROOT / spec['path'], allow_pickle=False) as archive:
                    actual = {name: torch.from_numpy(archive[name].copy()) for name in FIELDS}
                baseline.prepare_geometry(tuple(scene['counts']))
                retained = None
                for pattern, k in [('actual', 1), ('actual', 2), ('actual', 3), ('actual', 4),
                                   ('arbitrary', 4), ('zero', 2)]:
                    if time.monotonic() - started > reg['maximum_wall_seconds']:
                        raise TimeoutError('Finite native entry qualification deadline')
                    result['active'] = dict(identifier=case['identifier'], pattern=pattern, K=k)
                    if pattern == 'actual':
                        fields = actual
                    elif pattern == 'zero':
                        fields = {name: torch.zeros_like(v) for name, v in actual.items()}
                    else:
                        fields = {name: torch.randn_like(v) * .001 for name, v in actual.items()}
                    resident = upload_state(fields)
                    baseline.prepare_block(resident, scene, start, k, n)
                    expected = download_outputs(baseline.predict(resident, scene, start, k, n))
                    entry.prepare_native(fields, scene, start, k, n)
                    copied = entry.predict_native(fields, scene, start, k, n)
                    if len(copied) != k or hashes(copied) != hashes(expected):
                        raise ValueError('Complete native entry output bytes changed')
                    for a, b in zip(copied, expected):
                        for name in FIELDS:
                            if a[name].dtype != b[name].dtype or a[name].shape != b[name].shape:
                                raise ValueError('Native output dtype or shape changed')
                    if retained is not None and hashes(retained[0]) != retained[1]:
                        raise ValueError('Later replay overwrote retained CPU output')
                    if k > 1:
                        old = copied[0]['pressure'][0].clone()
                        copied[0]['pressure'][0] += 1.
                        if hashes(copied[1:]) != hashes(expected[1:]):
                            raise ValueError('Returned intermediate states overlap')
                        copied[0]['pressure'][0].copy_(old)
                    retained = copied, hashes(copied)
                    result['rows'].append(dict(identifier=case['identifier'], K=k, pattern=pattern,
                        complete_outputs=k, all_native_bytes_exact=True, retained_outputs_independent=True,
                        every_intermediate_required=True, native_output_sha256=retained[1]))
                # Check caller mutation after packing, reentrancy and real
                # cross-thread rejection at the protected actual replay seam.
                expected1 = download_outputs(baseline.predict(upload_state(actual), scene, start, 1, n))
                mutable = {name: v.clone() for name, v in actual.items()}
                replay = entry._replay_borrowed
                rejected = []
                stream = torch.cuda.current_stream()
                def interception(*a):
                    for value in mutable.values():
                        value.zero_()
                    try:
                        entry.predict_native(actual, scene, start, 1, n)
                    except RuntimeError:
                        rejected.append('reentrant')
                    def competing():
                        with torch.cuda.stream(stream):
                            try:
                                entry.predict_native(actual, scene, start, 1, n)
                            except RuntimeError:
                                rejected.append('concurrent')
                    thread = threading.Thread(target=competing)
                    thread.start()
                    thread.join(timeout=10)
                    if thread.is_alive():
                        raise TimeoutError('Concurrent rejection did not finish')
                    return replay(*a)
                entry._replay_borrowed = interception
                try:
                    guarded = entry.predict_native(mutable, scene, start, 1, n)
                finally:
                    entry._replay_borrowed = replay
                if set(rejected) != {'reentrant', 'concurrent'} or hashes(guarded) != hashes(expected1):
                    raise ValueError('Mutation/reentrancy/concurrency custody failed')
                # Keep normal GPU-returning ownership across a native replay.
                gpu_owned = entry.predict(upload_state(actual), scene, start, 1, n)
                gpu_hash = hashes(download_outputs(gpu_owned))
                entry.predict_native(actual, scene, start, 1, n)
                if hashes(download_outputs(gpu_owned)) != gpu_hash:
                    raise ValueError('Normal public GPU outputs lost independent ownership')
                copy_download = entry._copy_download
                def partial_failure(outputs, entries, pinned, gaps):
                    step, name, offset, length, shape, dtype = entries[0]
                    pinned.narrow(0, offset, length).view(dtype).reshape(shape).copy_(outputs[step][name], non_blocking=True)
                    raise RuntimeError('intentional failure after first native D2H copy')
                entry._copy_download = partial_failure
                try:
                    try:
                        entry.predict_native(actual, scene, start, 1, n)
                    except RuntimeError as error:
                        if str(error) != 'intentional failure after first native D2H copy':
                            raise
                    else:
                        raise AssertionError('Injected partial-copy failure was swallowed')
                finally:
                    entry._copy_download = copy_download
                recovered = entry.predict_native(actual, scene, start, 1, n)
                if hashes(recovered) != hashes(expected1) or hashes(retained[0]) != retained[1]:
                    raise ValueError('Failure drain/recovery changed output ownership or values')
                with torch.cuda.stream(torch.cuda.Stream()):
                    try:
                        entry.predict_native(actual, scene, start, 1, n)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('Wrong CUDA stream accepted')
                invalid = dict(actual, pressure=actual['pressure'].float())
                try:
                    entry.predict_native(invalid, scene, start, 1, n)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid CPU native dtype accepted')
                if not deterministic.fill_uninitialized_memory or not torch.are_deterministic_algorithms_enabled():
                    raise ValueError('Entry changed deterministic algorithms or allocation poisoning')
                result['guards'].append(dict(identifier=case['identifier'], caller_mutation_after_pack_safe=True,
                    reentrant_rejected=True, concurrent_same_stream_rejected=True, wrong_stream_rejected=True,
                    partial_D2H_failure_drained_and_recovered=True, normal_GPU_output_ownership_unchanged=True,
                    retained_CPU_output_ownership_verified=True, invalid_native_dtype_rejected=True,
                    deterministic_settings_unchanged=True))
                entry.release_graphs()
                baseline.release_graphs()
                del retained, copied, expected, recovered, guarded, gpu_owned, resident, actual, fields, mutable, expected1, invalid
                if (torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                        or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > reg['maximum_peak_rss_bytes']):
                    raise ValueError('Native entry qualification memory cap')
                write(args.output / 'results.json', result)
                print('Complete native entry bytes/ownership/failure custody', case['identifier'], 'passed', flush=True)
        if len(result['rows']) != 156 or len(result['guards']) != 26 or [inventory(m) for m in (entry, baseline)] != before:
            raise ValueError('All registered cases and unchanged learned weights required')
        for p, h in reg['source_sha256'].items():
            if digest(ROOT / p) != h:
                raise ValueError('Native entry source changed during qualification: ' + p)
        result.update(status='passed_all26_complete_native_entry_bytes_ownership_failure_custody', active=None,
                      complete_comparison_outputs=416, compared_typed_arrays=2080,
                      every_output_exact_to_qualified_GPU_baseline=True, learned_weights_unchanged=True,
                      gpu_model=torch.cuda.get_device_name(), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      elapsed_diagnostic_seconds=time.monotonic() - started)
    except BaseException as error:
        result.update(status='failed', active=None, error=repr(error))
        raise
    finally:
        entry.close()
        baseline.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
