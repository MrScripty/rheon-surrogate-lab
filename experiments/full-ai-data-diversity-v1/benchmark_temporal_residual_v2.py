"""Complete CPU-field-to-CPU-field GPU transaction timing after qualification."""
import argparse
import hashlib
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index, LongTemporalDataset, FIELDS
from temporal_residual_graph_v1 import TemporalFrozenGraph as TemporalFrozenBackbone
from temporal_residual_graph_v2 import TemporalFrozenGraph
from diagnose_causal_composition import inventory


def hashes(fields):
    return {n: hashlib.sha256(v.numpy().tobytes()).hexdigest() for n, v in fields.items()}


from temporal_native_transfer_v1 import transaction


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA benchmark required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen GPU timing dependency changed: ' + name)
    qualified = read(ROOT / reg['qualification'])
    if (qualified['status'] != 'completed_temporal_residual_zero_graph_controls'
            or not qualified['all_outputs_bitwise_equal_eager']
            or not qualified['fine_baseline_all_accuracy_pass']):
        raise ValueError('Actual graph/eager output equivalence and fine baseline accuracy required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    models = {}
    result = dict(status='running', rows=[], preparations=[],
        inference_device='cuda', display_label='GPU (CUDA)', cpu_model_forward=False,
        new_parameter_fits=0, reserved52_references_opened=0,
        residual_optimizer_updates=0, timing_only_no_accuracy_promotion=True,
        true_joint_temporal_generation=False,
        full_generator_target_pass=False, independent_confirmation=False,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        repetitions=reg['repetitions'], warmups=reg['warmups'],
        scope='Existing pinned packed-byte upload and asynchronous owned pinned output copies. Current nativeCPUfields through validation/H2D, known control packet construction/H2D, GPU force evaluation, all learned advances, allKowned output fields, D2H and synchronization. Divide only byKphysical outputs; loading/geometry/graph preparation and warmups separate.',
        event_scope='CUDA event span of complete transaction enqueue including scheduling/transfer gaps; wall latency is the acceptance observation, not an isolated kernel-time claim.',
        resource_clearance=reg['resource_clearance'],
        concurrent_reference_capture_session_at_registration=reg['concurrent_reference_capture_session'])

    def bounded():
        if (time.monotonic() - started > reg['maximum_wall_seconds']
                or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > reg['maximum_peak_rss_bytes']):
            raise ValueError('Registered benchmark resource budget exceeded')

    try:
        for name, cls in [('baseline_graph', TemporalFrozenBackbone), ('fused_graph', TemporalFrozenGraph)]:
            models[name] = cls(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
        before = {n: inventory(m) for n, m in models.items()}
        if before['baseline_graph'] != before['fused_graph']:
            raise ValueError('Identical restored learned parameters required')
        result.update(gpu_model=torch.cuda.get_device_name(), cuda_version=torch.version.cuda,
            torch_version=torch.__version__, pressure_dtype='float64', carrier_dtype='float32',
            tf32=False, autocast=False, parameter_inventory=before['baseline_graph'])
        index = build_index([ROOT / n for n in reg['audits']])
        dataset = LongTemporalDataset(index)
        expected = {(r['identifier'], r['start_fine_step'], r['output_steps_per_s'], r['K'], r['lead'], r['mode']): r
                    for r in qualified['rows']}
        result['independent_masters_per_split'] = index['independent_masters_per_split']
        examples = [e for e in index['examples'] if e['start_fine_step'] in reg['start_fine_steps']]
        examples.sort(key=lambda e:(e['identifier'], e['output_steps_per_s'], e['K'], e['start_fine_step']))
        for example in examples:
            identifier, start = example['identifier'], example['start_fine_step']
            rate, k = example['output_steps_per_s'], example['K']
            scene = dataset.receipts[identifier][1]['scene']
            inputs = dataset.inputs(example)
            fields = {n: torch.from_numpy(inputs['current'][n]) for n in FIELDS}
            original = hashes(fields)
            stride = 240 // rate
            for mode in reg['modes_by_output_rate'][str(rate)]:
                groups = {'fine_reference':(1,) * stride, 'half_passes':(2,) * (stride // 2),
                          'single_pass':(stride,)}[mode]
                expected_outputs = [expected[(identifier, start, rate, k, lead, mode)]['output_sha256']
                                    for lead in range(1, k + 1)]

                def verify(outputs):
                    if len(outputs) != k or [hashes(o) for o in outputs] != expected_outputs or hashes(fields) != original:
                        raise ValueError('Timed transaction changed qualified output/input bytes')

                samples = {name: [] for name in models}
                spans = {name: [] for name in models}
                for name, model in models.items():
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    outputs = transaction(model, fields, scene, start, rate, k, groups)
                    first_ms = 1000 * (time.perf_counter() - tick)
                    verify(outputs)
                    for _ in range(reg['warmups']):
                        verify(transaction(model, fields, scene, start, rate, k, groups))
                    result['preparations'].append(dict(identifier=identifier, start=start, output_steps_per_s=rate,
                        K=k, mode=mode, backend=name, first_prewarmup_complete_transaction_ms=first_ms,
                        includes_any_new_geometry_or_graph_setup=True, warmups=reg['warmups']))
                for repeat in range(reg['repetitions']):
                    for name in (('baseline_graph', 'fused_graph') if repeat % 2 == 0 else ('fused_graph', 'baseline_graph')):
                        bounded()
                        torch.cuda.synchronize()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        tick = time.perf_counter()
                        begin.record()
                        outputs = transaction(models[name], fields, scene, start, rate, k, groups)
                        end.record()
                        end.synchronize()
                        samples[name].append(1000 * (time.perf_counter() - tick))
                        spans[name].append(begin.elapsed_time(end))
                        verify(outputs)
                for name, values in samples.items():
                    result['rows'].append(dict(identifier=identifier, split=example['split'],
                        counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                        start_fine_step=start, output_steps_per_s=rate, output_dt_s=1 / rate,
                        K=k, block_duration_s=k / rate, mode=mode, backend=name,
                        groups_per_output=list(groups), actual_internal_advances=k * len(groups),
                        actual_learned_pressure_projections=k * sum(1 if width == 1 else 2 for width in groups),
                        residual_neural_forward_calls=k * sum(width > 1 for width in groups),
                        untrained_residual=True,
                        all_accuracy_pass=all(expected[(identifier, start, rate, k, lead, mode)]['all_accuracy_pass'] for lead in range(1, k + 1)),
                        block_wall_ms=values, pipeline_cuda_event_span_ms=spans[name],
                        median_block_ms=float(np.median(values)), p95_block_ms=float(np.percentile(values, 95)),
                        maximum_block_ms=max(values), maximum_per_physical_output_ms=max(values) / k,
                        median_simulated_seconds_per_wall_second=(k / rate) / (float(np.median(values)) / 1000),
                        every_output_hash_matches_qualification=True))
                write(args.output / 'results.json', result)
        if sum(len(r['block_wall_ms']) for r in result['rows']) != reg['expected_timing_samples']:
            raise ValueError('Incomplete registered timing sample population')
        if {n: inventory(m) for n, m in models.items()} != before:
            raise ValueError('Benchmark changed learned parameters')
        result.update(status='completed_untrained_temporal_residual_GPU_complete_transaction_benchmark',
            all_timed_outputs_bitwise_equal_qualification=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            elapsed_operational_seconds=time.monotonic() - started)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        for model in models.values():
            model.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
