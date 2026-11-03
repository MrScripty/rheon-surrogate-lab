"""Complete GPU cost and native errors of both untrained shared width variants."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index, LongTemporalDataset, FIELDS
from shared_temporal_graph_v1 import SharedTemporalGraph
from shared_temporal_native_transfer_v1 import transaction
from diagnose_causal_composition import inventory, measure


def hashes(fields):
    return {n: hashlib.sha256(v.numpy().tobytes()).hexdigest() for n, v in fields.items()}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only shared benchmark required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen benchmark source changed: ' + name)
    qualified = read(ROOT / reg['qualification'])
    if (qualified['status'] != 'passed_shared_temporal_graph_equality_controls_gradients_events_and_restore'
            or not qualified['all_complete_outputs_bitwise_equal_eager']):
        raise ValueError('Actual whole-output eager/graph qualification required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', rows=[], preparations=[], inference_device='cuda', display_label='GPU (CUDA)',
        cpu_model_forward=False, new_parameter_fits=0, residual_optimizer_updates=0,
        reserved52_references_opened=0, full_generator_target_pass=False,
        independent_confirmation=False, untrained_prototype=True,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        GPU=torch.cuda.get_device_name(), pressure_dtype='float64', carrier_dtype='float32',
        neural_dtype='float32', tf32=False, autocast=False,
        resource_clearance=reg['resource_clearance'],
        scope='Complete current nativeCPUfields through validation/packing/H2D; construction/H2D of whole prescribed forcepath and exacteventpackets; allGPUforceevaluation, one coarseendpointadvance, one sharedtrunk,endpointcorrectionandallinteriorlearnedpressureprojections/reconstruction,allKownedCPUoutputs,D2H/sync. Divide onlybyKphysicaloutputs. Geometry/graph/setup/prewarmup/warmups separate. All nativeaccuracychecks occur outside timing.')
    models = {}
    try:
        for hidden in reg['hidden_widths']:
            models[hidden] = SharedTemporalGraph(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'], hidden=hidden)
        before = {hidden: inventory(model) for hidden, model in models.items()}
        expected = {(r['identifier'], r['start_fine_step'], r['output_steps_per_s'], r['K'], r['hidden_width']): r['output_sha256']
                    for r in qualified['rows']}
        index = build_index([ROOT / p for p in reg['audits']])
        dataset = LongTemporalDataset(index)
        norm = read(ROOT / reg['normalization'])
        examples = [e for e in index['examples'] if e['start_fine_step'] in reg['starts']]
        examples.sort(key=lambda e: (e['identifier'], e['output_steps_per_s'], e['K'], e['start_fine_step']))
        for example in examples:
            identifier, start, rate, k = (example[n] for n in ('identifier', 'start_fine_step', 'output_steps_per_s', 'K'))
            scene = dataset.receipts[identifier][1]['scene']
            inputs = dataset.inputs(example)
            fields = {n: torch.from_numpy(inputs['current'][n]) for n in FIELDS}
            original = hashes(fields)
            targets = [dataset._load(identifier, step) for step in example['target_fine_steps']]
            metrics, samples, spans = {}, {h: [] for h in models}, {h: [] for h in models}

            def check(outputs, hidden):
                if (hashes(fields) != original or len(outputs) != k
                        or [hashes(o) for o in outputs] != expected[identifier, start, rate, k, hidden]):
                    raise ValueError('Timed call changed qualified outputs or caller inputs')

            for hidden, model in models.items():
                torch.cuda.synchronize()
                tick = time.perf_counter()
                outputs = transaction(model, fields, scene, start, rate, k)
                first_ms = 1000 * (time.perf_counter() - tick)
                check(outputs, hidden)
                metrics[hidden] = [measure(o, truth, scene, norm) for o, truth in zip(outputs, targets)]
                for _ in range(reg['warmups']):
                    check(transaction(model, fields, scene, start, rate, k), hidden)
                result['preparations'].append(dict(identifier=identifier, start_fine_step=start,
                    output_steps_per_s=rate, K=k, hidden_width=hidden,
                    first_prewarmup_complete_transaction_ms=first_ms, warmups=reg['warmups']))
            for repeat in range(reg['repetitions']):
                order = list(models) if repeat % 2 == 0 else list(reversed(models))
                for hidden in order:
                    if (time.monotonic() - started > reg['maximum_wall_seconds']
                            or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > reg['maximum_peak_rss_bytes']):
                        raise ValueError('Registered shared benchmark resource budget exceeded')
                    torch.cuda.synchronize()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    tick = time.perf_counter()
                    begin.record()
                    outputs = transaction(models[hidden], fields, scene, start, rate, k)
                    end.record()
                    end.synchronize()
                    samples[hidden].append(1000 * (time.perf_counter() - tick))
                    spans[hidden].append(begin.elapsed_time(end))
                    check(outputs, hidden)
            budget = reg['strict_ms_per_physical_output'] * max(1, math.prod(scene['counts']) / 32768)
            for hidden, values in samples.items():
                result['rows'].append(dict(identifier=identifier, split=example['split'],
                    counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                    output_steps_per_s=rate, K=k, block_duration_s=k / rate, start_fine_step=start,
                    hidden_width=hidden, neural_parameters=models[hidden].asset_identity['new_neural_parameters'],
                    coarse_endpoint_transports=1, shared_spatial_encoder_calls=1,
                    learned_pressure_projections=k + 1, native_metrics_per_output=metrics[hidden],
                    all_accuracy_pass=all(m['all_below_1e5'] for m in metrics[hidden]),
                    block_wall_ms=values, pipeline_cuda_event_span_ms=spans[hidden],
                    median_block_ms=float(np.median(values)), p95_block_ms=float(np.percentile(values, 95)),
                    maximum_block_ms=max(values), median_ms_per_physical_output=float(np.median(values)) / k,
                    maximum_ms_per_physical_output=max(values) / k,
                    strict_ms_per_physical_output_limit=budget,
                    all_observed_latency_samples_under_budget=max(values) / k < budget,
                    complete_output_hashes_match_qualification=True))
            write(args.output / 'results.json', result)
        if sum(len(r['block_wall_ms']) for r in result['rows']) != reg['expected_timing_samples']:
            raise ValueError('Incomplete declared timing population')
        if {h: inventory(m) for h, m in models.items()} != before:
            raise ValueError('Benchmark changed learned parameters')
        result.update(status='completed_untrained_shared_temporal_complete_GPU_cost_and_native_accuracy',
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            elapsed_operational_seconds=time.monotonic() - started)
    except BaseException as exception:
        result.update(status='failed', error=repr(exception))
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
