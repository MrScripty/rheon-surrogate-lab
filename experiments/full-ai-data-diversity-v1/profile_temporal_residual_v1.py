"""Instrument the qualified zero-head generator; profiler times are diagnostic."""
import argparse
import functools
import os
from pathlib import Path
import resource
import time

import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index, LongTemporalDataset, FIELDS
import temporal_residual_generator_v2 as generator
from temporal_native_transfer_v1 import transaction
from benchmark_temporal_residual_v1 import hashes
from diagnose_causal_composition import inventory


def instrument(label, function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(label):
            return function(*args, **kwargs)
    return wrapped


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only diagnostic required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Registered source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    index = build_index([ROOT / n for n in reg['audits']])
    dataset = LongTemporalDataset(index)
    qualified = read(ROOT / reg['qualification'])
    if qualified['status'] != 'completed_temporal_residual_zero_graph_controls':
        raise ValueError('Complete qualified outputs required')
    expected = {(r['identifier'], r['start_fine_step'], r['output_steps_per_s'], r['K'], r['lead'], r['mode']): r['output_sha256']
                for r in qualified['rows']}
    model = generator.TemporalResidualGenerator(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', cases=[], source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), inference_device='cuda', display_label='GPU (CUDA)',
        GPU=torch.cuda.get_device_name(), cpu_model_forward=False, new_parameter_fits=0,
        residual_optimizer_updates=0, reserved52_references_opened=0,
        acceptance_latency_qualification=False, full_generator_target_pass=False,
        scope='Profiler-instrumented eager complete CPU-field to allKownedCPU-field calls; named scopes overlap and are inclusive. Profiler/operator times are diagnostics, not complete uninstrumented acceptance latency. No kernel or scope time divided by fine internal substeps.')
    originals = (generator.features, generator.physical_deltas)
    generator.features = instrument('temporal/features', generator.features)
    generator.physical_deltas = instrument('temporal/physical_deltas', generator.physical_deltas)
    for obj, name, label in [(model, 'phase_packet', 'temporal/coarse_phase'),
            (model.network, 'encode_forces', 'temporal/force_encoder'),
            (model.network, 'forward', 'temporal/residual_network_including_force_encoder'),
            (model, 'emit', 'temporal/correction_projection_and_reconstruction')]:
        setattr(obj, name, instrument(label, getattr(obj, name)))
    try:
        for case in reg['cases']:
            example = next(e for e in index['examples'] if all(e[n] == case[n]
                for n in ('identifier', 'start_fine_step', 'output_steps_per_s', 'K')))
            scene = dataset.receipts[example['identifier']][1]['scene']
            inputs = dataset.inputs(example)
            fields = {n: torch.from_numpy(inputs['current'][n]) for n in FIELDS}
            initial = hashes(fields)
            start, rate, k = example['start_fine_step'], example['output_steps_per_s'], example['K']
            targets = [expected[(example['identifier'], start, rate, k, lead, 'single_pass')]
                       for lead in range(1, k + 1)]

            def check(outputs):
                if hashes(fields) != initial or [hashes(o) for o in outputs] != targets:
                    raise ValueError('Instrumented generator changed qualified outputs or caller fields')

            for _ in range(reg['warmups']):
                check(transaction(model, fields, scene, start, rate, k, (240 // rate,)))
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                for _ in range(reg['repetitions']):
                    with torch.profiler.record_function('temporal/complete_native_transaction'):
                        outputs = transaction(model, fields, scene, start, rate, k, (240 // rate,))
                    torch.cuda.synchronize()
            check(outputs)
            rows = [dict(name=e.key, device_type=str(e.device_type), calls=e.count,
                self_cpu_us=e.self_cpu_time_total, total_cpu_us=e.cpu_time_total,
                self_device_us=e.self_device_time_total, total_device_us=e.device_time_total,
                input_shapes=e.input_shapes) for e in profile.key_averages(group_by_input_shape=True)]
            result['cases'].append(dict(**case, counts_xyz=scene['counts'],
                cells_per_metre=scene['cells_per_metre'], repeats=reg['repetitions'], rows=rows,
                all_profiled_complete_outputs_match_qualification=True, caller_fields_unchanged=True))
            if (time.monotonic() - started > reg['maximum_wall_seconds']
                    or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                    or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > reg['maximum_peak_rss_bytes']):
                raise ValueError('Profile resource bound exceeded')
            write(args.output / 'results.json', result)
        if inventory(model) != before:
            raise ValueError('Profiling changed parameters')
        result.update(status='completed_temporal_residual_GPU_operator_diagnostic',
            elapsed_operational_seconds=time.monotonic() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), learned_parameters_unchanged=True)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        generator.features, generator.physical_deltas = originals
        model.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
