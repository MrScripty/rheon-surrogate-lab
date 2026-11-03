"""Score explicit-time GPU baseline and grouped controls on audited long data."""
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
from temporal_residual_graph_v2 import TemporalFrozenGraph as TemporalFrozenBackbone
from temporal_control_packets_v1 import control_packet
from scheduled_force_fine_controls import force
from diagnose_causal_composition import inventory, measure


def hashes(fields):
    return {n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
            for n, v in fields.items()}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only qualification required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen temporal model dependency changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    model = None
    result = dict(status='running', rows=[], control_rows=[], active=None,
        inference_device='cuda', display_label='GPU (CUDA)', cpu_model_forward=False,
        new_parameter_fits=0, reserved52_references_opened=0, latency_claim=False,
        full_generator_target_pass=False, source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol))

    def bounded():
        if (time.monotonic() - started > reg['maximum_wall_seconds']
                or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > reg['maximum_peak_rss_bytes']):
            raise ValueError('Registered finite qualification resource budget exceeded')

    try:
        index = build_index([ROOT / n for n in reg['audits']])
        dataset = LongTemporalDataset(index)
        model = TemporalFrozenBackbone(reg['transport_pair'], reg['shared_pair'],
            reg['build'], reg['control_build'])
        if inventory(model.network) != read(ROOT / reg['network_qualification'])['parameter_inventory']:
            raise ValueError('Exact qualified residual initialization required')
        before = inventory(model)
        result.update(gpu_model=torch.cuda.get_device_name(), model_identity=model.asset_identity,
            precision=dict(pressure='float64', carriers='float32', autocast=False, tf32=False),
            independent_masters_per_split=index['independent_masters_per_split'])
        eager_expected = {(r['identifier'],r['start_fine_step'],r['output_steps_per_s'],r['K'],r['lead'],r['mode']):r['output_sha256'] for r in read(ROOT / reg['eager_qualification'])['rows']}
        normalization = read(ROOT / reg['normalization'])
        with torch.inference_mode():
            for identifier, (_, receipt, _) in dataset.receipts.items():
                scene = receipt['scene']
                for fine in reg['control_fine_steps']:
                    packet = torch.from_numpy(control_packet(scene, fine, 1)).to('cuda')
                    actual = model.controls.generate_from_packet(tuple(scene['counts']), packet)[0]
                    expected = force(scene, fine)[0]
                    for c, (a, b) in enumerate(zip(actual, expected)):
                        b = b.to('cuda')
                        relative = float((a - b).norm() / b.norm().clamp_min(1e-30))
                        scaled = float((a - b).abs().max() / b.abs().max().clamp_min(1.))
                        if relative > reg['maximum_control_relative_error'] or scaled > reg['maximum_control_scaled_error']:
                            raise ValueError('Explicit-time CUDA known force differs from independent evaluator')
                        result['control_rows'].append(dict(identifier=identifier,
                            fine_step=fine, component=c, relative_l2=relative, scaled_absolute=scaled))
                examples = [e for e in index['examples'] if e['identifier'] == identifier
                            and e['start_fine_step'] in reg['start_fine_steps']]
                examples.sort(key=lambda e:(e['output_steps_per_s'],e['K'],e['start_fine_step']))
                prefix = {}
                for example in examples:
                    bounded()
                    start, rate, k = example['start_fine_step'], example['output_steps_per_s'], example['K']
                    inputs = dataset.inputs(example)
                    fields = {n: torch.from_numpy(inputs['current'][n]).to('cuda') for n in FIELDS}
                    initial_hash = hashes(fields)
                    stride = 240 // rate
                    for mode in reg['modes']:
                        groups = {'fine_reference': (1,) * stride,
                                  'half_passes': (2,) * (stride // 2),
                                  'single_pass': (stride,)}[mode]
                        result['active'] = dict(identifier=identifier, start=start, output_steps_per_s=rate, K=k, mode=mode)
                        outputs = model.predict(fields, scene, start, rate, k, groups)
                        if len(outputs) != k or hashes(fields) != initial_hash:
                            raise ValueError('Incomplete outputs or changed caller inputs')
                        identity = [hashes(o) for o in outputs]
                        key = (start, rate, mode)
                        previous = prefix.get(key, [])
                        if identity[:len(previous)] != previous:
                            raise ValueError('Common physical prefix changed acrossK')
                        prefix[key] = identity
                        for lead, output in enumerate(outputs, 1):
                            if identity[lead - 1] != eager_expected[(identifier,start,rate,k,lead,mode)]:
                                raise ValueError('Graph output changed qualified eager field bytes')
                            truth = dataset._load(identifier, start + lead * stride)
                            host = {n: value.cpu() for n, value in output.items()}
                            nonfinite = {n: int((~v.isfinite()).sum()) for n, v in host.items()}
                            metrics = None if any(nonfinite.values()) else measure(host, truth, scene, normalization)
                            result['rows'].append(dict(identifier=identifier, split=example['split'],
                                counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                                force_profile=scene['forcing_program']['profile'], force_motion=scene['forcing_program']['motion'],
                                start_fine_step=start, output_steps_per_s=rate, output_dt_s=1 / rate,
                                K=k, lead=lead, elapsed_s=lead / rate, block_duration_s=k / rate,
                                mode=mode, groups_per_output=list(groups), actual_internal_advances=k * len(groups),
                                metrics=metrics, nonfinite_counts=nonfinite,
                                all_accuracy_pass=bool(metrics and all(v <= reg['maximum_error'] for v in metrics['relative'].values())),
                                output_sha256=identity[lead - 1]))
                        bounded()
                    write(args.output / 'results.json', result)
                print('Scored explicit-time GPU baseline and grouped controls:', identifier, flush=True)
        if len(result['rows']) != reg['expected_complete_outputs'] or inventory(model) != before:
            raise ValueError('Incomplete population or changed learned parameters')
        result['summaries'] = []
        for mode in reg['modes']:
            rows = [r for r in result['rows'] if r['mode'] == mode]
            result['summaries'].append(dict(mode=mode, complete_outputs=len(rows),
                failures=sum(not r['all_accuracy_pass'] for r in rows),
                worst_relative={n:max(r['metrics']['relative'][n] for r in rows if r['metrics'])
                                for n in ('equation', 'pressure', 'velocity', 'tracer')}))
        result.update(status='completed_temporal_residual_zero_graph_controls', all_outputs_bitwise_equal_eager=True, all_zero_network_outputs_bitwise_equal_qualified_schedule=True, active=None,
            fine_baseline_all_accuracy_pass=all(r['all_accuracy_pass'] for r in result['rows'] if r['mode'] == 'fine_reference'),
            learned_parameters_unchanged=True, causal_prefix_and_input_ownership_pass=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            elapsed_operational_seconds=time.monotonic() - started)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if model is not None:
            model.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
