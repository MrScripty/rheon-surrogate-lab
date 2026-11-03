"""Qualify zero controls, shared computation, event sensitivity and restoration.

This performs no optimizer updates. Coarse zero-head endpoints and linearly
reconstructed interiors are controls, not an accuracy-qualified successor.
"""
import argparse
import hashlib
import os
from pathlib import Path
import time

import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index, LongTemporalDataset, FIELDS
from shared_temporal_graph_v3 import SharedTemporalGraph as SharedTemporalGenerator
from shared_temporal_generator_v3 import SharedTemporalGenerator as EagerStationaryGenerator
from shared_temporal_graph_v1 import SharedTemporalGraph as OriginalSharedGraph
from shared_temporal_network_v1 import SharedTemporalResidual
from temporal_frozen_history_v1 import TemporalFrozenBackbone
from diagnose_causal_composition import inventory


def hashes(fields):
    return {n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n, v in fields.items()}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only shared-generator qualification required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen shared-generator source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    result = dict(status='running', rows=[], neural_controls=[], new_parameter_fits=0,
        residual_optimizer_updates=0, inference_device='cuda', display_label='GPU (CUDA)',
        cpu_model_forward=False, reserved52_references_opened=0, latency_claim=False,
        accuracy_qualification=False, full_generator_target_pass=False,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    started = time.monotonic()
    model = None
    try:
        index = build_index([ROOT / p for p in reg['audits']])
        dataset = LongTemporalDataset(index)
        qualified = read(ROOT / reg['eager_qualification'])
        if qualified['status'] != 'passed_shared_temporal_zero_controls_gradients_events_and_restore':
            raise ValueError('Qualified eager shared prototype required')
        expected_graph = {(r['identifier'], r['start_fine_step'], r['output_steps_per_s'], r['K'], r['hidden_width']): r['output_sha256'] for r in qualified['rows']}
        examples = [e for e in index['examples'] if e['start_fine_step'] in reg['starts']]
        for hidden_width in reg['hidden_widths']:
            model = SharedTemporalGenerator(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'], hidden=hidden_width, force_sum_build=reg['force_sum_build'])
            before = inventory(model)
            for example in examples:
                scene = dataset.receipts[example['identifier']][1]['scene']
                inputs = dataset.inputs(example)
                fields = {n: torch.from_numpy(inputs['current'][n]).to('cuda') for n in FIELDS}
                initial = hashes(fields)
                rate, k, start = example['output_steps_per_s'], example['K'], example['start_fine_step']
                with torch.inference_mode():
                    outputs = model.predict(fields, scene, start, rate, k)
                    packet, event_values, event_mask = model.packets(scene, start, rate, k)
                    eager_outputs = EagerStationaryGenerator.predict_packet(model, fields, scene, rate, k, packet, event_values, event_mask)
                    if [hashes(o) for o in outputs] != [hashes(o) for o in eager_outputs]:
                        raise ValueError('Stationary graph differs from eager complete states')
                    previous_hashes = expected_graph[(example['identifier'], start, rate, k, hidden_width)]
                    changed_heads = []
                    for lead, (actual, old_hash) in enumerate(zip(outputs, previous_hashes), 1):
                        if hashes(actual) != old_hash:
                            target = dataset._load(example['identifier'], example['target_fine_steps'][lead - 1])
                            native_hash = {n: hashlib.sha256(target[n].tobytes()).hexdigest() for n in FIELDS}
                            if hashes(actual) != native_hash or hashes(actual) != initial:
                                raise ValueError('Changed stationary output differs from current/native state')
                            changed_heads.append(lead)
                    if hashes(fields) != initial:
                        raise ValueError('Shared generator changed caller fields')
                result['rows'].append(dict(identifier=example['identifier'], start_fine_step=start,
                    output_steps_per_s=rate, K=k, hidden_width=hidden_width,
                    endpoint_dt_s=k / rate, one_coarse_transport_per_block=True,
                    spatial_encoder_calls=1, pressure_projections=2 + (k - 1),
                    eager_stationary_graph_bytes_exact=True, changed_stationary_heads_native_exact=changed_heads,
                    output_sha256=[hashes(o) for o in outputs]))
                if (time.monotonic() - started > reg['maximum_wall_seconds']
                        or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                    raise ValueError('Shared qualification resource budget exceeded')
            network = model.network
            saved = {n: p.detach().clone() for n, p in network.state_dict().items()}
            # Nonzero readout probes exercise trunk gradients and conditioning;
            # assigning fixed diagnostic coefficients is not an optimizer fit.
            with torch.no_grad():
                for layer in (*network.endpoint_velocity, network.endpoint_tracer,
                              *network.interior_velocity, network.interior_tracer):
                    layer.weight.fill_(.01)
            with torch.inference_mode():
                model.release_graphs()
                original_model = OriginalSharedGraph(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'], hidden=hidden_width)
                try:
                    original_model.network.load_state_dict(network.state_dict(), strict=True)
                    expected_nonzero = original_model.predict(fields, scene, start, rate, k)
                    actual_nonzero = model.predict(fields, scene, start, rate, k)
                    if [hashes(o) for o in actual_nonzero] != [hashes(o) for o in expected_nonzero]:
                        raise ValueError('Batched force shared graph changed nonzero complete output bytes')
                finally:
                    original_model.close()
            network.requires_grad_(True)
            spatial = torch.linspace(-.1, .1, 26 * 5 * 7 * 9, device='cuda', dtype=torch.float32).reshape(1, 26, 5, 7, 9)
            controls = torch.linspace(-.2, .2, 24 * 8, device='cuda', dtype=torch.float32).reshape(1, 24, 8)
            events = torch.zeros((1, 5, 32), device='cuda', dtype=torch.float32)
            events[0, :, 0] = events.new_tensor([.1, .1, 0., 1., .2])
            event_mask = torch.zeros((1, 1, 32), device='cuda', dtype=torch.float32)
            event_mask[0, 0, 0] = 1
            active = torch.ones((), device='cuda', dtype=torch.bool)
            first = network.encode(spatial, controls, events, event_mask)
            early = network.endpoint(first, active)
            late_events = events.clone()
            late_events[0, :2, 0] = .9
            late = network.endpoint(network.encode(spatial, controls, late_events, event_mask), active)
            gap = max(float((a - b).detach().abs().max()) for a, b in zip(early, late))
            if not gap > 0:
                raise ValueError('Exact event phase does not reach the endpoint decoder')
            condition = torch.linspace(-.1, .1, 8 * 5 * 7 * 9, device='cuda', dtype=torch.float32).reshape(1, 8, 5, 7, 9)
            interior = network.interior(first, condition, active)
            objective = sum(v.square().mean() for v in [*early, *interior])
            objective.backward()
            for name, parameter in network.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError('Nonfinite or missing shared network gradient: ' + name)
            if not all(float(getattr(network, name).weight.grad.norm()) > 0
                       for name in ('force1', 'event1', 'mix', 'local', 'interior_mix')):
                raise ValueError('Nonzero probe did not train shared trunks and both conditioning paths')
            with torch.no_grad():
                quiet = network.endpoint(first, torch.zeros_like(active))
                if any(torch.count_nonzero(v) for v in quiet):
                    raise ValueError('Quiescent readout gate differs from zero')
                origin = [v.detach().cpu().clone() for v in early]
                cpu = SharedTemporalResidual(hidden=hidden_width, device='cpu')
                cpu.load_state_dict({n: v.detach().cpu().clone() for n, v in network.state_dict().items()}, strict=True)
                if any(not torch.equal(cpu.state_dict()[n], v.detach().cpu()) for n, v in network.state_dict().items()):
                    raise ValueError('CPU typed parameter restoration differs')
                restored = cpu.to('cuda').eval()
                copied = restored.endpoint(restored.encode(spatial, controls, events, event_mask), active)
                if any(not torch.equal(a, b.cpu()) for a, b in zip(origin, copied)):
                    raise ValueError('Copied nonzero GPU probe differs after typed CPU restore')
                network.load_state_dict(saved, strict=True)
            network.requires_grad_(False)
            if inventory(model) != before:
                raise ValueError('Diagnostic changed prototype parameters')
            result['neural_controls'].append(dict(hidden_width=hidden_width,
                neural_parameters=sum(p.numel() for p in network.parameters()),
                nonzero_event_offset_output_maximum_gap=gap, nonzero_complete_shared_outputs_bitwise_equal_original=True, shared_trunk_and_conditioning_gradients_pass=True,
                typed_CPU_restore_exact=True, cpu_forward_executed=False,
                copied_nonzero_GPU_probe_exact=True, initial_zero_heads_restored=True))
            model.close()
            model = None
            write(args.output / 'results.json', result)
        if len(result['rows']) != reg['expected_blocks']:
            raise ValueError('Incomplete all12/shared-width qualification population')
        result.update(status='passed_stationary_prefix_graph_native_controls_gradients_events_and_restore',
            all_complete_outputs_bitwise_equal_stationary_eager=True,
            complete_output_checks=sum(r['K'] for r in result['rows']),
            elapsed_operational_seconds=time.monotonic() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exception:
        result.update(status='failed', error=repr(exception))
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
