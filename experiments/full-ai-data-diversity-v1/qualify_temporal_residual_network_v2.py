"""Zero-update CUDA model, coupled loss, temporal encoder and restore checks."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from temporal_group_projector_v1 import TemporalGroupProjector
from temporal_residual_network_v1 import TemporalResidual, features, physical_deltas, PARAMETERS
from long_temporal_dataset_v1 import LongTemporalDataset, FIELDS
from recurrent_sampler_native import dimensions
from internal_residual_loss_v1 import metrics as physical_losses
from diagnose_causal_composition import inventory, measure


def hashes(values):
    return [hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for v in values]


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only neural qualification required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen neural qualification source changed: ' + name)
    targets = read(ROOT / reg['target_qualification'])
    if not targets['all_teacher_injections_pass'] or not targets['all_gradient_controls_pass']:
        raise ValueError('Native temporal targets and physical gradients must first pass')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    projector = None
    result = dict(status='running', rows=[], probes=[], optimizer_updates=0,
        new_parameter_fits=0, cpu_model_forward=False, reserved52_references_opened=0,
        inference_device='cuda', display_label='GPU (CUDA)', full_generator_target_pass=False,
        deployable_accuracy_claim=False, latency_claim=False,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    started = time.monotonic()
    try:
        dataset = LongTemporalDataset(read(ROOT / reg['index']))
        projector = TemporalGroupProjector(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
        network = TemporalResidual()
        original = inventory(network)
        frozen = inventory(projector)
        probe = copy.deepcopy(network)
        with torch.no_grad():
            for layer in (*probe.velocity, probe.tracer):
                layer.weight.copy_(torch.linspace(-.0001, .0001, layer.weight.numel(), device='cuda').reshape_as(layer.weight))
                layer.bias.fill_(.0001)
        norm = read(ROOT / reg['normalization'])
        for row in targets['rows']:
            identifier, start, width = row['identifier'], row['start_fine_step'], row['width']
            scene = dataset.receipts[identifier][1]['scene']
            initial = dataset._load(identifier, start)
            truth = dataset._load(identifier, start + width)
            fields = {n: torch.from_numpy(initial[n]).to('cuda') for n in FIELDS}
            phase = projector.phase(fields, scene, start, width)
            spatial, ordered, active = features(phase)
            network.zero_grad(set_to_none=True)
            values = network(spatial, ordered, active)
            if any(bool(v.count_nonzero()) for v in values):
                raise ValueError('Initial correction heads must produce exact zero')
            delta_v, delta_t = physical_deltas(values, phase)
            output = projector.emit(phase, delta_v, delta_t, float_carriers=False)
            shapes = dict(pressure=phase['shape'], **dict(zip(FIELDS[1:], dimensions(phase['counts']))))
            sample = dict(h=phase['h'], rho=phase['rho'],
                target_rhs=torch.from_numpy(truth['rhs']).to('cuda').reshape(phase['shape']))
            for n in FIELDS:
                sample['target_' + n] = torch.from_numpy(truth[n]).to('cuda').reshape(shapes[n])

            def losses(prediction):
                return physical_losses({n: v.reshape(shapes[n]) for n, v in prediction.items()}, sample, norm)

            components = losses(output)
            objective = sum(components.values())
            objective.backward()
            if any(p.grad is not None and not bool(p.grad.isfinite().all()) for p in network.parameters()):
                raise ValueError('Neural coupled objective has nonfinite gradients')
            head_gradient = sum(float(p.grad.double().square().sum()) for layer in (*network.velocity, network.tracer)
                                for p in layer.parameters() if p.grad is not None) ** .5
            if bool(active) and float(objective.detach()) > reg['nonzero_objective_floor'] and head_gradient == 0:
                raise ValueError('Active physical correction has no output-head gradient')
            if start == 480 and width in (8, 32):
                probe.zero_grad(set_to_none=True)
                dv, dt = physical_deltas(probe(spatial, ordered, active), phase)
                probe_loss = sum(losses(projector.emit(phase, dv, dt, float_carriers=False)).values())
                probe_loss.backward()
                norms = {name: float(layer.weight.grad.double().norm())
                         for name, layer in [('force1', probe.force1), ('force2', probe.force2), ('spatial1', probe.conv1)]}
                has_force = bool(ordered.count_nonzero())
                required = [norms['spatial1']] if bool(active) else []
                if has_force:
                    required.extend((norms['force1'], norms['force2']))
                if any(not np.isfinite(v) or v <= 0 for v in required):
                    raise ValueError('Nonzero-head probe does not train temporal/spatial encoders')
                result['probes'].append(dict(identifier=identifier, width=width, gradient_norms=norms,
                    force_path_has_nonzero_input=has_force, zero_force_input_weight_gradient_may_be_zero=True,
                    weights_are_diagnostic_only=True))
            with torch.no_grad():
                actual = projector.emit(phase, *physical_deltas(network(spatial, ordered, active), phase))
                measured = measure({n: v.cpu() for n, v in actual.items()}, truth, scene, norm)
            for name in ('equation', 'pressure', 'velocity', 'tracer'):
                expected = row['baseline_metrics']['relative'][name]
                if not np.isclose(measured['relative'][name], expected, rtol=1e-10, atol=1e-20):
                    raise ValueError('Zero network changed the frozen physical baseline')
            result['rows'].append(dict(identifier=identifier, start_fine_step=start, width=width,
                local_dt_s=phase['dt'], cells_per_metre=scene['cells_per_metre'], counts_xyz=scene['counts'],
                spatial_shape=list(spatial.shape), ordered_force_shape=list(ordered.shape), active=bool(active),
                zero_physical_baseline_preserved=True, head_gradient_norm=head_gradient,
                squared_physical_losses={n: float(v.detach()) for n, v in components.items()}))
            del objective, output, values, delta_v, delta_t, actual, components, phase, fields, sample
            if (time.monotonic() - started > reg['maximum_wall_seconds']
                    or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                raise ValueError('Finite neural qualification resource budget exceeded')
            if len(result['rows']) % 30 == 0:
                write(args.output / 'results.json', result)
                print('Temporal neural zero/gradient cases', len(result['rows']), '/120', flush=True)

        # Genuine pulse-order witness: same first/last values, integral and first
        # temporal moment, but different adjacency/order of interior impulses.
        a = torch.zeros((1, 6, 8, 5, 5, 5), device='cuda')
        b = torch.zeros_like(a)
        a[:, :, [2, 5]] = 1.
        b[:, :, [3, 4]] = 1.
        with torch.no_grad():
            order_gap = float((network.encode_forces(a) - network.encode_forces(b)).norm())
            if order_gap <= reg['minimum_order_witness_difference']:
                raise ValueError('Chosen equal-moment pulse-order witness collapsed')
        # CPU construction and exact restore only; every probe forward is CUDA.
        host = TemporalResidual(device='cpu')
        host.load_state_dict({n: v.detach().cpu().clone() for n, v in probe.state_dict().items()})
        if inventory(host) != inventory(probe):
            raise ValueError('Exact typed CPU parameter restore failed')
        restored = host.to('cuda')
        zero_spatial = torch.zeros((1, 34, 5, 5, 5), device='cuda')
        enabled = torch.ones((), device='cuda', dtype=torch.bool)
        with torch.no_grad():
            expected = probe(zero_spatial, a, enabled)
            if not any(bool(v.count_nonzero()) for v in expected):
                raise ValueError('Restored GPU probe is vacuous')
            if hashes(expected) != hashes(restored(zero_spatial, a, enabled)):
                raise ValueError('Restored copied-GPU probe output differs')
        quiet_receipt = read(ROOT / reg['quiescent_receipt'])
        quiet_states = {s['fine_step']: s for s in quiet_receipt['states']}
        quiet = {}
        for step in (0, 8):
            source = (ROOT / reg['quiescent_receipt']).parent / quiet_states[step]['path']
            if digest(source) != quiet_states[step]['sha256']:
                raise ValueError('Native quiescent control changed')
            with np.load(source, allow_pickle=False) as archive:
                quiet[step] = {n: archive[n].copy() for n in (*FIELDS, 'rhs')}
        quiet_phase = projector.phase({n: torch.from_numpy(quiet[0][n]).to('cuda') for n in FIELDS},
                                      quiet_receipt['scene'], 0, 8)
        quiet_features = features(quiet_phase)
        if bool(quiet_features[2]) or not bool(quiet_features[0][:, 7].count_nonzero()):
            raise ValueError('Varied native tracer must not activate a quiescent correction')
        with torch.no_grad():
            if any(bool(v.count_nonzero()) for v in probe(*quiet_features)):
                raise ValueError('Actual quiescent phase gives a nonzero correction')
            if not any(bool(v.count_nonzero()) for v in probe(quiet_features[0], quiet_features[1], enabled)):
                raise ValueError('Native quiet gate control is vacuous')
            quiet_output = projector.emit(quiet_phase, *physical_deltas(probe(*quiet_features), quiet_phase))
            quiet_metrics = measure({n: v.cpu() for n, v in quiet_output.items()}, quiet[8], quiet_receipt['scene'], norm)
            if not quiet_metrics['all_below_1e5']:
                raise ValueError('Quiet gate fails actual native physical output')
            alternate = dict(quiet_phase, dt=quiet_phase['dt'] / 2,
                             reference_dt_s=quiet_phase['reference_dt_s'] / 2)
            alternate_features = features(alternate)
            if torch.equal(quiet_features[0][:, 28:30], alternate_features[0][:, 28:30]):
                raise ValueError('Explicit physical local/fine timesteps were lost')
        if len(result['rows']) != reg['expected_groups'] or inventory(network) != original or inventory(projector) != frozen:
            raise ValueError('All native target cases and unchanged model weights required')
        result.update(status='passed_native_temporal_neural_features_objective_and_restore',
            parameter_count=PARAMETERS, parameter_inventory=original, native_training_pilots=4,
            pulse_order_witness_difference=order_gap, pulse_order_test_is_not_universal_injectivity=True,
            exact_typed_CPU_restore=True, copied_GPU_nonzero_probe_bitwise_equal=True,
            inactive_nonzero_head_probe_exact_zero=True, learned_parameters_unchanged=True,
            native_quiescent_metrics=quiet_metrics, explicit_timestep_channels_distinguish_empty_flow=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if projector is not None:
            projector.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
