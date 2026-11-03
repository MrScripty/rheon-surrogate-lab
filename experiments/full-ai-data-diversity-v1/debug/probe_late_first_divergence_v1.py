"""DEBUG: isolate known force versus learned stages at actual native102->103.

The native-pressure intervention is a diagnostic oracle only. All learned
operations remain on CUDA and no deployment source or parameter is changed.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT, read, write, digest
from complete_cuda_graph_v2 import CompleteCudaGraphGeneratorV2
from cuda_native_transfer_v1 import upload_state, download_outputs
from expanded_cuda_generator_v1 import prescribed_controls
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory, measure
from probe_high_volume_first_step_v1 import difference

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')


def stages(core, state, external, scene):
    counts = tuple(scene['counts'])
    shape = tuple(reversed(counts))
    sizes = dimensions(counts)
    dt, h, rho = 1 / 60, 1 / scene['cells_per_metre'], 1000.
    current = [state[n].reshape(s) for n, s in zip(FIELDS[1:], sizes)]
    previous = state['pressure'].reshape(1, 1, *shape)
    advected = core.transport(current, counts, dt / h)
    entering = [(v.double() + f).float() for v, f in zip(advected[:3], external)]
    provisional = [core.wall((entering[c].double() + core.linear(entering, s, 'viscosity', c)
                              * (.001 / rho * dt / h**2)).float(), c)
                   for c, s in enumerate(sizes[:3])]
    rhs = torch.zeros(shape, dtype=torch.float64, device='cuda')
    for channel in range(3):
        rhs = rhs + (h * h) * core.linear(provisional, shape, 'flux', 0, channel)
    rhs = rhs / torch.full((), dt, dtype=torch.float64, device='cuda')
    history = [core.wall(core.linear([previous[0, 0]], s, 'momentum', c)
                         * ((h * h) * (1 / (rho * h))), c)
               for c, s in enumerate(sizes[:3])]
    residual = rhs - core.linear(history, shape, 'flux', 0, sign=-1)
    predicted = previous + core.pressure((residual * (rho / h)).reshape_as(previous), counts)
    predicted = predicted - predicted[:, :, :1, :1, :1]

    def output(pressure):
        velocity = [core.wall((provisional[c].double()
                              + core.linear([pressure[0, 0]], s, 'momentum', c)
                              * (dt * (1 / (rho * h)))).float(), c)
                    for c, s in enumerate(sizes[:3])]
        return dict(pressure=pressure.contiguous().flatten(),
                    vx=velocity[0].flatten(), vy=velocity[1].flatten(),
                    vz=velocity[2].flatten(), tracer=advected[3].flatten())
    return rhs, residual, predicted, output


def run(args):
    if args.output.exists():
        raise ValueError('Fresh actual boundary probe required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('GPU-only learned probe required')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen diagnostic source changed: ' + path)
    os.sched_setaffinity(0, {4})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    case = reg['case']
    scene = dict(read(ROOT / case['receipt'])['scene'], reference_substeps=1, forcing_control_substeps=1)
    fixture = read(ROOT / reg['fixture_receipt'])
    states = {}
    for spec in fixture['fixtures']:
        if digest(ROOT / spec['path']) != spec['sha256']:
            raise ValueError('Actual native boundary fixture changed')
        with np.load(ROOT / spec['path'], allow_pickle=False) as archive:
            states[spec['step']] = {n: archive[n].copy() for n in archive.files}
    initial = {n: torch.from_numpy(states[102][n]) for n in FIELDS}
    truth = states[103]
    observed = read(ROOT / reg['observation'])
    expected = next(r['output_sha256'] for r in observed['rows'] if r['K'] == 1 and r['step'] == 103)
    model = CompleteCudaGraphGeneratorV2(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model)
    args.output.mkdir(parents=True)
    rows = []
    try:
        with torch.inference_mode():
            model.prepare_geometry(tuple(scene['counts']))
            state = upload_state(initial)
            gpu_force = model.controls.generate(scene, 102, 1, 1)[0]
            cpu_force = prescribed_controls(scene, 102, 1, 1)[0]
            forces = [difference(a, b.cpu()) for a, b in zip(gpu_force, cpu_force)]
            for provider, external in [('gpu_dynamic_known_input', gpu_force), ('cpu_known_input_only', cpu_force)]:
                rhs, residual, predicted, output = stages(model.model, state, external, scene)
                oracle = torch.from_numpy(truth['pressure']).to('cuda').reshape_as(predicted)
                for intervention, pressure in [('no_intervention', predicted), ('native_pressure_oracle_only', oracle)]:
                    emitted = download_outputs([output(pressure)])[0]
                    hashes = {n: hashlib.sha256(v.numpy().tobytes()).hexdigest() for n, v in emitted.items()}
                    actual = all(hashes[n] == expected[n] for n in ('pressure', 'vx', 'vy', 'vz'))
                    if provider == 'gpu_dynamic_known_input' and intervention == 'no_intervention' and not actual:
                        raise ValueError('Stage probe failed to reproduce actual first pressure/velocity divergence')
                    metrics = measure(emitted, truth, scene, read(ROOT / reg['normalization']))
                    rows.append(dict(provider=provider, intervention=intervention, metrics=metrics,
                        actual_observed_pressure_velocity_bytes_equal=actual,
                        rhs_vs_native=difference(rhs, truth['rhs']),
                        residual_vs_native=difference(residual, truth['native_input_residual']),
                        pressure_vs_native=difference(pressure, truth['pressure']),
                        velocity_vs_native={n: difference(emitted[n], truth[n]) for n in ('vx', 'vy', 'vz')}))
                    print(provider, intervention, 'RHS', rows[-1]['rhs_vs_native'],
                          'P/V errors', metrics['relative']['pressure'], metrics['relative']['velocity'], flush=True)
        if inventory(model) != before:
            raise ValueError('Probe changed learned parameters')
        write(args.output / 'results.json', dict(status='completed_actual_first_divergence_stage_probe',
            rows=rows, known_force_difference=forces, cpu_model_forward=False, optimizer_updates=0,
            native_future_pressure_oracle_diagnostic_only=True, no_candidate_change=True,
            latency_claim=False, protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256']))
    finally:
        model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
