"""DEBUG ONLY: native transport/force/viscosity stage boundaries at102->103."""
import argparse
import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT, read, write, digest
from continuous_reference_client import ContinuousReferenceClient
from cuda_native_transfer_v1 import upload_state, download_outputs
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory, measure
from probe_high_volume_first_step_v1 import difference
from probe_late_first_divergence_v1 import stages

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')


def run(args):
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh native-stage probe required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('All learned computation must be GPU')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen native-stage diagnostic changed: ' + path)
    os.sched_setaffinity(0, {4})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    case = reg['case']
    scene = dict(read(ROOT / case['receipt'])['scene'], reference_substeps=1, forcing_control_substeps=1)
    fixtures = read(ROOT / reg['fixture_receipt'])
    native_states = {}
    for spec in fixtures['fixtures']:
        with np.load(ROOT / spec['path'], allow_pickle=False) as archive:
            native_states[spec['step']] = {n: archive[n].copy() for n in archive.files}
    args.output.mkdir(parents=True)
    binary = ROOT / reg['native_binary']
    sizes = dimensions(scene['counts'])

    def primitive(mode, values, tag):
        directory = args.output / tag
        directory.mkdir()
        for name, value in values.items():
            np.asarray(value, dtype='<f4').tofile(directory / (name + '.f32'))
        output = directory / 'native-output'
        command = [str(binary), mode, *map(str, scene['counts']),
                   str(1 / scene['cells_per_metre']), str(1 / 60), str(directory), str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise ValueError('Native primitive failed: ' + result.stderr)
        return [np.fromfile(output / (name + '.f32'), dtype='<f4').reshape(size)
                for name, size in zip(FIELDS[1:] if mode == 'transport' else FIELDS[1:4], sizes)]

    cls = getattr(importlib.import_module(reg['candidate_module']), reg['candidate_class'])
    model = cls(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model)
    core = model.model
    original_transport, original_linear = core.transport, core.linear
    captured = {}

    def transport(fields, counts, alpha):
        captured['advected'] = original_transport(fields, counts, alpha)
        return captured['advected']

    def linear(fields, target_shape, name, head, selected_channel=None, sign=1):
        value = original_linear(fields, target_shape, name, head, selected_channel, sign)
        if name == 'viscosity':
            captured['entering'] = fields
            captured.setdefault('viscosity', {})[head] = value
        return value

    native = None
    report = dict(status='running', cpu_model_forward=False, native_oracles_diagnostic_only=True,
                  optimizer_updates=0, latency_claim=False)
    try:
        with torch.inference_mode():
            model.prepare_geometry(tuple(scene['counts']))
            if hasattr(core, 'configure_transport'):
                core.configure_transport(1 / scene['cells_per_metre'], 1 / 60)
            state = upload_state({n: torch.from_numpy(native_states[102][n]) for n in FIELDS})
            external = model.controls.generate(scene, 102, 1, 1)[0]
            core.transport, core.linear = transport, linear
            rhs, residual, pressure, output = stages(core, state, external, scene)
            emitted = download_outputs([output(pressure)])[0]
            observed = read(ROOT / reg['observation'])
            expected = next(r['output_sha256'] for r in observed['rows'] if r['K'] == 1 and r['step'] == 103)
            observed_bytes_equal = all(hashlib.sha256(emitted[n].numpy().tobytes()).hexdigest() == expected[n]
                                       for n in ('pressure', 'vx', 'vy', 'vz'))
            if reg['require_observed_bytes'] and not observed_bytes_equal:
                raise ValueError('Native-stage baseline does not reproduce actual pressure/velocity divergence')
            core.transport, core.linear = original_transport, original_linear
            true_advected = primitive('transport', {n: native_states[102][n] for n in FIELDS[1:]}, 'transport')
            report['transport_vs_native'] = {n: difference(a, b) for n, a, b in zip(FIELDS[1:], captured['advected'], true_advected)}
            native = ContinuousReferenceClient(scene, args.output / 'native-force-preview', io_timeout_s=120.)
            true_entering = native.preview_force(102, true_advected[:3])
            same_advected_entering = native.preview_force(102, [v.cpu().numpy() for v in captured['advected'][:3]])
            custody = native.close()
            native = None
            report['native_process'] = custody
            if not custody['reaped'] or custody['returncode'] != 0 or not custody['graceful_protocol_complete']:
                raise ValueError('Native force preview custody failed')
            true_entering = [v.reshape(s) for v, s in zip(true_entering, sizes[:3])]
            same_advected_entering = [v.reshape(s) for v, s in zip(same_advected_entering, sizes[:3])]
            report['entering_vs_actual_native'] = {n: difference(a, b) for n, a, b in zip(FIELDS[1:4], captured['entering'], true_entering)}
            report['force_same_advected_input_vs_native'] = {n: difference(a, b) for n, a, b in zip(FIELDS[1:4], captured['entering'], same_advected_entering)}
            true_provisional = primitive('viscosity', dict(zip(FIELDS[1:4], true_entering)), 'true-viscosity')
            same_input_provisional = primitive('viscosity', {n: v.cpu().numpy() for n, v in zip(FIELDS[1:4], captured['entering'])}, 'same-input-viscosity')
            beta = .001 / 1000. * (1 / 60) / (1 / scene['cells_per_metre'])**2
            actual_provisional = [core.wall((captured['entering'][c].double() + captured['viscosity'][c] * beta).float(), c) for c in range(3)]
            report['viscosity_same_entering_input_vs_native'] = {n: difference(a, b) for n, a, b in zip(FIELDS[1:4], actual_provisional, same_input_provisional)}
            report['provisional_vs_actual_native'] = {n: difference(a, b) for n, a, b in zip(FIELDS[1:4], actual_provisional, true_provisional)}
            # Tie extracted native primitive stages to the actual published
            # endpoint through GPU learned momentum with native-pressure-only.
            p = torch.from_numpy(native_states[103]['pressure']).to('cuda').reshape(tuple(reversed(scene['counts'])))
            factor = (1 / 60) * (1 / (1000. * (1 / scene['cells_per_metre'])))
            velocity = [core.wall((torch.from_numpy(v).to('cuda').double()
                                  + core.linear([p], s, 'momentum', c) * factor).float(), c)
                        for c, (v, s) in enumerate(zip(true_provisional, sizes[:3]))]
            report['native_stage_pressure_oracle_velocity_vs_actual'] = {n: difference(v, native_states[103][n]) for n, v in zip(FIELDS[1:4], velocity)}
            if any(v['different_values'] for v in report['native_stage_pressure_oracle_velocity_vs_actual'].values()):
                raise ValueError('Extracted native stages do not reconstruct actual velocity endpoint')
            report.update(status='completed_verified_native_primitive_boundary_probe',
                          actual_observed_pressure_velocity_bytes_equal=observed_bytes_equal,
                          candidate_boundary_metrics=measure(emitted, native_states[103], scene, read(ROOT / reg['normalization'])),
                          learned_parameters_unchanged=inventory(model) == before)
            if not report['learned_parameters_unchanged']:
                raise ValueError('Native-stage probe changed parameters')
            for key in ['transport_vs_native', 'force_same_advected_input_vs_native',
                        'viscosity_same_entering_input_vs_native', 'provisional_vs_actual_native']:
                print(key, report[key], flush=True)
    finally:
        core.transport, core.linear = original_transport, original_linear
        if native is not None:
            report['native_process'] = native.close()
        report.update(protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
        write(args.output / 'results.json', report)
        model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
