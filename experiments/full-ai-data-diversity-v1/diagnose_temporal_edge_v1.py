"""Localize the observed pulse-off baseline failure without fitting or timing."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from temporal_frozen_backbone_v2 import TemporalFrozenBackbone
from temporal_control_packets_v1 import control_packet
from bound_sampler_pressure_graph_v1 import BoundSamplerPressureGraph
from cell_coordinate_pressure_graph_v1 import CellCoordinatePressureGraph
from absolute_pressure_graph_v2 import AbsolutePressureGraph
from scheduled_force_fine_controls import force
from diagnose_causal_composition import inventory, measure

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')


class Variant(TemporalFrozenBackbone):
    def __init__(self, cls, reg):
        self.backend = cls(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
        self.model, self.controls = self.backend.model, self.backend.controls
        self.asset_identity = self.backend.asset_identity


class OfflineKnownForce:
    """Independent CPU construction of prescribed fields, no learned forward."""
    def __init__(self, scene):
        self.scene, self.step = scene, None

    def generate_from_packet(self, counts, packet):
        return [[v.to('cuda') for v in force(self.scene, self.step)[0]]]


def hashes(fields):
    return {n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
            for n, v in fields.items()}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU edge diagnosis required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen edge-diagnosis source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    receipt_path = ROOT / reg['receipt']
    receipt = read(receipt_path)
    scene = receipt['scene']
    states = {s['fine_step']: s for s in receipt['states']}

    def load(step):
        spec = states[step]
        path = receipt_path.parent / spec['path']
        if digest(path) != spec['sha256']:
            raise ValueError('Frozen native internal state changed')
        with np.load(path, allow_pickle=False) as a:
            return {n: a[n].copy() for n in (*FIELDS, 'rhs')}

    bad = [r for r in read(ROOT / reg['failed_qualification'])['rows']
           if r['mode'] == 'fine_reference' and not r['all_accuracy_pass']]
    if len(bad) != 1 or bad[0]['start_fine_step'] != 0 or bad[0]['elapsed_s'] != 32 / 240:
        raise ValueError('Exact previously observed32fine-update failure required')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', cases=[], new_parameter_fits=0, cpu_model_forward=False,
        inference_device='cuda', display_label='GPU (CUDA)', latency_claim=False,
        reserved52_references_opened=0, full_generator_target_pass=False,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    try:
        for label, cls in [('nested_physical', BoundSamplerPressureGraph),
                           ('cell_coordinates', CellCoordinatePressureGraph),
                           ('original_geometry', AbsolutePressureGraph)]:
            for forcing in ('GPU_dynamic', 'offline_native_known_force'):
                model = Variant(cls, reg)
                before = inventory(model)
                try:
                    if forcing == 'offline_native_known_force':
                        model.controls = OfflineKnownForce(scene)
                    initial = load(0)
                    fields = {n: torch.from_numpy(initial[n]).to('cuda') for n in FIELDS}
                    rows = []
                    for step in range(1, 33):
                        if time.monotonic() - started > reg['maximum_wall_seconds']:
                            raise TimeoutError('Finite edge-diagnosis wall budget')
                        if forcing == 'offline_native_known_force':
                            model.controls.step = step - 1
                        packet = torch.from_numpy(control_packet(scene, step - 1, 1)).to('cuda')
                        fields = model.predict_packet(fields, scene, 1, (1,), packet)[0]
                        host = {n: v.cpu() for n, v in fields.items()}
                        truth = load(step)
                        metrics = measure(host, truth, scene, read(ROOT / reg['normalization']))
                        mismatch = {n: int(np.count_nonzero(host[n].numpy().view(np.uint32)
                            != truth[n].view(np.uint32))) for n in FIELDS[1:]}
                        rows.append(dict(fine_step=step, physical_time_s=step / 240,
                            metrics=metrics, carrier_u32_mismatch_counts=mismatch,
                            output_sha256=hashes(fields)))
                    if label == 'nested_physical' and forcing == 'GPU_dynamic' and rows[-1]['output_sha256'] != bad[0]['output_sha256']:
                        raise ValueError('Fine-by-fine probe did not reproduce actual failing output bytes')
                    if inventory(model) != before:
                        raise ValueError('Diagnosis changed learned parameters')
                    result['cases'].append(dict(sampler=label, known_forcing=forcing, rows=rows,
                        first_carrier_mismatch_step=next((r['fine_step'] for r in rows
                            if any(r['carrier_u32_mismatch_counts'].values())), None),
                        maximum_relative={n: max(r['metrics']['relative'][n] for r in rows)
                            for n in ('equation', 'pressure', 'velocity', 'tracer')},
                        failing_fine_steps=[r['fine_step'] for r in rows if not r['metrics']['all_below_1e5']],
                        final_failed_case_passes=rows[-1]['metrics']['all_below_1e5']))
                    write(args.output / 'results.json', result)
                finally:
                    model.close()
        result.update(status='completed_pulse_off_GPU_baseline_sampler_and_force_diagnosis',
            failed_case_reproduced_bitwise=True, variants_are_diagnostics_only=True,
            wider_model_qualification_not_proven=True)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
