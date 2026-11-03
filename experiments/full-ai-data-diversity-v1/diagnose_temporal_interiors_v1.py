"""Offline oracle-endpoint interpolation diagnostic, never a candidate model.

Answers how much correction a cheap interior decoder must learn even if the
endpoint were exact. True future endpoints are used only in this diagnostic.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from long_temporal_dataset_v1 import ROOT, FIELDS, LongTemporalDataset, digest


def metric(actual, reference, names):
    numerator = sum(float(np.square(actual[n].astype(np.float64) - reference[n]).sum())
                    for n in names) ** .5
    norm = sum(float(np.square(reference[n].astype(np.float64)).sum()) for n in names) ** .5
    return dict(relative_l2=numerator / max(norm, 1e-30), absolute_l2=numerator,
                reference_l2=norm)


def run(args):
    if args.output.exists():
        raise ValueError('Fresh diagnostic output required')
    reg = json.loads(args.protocol.read_text())
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen offline diagnostic source changed: ' + name)
    qualified = json.loads((ROOT / reg['dataset_qualification']).read_text())
    if qualified['status'] != 'passed_real_long_temporal_dataset_conditioning_and_label_isolation':
        raise ValueError('Qualified temporal loader required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    index = json.loads((ROOT / reg['index']).read_text())
    dataset = LongTemporalDataset(index)
    args.output.mkdir(parents=True)
    result = dict(status='running', rows=[],
        offline_oracle_endpoint_only=True, future_true_endpoint_used=True,
        deployable_model=False, learned_forwards=0, cpu_model_forward=False,
        new_parameter_fits=0, reserved52_references_opened=0,
        full_generator_target_pass=False, speed_measured=False,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        interpretation='Oracle diagnostic gives the interpolation residual when endpoints are exact. It does not predict endpoints, qualify a model, or prove a lower bound for other interior decoders.')
    try:
        for example in index['examples']:
            if example['K'] == 1:
                continue
            inputs, labels = dataset.inputs(example), dataset.labels(example)
            endpoint = labels['endpoint']
            for j, target in enumerate(labels['outputs'][:-1], 1):
                alpha = j / example['K']
                estimate = {n: ((1 - alpha) * inputs['current'][n].astype(np.float64)
                                + alpha * endpoint[n].astype(np.float64)).astype(target[n].dtype)
                            for n in FIELDS}
                errors = {name: metric(estimate, target, fields)
                          for name, fields in [('pressure', ('pressure',)),
                                               ('velocity', ('vx', 'vy', 'vz')),
                                               ('tracer', ('tracer',))]}
                result['rows'].append(dict(identifier=example['identifier'], split=example['split'],
                    start_fine_step=example['start_fine_step'],
                    output_steps_per_s=example['output_steps_per_s'], K=example['K'],
                    lead_index=j, elapsed_s=j / example['output_steps_per_s'],
                    counts_xyz=inputs['counts_xyz'], cells_per_metre=1 / inputs['cell_size_m'],
                    errors=errors))
        result['per_configuration'] = []
        for rate in (30, 60, 120):
            for k in (2, 3, 4):
                rows = [r for r in result['rows'] if r['output_steps_per_s'] == rate and r['K'] == k]
                result['per_configuration'].append(dict(output_steps_per_s=rate, K=k,
                    interior_outputs=len(rows), worst_relative_l2={
                        n: max(r['errors'][n]['relative_l2'] for r in rows)
                        for n in ('pressure', 'velocity', 'tracer')}))
        result.update(status='completed_true_endpoint_linear_interior_diagnostic',
            independent_masters_per_split=index['independent_masters_per_split'],
            interior_outputs=len(result['rows']),
            worst_relative_l2={n: max(r['errors'][n]['relative_l2'] for r in result['rows'])
                               for n in ('pressure', 'velocity', 'tracer')},
            endpoint_predictions=0, K1_predictions=0)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
