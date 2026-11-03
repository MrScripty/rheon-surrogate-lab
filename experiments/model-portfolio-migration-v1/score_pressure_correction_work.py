"""Independently rescore retained policy outputs using a NumPy FV stencil."""
import argparse
from contextlib import ExitStack
import hashlib
import math
from pathlib import Path
import time

from pressure_gpu_timing import ROOT, read, save, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def score(np, pressure, sample):
    # Independent array implementation: negative finite-volume Laplacian with
    # zero exterior flux, using the same captured cubic spacing and fluid rho.
    h, rho = float(sample['cell_size_m'][0]), float(sample['rho'][0])
    weight = (h * h) * (1.0 / (rho * h))
    applied = np.zeros_like(pressure)
    for axis in (4, 3, 2):
        face = np.diff(pressure, axis=axis)
        prepend = [(0, 0)] * 5
        append = [(0, 0)] * 5
        prepend[axis] = (1, 0)
        append[axis] = (0, 1)
        applied += weight * (np.pad(face, prepend) - np.pad(face, append))
    residual = applied - sample['rhs'].numpy()
    difference = pressure - sample['reference'].numpy()
    difference -= difference.mean()
    centered_reference = sample['reference'].numpy() - sample['reference'].numpy().mean()
    return dict(common_equation_relative_l2=float(np.linalg.norm(residual.ravel())) /
                    max(float(np.linalg.norm(sample['rhs'].numpy().ravel())), 1e-30),
                pressure_mean_free_relative_l2=float(np.linalg.norm(difference.ravel())) /
                    max(float(np.linalg.norm(centered_reference.ravel())), 1e-30),
                pressure_absolute_rms=float(np.sqrt(np.mean(difference * difference))),
                pressure_absolute_max=float(np.max(np.abs(difference))))


def run(protocol_path, report_path):
    require(not report_path.exists(), 'Fresh independent numerical report required')
    protocol = read(protocol_path)
    job = ROOT / protocol['output_root']
    batch = read(job / 'status.json')
    require(batch['status'] == 'complete_original_GPU_pressure_correction_work_matrix'
            and batch['protocol_sha256'] == sha(protocol_path) and batch['active'] is None
            and not batch['pending'], 'Complete original execution required')
    for name, expected in protocol['source_sha256'].items():
        require(sha(ROOT / name) == expected, 'Preregistered source changed: ' + name)
    import numpy as np
    import torch
    import pressure_cases
    torch.set_num_threads(1)
    cases = pressure_cases.load(torch, np)
    frozen = read(ROOT / protocol['frozen_cases']['path'])
    require(sha(ROOT / protocol['frozen_cases']['path']) == protocol['frozen_cases']['sha256'],
            'Frozen equation bindings changed')
    completed = {row['identifier']:row for row in batch['completed']}
    models, policy_rows = {}, {}
    total_fields = 0
    maximum_relative_metric_difference = 0.0
    for identifier, specification in protocol['models'].items():
        captured = completed[identifier]
        result_path = ROOT / captured['result']
        require(sha(result_path) == captured['sha256'], 'Execution result changed')
        result = read(result_path)
        require(result['status'] == 'complete_GPU_pressure_correction_work_matrix'
                and result['protocol_sha256'] == sha(protocol_path)
                and result['device_calls']['forbidden'] == 0
                and result['emitted_pressure_fields'] == 128 * len(protocol['policies']),
                'Model execution is incomplete or unqualified')
        reference_path = ROOT / specification['reference']['path']
        require(sha(reference_path) == specification['reference']['sha256'], 'Original output capture changed')
        original = read(reference_path)
        require(sha(ROOT / specification['weights_path']) == original['weights_sha256'], 'Checkpoint bytes changed')
        archive_path = reference_path.parent / original['output_archive']['path']
        require(sha(archive_path) == original['output_archive']['sha256'], 'Original output archive changed')
        with ExitStack() as stack:
            original_outputs = stack.enter_context(np.load(archive_path, allow_pickle=False))
            generated = None
            if result.get('new_output_archive'):
                generated_path = result_path.parent / result['new_output_archive']['path']
                require(sha(generated_path) == result['new_output_archive']['sha256'], 'New output archive changed')
                generated = stack.enter_context(np.load(generated_path, allow_pickle=False))
            rows = {}
            new_members = set()
            actual_default_parity = 0
            for index, ((identity, sample), frozen_case, reference_row) in enumerate(zip(cases, frozen['cases'], original['rows'], strict=True)):
                for offset, policy in enumerate(protocol['policies']):
                    row = result['rows'][index * len(protocol['policies']) + offset]
                    require(row['case_id'] == frozen_case['id'] and row['policy'] == policy
                            and all(row[key] == value for key,value in identity.items()), 'Policy output is bound to another case')
                    for name in ('base', 'native_input_residual', 'cell_size_m', 'rho', 'dt'):
                        require(array_sha(sample[name].numpy()) == row['input_sha256'][name]
                                == reference_row['input_sha256'][name], 'Numerical input identity changed')
                    for field, key in (('reference','reference_pressure_sha256'),('rhs','reference_rhs_sha256')):
                        require(array_sha(sample[field].numpy()) == row[key] == reference_row[key], 'Numerical reference identity changed')
                    location = row['output_location']
                    if 'new_archive' in location:
                        require(generated is not None and location['new_archive'] == result['new_output_archive']['path'],
                                'Unknown new output archive')
                        pressure = generated[location['member']]
                        new_members.add(location['member'])
                    else:
                        require(location['original_capture'] == specification['reference']
                                and location['original_archive'] == original['output_archive']
                                and location['original_member'] == f'case{index:03d}_pressure', 'Wrong original output location')
                        pressure = original_outputs[location['original_member']]
                    require(pressure.dtype.str == '<f8' and pressure.shape == tuple(sample['reference'].shape)
                            and bool(np.isfinite(pressure).all()) and array_sha(pressure) == row['output_sha256'],
                            'Actual pressure output bytes/dtype/shape differ')
                    parity = row['output_sha256'] == reference_row['output_sha256']
                    require(parity == row['matches_original_reference_output'], 'Recorded source parity differs')
                    if policy == 'captured_default':
                        actual_default_parity += int(parity)
                    requested = result['configured_correction']['steps'] if policy == 'captured_default' else int(policy.split('_')[1])
                    require(row['actual_network_root_calls'] == requested, 'Correction work is not the stated budget')
                    metrics = score(np, pressure, sample)
                    for key,value in metrics.items():
                        require(math.isfinite(value) and math.isclose(value, row['metrics'][key], rel_tol=1e-12, abs_tol=1e-30),
                                'Independent NumPy score differs: ' + identifier + '/' + policy + '/' + key)
                        maximum_relative_metric_difference = max(maximum_relative_metric_difference,
                            abs(value-row['metrics'][key]) / max(abs(value),abs(row['metrics'][key]),1e-30))
                    rows[(policy,index)] = dict(metrics=metrics, root_calls=requested,
                        counts_xyz=identity['counts_xyz'], cells_per_metre=1.0/identity['cell_size_m'])
                    total_fields += 1
            require(actual_default_parity == result['original_default_output_parity_cases']
                    and (not specification['require_reference_parity'] or actual_default_parity == 128),
                    'Default policy source parity is unqualified')
            require((generated is None and not new_members) or
                    (generated is not None and new_members == set(generated.files)), 'Unexpected or unused generated output fields')
        summaries = {}
        for policy in protocol['policies']:
            own = [rows[(policy,index)] for index in range(128)]
            worst = {key:max(row['metrics'][key] for row in own) for key in own[0]['metrics']}
            require(all(math.isclose(value,result['summary'][policy]['worst'][key],rel_tol=1e-12,abs_tol=1e-30)
                        for key,value in worst.items()), 'Stored policy maximum differs')
            summaries[policy] = dict(cases=128, root_calls=sorted({row['root_calls'] for row in own}), worst=worst,
                by_cells_per_metre={str(cpm):dict(cases=sum(row['cells_per_metre']==cpm for row in own),
                    worst={key:max(row['metrics'][key] for row in own if row['cells_per_metre']==cpm)
                        for key in worst}) for cpm in sorted({row['cells_per_metre'] for row in own})})
        models[identifier] = dict(result=captured, original_default_output_parity_cases=actual_default_parity,
            reference_role=specification['reference_role'], summary=summaries,
            new_archive=result.get('new_output_archive'))
        policy_rows[identifier] = rows
        print(identifier, '640retained outputs independently rescored',flush=True)
    comparisons = {}
    for identifier in models:
        if '-' not in identifier:
            continue
        parent = identifier.split('-')[0]
        comparison = {}
        for policy in protocol['policies']:
            fine = models[identifier]['summary'][policy]['worst']
            original = models[parent]['summary'][policy]['worst']
            comparison[policy] = dict(ratio_of_worst_to_parent={key:fine[key]/original[key] for key in fine},
                cases_with_lower_equation_error=sum(policy_rows[identifier][(policy,i)]['metrics']['common_equation_relative_l2']
                    < policy_rows[parent][(policy,i)]['metrics']['common_equation_relative_l2'] for i in range(128)),
                cases_with_lower_pressure_error=sum(policy_rows[identifier][(policy,i)]['metrics']['pressure_mean_free_relative_l2']
                    < policy_rows[parent][(policy,i)]['metrics']['pressure_mean_free_relative_l2'] for i in range(128)))
        comparisons[identifier] = comparison
    report = dict(status='all_7680_retained_pressure_policy_fields_independently_rescored',
        created_unix=time.time(), models=models, matched_policy_parent_comparisons=comparisons,
        protocol=dict(path=str(protocol_path.relative_to(ROOT)),sha256=sha(protocol_path)),
        producer=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=sha(Path(__file__))),
        emitted_fields=total_fields, metric_relative_tolerance=1e-12, metric_absolute_tolerance=1e-30,
        maximum_observed_relative_metric_difference=maximum_relative_metric_difference,
        independent_operator='NumPy negative finite-volume Laplacian with zero exterior flux',
        new_model_forwards=0, new_reference_generation=False, latency_claim=False,
        physical_simulation_steps_advanced=0, full_portfolio_complete=False)
    require(total_fields == 7680 and len(models) == 12, 'Full admitted matrix required')
    save(report_path,report)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    run(args.protocol.resolve(),args.report.resolve())
