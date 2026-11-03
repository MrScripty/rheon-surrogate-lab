"""Independently join balanced GPU pressure timings to frozen common accuracy.

This reads retained JSON/source/weight bytes; it performs no inference. It admits
only the reconciled 98-block prefix plus the explicit 156-block continuation,
never the excluded contended C102 block. Pressure requests are not full steps.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1/reports'
IDENTITY = ('trajectory_id', 'parent', 'native_frame', 'counts_xyz', 'cell_size_m')
METRICS = ('common_equation_relative_l2', 'pressure_mean_free_relative_l2',
           'pressure_absolute_rms', 'pressure_absolute_max')
SETTINGS = ('gpu_model', 'torch_version', 'cuda_version', 'host_affinity',
            'torch_threads', 'interop_threads', 'tf32', 'network_dtype',
            'pressure_dtype', 'autocast', 'deterministic_algorithms', 'cudnn_benchmark')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def summarize(samples):
    require(samples and all(type(v) is int and v > 0 for v in samples),
            'Positive integer wall nanoseconds required')
    values = sorted(samples)
    rank = .95 * (len(values) - 1)
    lower, upper = math.floor(rank), math.ceil(rank)
    return dict(samples=len(values), minimum_ms=values[0] / 1e6,
                median_ms=statistics.median(values) / 1e6,
                p95_ms=(values[lower] + (values[upper] - values[lower]) * (rank - lower)) / 1e6,
                maximum_ms=values[-1] / 1e6)


def verify_row(actual, case, reference, scored, samples_per_case):
    require(actual['case_id'] == case['id'] and
            all(actual[k] == reference[k] == scored[k] == case['original_case'][k] for k in IDENTITY),
            'Native request identity or domain differs')
    require(actual['cells_per_metre'] == case['cells_per_metre']
            and actual['input_dt_s'] == case['input_dt_s']
            and actual['input_sha256'] == reference['input_sha256'] == case['original_case']['input_sha256']
            and actual['output_sha256'] == reference['output_sha256'] == scored['output_sha256']
            and actual['output_independent_of_inputs'], 'Native values/output/density/cadence differs')
    require(reference['reference_pressure_sha256'] == case['original_case']['reference_pressure_sha256']
            and reference['reference_rhs_sha256'] == case['original_case']['reference_rhs_sha256'],
            'Frozen old-reference fields differ')
    require(set(scored['metrics']) == set(METRICS) and all(type(v) in (int, float)
            and math.isfinite(v) and v >= 0 for v in scored['metrics'].values()), 'Common score is invalid')
    samples = actual['raw_wall_nanoseconds']
    require(len(samples) == samples_per_case, 'A process did not retain exactly its declared samples')
    summary = summarize(samples)
    require(all(actual[k] == summary[k] for k in ('minimum_ms', 'median_ms', 'maximum_ms')),
            'Worker statistic differs from raw samples')
    return samples


def aggregate(rows):
    return dict(cases=len(rows), median_case_median_ms=statistics.median(r['median_ms'] for r in rows),
                worst_case_median_ms=max(r['median_ms'] for r in rows),
                worst_case_p95_ms=max(r['p95_ms'] for r in rows),
                maximum_sample_ms=max(r['maximum_ms'] for r in rows),
                worst_accuracy={metric: max(r['accuracy'][metric] for r in rows) for metric in METRICS})


def frontier(models, metric):
    """Model-level worst-error versus worst case median pressure latency."""
    points = {name: (m['summary']['worst_case_median_ms'], m['summary']['worst_accuracy'][metric])
              for name, m in models.items()}
    return [name for name, point in points.items() if not any(
        other[0] <= point[0] and other[1] <= point[1] and other != point for other in points.values())]


def reduce(protocol_path, output):
    require(not output.exists() and output.is_relative_to(DOC), 'Fresh task-owned report required')
    protocol = read(protocol_path)
    state_path = ROOT / protocol['output_root'] / 'status.json'
    state = read(state_path)
    require(state['status'] == 'all254_pressure_timing_blocks_collected_independent_reduction_pending'
            and len(state['completed']) == 254 and state['active'] is None
            and not Path('/proc', str(state['pid'])).exists(), 'Original continuation must be terminal and complete')
    require(state['protocol']['sha256'] == sha(protocol_path), 'Continuation protocol changed')
    original_path = ROOT / protocol['original_protocol']['path']
    require(sha(original_path) == protocol['original_protocol']['sha256'], 'Original protocol changed')
    original = read(original_path)
    activity_path = DOC / 'pressure-timing-host-activity-01.json'
    activity = read(activity_path)
    require(activity['status'] == 'known_host_metadata_activity_conservatively_disclosed',
            'Known host activity requires explicit disclosure')
    activity_models = set(activity['models'])
    require([entry['block'] for entry in activity['potentially_overlapping_original_blocks']] == list(range(196,206))
            and activity_models == {entry['identifier'] for entry in activity['potentially_overlapping_original_blocks']}
            and all({k:entry[k] for k in ('identifier','phase')} == original['block_order'][entry['block']-1]
                    for entry in activity['potentially_overlapping_original_blocks']),
            'Known host-activity envelope/model attribution differs from original process order')
    reconciliation_path = DOC / 'pressure-timing-contention-reconciliation-01.json'
    reconciliation = read(reconciliation_path)
    require(reconciliation['status'] == '98_original_exact_blocks_retained_C102_contention_excluded'
            and reconciliation['continuation_protocol']['sha256'] == sha(protocol_path)
            and state['completed'][:98] == protocol['accepted_prefix'], 'Original prefix/exclusion lacks reconciliation')
    require(sha(ROOT / protocol['original_status']['path']) == protocol['original_status']['sha256']
            and sha(ROOT / protocol['excluded_contended_block']['path'])
            == protocol['excluded_contended_block']['sha256'], 'Original terminal/excluded evidence changed')
    require(protocol['models'] == [f'C{i:02}' for i in range(3, 131) if i != 68]
            and protocol['samples_per_case'] == 10 and protocol['pooled_samples_per_case'] == 20
            and protocol['cases_per_model'] == 128 and protocol['expected_timed_calls'] == 325120,
            'Complete registered population/sample contract required')
    for name, expected in protocol['source_sha256'].items():
        require(sha(ROOT / name) == expected, 'Preregistered numerical source differs: ' + name)
    material = read(ROOT / protocol['material'])
    entries = {entry['id']: entry for entry in material['models']}
    bindings = {}
    for entry in entries.values():
        for name, digest in {**entry['identities'], **entry['source_bindings']}.items():
            require(name not in bindings or bindings[name] == digest, 'Conflicting original material bindings')
            bindings[name] = digest
    require(len(bindings) == protocol['material_bindings'] and set(entries) == set(protocol['models']),
            'Original model/source membership differs')
    for name, digest in bindings.items():
        require(sha(ROOT / name) == digest, 'Captured source/parameter bytes differ: ' + name)
    frozen_path = ROOT / protocol['frozen_cases']['path']
    require(sha(frozen_path) == protocol['frozen_cases']['sha256'], 'Frozen native cases differ')
    cases = read(frozen_path)['cases']
    require(len(cases) == len({c['id'] for c in cases}) == 128, 'All unique native pressure requests required')
    score = read(ROOT / protocol['score'])
    detail_path = ROOT / score['details']['path']
    require(sha(detail_path) == score['details']['sha256'] and score['independent_pressure_scoring_exact'],
            'Independent common-pressure scoring changed')
    details = read(detail_path)['records']
    captures = {}
    for name in protocol['models']:
        spec = protocol['accuracy'][name]
        require(spec == score['models'][name]['result'] and sha(ROOT/spec['path']) == spec['sha256'],
                'Another accuracy capture selected')
        captures[name] = read(ROOT/spec['path'])
        require(captures[name]['weights_sha256'] == entries[name]['weights_sha256']
                == score['models'][name]['weights_sha256'], 'Another checkpoint selected')
    collected = {name: [] for name in protocol['models']}
    hardware = settings = None
    timed_calls = 0
    for number, (entry, expected) in enumerate(zip(state['completed'], original['block_order'], strict=True), 1):
        name, phase = expected['identifier'], expected['phase']
        path = ROOT / entry['result']
        require(entry['block'] == number and entry['identifier'] == name and entry['phase'] == phase
                and sha(path) == entry['sha256'] and entry['cases'] == 128,
                'Original process order/receipt differs')
        require(path != ROOT / protocol['excluded_contended_block']['path'], 'Contended original C102 cannot qualify')
        raw = read(path)
        expected_protocol = sha(original_path) if number <= 98 else sha(protocol_path)
        require(raw['status'] == 'complete_pressure_GPU_timing_pilot' and raw['identifier'] == name
                and raw['protocol_sha256'] == expected_protocol and raw['cases'] == 128
                and raw['samples_per_case'] == 10 and raw['warmups_per_case'] == 5
                and raw['task_calls_timed'] == 1280 and raw['learned_parameters_unchanged']
                and raw['architecture_unchanged'] and raw['all_output_hashes_exact']
                and not raw['device_qualification']['forbidden_module_calls']
                and raw['device_qualification']['cuda_module_calls'] > 0
                and not raw['CPU_model_forward'] and not raw['timed_module_hooks_enabled']
                and not raw['full_simulation_step_latency'] and not raw['new_optimizer_updates']
                and raw['inference_device'] == 'cuda' and raw['timing_units'] == 'ms/pressure_request'
                and raw['timing_scope'] == protocol['timing_scope']
                and raw['weights_sha256'] == captures[name]['weights_sha256']
                and raw['source_entry'] == entries[name], 'Complete exact pressure worker qualification differs')
        observed = {k: raw[k] for k in SETTINGS}
        require(settings is None or observed == settings, 'GPU runtime/precision/thread settings differ')
        settings = observed
        require(raw['host_affinity'] == [4] and raw['torch_threads'] == raw['interop_threads'] == 1
                and not any(raw['tf32'].values()) and raw['deterministic_algorithms']
                and not raw['cudnn_benchmark'] and not raw['autocast']
                and raw['network_dtype'] == 'float32' and raw['pressure_dtype'] == 'float64',
                'Registered numerical/resource policy differs')
        require(raw['resource_observations'], 'Missing sampled resource observations')
        for observation in raw['resource_observations']:
            require(not observation['other_compute_pids'], 'Unrelated GPU compute invalidates latency')
            devices = list(csv.reader(observation['gpu'].splitlines(), skipinitialspace=True))
            require(len(devices) == 1 and len(devices[0]) >= 4, 'Single GPU identity required')
            identity = devices[0][:4]
            require(hardware is None or hardware == identity, 'GPU or driver changed')
            hardware = identity
        require(len(raw['rows']) == len(captures[name]['rows']) == len(details[name]['rows']) == 128,
                'Complete literal case coverage required')
        for actual, case, reference, scored in zip(raw['rows'], cases, captures[name]['rows'], details[name]['rows'], strict=True):
            timed_calls += len(verify_row(actual, case, reference, scored, 10))
        require(raw['case_median_median_ms'] == statistics.median(r['median_ms'] for r in raw['rows'])
                and raw['worst_case_median_ms'] == max(r['median_ms'] for r in raw['rows']),
                'Worker aggregate does not reduce from exact rows')
        collected[name].append(raw)
    models = {}
    for name, own in collected.items():
        require(len(own) == 2 and own[0]['pid'] != own[1]['pid'], 'Two distinct process passes/model required')
        rows = []
        for index, case in enumerate(cases):
            samples = [v for block in own for v in block['rows'][index]['raw_wall_nanoseconds']]
            require(len(samples) == 20, 'Exactly20pooled samples/native case required')
            rows.append(dict(case_id=case['id'], dimensions=case['dimensions'], raw_wall_nanoseconds=samples,
                **summarize(samples), input_sha256=case['original_case']['input_sha256'],
                output_sha256=captures[name]['rows'][index]['output_sha256'],
                accuracy=details[name]['rows'][index]['metrics']))
        summary = aggregate(rows)
        require(summary['worst_accuracy'] == score['models'][name]['worst'], 'Joined worst accuracy changed')
        density_groups = {}
        domain_groups = []
        for density in sorted({c['cells_per_metre'] for c in cases}):
            selected = [r for r in rows if r['dimensions']['cells_per_metre'] == density]
            density_groups[str(density)] = aggregate(selected)
            require(density_groups[str(density)]['worst_accuracy']
                    == score['models'][name]['by_cells_per_metre'][str(density)]['worst'], 'Density-scoped accuracy differs')
            shapes = sorted({tuple(r['dimensions'][f'grid_{axis}'] for axis in 'xyz') for r in selected})
            for shape in shapes:
                group = [r for r in selected if tuple(r['dimensions'][f'grid_{axis}'] for axis in 'xyz') == shape]
                domain_groups.append(dict(cells_per_metre=density, counts_xyz=list(shape), **aggregate(group)))
        models[name] = dict(weights_sha256=own[0]['weights_sha256'], execution_variant=own[0]['execution_variant'],
            host_metadata_overlap_in_one_pass=name in activity_models,
            rows=rows, summary=summary, by_cells_per_metre=density_groups, by_density_and_domain=domain_groups,
            process_case_median_medians_ms=[b['case_median_median_ms'] for b in own],
            raw_blocks=[e for e in state['completed'] if e['identifier'] == name])
    require(timed_calls == 325120, 'Complete timed-call denominator differs')
    ratios = [b['median_ms']/a['median_ms'] for a, b in zip(models['C106']['rows'], models['C107']['rows'], strict=True)]
    report = dict(schema='rheon.complete-pressure-portfolio-GPU-timing-reduction.v1',
        status='all127_named_pressure_models_two_pass_GPU_timings_and_exact_accuracy_join_verified',
        created_unix=time.time(), models=models, named_models=127, process_blocks=254,
        unique_pressure_cases=128, pooled_model_cases=127*128, timed_pressure_calls=timed_calls,
        units='ms/pressure_request', full_simulation_step_timing=False, K_division_applied=False,
        settings=settings, GPU_identity=hardware, timing_scope=protocol['timing_scope'],
        percentile_convention='Linear interpolation at rank .95*(N-1) over20positive integer wall-nanosecond samples.',
        warmups_setup_first_call_and_hashing_excluded=True, all_native_input_and_output_hashes_exact=True,
        every_common_accuracy_row_joined=True, independent_numerical_reexecution=False,
        source_bindings_verified=len(bindings), CPU_neural_forwards=0, new_parameter_fits=0,
        sampled_resources_not_continuous_exclusivity_proof=True,
        balanced_order_not_controlled_thermal_environment=True,
        excluded_contended_block=protocol['excluded_contended_block'],
        missing_named_checkpoint_dispositions=protocol['excluded_identifiers'],
        pressure_only_model_frontiers={m:frontier(models,m) for m in METRICS[:2]},
        known_host_activity_disclosure=activity,
        pressure_only_frontiers_excluding_known_host_activity={m:frontier(
            {name:value for name,value in models.items() if name not in activity_models},m) for m in METRICS[:2]},
        frontier_scope='Same128native pressure requests. Model-wide worst error versus worst case pooled median latency; these extrema can occur on different cases. Does not compare full-state generators.',
        paired_C107_over_C106_median_latency_ratio=dict(minimum=min(ratios),median=statistics.median(ratios),
            maximum=max(ratios),C107_faster_cases=sum(v<1 for v in ratios),paired_cases=128),
        full205_fit_524_state_portfolio_complete=False,
        evidence=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in
            (protocol_path, original_path, state_path, reconciliation_path, frozen_path, detail_path, ROOT/protocol['score'], activity_path)],
        producer=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=sha(Path(__file__))))
    output.write_text(json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n')
    print(report['status'], timed_calls, 'timed pressure requests', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    reduce(args.protocol.resolve(), args.report.resolve())
