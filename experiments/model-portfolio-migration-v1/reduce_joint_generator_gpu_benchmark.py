"""Independently reduce all original paired joint-generator GPU observations."""
from collections import defaultdict
import math
from pathlib import Path
import statistics
import time

from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROTOCOL = RUN/'joint-generator-GPU-benchmark-protocol-01.json'
BASE = RUN/'joint-generator-GPU-benchmark-01'
OUTPUT = DOC/'reports/joint-generator-GPU-benchmark-01.json'


def quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered)-1)*fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)


def summarize(cases):
    names = sorted({key for case in cases for row in case['metrics'] for key in row['values']})
    worst = {}
    undefined = {}
    witnesses = {}
    for name in names:
        values = [(row['values'][name],case['identifier'],case['start'],row['lead']) for case in cases for row in case['metrics']]
        undefined[name] = sum(value is None for value,*_ in values)
        defined = [value for value in values if value[0] is not None]
        largest = max(defined) if defined else None
        worst[name] = largest[0] if largest else None
        witnesses[name] = dict(identifier=largest[1],start=largest[2],lead=largest[3]) if largest else None
    return dict(cases=len(cases),physical_outputs=sum(case['K'] for case in cases),
        median_case_median_block_ms=statistics.median(statistics.median(c['block_ms']) for c in cases),
        median_case_median_per_output_ms=statistics.median(statistics.median(c['per_output_ms']) for c in cases),
        worst_case_median_per_output_ms=max(statistics.median(c['per_output_ms']) for c in cases),
        worst_case_p95_per_output_ms=max(quantile(c['per_output_ms'],.95) for c in cases),
        worst_metrics=worst,undefined_metrics=undefined,worst_metric_cases=witnesses)


def run():
    require(not OUTPUT.exists(), 'Preserve original complete reduction')
    protocol = read(PROTOCOL)
    protocol_binding = binding(PROTOCOL)
    owner = read(BASE/'status.json')
    require(owner['status'] == 'all8_original_workers_complete_reduction_pending'
            and len(owner['completed']) == 8 and owner['active'] is None
            and owner['protocol_sha256'] == protocol_binding['sha256'], 'All8original workers must finish')
    for spec in protocol['dependencies']:
        require(binding(ROOT/spec['path']) == spec, 'Admitted source/reference/weight input changed')
    require(binding(ROOT/protocol['cases']['path']) == protocol['cases'], 'Admitted case selection changed')
    prepared = read(ROOT/protocol['cases']['path'])
    expected_cases = {(c['identifier'],c['start']):c for c in prepared['cases']}
    models = {m['identifier']:m for m in prepared['models']}
    reports = defaultdict(dict)
    raw_bindings = []
    competing, module_calls, network_calls, timed_calls, timed_outputs, all_outputs = [], 0, 0, 0, 0, 0
    for completed in owner['completed']:
        path = ROOT/completed['result']
        require(binding(path) == completed['result_binding'], 'Original raw worker result changed')
        d = read(path)
        name, phase = d['identifier'], d['phase']
        model = models[name]
        require(d['status'] == 'all64_native_cases_measured' and d['fit_item_id'] == model['fit_item_id']
                and d['K'] == model['K'] and d['protocol_sha256'] == protocol_binding['sha256']
                and d['CPU_neural_forwards'] == d['forbidden_forward_attempts'] == 0
                and d['CUDA_network_calls'] == 64*(1+protocol['warmups']+protocol['samples'])
                and len(d['rows']) == 64 and phase in (0,1) and phase not in reports[name],
                'Original membership/device/work differs')
        observed = set()
        for row in d['rows']:
            key = (row['identifier'],row['start'])
            case = expected_cases[key]
            require(key not in observed and row['counts_xyz'] == case['scene']['counts']
                    and row['cells_per_metre'] == case['scene']['cells_per_metre']
                    and row['K'] == model['K'] and row['output_dt_s'] == 1/60
                    and row['block_advance_s'] == row['K']/60 and row['all_repeat_and_ownership_checks_passed']
                    and len(row['block_ms']) == len(row['per_output_ms']) == 10
                    and len(row['warmup_block_ms']) == 3 and len(row['metrics']) == len(row['output_hashes']) == row['K'],
                    'A physical case/output/warmup/sample is missing or mismatched')
            observed.add(key)
            for block, per_output in zip(row['block_ms'],row['per_output_ms'],strict=True):
                require(math.isfinite(block) and block > 0 and per_output == block/row['K'], 'Only Kphysical outputs may divide blocktime')
            for lead, metric in enumerate(row['metrics'],1):
                require(metric['lead'] == lead and metric['time_s'] == case['reference_leads'][lead-1]['time_s'], 'Target lead/timestamp differs')
                require(all(value is None or (math.isfinite(value) and value >= 0) for value in metric['values'].values()),
                        'Invalid scientific scalar')
                require(set(row['output_hashes'][lead-1]) == {'pressure','vx','vy','vz','tracer'}, 'A generated field is missing')
            timed_calls += len(row['block_ms'])
            timed_outputs += len(row['block_ms'])*row['K']
            all_outputs += (1+len(row['warmup_block_ms'])+len(row['block_ms']))*row['K']
        require(observed == expected_cases.keys(), 'Raw worker omitted an original case')
        for where in ('resources_before','resources_after'):
            if d[where]['other_compute_pids']:
                competing.append(dict(identifier=name,phase=phase,where=where,pids=d[where]['other_compute_pids']))
        network_calls += d['CUDA_network_calls']
        module_calls += d['CUDA_module_calls']
        reports[name][phase] = d
        raw_bindings.append(binding(path))
    require(timed_calls == protocol['expected_timed_network_calls'] == 5120
            and timed_outputs == protocol['expected_timed_physical_outputs'] == 12800
            and network_calls == protocol['expected_all_network_calls_including_first_and_warmups'] == 7168
            and all_outputs == protocol['expected_all_physical_outputs_including_first_and_warmups'] == 17920,
            'Complete aggregate work differs')
    summaries, max_phase_metric_difference = {}, 0.
    for name, pair in reports.items():
        require(set(pair) == {0,1}, 'Both independent process/order phases required')
        a = {(r['identifier'],r['start']):r for r in pair[0]['rows']}
        b = {(r['identifier'],r['start']):r for r in pair[1]['rows']}
        pooled = []
        for key, first in a.items():
            second = b[key]
            require(first['output_hashes'] == second['output_hashes'], 'Independent phases changed physical output bytes')
            for left,right in zip(first['metrics'],second['metrics'],strict=True):
                require(left['lead'] == right['lead'] and left['time_s'] == right['time_s'], 'Paired lead differs')
                require(left['values'].keys() == right['values'].keys(), 'A metric disappeared between phases')
                for metric,value in left['values'].items():
                    other = right['values'][metric]
                    require((value is None and other is None) or (value is not None and other is not None
                            and math.isclose(value,other,rel_tol=1e-12,abs_tol=1e-14)), 'Paired metric differs beyond declared numerical tolerance')
                    if value is not None:
                        max_phase_metric_difference = max(max_phase_metric_difference,abs(value-other))
            pooled.append(dict(first,block_ms=first['block_ms']+second['block_ms'],
                               per_output_ms=first['per_output_ms']+second['per_output_ms']))
        def grouped(key):
            groups = defaultdict(list)
            for row in pooled:
                groups[key(row)].append(row)
            return {label:summarize(rows) for label,rows in groups.items()}
        summaries[name] = dict(fit_item_id=models[name]['fit_item_id'],K=models[name]['K'],
            summary=summarize(pooled),by_start=grouped(lambda r:str(r['start'])),
            by_grid_density=grouped(lambda r:str(r['cells_per_metre'])),
            by_grid_dimensions=grouped(lambda r:'x'.join(map(str,r['counts_xyz']))),
            by_density_and_dimensions=grouped(lambda r:str(r['cells_per_metre'])+'cpm-'+ 'x'.join(map(str,r['counts_xyz']))),
            peak_CUDA_allocated_bytes=max(d['peak_CUDA_allocated_bytes'] for d in pair.values()),
            cases=pooled)
    probe_path = DOC/'reports/joint-generator-retained-probe-check-01.json'
    probe = read(probe_path)
    require(probe['status'] == 'four_original_saved_probe_field_metrics_reproduced_by_GPU_boundary', 'Original saved-probe boundary check required')
    for spec in probe['evidence']:
        require(binding(ROOT/spec['path']) == spec, 'Saved-probe evidence changed')
    result = dict(status='all4_joint_generators_8blocks_independently_reduced',created_unix=time.time(),models=summaries,
        cases=64,models_measured=4,timed_joint_calls=timed_calls,timed_physical_outputs=timed_outputs,
        verified_network_calls_including_first_and_warmups=network_calls,verified_physical_outputs_including_first_and_warmups=all_outputs,
        actual_CUDA_module_calls=module_calls,CPU_neural_forwards=0,all_phase_output_hashes_equal=True,
        max_phase_metric_difference=max_phase_metric_difference,raw_samples_per_model_case=20,
        timing_units='GPU wall ms/physical_output; fullKblock ms retained; complete native CPUinput/output boundary',
        timing_qualified_at_observed_resource_scope=not competing,observed_competing_GPU_compute=competing,
        continuous_hardware_exclusivity_proven=False,resource_scope='No competing compute at every worker start; explicit before/after snapshots retained. Unobserved activity/ordinary system jitter not excluded.',
        reference_method=prepared['reference_method'],reference_native_probe_binary_sha256=prepared['reference_native_probe_binary_sha256'],
        reference_equation_relative_l2_max=prepared['reference_equation_relative_l2_max'],
        original_saved_CPU_probe_metrics_agree_on_first_case=True,new_CPU_probe_forward_executed=False,
        exposure='All16original fitting scenes; consumed diagnostics, not independent holdout.',
        interpretation='Joint heads amortize complete-pass cost, but these original F32models have large pressure/velocity tails, particularly atstart0. The errors greatly exceed the recorded label-quantization error. This is not a benchmark of all35full generators, all205fits, or the quoted F64anchors on these same cases.',
        native_publication_complete=False,full_portfolio_complete=False,new_reference_generation=False,array_copies_to_disk=0,
        producer=binding(Path(__file__)),evidence=[protocol_binding,binding(ROOT/protocol['cases']['path']),binding(BASE/'status.json'),binding(probe_path),*raw_bindings])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    for name,row in summaries.items():
        s = row['summary']
        print(name,'K',row['K'],'GPUms/output',s['median_case_median_per_output_ms'],
              'pressure worst',s['worst_metrics']['pressure_relative_l2'])


if __name__ == '__main__':
    run()
