"""Join two original future-q heads without treating requests as simulation steps."""
from collections import Counter
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, PROJECT, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output = DOC/'reports/common-fit-benchmark-coverage-09.json'
    require(not output.exists(), 'Preserve the original causal-request coverage decision')
    publication, _ = current_publication(ROOT)
    previous_path = DOC/'reports/common-fit-benchmark-coverage-08.json'
    result_path = DOC/'reports/causal-request-component-GPU-benchmark-01.json'
    objective_path = DOC/'reports/causal-q-original-objective-GPU-01.json'
    probes_path = DOC/'reports/causal-q-retained-CPU-probe-comparison-01.json'
    previous, measured, objective, probes = map(read, [previous_path, result_path, objective_path, probes_path])
    expected = {'fit-a9e199535346c3d9', 'fit-ccb969376d283836'}
    require(measured['status'] == 'two_original_causal_q_GPU_benchmarks_independently_reduced'
            and measured['cases_per_fit'] == 2048 and measured['fresh_processes'] == 4
            and measured['primary_calls'] == 81920 and measured['logical_calls'] == 196608
            and measured['CPU_neural_forwards'] == measured['physical_simulation_steps'] == 0
            and measured['timing_units'] == 'ms/normalized_future_q_request'
            and measured['all_cross_fit_input_target_and_membership_bindings_exact']
            and measured['all_process_output_hashes_exact'], 'Complete matched original GPU request benchmarks required')
    require(objective['status'] == 'original_q_training_objective_GPU_evaluated_on_all_saved_outputs'
            and objective['original_loss_evaluations'] == 4096
            and objective['CPU_neural_forwards'] == objective['new_learned_GPU_forwards'] == 0
            and objective['benchmark'] == binding(result_path)
            and not objective['inference_timing_includes_this_diagnostic']
            and probes['status'] == 'both_original_saved_CPU_q_probes_compared_without_model_execution'
            and probes['benchmark'] == binding(result_path) and probes['fields_per_fit'] == 12
            and probes['CPU_neural_forwards'] == 0, 'Separate original-objective and retained-probe diagnostics required')
    specs = [measured[key] for key in ('protocol', 'owner', 'producer', 'material', 'contract', 'sampling_membership')]
    specs += [objective['producer'], objective['original_objective_source'], probes['producer']]
    for model in measured['models']:
        specs.extend([model['prediction_archive'], model['weights'], *model['raw_results']])
    specs.extend(model['original_saved_CPU_probe'] for model in probes['models'])
    for spec in specs:
        observed_file(ROOT, spec['path'], spec)
    index_path = PROJECT/'notes/portfolio-index.json'
    index = read(index_path)
    require(binding(index_path) == previous['source_fit_index'], 'Accepted native fit index changed')
    native = {row['item_id']: row for row in index['fits']}
    rows = copy.deepcopy(previous['rows'])
    fits = {row['fit_item_id']: row for row in rows}
    models = {row['fit_item_id']: row for row in measured['models']}
    objectives = {row['fit_item_id']: row for row in objective['models']}
    probe_models = {row['fit_item_id']: row for row in probes['models']}
    material = {row['fit_item_id']: row for row in read(ROOT/measured['material']['path'])['models']}
    require(len(rows) == len(fits) == len(native) == 205 and set(fits) == set(native)
            and len(measured['models']) == len(objective['models']) == len(probes['models']) == 2
            and set(models) == set(objectives) == set(probe_models) == expected, 'Exact original fit identities required')
    for fid in sorted(expected):
        row, entry, model = fits[fid], material[fid], models[fid]
        require(row['task'] == entry['task'] == native[fid]['task'] == 'pressure_request_component'
                and row['status'] == 'common_GPU_benchmark_pending' and row['default_benchmark'] is None
                and model['weights'] == entry['outputs']['weights.npz']
                and native[fid]['receipt_sha256'] == entry['original_receipt']['sha256']
                and any(item['sha256'] == model['weights']['sha256'] for item in native[fid]['checkpoints']),
                'Pending original request fit, receipt and selected published tensors must join')
        groups = model['groups']
        require(groups['overall']['cases'] == 2048
                and groups['by_main_q_fit_seen']['sampled']['cases'] == 1834
                and groups['by_main_q_fit_seen']['not_sampled']['cases'] == 214,
                'Actual main-loss sampling membership must remain explicit')
        row.update(status='default_common_GPU_accuracy_and_timing_available',
            default_benchmark=dict(group='causal_future_q_component', identifier=fid, cases=2048,
                source='original_published_F32_normalized_future_q_head_CUDA_restore',
                benchmark_authority=binding(result_path), accuracy_authority=binding(result_path),
                timing_units='GPU ms/normalized future-q request; not pressure or simulation-step latency',
                case_median_median_ms=groups['overall']['median_case_GPU_ms'], population=measured['scope'],
                native_publication_pending=True, physical_simulation_steps=0, independent_holdout=False),
            accuracy_benchmark=dict(independent_reduction=binding(result_path), latency_qualified=True,
                metric='Relative L2 and absolute normalized/physical q errors against original F32 targets',
                worst=groups['overall'], main_loss_membership=groups['by_main_q_fit_seen'],
                precision_limit=measured['precision_limit'], original_training_objective=binding(objective_path),
                original_training_objective_summary={k: v for k, v in objectives[fid].items() if k != 'rows'},
                saved_CPU_probe_comparison=binding(probes_path),
                maximum_saved_CPU_probe_relative_difference=probe_models[fid]['maximum_GPU_minus_saved_CPU_relative_l2']))
    pending = [row for row in rows if row['default_benchmark'] is None]
    require(len(pending) == 10 and sum(row['default_benchmark'] is not None for row in rows) == 195,
            'Only the two original request fits may advance beyond 193')
    report = dict(previous, status='all205_fit_identities_joined_to195_role_scoped_GPU_benchmarks_and10_pending',
        created_unix=time.time(), previous_coverage=binding(previous_path), source_fit_index=binding(index_path),
        current_publication_acceptance=binding(publication), new_component_result=binding(result_path), rows=rows,
        default_common_benchmark_fits=195, common_GPU_accuracy_fits=195, causal_future_q_component_fits=2,
        common_benchmark_pending_fits=10, common_accuracy_pending_fits=10,
        pending_fit_ids=[row['fit_item_id'] for row in pending],
        pending_by_task=dict(sorted(Counter(row['task'] for row in pending).items())),
        scope=previous['scope']+' Two causal future-q heads add matched 2048 same-parent transitions each, '
              '1834 actually sampled/214 not sampled in main q fitting; no independent holdout or full-state claim.',
        evidence=[*previous['evidence'], binding(result_path), binding(objective_path), binding(probes_path)],
        producer=binding(Path(__file__)), new_model_forwards=0, native_mutations=0,
        weight_or_dataset_copies=0, full_portfolio_complete=False)
    save(output, report)
    print('195/205 original default fits benchmarked; 10 components remain.', flush=True)


if __name__ == '__main__':
    run()
