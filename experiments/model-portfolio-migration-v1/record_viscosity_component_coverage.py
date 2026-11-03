"""Join the qualified viscosity result to its original portfolio fit identity."""
from collections import Counter
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, PROJECT, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output = DOC/'reports/common-fit-benchmark-coverage-04.json'
    require(not output.exists(), 'Preserve the original component coverage decision')
    publication, _ = current_publication(ROOT)
    previous_path = DOC/'reports/common-fit-benchmark-coverage-03.json'
    previous = read(previous_path)
    result_path = DOC/'reports/viscosity-component-GPU-benchmark-01.json'
    measured = read(result_path)
    fid = 'fit-d1c3dbdd9793117a'
    require(measured['status'] == 'original_viscosity_80pattern_GPU_benchmark_independently_reduced'
            and measured['fit_item_id'] == fid and measured['cases'] == 80
            and measured['fresh_processes'] == 2 and measured['primary_calls'] == 1600
            and measured['CPU_model_forwards'] == 0 and measured['all_original_tensors_and_architecture_unchanged']
            and measured['timing_units'] == 'ms/local_viscosity_request'
            and measured['physical_simulation_steps'] == 0, 'Qualified original component result required')
    for spec in [measured['protocol'], measured['owner'], measured['material'], measured['producer'],
                 measured['prediction_archive'], measured['original_model_source'], measured['original_weights'],
                 *measured['raw_results']]:
        observed_file(ROOT, spec['path'], spec)
    material = read(ROOT/measured['material']['path'])
    selected = next(row for row in material['models'] if row['fit_item_id'] == fid)
    index_path = PROJECT/'notes/portfolio-index.json'
    index = read(index_path)
    original_fits = {row['item_id']: row for row in index['fits']}
    rows = copy.deepcopy(previous['rows'])
    require(len(rows) == len(original_fits) == 205
            and {row['fit_item_id'] for row in rows} == set(original_fits), 'Exact original205fit roster required')
    for row in rows:
        require(row['task'] == original_fits[row['fit_item_id']]['task'], 'Original task identity changed')
    row = next(row for row in rows if row['fit_item_id'] == fid)
    require(row['task'] == selected['task'] == 'viscosity_component'
            and row['default_benchmark'] is None and row['status'] == 'common_GPU_benchmark_pending'
            and measured['original_weights'] == selected['outputs']['weights.npz']
            and original_fits[fid]['receipt_sha256'] == selected['original_receipt']['sha256']
            and any(checkpoint['sha256'] == measured['original_weights']['sha256']
                    for checkpoint in original_fits[fid]['checkpoints'])
            and selected['original_receipt']['sha256'][:16] == fid.removeprefix('fit-'),
            'Result must join the pending original fit and selected weights')
    row.update(status='default_common_GPU_accuracy_and_timing_available',
        default_benchmark=dict(group='viscosity_component', identifier=fid, cases=80,
            source='original_training_output_CUDA_restore', benchmark_authority=binding(result_path),
            accuracy_authority=binding(result_path), timing_units='GPU ms/local viscosity request; no simulation advancement',
            case_median_median_ms=measured['groups']['all']['median_case_GPU_ms'],
            population=measured['population'], native_publication_pending=True),
        accuracy_benchmark=dict(independent_reduction=binding(result_path), latency_qualified=True,
            metric='masked dimensionless stiffness relativeL2; not pressure or trajectory error',
            worst=measured['groups']['all']['worst_masked_relative_l2_nonzero_targets']))
    pending = [row for row in rows if row['default_benchmark'] is None]
    require(len(pending) == 19 and sum(row['default_benchmark'] is not None for row in rows) == 186,
            'Exactly one new original component fit may advance')
    report = dict(previous, status='all205_fit_identities_joined_to186_role_scoped_GPU_benchmarks_and19_pending',
        created_unix=time.time(), previous_coverage=binding(previous_path), source_fit_index=binding(index_path),
        current_publication_acceptance=binding(publication), new_component_result=binding(result_path), rows=rows,
        default_common_benchmark_fits=186, common_GPU_accuracy_fits=186, viscosity_component_fits=1,
        common_benchmark_pending_fits=19, common_accuracy_pending_fits=19,
        pending_fit_ids=[row['fit_item_id'] for row in pending],
        pending_by_task=dict(sorted(Counter(row['task'] for row in pending).items())),
        scope='150pressure fits/128requests,35full generators/64cases,one viscosity component/80patterns. '
              'Role-specific populations and units remain distinct; not524states,intended variants,holdout,drift or clean retraining.',
        evidence=[*previous['evidence'], binding(result_path)], producer=binding(Path(__file__)),
        new_model_forwards=0, native_mutations=0, weight_or_dataset_copies=0, full_portfolio_complete=False)
    save(output, report)
    print('186/205original default fits benchmarked on role-specific populations;19components remain.', flush=True)


if __name__ == '__main__':
    run()
