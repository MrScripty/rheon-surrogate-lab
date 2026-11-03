"""Join the original recurrent sampler head and its full local training population."""
from collections import Counter
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, PROJECT, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output = DOC/'reports/common-fit-benchmark-coverage-08.json'
    require(not output.exists(), 'Preserve original sampler coverage decision')
    publication,_ = current_publication(ROOT)
    previous_path = DOC/'reports/common-fit-benchmark-coverage-07.json';previous = read(previous_path)
    result_path = DOC/'reports/sampler-component-GPU-benchmark-01.json';measured = read(result_path)
    fid = 'fit-11a2f63a94a9168e'
    require(measured['status'] == 'original_sampler_77998point_GPU_benchmark_independently_reduced'
            and measured['fit_item_id'] == fid and measured['cases'] == 576 and measured['points'] == 77998
            and measured['fresh_processes'] == 2 and measured['primary_calls'] == 11520
            and measured['CPU_neural_forwards'] == measured['physical_simulation_steps'] == 0
            and measured['timing_units'] == 'ms/local_sampler_group', 'Complete independently reduced original sampler required')
    for spec in [measured['protocol'],measured['owner'],measured['producer'],measured['material'],
                 measured['prediction_archive'],measured['original_weights'],*measured['raw_results']]:
        observed_file(ROOT,spec['path'],spec)
    material = read(ROOT/measured['material']['path']);entry = next(row for row in material['models'] if row['fit_item_id'] == fid)
    index_path = PROJECT/'notes/portfolio-index.json';index = read(index_path)
    require(binding(index_path) == previous['source_fit_index'], 'Accepted native fit index changed')
    native = {row['item_id']:row for row in index['fits']};rows = copy.deepcopy(previous['rows'])
    fits = {row['fit_item_id']:row for row in rows}
    require(len(rows) == len(fits) == len(native) == 205 and set(fits) == set(native), 'Exact205fit roster required')
    row = fits[fid]
    require(row['task'] == entry['task'] == native[fid]['task'] == 'recurrent_transport_component'
            and row['status'] == 'common_GPU_benchmark_pending' and row['default_benchmark'] is None
            and measured['original_weights'] == entry['outputs']['weights.npz']
            and native[fid]['receipt_sha256'] == entry['original_receipt']['sha256']
            and any(item['sha256'] == measured['original_weights']['sha256'] for item in native[fid]['checkpoints']),
            'Original pending recurrent fit/receipt/published weight identity required')
    summary = measured['groups']['overall']
    require(summary['groups'] == 576 and summary['points'] == 77998, 'All original local groups and points must be scored')
    row.update(status='default_common_GPU_accuracy_and_timing_available',
        default_benchmark=dict(group='recurrent_transport_component',identifier=fid,cases=576,points=77998,
            source='original_published_F64_eager_Torch_local_head_CUDA_restore',benchmark_authority=binding(result_path),
            accuracy_authority=binding(result_path),timing_units='GPU ms/local group of130-136points;not a fullgrid orsimulationstep',
            case_median_median_ms=summary['median_case_GPU_ms'],population=measured['scope'],native_publication_pending=True,
            optimized_full_generator_performance_claim=False),
        accuracy_benchmark=dict(independent_reduction=binding(result_path),latency_qualified=True,
            metric='Physical middle/end vector errors onall77998consumed trainingpoints;group/pointrelative statistics andzero-target absoluteerrors separate',
            worst=summary['stages']))
    pending = [row for row in rows if row['default_benchmark'] is None]
    require(len(pending) == 12 and sum(row['default_benchmark'] is not None for row in rows) == 193,
            'Only one original fit may advance beyond192')
    report = dict(previous,status='all205_fit_identities_joined_to193_role_scoped_GPU_benchmarks_and12_pending',
        created_unix=time.time(),previous_coverage=binding(previous_path),source_fit_index=binding(index_path),
        current_publication_acceptance=binding(publication),new_component_result=binding(result_path),rows=rows,
        default_common_benchmark_fits=193,common_GPU_accuracy_fits=193,recurrent_transport_component_fits=1,
        common_benchmark_pending_fits=12,common_accuracy_pending_fits=12,
        pending_fit_ids=[row['fit_item_id'] for row in pending],pending_by_task=dict(sorted(Counter(row['task'] for row in pending).items())),
        scope='150CNNpressure fits/128requests,35full generators/64cases,viscosity/point components/80patterns each,'
              'four modal/one sharedspectral component/144nativeRHSrequests,andone recurrenthead/576localgroups/77998points. '
              'Distinct populations/boundaries;not524states,intendedvariants,holdout,drift orclean retraining.',
        evidence=[*previous['evidence'],binding(result_path)],producer=binding(Path(__file__)),
        new_model_forwards=0,native_mutations=0,weight_or_dataset_copies=0,full_portfolio_complete=False)
    save(output,report)
    print('193/205original default fits benchmarked;12components remain.',flush=True)


if __name__ == '__main__':
    run()
