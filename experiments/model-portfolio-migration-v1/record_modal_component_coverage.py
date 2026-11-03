"""Join the four qualified original modal components to the205-fit inventory."""
from collections import Counter
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, PROJECT, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output = DOC/'reports/common-fit-benchmark-coverage-06.json'
    require(not output.exists(), 'Preserve original modal coverage decision')
    publication,_ = current_publication(ROOT)
    previous_path = DOC/'reports/common-fit-benchmark-coverage-05.json';previous = read(previous_path)
    result_path = DOC/'reports/modal-component-GPU-benchmark-01.json';measured = read(result_path)
    require(measured['status'] == 'four_original_modal_GPU_benchmarks_independently_reduced'
            and measured['fits'] == 4 and measured['cases_per_fit'] == 144 and measured['fresh_processes'] == 8
            and measured['primary_calls'] == 11520 and measured['CPU_neural_forwards'] == 0
            and measured['physical_simulation_steps'] == 0 and not measured['learned_gain_cache']
            and measured['timing_units'] == 'ms/native_RHS_pressure_request', 'Complete independently reduced four-modal result required')
    for spec in [measured['protocol'],measured['contract'],measured['owner'],measured['producer']]:
        observed_file(ROOT,spec['path'],spec)
    contract = read(ROOT/measured['contract']['path'])
    original = read(ROOT/contract['material']['path'])
    material = {row['fit_item_id']:row for row in original['models']}
    index_path = PROJECT/'notes/portfolio-index.json';index = read(index_path)
    require(binding(index_path) == previous['source_fit_index'], 'Accepted native fit index changed')
    native = {row['item_id']:row for row in index['fits']}
    rows = copy.deepcopy(previous['rows']);fits = {row['fit_item_id']:row for row in rows}
    require(len(rows) == len(fits) == len(native) == 205 and set(fits) == set(native), 'Exact205fit roster required')
    require({row['fit_item_id'] for row in measured['models']} == {row['fit_item_id'] for row in contract['fits']}
            and len(measured['models']) == 4, 'Exact four original modal fits required')
    for model in measured['models']:
        fid = model['fit_item_id'];row = fits[fid];entry = material[fid]
        require(row['task'] == entry['task'] == native[fid]['task'] == 'pressure_component'
                and row['default_benchmark'] is None and row['status'] == 'common_GPU_benchmark_pending'
                and model['weights'] == entry['outputs']['weights.npz']
                and native[fid]['receipt_sha256'] == entry['original_receipt']['sha256']
                and any(value['sha256'] == model['weights']['sha256'] for value in native[fid]['checkpoints']),
                'Qualified fields must join original pending fit/receipt/checkpoint identity')
        for spec in [model['weights'],model['prediction_archive'],*model['raw_results']]:
            observed_file(ROOT,spec['path'],spec)
        summary = model['groups']['overall']
        row.update(status='default_common_GPU_accuracy_and_timing_available',
            default_benchmark=dict(group='modal_pressure_component',identifier=fid,cases=144,
                source='original_published_F64_modal_table_CUDA_restore',benchmark_authority=binding(result_path),
                accuracy_authority=binding(result_path),timing_units='GPU ms/native RHS pressure request;no simulation advancement',
                case_median_median_ms=summary['median_case_GPU_ms'],population=measured['scope'],
                native_publication_pending=True,not_same_as_CNN_five_input_boundary=True),
            accuracy_benchmark=dict(independent_reduction=binding(result_path),latency_qualified=True,
                metric='physical mean-free pressure relativeL2 on144consumed native fitting frames',worst=summary['worst']))
    pending = [row for row in rows if row['default_benchmark'] is None]
    require(len(pending) == 14 and sum(row['default_benchmark'] is not None for row in rows) == 191,
            'Exactly four original fits may advance beyond187')
    report = dict(previous,status='all205_fit_identities_joined_to191_role_scoped_GPU_benchmarks_and14_pending',
        created_unix=time.time(),previous_coverage=binding(previous_path),source_fit_index=binding(index_path),
        current_publication_acceptance=binding(publication),new_component_result=binding(result_path),rows=rows,
        default_common_benchmark_fits=191,common_GPU_accuracy_fits=191,modal_pressure_component_fits=4,
        common_benchmark_pending_fits=14,common_accuracy_pending_fits=14,
        pending_fit_ids=[row['fit_item_id'] for row in pending],pending_by_task=dict(sorted(Counter(row['task'] for row in pending).items())),
        scope='150CNNpressure fits/128requests,35full generators/64cases,viscosity andpoint components/80patterns each,'
              'four modal components/144native RHSrequests. Distinct role-specific populations andunits;not524states,intended variants,holdout,drift orclean retraining.',
        evidence=[*previous['evidence'],binding(result_path)],producer=binding(Path(__file__)),
        new_model_forwards=0,native_mutations=0,weight_or_dataset_copies=0,full_portfolio_complete=False)
    save(output,report)
    print('191/205original default fits have role-scoped GPU accuracy/timing;14components remain.',flush=True)


if __name__ == '__main__':
    run()
