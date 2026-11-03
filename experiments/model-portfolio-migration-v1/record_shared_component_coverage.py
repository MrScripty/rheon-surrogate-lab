"""Join one original shared fit, with two decoder/cache execution configurations."""
from collections import Counter
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, PROJECT, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output = DOC/'reports/common-fit-benchmark-coverage-07.json'
    require(not output.exists(), 'Preserve original shared coverage decision')
    publication,_ = current_publication(ROOT)
    previous_path = DOC/'reports/common-fit-benchmark-coverage-06.json';previous = read(previous_path)
    result_path = DOC/'reports/shared-component-GPU-benchmark-02.json';measured = read(result_path)
    require(measured['status'] == 'shared_original_GPU_decoder_cache_comparison_independently_reduced'
            and measured['fits'] == 1 and measured['configurations'] == 2 and measured['fresh_processes'] == 4
            and measured['cases_per_fit'] == 144 and measured['primary_calls'] == 5760
            and measured['CPU_neural_forwards'] == measured['physical_simulation_steps'] == 0
            and measured['all_mode_and_process_outputs_exact'], 'Both original shared cache modes must qualify')
    for spec in [measured['protocol'],measured['contract'],measured['owner'],measured['producer'],measured['unique_prediction_archive']]:
        observed_file(ROOT,spec['path'],spec)
    protocol = read(ROOT/measured['protocol']['path']);fid = protocol['selected']['fit_item_id']
    require(fid == 'fit-9c914f77b1f2ec81', 'Exact original shared fit required')
    contract = read(ROOT/measured['contract']['path'])
    material = read(ROOT/contract['material']['path'])
    entry = next(row for row in material['models'] if row['fit_item_id'] == fid)
    index_path = PROJECT/'notes/portfolio-index.json';index = read(index_path)
    require(binding(index_path) == previous['source_fit_index'], 'Accepted native fit index changed')
    native = {row['item_id']:row for row in index['fits']}
    rows = copy.deepcopy(previous['rows']);fits = {row['fit_item_id']:row for row in rows}
    require(len(rows) == len(fits) == len(native) == 205 and set(fits) == set(native), 'Original205fit roster required')
    row = fits[fid]
    require(row['task'] == entry['task'] == native[fid]['task'] == 'pressure_component'
            and row['status'] == 'common_GPU_benchmark_pending' and row['default_benchmark'] is None
            and protocol['selected']['weights'] == entry['outputs']['weights.npz']
            and native[fid]['receipt_sha256'] == entry['original_receipt']['sha256']
            and any(item['sha256'] == entry['outputs']['weights.npz']['sha256'] for item in native[fid]['checkpoints']),
            'Result must join original pending fit/receipt/published weight identity')
    modes = {item['mode']:item for item in measured['modes']}
    require(set(modes) == {'prepared','unprepared'} and len(measured['modes']) == 2, 'Two execution modes,one fit required')
    benchmarks = {}
    for mode,item in modes.items():
        require(item['fit_item_id'] == fid and item['weights'] == entry['outputs']['weights.npz']
                and item['prediction_archive'] == measured['unique_prediction_archive'], 'Original parameters and shared field archive required')
        for spec in [item['weights'],*item['raw_results']]:
            observed_file(ROOT,spec['path'],spec)
        benchmarks[mode] = dict(group='shared_spectral_pressure_component',identifier=fid,cases=144,cache_policy=mode,
            source='original_F64_CUDA_decoder_with_original_standalone_GPU_basis_policy',benchmark_authority=binding(result_path),
            accuracy_authority=binding(result_path),timing_units='GPU ms/native RHS pressure request;no simulation advancement',
            case_median_median_ms=item['groups']['overall']['median_case_GPU_ms'],population=measured['scope'],
            native_publication_pending=True,not_same_as_CNN_five_input_boundary=True,
            setup_separate=True,full_generator_bitwise_parity_claim=False)
    row.update(status='default_common_GPU_accuracy_and_timing_available',default_benchmark=benchmarks['prepared'],
        alternate_benchmarks=[benchmarks['unprepared']],
        accuracy_benchmark=dict(independent_reduction=binding(result_path),latency_qualified=True,
            metric='physical mean-free pressure relativeL2 on144posthoc legacy native requests;not an independentholdout',
            worst=modes['prepared']['groups']['overall']['worst']))
    pending = [row for row in rows if row['default_benchmark'] is None]
    require(len(pending) == 13 and sum(row['default_benchmark'] is not None for row in rows) == 192,
            'Only one original fit may advance beyond191')
    report = dict(previous,status='all205_fit_identities_joined_to192_role_scoped_GPU_benchmarks_and13_pending',
        created_unix=time.time(),previous_coverage=binding(previous_path),source_fit_index=binding(index_path),
        current_publication_acceptance=binding(publication),new_component_result=binding(result_path),rows=rows,
        default_common_benchmark_fits=192,common_GPU_accuracy_fits=192,shared_spectral_component_fits=1,
        common_benchmark_pending_fits=13,common_accuracy_pending_fits=13,
        pending_fit_ids=[row['fit_item_id'] for row in pending],pending_by_task=dict(sorted(Counter(row['task'] for row in pending).items())),
        scope='150CNNpressure fits/128requests,35full generators/64cases,viscosity andpoint components/80patterns each,'
              'four modal andone sharedspectral component/144native RHSrequests. Sharedfit hasprepared/unprepared modes,countedonce. '
              'Distinct populations/boundaries;not524states,intended variants,holdout,drift orclean retraining.',
        evidence=[*previous['evidence'],binding(result_path)],producer=binding(Path(__file__)),
        new_model_forwards=0,native_mutations=0,weight_or_dataset_copies=0,full_portfolio_complete=False)
    save(output,report)
    print('192/205original default fits benchmarked;shareddecoder modescountonce;13components remain.',flush=True)


if __name__ == '__main__':
    run()
