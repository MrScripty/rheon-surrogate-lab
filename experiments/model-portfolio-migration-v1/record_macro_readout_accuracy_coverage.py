"""Advance two original field-accuracy rows while keeping latency pending."""
from collections import Counter
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, PROJECT, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output = DOC/'reports/common-fit-benchmark-coverage-10.json'
    require(not output.exists(), 'Preserve original field-accuracy coverage decision')
    publication,_ = current_publication(ROOT)
    previous_path = DOC/'reports/common-fit-benchmark-coverage-09.json'
    previous = read(previous_path)
    result_path = DOC/'reports/macro-readout-field-GPU-accuracy-01.json'
    measured = read(result_path)
    require(measured['status'] == 'two_original_macro_readout_causal_field_GPU_accuracy_independently_reduced'
            and measured['fresh_GPU_processes'] == 2 and measured['active_cases_per_fit'] == 58
            and measured['physical_outputs_checked_across_processes'] == 232
            and measured['original_QR_factor_bitwise_checks'] == 232
            and measured['all_cross_process_input_and_output_hashes_exact']
            and not measured['latency_qualified'] and measured['primary_timing_samples'] == 0
            and measured['CPU_neural_forwards'] == 0, 'Complete independent original CUDA field-accuracy evidence required')
    protocol = read(ROOT/measured['protocol']['path'])
    contract = read(ROOT/measured['contract']['path'])
    material = {row['fit_item_id']:row for row in read(ROOT/contract['material']['path'])['models']}
    index_path = PROJECT/'notes/portfolio-index.json'
    require(binding(index_path) == previous['source_fit_index'], 'Accepted205fit source index changed')
    native = {row['item_id']:row for row in read(index_path)['fits']}
    rows = copy.deepcopy(previous['rows']);lookup = {row['fit_item_id']:row for row in rows}
    require(len(rows) == len(lookup) == len(native) == 205 and set(lookup) == set(native), 'Exact original205fit roster required')
    expected = {'fit-6eadd58ed412cffc','fit-e6d5423babcc9478'}
    require({row['fit_item_id'] for row in measured['models']} == expected, 'Exactly the two original readout fits required')
    for spec in [measured['protocol'],measured['contract'],measured['producer'],*measured['raw_results']]:
        observed_file(ROOT,spec['path'],spec)
    for model in measured['models']:
        fid = model['fit_item_id'];row,entry = lookup[fid],material[fid]
        require(row['task'] == entry['task'] == native[fid]['task'] == 'full_state_readout_component'
                and row['default_benchmark'] is None and row['status'] == 'common_GPU_benchmark_pending'
                and model['weights'] == entry['outputs']['weights.npz']
                and native[fid]['receipt_sha256'] == entry['original_receipt']['sha256']
                and any(value['sha256'] == model['weights']['sha256'] for value in native[fid]['checkpoints']),
                'Original pending readout fit,receipt and published coefficient identity must join')
        observed_file(ROOT,model['prediction_archive']['path'],model['prediction_archive'])
        row.update(status='common_GPU_accuracy_available_timing_pending',accuracy_benchmark=dict(
            group='full_state_readout_component',independent_reduction=binding(result_path),latency_qualified=False,
            cases=58,population='29originalN8training parents/two starts;119N1parents/238windows bypass the head',
            source='Unchanged original3/4stage causal CUDA backbone plus raw published F64readout;final native F64pressure/F32carriers',
            scope=measured['scope'],worst=model['overall'],original_QR_factors_bitwise_reproduced=True,
            native_publication_pending=True,training_objective_only=False,
            independent_holdout=False,own_history_drift=False))
    timed_pending = [row for row in rows if row['default_benchmark'] is None]
    accuracy_pending = [row for row in timed_pending if row.get('accuracy_benchmark') is None]
    require(len(timed_pending) == 10 and len(accuracy_pending) == 8,
            'Advance field accuracy only; preserve all ten outstanding complete timing qualifications')
    result = dict(previous,status='all205_fits_joined_to197_GPU_accuracy_and195_timed_default_benchmarks',
        created_unix=time.time(),previous_coverage=binding(previous_path),source_fit_index=binding(index_path),
        current_publication_acceptance=binding(publication),new_component_accuracy_result=binding(result_path),rows=rows,
        default_common_benchmark_fits=195,common_GPU_accuracy_fits=197,common_accuracy_pending_fits=8,
        common_benchmark_pending_fits=10,readout_field_accuracy_fits=2,readout_latency_pending_fits=2,
        pending_fit_ids=[row['fit_item_id'] for row in timed_pending],
        pending_accuracy_fit_ids=[row['fit_item_id'] for row in accuracy_pending],
        pending_accuracy_by_task=dict(sorted(Counter(row['task'] for row in accuracy_pending).items())),
        scope=previous['scope']+' Two raw readout field accuracies add58active expanded windows each;'
            '238fixed bypass windows not counted as learned predictions. These two still lack qualified timing.',
        evidence=[*previous['evidence'],binding(result_path)],producer=binding(Path(__file__)),
        new_model_forwards=0,native_mutations=0,weight_or_dataset_copies=0,full_portfolio_complete=False)
    save(output,result)
    print('197/205GPU field/task accuracies;195/205complete timed defaults;8accuracy and10timing qualifications remain.',flush=True)


if __name__ == '__main__':
    run()
