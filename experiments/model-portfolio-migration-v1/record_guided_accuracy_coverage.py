"""Join two original guided composition accuracies;keep timing pending."""
from collections import Counter
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, PROJECT, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output=DOC/'reports/common-fit-benchmark-coverage-12.json'
    require(not output.exists(), 'Preserve original guided accuracy coverage decision')
    publication,_=current_publication(ROOT)
    previous_path=DOC/'reports/common-fit-benchmark-coverage-11.json';previous=read(previous_path)
    measured_path=DOC/'reports/guided-GPU-accuracy-01.json';measured=read(measured_path)
    require(measured['status']=='two_original_guided_full_populations_GPU_accuracy_independently_reduced'
            and measured['fresh_GPU_processes']==4 and measured['blocks_per_fit']==2880
            and measured['physical_outputs_per_fit']==7200 and measured['physical_outputs_across_processes']==28800
            and measured['original_selected_training_objectives_and_worst_errors_reproduced']
            and measured['all_cross_process_input_output_metric_and_work_rows_exact']
            and not measured['latency_qualified'] and measured['primary_timing_samples']==0
            and measured['CPU_neural_forwards']==0, 'Complete original guided accuracy evidence required')
    contract=read(ROOT/measured['contract']['path'])
    material={r['fit_item_id']:r for r in read(ROOT/contract['material']['path'])['models']}
    index_path=PROJECT/'notes/portfolio-index.json'
    require(binding(index_path)==previous['source_fit_index'], 'Accepted original205fit index changed')
    native={r['item_id']:r for r in read(index_path)['fits']}
    rows=copy.deepcopy(previous['rows']);lookup={r['fit_item_id']:r for r in rows}
    require(len(rows)==len(lookup)==len(native)==205 and set(lookup)==set(native), 'Exact205fit roster required')
    expected={r['fit_item_id'] for r in contract['models']}
    require(len(expected)==2 and {m['fit_item_id'] for m in measured['models']}==expected, 'Exactly both original guided fits required')
    for spec in [measured['protocol'],measured['contract'],measured['producer'],measured['compaction'],measured['failure_reconciliation'],*measured['raw_results']]:
        observed_file(ROOT,spec['path'],spec)
    for model in measured['models']:
        fid=model['fit_item_id'];row,entry=lookup[fid],material[fid]
        require(row['task']==entry['task']==native[fid]['task'] and row['task'] in ('guided_shared_component','guided_pressure_component')
                and row['default_benchmark'] is None and row['status']=='common_GPU_benchmark_pending'
                and model['weights']==entry['outputs']['weights.npz']
                and native[fid]['receipt_sha256']==entry['original_receipt']['sha256']
                and any(value['sha256']==model['weights']['sha256'] for value in native[fid]['checkpoints']),
                'Original pending guided fit,receipt and selected weights must join')
        observed_file(ROOT,model['worst_field_capture']['path'],model['worst_field_capture'])
        row.update(status='common_GPU_accuracy_available_timing_pending',accuracy_benchmark=dict(
            group='guided_component_complete_composition',independent_reduction=binding(measured_path),latency_qualified=False,
            cases=2880,physical_outputs=7200,population='24consumed training parents;240current windows;30/60/120outputs_s andK1–4',
            source='Original captured guided composition,including fixed guide and raw selected F32residual network;all causal prediction CUDA',
            actual_model_module=model['actual_model_module'],worst=model['overall'],
            original_selected_training_objective_reproduced=True,original_training_worst_errors_reproduced=True,
            scope='Complete original guided composition accuracy;not standalone residual-subnet accuracy or an independent holdout.',
            retention_scope=measured['retention_scope'],native_publication_pending=True,independent_holdout=False,own_history_drift=False))
    timed_pending=[r for r in rows if r['default_benchmark'] is None]
    accuracy_pending=[r for r in timed_pending if r.get('accuracy_benchmark') is None]
    require(len(timed_pending)==10 and len(accuracy_pending)==2, 'Only two guided accuracies advance;ten timings remain')
    result=dict(previous,status='all205_fits_joined_to203_GPU_accuracy_and195_timed_default_benchmarks',
        created_unix=time.time(),producer=binding(Path(__file__)),previous_coverage=binding(previous_path),
        source_fit_index=binding(index_path),current_publication_acceptance=binding(publication),
        new_component_accuracy_result=binding(measured_path),rows=rows,common_GPU_accuracy_fits=203,
        default_common_benchmark_fits=195,common_accuracy_pending_fits=2,common_benchmark_pending_fits=10,
        guided_composition_accuracy_fits=2,guided_latency_pending_fits=2,
        pending_fit_ids=[r['fit_item_id'] for r in timed_pending],pending_accuracy_fit_ids=[r['fit_item_id'] for r in accuracy_pending],
        pending_accuracy_by_task=dict(sorted(Counter(r['task'] for r in accuracy_pending).items())),
        scope=previous['scope']+' Two original guided compositions add2880blocks/7200physical-output accuracies each from24training parents;timing remains pending.',
        evidence=[*previous['evidence'],binding(measured_path)],new_model_forwards=0,native_mutations=0,
        weight_or_dataset_copies=0,full_portfolio_complete=False)
    save(output,result)
    print('203/205GPUtask accuracies;195/205timed;2accuracy and10timing qualifications remain.',flush=True)


if __name__=='__main__':run()
