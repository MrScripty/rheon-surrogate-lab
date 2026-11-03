"""Join four original component accuracies;keep all outstanding timings visible."""
from collections import Counter
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, PROJECT, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output=DOC/'reports/common-fit-benchmark-coverage-11.json'
    require(not output.exists(), 'Preserve original correction accuracy coverage decision')
    publication,_=current_publication(ROOT)
    previous_path=DOC/'reports/common-fit-benchmark-coverage-10.json'
    previous=read(previous_path)
    measured_path=DOC/'reports/internal-residual-GPU-accuracy-01.json'
    measured=read(measured_path)
    require(measured['status']=='four_original_internal_residual_GPU_accuracies_independently_scored'
            and measured['fresh_GPU_processes']==2 and measured['active_requests_per_fit']==783
            and measured['CUDA_correction_requests_across_processes']==6264
            and measured['all_cross_process_input_output_and_metric_rows_exact']
            and measured['generated_field_hashes_checked_across_processes']==31320
            and not measured['latency_qualified'] and measured['primary_timing_samples']==0
            and measured['CPU_neural_forwards']==0 and not measured['complete_generator_accuracy_claim'],
            'Complete independent original component accuracy evidence required')
    contract=read(ROOT/measured['contract']['path'])
    material={r['fit_item_id']:r for r in read(ROOT/contract['material']['path'])['models']}
    index_path=PROJECT/'notes/portfolio-index.json'
    require(binding(index_path)==previous['source_fit_index'], 'Accepted original205fit source index changed')
    native={r['item_id']:r for r in read(index_path)['fits']}
    rows=copy.deepcopy(previous['rows']);lookup={r['fit_item_id']:r for r in rows}
    require(len(rows)==len(lookup)==len(native)==205 and set(lookup)==set(native), 'Exact original205fit roster required')
    expected={r['fit_item_id'] for r in contract['models']}
    require(len(expected)==4 and {r['fit_item_id'] for r in measured['models']}==expected, 'Exactly the four original correction fits required')
    for spec in [measured['protocol'],measured['contract'],measured['producer'],measured['failure_reconciliation'],*measured['raw_results']]:
        observed_file(ROOT,spec['path'],spec)
    for model in measured['models']:
        fid=model['fit_item_id'];row,entry=lookup[fid],material[fid]
        require(row['task']==entry['task']==native[fid]['task']=='full_state_correction_component'
                and row['default_benchmark'] is None and row['status']=='common_GPU_benchmark_pending'
                and model['weights']==entry['outputs']['weights.npz']
                and native[fid]['receipt_sha256']==entry['original_receipt']['sha256']
                and any(value['sha256']==model['weights']['sha256'] for value in native[fid]['checkpoints']),
                'Original pending fit,receipt and selected checkpoint identity must join')
        observed_file(ROOT,model['worst_field_capture']['path'],model['worst_field_capture'])
        row.update(status='common_GPU_accuracy_available_timing_pending',accuracy_benchmark=dict(
            group='full_state_correction_component',independent_reduction=binding(measured_path),
            latency_qualified=False,cases=783,population='29originalN8training parents,nine starts,three internal2/2/3groups;119N1parents bypass head',
            source='Original F32correction head with frozen learned F64coupled response;supplied internal-stage context',
            scope=measured['scope'],worst=model['overall'],retention_scope=measured['retention_scope'],
            native_publication_pending=True,independent_holdout=False,own_history_drift=False,
            complete_generator_accuracy_claim=False))
    timed_pending=[r for r in rows if r['default_benchmark'] is None]
    accuracy_pending=[r for r in timed_pending if r.get('accuracy_benchmark') is None]
    require(len(timed_pending)==10 and len(accuracy_pending)==4,
            'Advance accuracy only;keep ten outstanding timing qualifications')
    result=dict(previous,status='all205_fits_joined_to201_GPU_accuracy_and195_timed_default_benchmarks',
        created_unix=time.time(),producer=binding(Path(__file__)),previous_coverage=binding(previous_path),
        source_fit_index=binding(index_path),current_publication_acceptance=binding(publication),
        new_component_accuracy_result=binding(measured_path),rows=rows,
        common_GPU_accuracy_fits=201,default_common_benchmark_fits=195,
        common_accuracy_pending_fits=4,common_benchmark_pending_fits=10,
        correction_field_accuracy_fits=4,correction_latency_pending_fits=4,
        pending_fit_ids=[r['fit_item_id'] for r in timed_pending],
        pending_accuracy_fit_ids=[r['fit_item_id'] for r in accuracy_pending],
        pending_accuracy_by_task=dict(sorted(Counter(r['task'] for r in accuracy_pending).items())),
        scope=previous['scope']+' Four original correction heads add783supplied internal-stage component request accuracies each;not causal full-step predictions and timing remains pending.',
        evidence=[*previous['evidence'],binding(measured_path)],new_model_forwards=0,native_mutations=0,
        weight_or_dataset_copies=0,full_portfolio_complete=False)
    save(output,result)
    print('201/205GPUtask accuracies;195/205timed;4accuracy and10timing qualifications remain.',flush=True)


if __name__=='__main__':
    run()
