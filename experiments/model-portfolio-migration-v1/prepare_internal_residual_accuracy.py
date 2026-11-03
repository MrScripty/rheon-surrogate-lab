"""Register complete original correction accuracy without collecting latency."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from component_material import current_publication


def run():
    path=RUN/'internal-residual-GPU-accuracy-protocol-02.json'
    require(not path.exists(), 'Preserve original correction accuracy registration')
    current_publication(ROOT)
    contract_path=RUN/'internal-residual-component-contract-01.json'
    contract=read(contract_path)
    reconciliation_path=DOC/'reports/internal-residual-normalization-reader-reconciliation-01.json'
    reconciliation=read(reconciliation_path)
    require(reconciliation['failed_worker_terminal'] and reconciliation['accepted_accuracy_results']==0
            and reconciliation['original_normalization_equals_both_embedded_training_configs'],
            'Original failed first request must be reconciled before fresh campaign')
    manifest=read(ROOT/contract['manifest']['path'])
    evaluation_path=ROOT/manifest['original_eval_normalization']
    evaluation=read(evaluation_path)
    require(isinstance(evaluation,dict) and 'relative_denominator_rms_floor' in evaluation,
            'Original evaluation normalization document required,not its path string')
    originals=[read(ROOT/item['original_configuration']['path'])['evaluation_normalization']
               for item in contract['models'] if 'evaluation_normalization' in read(ROOT/item['original_configuration']['path'])]
    require(len(originals)==2 and all(value==evaluation for value in originals),
            'Normalization must equal both original embedded S1/S2 training authorities')
    probe_path=DOC/'reports/internal-residual-original-GPU-probe-01.json'
    probe=read(probe_path)
    require(probe['status']=='four_original_internal_residual_GPU_probes_replayed'
            and probe['all_retained_GPU_probes_bitwise_equal'] and probe['cleanup_complete']
            and not Path('/proc',str(probe['pid'])).exists()
            and probe['contract']==binding(contract_path)
            and probe['module_forward_guard']['forbidden_forward_attempts']==0,
            'Original four CUDA probe matches must be terminal before full accuracy')
    sources=dict(contract['source_sha256'])
    for entry in Path(__file__).parent.glob('*.py'):
        spec=binding(entry);sources[spec['path']]=spec['sha256']
    for entry in (contract_path,probe_path,evaluation_path,reconciliation_path):
        spec=binding(entry);sources[spec['path']]=spec['sha256']
    save(path,dict(status='registered_before_four_original_correction_full_request_GPU_accuracy',
        created_unix=time.time(),producer=binding(Path(__file__)),contract=binding(contract_path),
        original_probe_replay=binding(probe_path),source_sha256=sources,
        evaluation_normalization=binding(evaluation_path),failure_reconciliation=binding(reconciliation_path),
        expected_GPU=contract['expected_GPU'],host_cpu=4,threads=1,phases=[0,1],
        output_root='runs/model-portfolio-migration-v1/internal-residual-GPU-accuracy-02',
        requests_per_fit=783,fit_ids=[m['fit_item_id'] for m in contract['models']],
        worker_deadline_seconds=1200,maximum_CUDA_allocated_bytes=6*1024**3,
        maximum_RSS_bytes=8*1024**3,minimum_free_disk_bytes=10*1024**3,
        retained_prediction_policy='Every request retains exact generated field hashes and original Torch plus independent NumPy physical scores. Phase0 retains the union of four metric-worst complete field outputs perfit;phase1 reuses those captures and requires every generated field hash to match. No whole-prediction-corpus or dataset copy.',
        independent_metric_tolerance=dict(relative=1e-10,absolute=1e-12),
        primary_timing_samples=0,latency_qualified=False,concurrent_compute_allowed_for_accuracy=True,
        targets_forwarded_to_head_or_response=False,CPU_neural_forwards=0,
        complete_generator_accuracy_claim=False,independent_holdout=False,
        full_portfolio_complete=False))
    print('Four fits ×783requests ×two phases registered;compact worst-field retention;no timing.',flush=True)


if __name__=='__main__':
    run()
