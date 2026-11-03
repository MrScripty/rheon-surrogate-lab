"""Register both full original guided populations after compact metadata custody."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from component_material import current_publication


def run():
    path=RUN/'guided-GPU-accuracy-protocol-02.json'
    require(not path.exists(), 'Preserve original guided full accuracy registration')
    current_publication(ROOT)
    scorer_path=DOC/'reports/guided-scoring-isolation-qualification-01.json';scorer=read(scorer_path)
    failure_path=DOC/'reports/guided-scoring-isolation-reconciliation-01.json';failure=read(failure_path)
    require(scorer['status']=='actual_scorer_isolation_failure_reproduced_and_six_literal_oracles_passed'
            and scorer['positive_literal_checks']==6 and failure['failed_worker_terminal']
            and failure['accepted_result_rows']==0, 'Original scoring-isolation failure must be reconciled')
    compact_path=RUN/'guided-component-contract-02.json';contract=read(compact_path)
    disposition_path=DOC/'reports/guided-contract-compaction-01.json';disposition=read(disposition_path)
    require(disposition['compact_contract']['sha256']==binding(compact_path)['sha256']
            and disposition['all2880cases_and7200outputs_preserved']
            and disposition['exact_old_byte_reconstruction_sha256']==disposition['previous_expanded_contract']['sha256']
            and not contract['scene_definitions_copied'], 'Lossless compact source authority required')
    old_probe_protocol=read(RUN/'guided-original-GPU-probe-protocol-01.json')
    probes=[]
    for item in contract['models']:
        p=ROOT/old_probe_protocol['output_root']/item['fit_item_id']/'result.json';probe=read(p)
        require(probe['status']=='original_guided_selected_GPU_probe_replayed'
                and probe['original_selected_GPU_probe_bitwise_equal'] and probe['cleanup_complete']
                and probe['temporary_source_view_removed'] and not Path('/proc',str(probe['pid'])).exists()
                and not probe['ambient_access_attempts'] and probe['module_forward_guard']['forbidden_forward_attempts']==0,
                'Both original captured CUDA probe controls must be terminal and exact')
        probes.append(binding(p))
    sources={s['path']:s['sha256'] for s in contract['original_material_observations']}
    for p in Path(__file__).parent.glob('*.py'):
        spec=binding(p);sources[spec['path']]=spec['sha256']
    for p in (compact_path,disposition_path,scorer_path,failure_path,*[ROOT/s['path'] for s in probes]):
        spec=binding(p);sources[spec['path']]=spec['sha256']
    save(path,dict(status='registered_before_both_original_guided_full_population_GPU_accuracies',created_unix=time.time(),
        producer=binding(Path(__file__)),contract=binding(compact_path),compaction=binding(disposition_path),
        scoring_isolation_qualification=binding(scorer_path),failure_reconciliation=binding(failure_path),
        original_probe_controls=probes,source_sha256=sources,fit_ids=[m['fit_item_id'] for m in contract['models']],
        output_root='runs/model-portfolio-migration-v1/guided-GPU-accuracy-02',phases=[0,1],
        expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',host_cpu=4,threads=1,
        worker_deadline_seconds=1200,maximum_CUDA_allocated_bytes=6*1024**3,maximum_RSS_bytes=8*1024**3,
        blocks_per_fit=2880,physical_outputs_per_fit=7200,
        independent_metric_tolerance=dict(relative=1e-10,absolute=1e-12),
        original_training_objective_tolerance=dict(relative=1e-9,absolute=1e-14),
        retained_prediction_policy='Every physical output retains hashes and NumPy/original Torch field scores. Only the union of four metric-worst outputs perfit is retained as arrays. Phase1 reuses captures and requires every field/score to repeat exactly.',
        primary_timing_samples=0,latency_qualified=False,concurrent_compute_allowed_for_accuracy=True,
        CPU_neural_forwards=0,new_parameter_fits=0,model_dataset_or_project_copies=0,
        independent_holdout=False,own_history_drift=False,full_portfolio_complete=False))
    print('Both original guided fits ×2880blocks/7200outputs ×two phases registered;no timing.',flush=True)


if __name__=='__main__':run()
