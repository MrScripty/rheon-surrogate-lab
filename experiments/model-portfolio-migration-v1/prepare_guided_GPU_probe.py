"""Freeze source-preserving original guided CUDA probe restoration."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from component_material import current_publication


def run():
    path=RUN/'guided-original-GPU-probe-protocol-01.json'
    require(not path.exists(), 'Preserve original guided GPU probe registration')
    current_publication(ROOT)
    contract_path=RUN/'guided-component-contract-01.json';contract=read(contract_path)
    qualification_path=DOC/'reports/guided-captured-runtime-qualification-01.json';q=read(qualification_path)
    require(q['status']=='both_actual_guided_source_views_and_access_boundaries_qualified'
            and q['negative_controls_passed']==6 and q['contract']==binding(contract_path)
            and all(m['temporary_view_removed'] for m in q['models']), 'Actual source-view access qualification required')
    sources={spec['path']:spec['sha256'] for spec in contract['original_material_observations']}
    for p in Path(__file__).parent.glob('*.py'):
        spec=binding(p);sources[spec['path']]=spec['sha256']
    for p in (contract_path,qualification_path):
        spec=binding(p);sources[spec['path']]=spec['sha256']
    save(path,dict(status='registered_before_original_guided_GPU_probe_restores',created_unix=time.time(),
        producer=binding(Path(__file__)),contract=binding(contract_path),source_view_qualification=binding(qualification_path),
        source_sha256=sources,fit_ids=[m['fit_item_id'] for m in contract['models']],
        expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',host_cpu=4,threads=1,
        output_root='runs/model-portfolio-migration-v1/guided-original-GPU-probe-01',
        worker_deadline_seconds=300,maximum_CUDA_allocated_bytes=6*1024**3,maximum_RSS_bytes=8*1024**3,
        primary_timing_samples=0,latency_qualified=False,concurrent_compute_allowed_for_accuracy=True,
        CPU_neural_forwards=0,new_parameter_fits=0,model_dataset_or_project_copies=0,full_portfolio_complete=False))
    print('Both original guided CUDA probe restores registered;no timing or training.',flush=True)


if __name__=='__main__':run()
