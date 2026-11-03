"""Reconcile a zero-prediction launch and bind the task's whole tooling context.

The scientific worker, model, references and measurement boundary stay exact.
An explicit process-local selector supplies a fresh protocol/output to the
unchanged owner/reducer. The failed original protocol and source remain intact.
"""
from pathlib import Path
import time

from review_water_learning_sequence import ROOT, binding, canonical, read, require


def run():
    runs = ROOT/'runs/model-portfolio-migration-v1'
    docs = ROOT/'docs/plans/model-portfolio-migration-v1'
    destination = runs/'same-case-F64-GPU-protocol-02.json'
    report_path = docs/'reports/same-case-F64-dependency-correction-01.json'
    require(not destination.exists() and not report_path.exists(),'Preserve corrected F64 admission')
    original_path = runs/'same-case-F64-GPU-protocol-01.json'
    protocol = read(original_path)
    owner_path = ROOT/protocol['output']/'status.json'
    result_path = ROOT/protocol['output']/'K1-phase0.json'
    owner,result = read(owner_path),read(result_path)
    require(owner['status'] == 'original_F64_campaign_requires_reconciliation' and owner['active'] is None
            and not owner['completed'] and not Path(f'/proc/{owner["pid"]}').exists()
            and result['status'] == 'original_F64_worker_failed_requires_reconciliation'
            and not Path(f'/proc/{result["pid"]}').exists() and not result['rows'] and result['cleanup_complete']
            and result['CPU_neural_forwards'] == result['forbidden_forward_attempts'] == result['guarded_module_calls'] == 0
            and result['error'] == 'Imported project source was not preregistered: experiments/model-portfolio-migration-v1/prepare_method_review_revisions.py',
            'Exact terminal zero-prediction failure must be reconciled,not inferred or replayed')
    dependencies = {s['path']:s for s in protocol['dependencies']}
    for spec in dependencies.values():
        require(binding(ROOT/spec['path']) == spec,'Original scientific/source identity changed')
    tooling = sorted(Path(__file__).parent.glob('*.py'))
    for path in tooling:
        dependencies[str(path.relative_to(ROOT))] = binding(path)
    dependencies[str(original_path.relative_to(ROOT))] = binding(original_path)
    protocol.update(created_unix=time.time(),status='corrected_dependencies_preregistered_not_dispatched',
        dependencies=sorted(dependencies.values(),key=lambda s:s['path']),
        output='runs/model-portfolio-migration-v1/same-case-F64-GPU-benchmark-02',
        previous_zero_prediction_failure=dict(owner=binding(owner_path),result=binding(result_path)),
        producer=binding(Path(__file__)),input_selector=binding(Path(__file__).with_name('select_same_case_F64_protocol_02.py')),
        source_selection=protocol['source_selection']+' Also bind every existing migration-task Python tool,including transitive metadata helper imports; actual imported-source enforcement unchanged.')
    for block in protocol['process_blocks']:
        block['result'] = block['result'].replace('same-case-F64-GPU-benchmark-01/','same-case-F64-GPU-benchmark-02/')
    protocol['bound_dependency_bytes'] = sum((ROOT/s['path']).stat().st_size for s in dependencies.values())
    destination.write_bytes(canonical(protocol)+b'\n')
    report = dict(status='terminal_zero_prediction_F64_failure_reconciled_complete_task_tooling_bound',created_unix=time.time(),
        classification='Tooling source-closure omission; no field predictions or CPUneural execution occurred.',
        original_protocol=binding(original_path),original_owner=binding(owner_path),original_result=binding(result_path),
        corrected_protocol=binding(destination),bound_task_Python_files=len(tooling),
        unchanged_scientific_worker=True,unchanged_weights_kernels_references=True,original_failed_directory_replayed=False,
        new_model_forwards=0,native_mutations=0,full_portfolio_complete=False,producer=binding(Path(__file__)))
    report_path.write_bytes(canonical(report)+b'\n')
    print(report['status'],'with',len(tooling),'task tools;fresh output02;no inference dispatched.')


if __name__ == '__main__':
    run()
