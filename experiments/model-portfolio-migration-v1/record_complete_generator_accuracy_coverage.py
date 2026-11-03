"""Reconcile common-case accuracy for all35fits without waiting on timing phases."""
from pathlib import Path
import time

from reduce_complete_generator_gpu_benchmark import historical_results, validate_row
from reduce_inherited_generator_qualification import summarize
from review_water_learning_sequence import ROOT, binding, canonical, read, require


def run():
    docs = ROOT/'docs/plans/model-portfolio-migration-v1'
    runs = ROOT/'runs/model-portfolio-migration-v1'
    output = docs/'reports/common-generator-accuracy-coverage-02.json'
    require(not output.exists(),'Preserve original complete accuracy reconciliation')
    protocol_path = runs/'complete-generator-GPU-protocol-01.json'
    protocol = read(protocol_path)
    expected = {(c['identifier'],c['start']):c for c in read(ROOT/protocol['cases']['path'])['cases']}
    previous = historical_results(protocol)
    owner_path = ROOT/protocol['output']/'status.json'
    owner = read(owner_path)
    require(owner['protocol'] == binding(protocol_path),'Original campaign identity changed')
    completed = {b['fit_item_id']:b for b in owner['completed'] if b['phase'] == 0}
    rows,evidence = [],[]
    for model in protocol['models']:
        fit = model['fit_item_id']
        if fit in previous:
            source = previous[fit]
            values = source['rows']
            spec = source['binding']
            scope = 'Previously qualified complete64case accuracy; historical timing is separate.'
        else:
            require(fit in completed,'A previously unmeasured fit has not completed its original worker')
            spec = completed[fit]['result_binding']
            require(binding(ROOT/spec['path']) == spec,'Original completed worker changed')
            d = read(ROOT/spec['path'])
            require(d['status'] == 'all64_original_cases_uninstrumented_primary_and_paired_hooks_measured'
                    and d['protocol'] == binding(protocol_path) and d['fit_item_id'] == fit
                    and d['phase'] == 0 and d['CPU_neural_forwards'] == d['forbidden_forward_attempts'] == 0,
                    'Complete first-phase CUDA result required')
            values = d['rows']
            for row in values:
                validate_row(row,expected[(row['identifier'],row['start'])],model,protocol,0)
            scope = 'New original phase0complete64case CUDA accuracy; final two-phase timing reduction remains pending.'
        require(len(values) == 64 and {(v['identifier'],v['start']) for v in values} == expected.keys(),
                'Exact common physical population required')
        rows.append(dict(fit_item_id=fit,identifier=model['identifier'],K=model['K'],
            source=spec,scope=scope,accuracy=summarize(values)))
        evidence.append(spec)
    require(len(rows) == len({r['fit_item_id'] for r in rows}) == 35,'All35distinct original fits required')
    result = dict(status='all35_original_full_generators_have_common64case_GPU_accuracy',created_unix=time.time(),
        fits=35,cases_per_fit=64,unique_fit_cases=2240,unique_physical_outputs_scored=sum(r['K'] for r in rows)*64,
        newly_evaluated_fits=9,rows=rows,CPU_neural_forwards=0,
        final_two_phase_timing_qualified=False,full_portfolio_complete=False,
        exposure='All16consumed fitting scenes,four starts,1/60seconds/output;no independent holdout or own-history drift.',
        evidence=[binding(protocol_path),binding(owner_path),*evidence],producer=binding(Path(__file__)))
    output.write_bytes(canonical(result)+b'\n')
    print(result['status'],result['unique_fit_cases'],'fit/cases;',result['unique_physical_outputs_scored'],'physical outputs.')


if __name__ == '__main__':
    run()
