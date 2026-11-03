"""Join retained common benchmarks to exact fit identities without executing models.

Pressure-only and complete-state measurements remain different groups. A fit
with a qualified default benchmark may still need intended solver variants,
checkpoint states, reproduction, holdout and drift testing.
"""
from collections import Counter,defaultdict
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,DOC,PROJECT,read,binding,require,save

REPORT = DOC/'reports/common-fit-benchmark-coverage-01.json'


def run():
    require(not REPORT.exists(),'Preserve the original common-fit coverage report')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_binding = binding(index_path)
    index = read(index_path)
    pressure_path = DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    accuracy_path = DOC/'reports/pressure-common-score-01.json'
    generator_path = DOC/'reports/complete-generator-GPU-benchmark-01.json'
    pressure,accuracy,generators = read(pressure_path),read(accuracy_path),read(generator_path)
    require(pressure['status'] == 'all127_named_pressure_models_two_pass_GPU_timings_and_exact_accuracy_join_verified'
            and accuracy['status'] == 'all127_available_named_checkpoints_scored_on_same128_native_requests'
            and generators['status'] == 'all35_original_generators_common_GPU_primary_and_hook_calibration_reduced'
            and not pressure['CPU_neural_forwards'] and not generators['CPU_neural_forwards'],
            'Accepted common GPU benchmark authorities required')
    fits = {f['item_id']:f for f in index['fits']}
    require(len(fits) == len(index['fits']) == 205,'Exact205fit inventory required')
    aliases = defaultdict(list)
    for alias in index['aliases']:
        if alias['kind'] == 'model_alias':
            aliases[alias['label']].append(alias)
    measured = defaultdict(list)
    for name,model in pressure['models'].items():
        score = accuracy['models'][name]
        candidates = [a for a in aliases[name] if a['declared_weights_sha256'] == model['weights_sha256']]
        require(len(candidates) == 1 and candidates[0]['fit_item_id'] in fits,'Pressure alias must join exactly one inventoried fit')
        fit_id = candidates[0]['fit_item_id']
        require(score['weights_sha256'] == model['weights_sha256']
                and any(c['sha256'] == model['weights_sha256'] for c in fits[fit_id]['checkpoints'])
                and score['cases'] == len(model['rows']) == 128,'Pressure checkpoint/case population differs')
        for metric,value in score['worst'].items():
            require(max(r['accuracy'][metric] for r in model['rows']) == value,'Pressure timing/accuracy worst metric differs')
        measured[fit_id].append(dict(group='pressure_request',identifier=name,cases=128,weights_sha256=model['weights_sha256'],
            execution_variant=model['execution_variant'],timing_units='GPU ms/pressure request; not a simulation step',
            benchmark_authority=binding(pressure_path),accuracy_authority=binding(accuracy_path)))
    for model in generators['models']:
        fit_id = model['fit_item_id']
        require(fit_id in fits and fits[fit_id]['task'] == 'full_state_generator'
                and len(model['pooled_rows']) == generators['cases_per_model'] == 64,'Full generator fit/case population differs')
        measured[fit_id].append(dict(group='complete_state',identifier=model['identifier'],K=model['K'],cases=64,
            timing_units='GPU ms/block and GPU ms/physical output; full CPU input to owned CPU output',
            benchmark_authority=binding(generator_path)))
    rows = []
    for fit_id,fit in sorted(fits.items()):
        scopes = measured[fit_id]
        require(len(scopes) <= 1,'One fit unexpectedly maps to multiple default benchmark identities')
        rows.append(dict(fit_item_id=fit_id,name=fit['name'],task=fit['task'],named_entries=fit['named_entries'],
            status='default_common_GPU_accuracy_and_timing_available' if scopes else 'common_GPU_benchmark_pending',
            default_benchmark=scopes[0] if scopes else None,checkpoint_states=len(fit['checkpoints']),
            complete_variant_and_checkpoint_coverage_qualified=False,clean_retraining_qualified=False))
    pending = [r for r in rows if r['default_benchmark'] is None]
    require(len(measured) == 205 and sum(bool(r['default_benchmark']) for r in rows) == 162 and len(pending) == 43,
            'Measured35generator+127pressurefits and43pendingfits expected')
    require(binding(index_path) == index_binding,'Fit inventory changed during read-only coverage join')
    save(REPORT,dict(status='all205_fit_identities_joined_to162_default_common_benchmarks_and43_pending',created_unix=time.time(),
        fit_identities=205,default_common_benchmark_fits=162,complete_state_generator_fits=35,pressure_request_fits=127,
        common_benchmark_pending_fits=43,pending_by_task=dict(sorted(Counter(r['task'] for r in pending).items())),
        rows=rows,pending_fit_ids=[r['fit_item_id'] for r in pending],
        scope='Exact fit-identity coverage for retained default common GPU accuracy/timing. Two metric populations remain distinct. No claim of complete524state or intended-variant coverage, holdout, long drift or clean reproduction.',
        source_fit_index=index_binding,evidence=[binding(p) for p in [pressure_path,accuracy_path,generator_path]],
        new_model_forwards=0,native_mutations=0,weight_or_dataset_copies=0,full_portfolio_complete=False,producer=binding(Path(__file__))))
    print('Exact fit coverage:127pressure-only +35full-state defaults;43fits pending; variants/checkpoint coverage remains open.',flush=True)


if __name__ == '__main__':
    run()
