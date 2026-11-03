"""Bind all16CUDA constructor/accuracy results and reduce every native lead.

Independently checks process/source/case/work/route coverage and recomputes
aggregate extrema from the original per-lead scalars. It does not rerun neural
inference or claim independently recomputed physical fields or latency.
"""
from collections import Counter,defaultdict
import math
from pathlib import Path
import time

from review_water_learning_sequence import ROOT,binding,canonical,read,require

RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROTOCOL = RUN/'inherited-generator-CUDA-qualification-protocol-01.json'
OUTPUT = DOC/'reports/inherited-generator-CUDA-qualification-01.json'
FIELDS = {'pressure','vx','vy','vz','tracer'}


def summarize(rows):
    names = sorted({name for row in rows for lead in row['metrics'] for name in lead['values']})
    worst,undefined,witnesses = {},{},{}
    for name in names:
        values = [(lead['values'][name],row['identifier'],row['start'],lead['lead'])
            for row in rows for lead in row['metrics']]
        undefined[name] = sum(v is None for v,*_ in values)
        valid = [r for r in values if r[0] is not None]
        largest = max(valid) if valid else None
        worst[name] = largest[0] if largest else None
        witnesses[name] = None if largest is None else dict(identifier=largest[1],start=largest[2],lead=largest[3])
    return dict(cases=len(rows),physical_outputs=sum(row['K'] for row in rows),worst_metrics=worst,
        undefined_metrics=undefined,worst_metric_cases=witnesses)


def run():
    require(not OUTPUT.exists(),'Preserve original independent family reduction')
    protocol = read(PROTOCOL)
    base = ROOT/protocol['output']
    owner = read(base/'status.json')
    require(owner['status'] == 'all16_original_workers_complete_reduction_pending' and owner['active'] is None
            and not Path(f'/proc/{owner["pid"]}').exists() and len(owner['completed']) == 16
            and owner['protocol'] == binding(PROTOCOL),'All16original workers and terminal owner required')
    for spec in [protocol['cases'],*protocol['dependencies']]:
        require(binding(ROOT/spec['path']) == spec,'Admitted constructor qualification authority changed')
    original = read(ROOT/protocol['cases']['path'])
    cases = {(c['identifier'],c['start']):c for c in original['cases']}
    models = {m['identifier']:m for m in protocol['models']}
    require(len(cases) == 64 and len(models) == 16
            and {r['identifier'] for r in owner['completed']} == models.keys(),'Exact model/physical population required')
    results,raw_bindings = {},[]
    prediction_calls,physical_outputs,construction_roots = 0,0,0
    for finished in owner['completed']:
        spec = finished['result']
        require(binding(ROOT/spec['path']) == spec,'Original CUDA result changed')
        raw_bindings.append(spec)
        d = read(ROOT/spec['path'])
        name = finished['identifier']
        model = models[name]
        k,readiness = model['K'],model['readiness']
        require(d['status'] == 'all64_cases_and_original_inherited_CUDA_constructor_qualified'
                and d['identifier'] == name and d['fit_item_id'] == model['fit_item_id'] and d['K'] == k
                and d['protocol'] == binding(PROTOCOL) and d['CPU_neural_forwards'] == 0
                and d['inference_latency_qualified'] is False and not d['output_arrays_written']
                and not Path(f'/proc/{d["pid"]}').exists()
                and d['reference_binary_sha256'] == protocol['original_reference_binary_sha256']
                and d['common_reference_method'] == original['reference_method'],
                'Original worker membership,device,reference or untimed scope differs')
        constructor = d['constructor']
        core = read(ROOT/model['project_path']/'expert-assets/pressure/configuration.json')
        expected_queries = readiness['fine_root_queries_in_batches_of4']+core['steps']*sum(
            (n+3)//4 for n in readiness['full_unit_query_counts'] if n > 2048)
        counters = constructor['neural_calls_by_phase']
        require(constructor['CPU_neural_forwards'] == constructor['forbidden_forward_attempts'] == 0
                and constructor['scoped_hooks_removed'] and constructor['original_final_tensor_inventory_exact']
                and constructor['all_parameters_and_buffers_CUDA']
                and counters['construction']['FrozenSpatial'] == expected_queries
                and counters['prediction']['BundleNet'] == 128
                and d['peak_CUDA_allocated_bytes'] <= 2*1024**3,
                'Actual CUDA setup/prediction work,learned tensors or cleanup differs')
        maps = constructor['derived_GPU_maps']
        require(sum(r['bytes'] for r in maps if r['name'].endswith('_fine_matrix')) == readiness['fine_derived_matrix_bytes']
                and sum(r['bytes'] for r in maps if r['name'].endswith('_full_matrix')) == readiness['full_derived_matrix_bytes'],
                'GPU-built derived fine/full buffer coverage differs')
        require(all(len(r['sha256']) == 64 and all(c in '0123456789abcdef' for c in r['sha256']) for r in maps),
                'Every GPU-built map needs a retained byte identity')
        observed,routes = set(),Counter()
        for row in d['rows']:
            key = row['identifier'],row['start']
            case = cases[key]
            require(key not in observed and row['K'] == k and row['counts_xyz'] == case['scene']['counts']
                    and row['cells_per_metre'] == case['scene']['cells_per_metre']
                    and row['output_dt_s'] == 1/60 and row['block_advance_s'] == k/60
                    and row['repeat_and_ownership_checks_passed'] and len(row['metrics']) == len(row['complete_output_hashes']) == k,
                    'Missing,duplicated or mismatched native physical case/output')
            observed.add(key)
            cells = math.prod(case['scene']['counts'])
            if readiness['full_unit_query_counts']:
                expected_route = 'complete_F64_unit_map' if cells in readiness['full_unit_query_counts'] else 'core_CNN_fallback'
            else:
                expected_route = 'fine_F64_unit_map' if cells in readiness['fine_unit_query_counts'] else 'fine_F32_CNN'
            require(row['pressure_route'] == expected_route,'Actual route disagrees with captured geometry/buffer authority')
            routes[expected_route] += 1
            for lead,(metric,hashes) in enumerate(zip(row['metrics'],row['complete_output_hashes'],strict=True),1):
                require(metric['lead'] == lead and metric['time_s'] == case['reference_leads'][lead-1]['time_s']
                        and set(hashes) == FIELDS
                        and all(len(v) == 64 and all(c in '0123456789abcdef' for c in v) for v in hashes.values())
                        and all(v is None or (type(v) in (int,float) and math.isfinite(v) and v >= 0) for v in metric['values'].values()),
                        'Physical timestamp,field hash or metric scalar differs')
        require(observed == cases.keys(),'Worker omitted a common case')
        if readiness['full_unit_query_counts']:
            require(routes == Counter(complete_F64_unit_map=48,core_CNN_fallback=16)
                    and counters['prediction']['FullLearnedPressureMap'] == 128*k
                    and counters['prediction']['FrozenSpatial'] == 64*k,
                    'Full-map and32cubed fallback paths must both actually execute')
        else:
            require(routes == Counter(fine_F64_unit_map=32,fine_F32_CNN=32)
                    and counters['prediction']['FrozenPressureCore'] == 128
                    and counters['prediction']['FrozenSpatial'] == 128,
                    'Rectangular fine-map and cube CNN paths must both actually execute')
        probe = d['original_saved_probe_metric_comparison']
        require(binding(ROOT/probe['probe']['path']) == probe['probe'] and len(probe['comparisons']) == k,
                'Original saved-probe identity or lead coverage differs')
        for expected_lead,comparison in enumerate(probe['comparisons'],1):
            require(comparison['lead'] == expected_lead,'Saved-probe physical lead differs')
            a,b = comparison['original_saved_CPU_metrics'],comparison['fresh_GPU_metrics']
            require(a.keys() == b.keys() and all((v is None and b[n] is None) or
                (v is not None and b[n] is not None and math.isclose(v,b[n],rel_tol=5e-4,abs_tol=5e-6)) for n,v in a.items()),
                'Original saved CPU probe metric tolerance failed')
        def grouped(key):
            groups = defaultdict(list)
            for row in d['rows']:
                groups[key(row)].append(row)
            return {label:summarize(rows) for label,rows in groups.items()}
        results[name] = dict(fit_item_id=model['fit_item_id'],K=k,architecture=model['architecture'],
            summary=summarize(d['rows']),by_start=grouped(lambda r:str(r['start'])),
            by_grid_density=grouped(lambda r:str(r['cells_per_metre'])),
            by_density_and_dimensions=grouped(lambda r:str(r['cells_per_metre'])+'cpm-'+'x'.join(map(str,r['counts_xyz']))),
            by_pressure_route=grouped(lambda r:r['pressure_route']),route_cases=dict(routes),
            constructor_CUDA_unit_query_batches=expected_queries,CPU_parameters_moved=constructor['CPU_parameters_moved'],
            peak_CUDA_allocated_bytes=d['peak_CUDA_allocated_bytes'],raw_result=spec)
        prediction_calls += 128
        physical_outputs += 128*k
        construction_roots += expected_queries
    require(prediction_calls == protocol['expected_prediction_calls'] == 2048
            and physical_outputs == protocol['expected_physical_outputs_including_repeat'] == 4608,
            'All16models and repeated physical work required')
    value = dict(status='all16_inherited_CUDA_constructors_and64case_accuracy_coverage_reduced',created_unix=time.time(),
        models=results,models_qualified=16,physical_cases_per_model=64,complete_prediction_calls=prediction_calls,
        physical_outputs_including_repeat=physical_outputs,constructor_CUDA_unit_query_batches=construction_roots,
        CPU_neural_forwards=0,all_learned_final_tensors_exact=True,all_required_matrix_and_CNN_routes_exercised=True,
        inference_latency_qualified=False,independent_neural_or_physical_reexecution=False,
        reduction_scope='Independent original process/source/case/device/work/route coverage and extrema over retained per-lead scalars.',
        precision_variant='GPU-built F64derived maps/bases;no bitwise CPU-derived-map equivalence claim.',
        exposure='Original16fitting scenes/four starts;consumed diagnostic only,no independent holdout or own-history drift.',
        native_measurement_ingestion_complete=False,new_reference_generation=False,output_arrays_written=False,
        full_portfolio_complete=False,producer=binding(Path(__file__)),
        evidence=[binding(PROTOCOL),binding(base/'status.json'),*raw_bindings])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    for name,model in results.items():
        print(name,'K',model['K'],'worst pressure relativeL2',model['summary']['worst_metrics']['pressure_relative_l2'])


if __name__ == '__main__':
    run()
