"""Reduce all512matched F64 cases and compare with the35original fit results."""
from collections import defaultdict
from pathlib import Path
import time

from reduce_complete_generator_gpu_benchmark import validate_row
from reduce_joint_generator_gpu_benchmark import summarize
from review_water_learning_sequence import ROOT, binding, canonical, read, require

RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROTOCOL = RUN/'same-case-F64-GPU-protocol-01.json'
OUTPUT = DOC/'reports/same-case-F64-GPU-benchmark-01.json'
COMPARISON = DOC/'reports/matched-generator-F64-comparison-01.json'


def run():
    require(not OUTPUT.exists() and not COMPARISON.exists(),'Preserve original matched F64 reduction')
    protocol = read(PROTOCOL)
    base = ROOT/protocol['output']
    owner = read(base/'status.json')
    require(owner['status'] == 'all8_original_matched_F64_workers_complete_reduction_pending'
            and owner['active'] is None and len(owner['completed']) == 8
            and owner['protocol'] == binding(PROTOCOL) and not Path(f'/proc/{owner["pid"]}').exists(),
            'Original8F64workers must be terminal')
    for spec in protocol['dependencies']:
        require(binding(ROOT/spec['path']) == spec,'Preregistered F64 authority changed')
    prepared = read(ROOT/protocol['cases']['path'])
    cases = {(c['identifier'],c['start']):c for c in prepared['cases']}
    require(len(cases) == 64,'Original64native cases required')
    admitted = {(b['K'],b['phase'],b['result']) for b in protocol['process_blocks']}
    require(admitted == {(b['K'],b['phase'],b['result']) for b in owner['completed']},'Original F64worker membership differs')
    reports,evidence,competing = defaultdict(dict),[],[]
    calls,outputs = 0,0
    imported = {s['path'] for s in protocol['dependencies']}
    for block in owner['completed']:
        path = ROOT/block['result']
        require(binding(path) == block['result_binding'],'Raw original F64worker changed')
        d = read(path)
        k,phase = d['K'],d['phase']
        require(k == block['K'] and phase == block['phase'] and phase not in reports[k]
                and d['status'] == 'all64_original_cases_F64_N1_GPU_measured'
                and d['protocol'] == binding(PROTOCOL) and len(d['rows']) == 64
                and d['CPU_neural_forwards'] == d['forbidden_forward_attempts'] == 0
                and not d['primary_validation_hook_active'] and not d['reference_fields_passed_to_model']
                and d['cleanup_complete'] and d['peak_CUDA_allocated_bytes'] <= protocol['max_CUDA_allocated_bytes']
                and d['original_reference_binary_sha256'] == protocol['original_reference_binary_sha256']
                and d['imported_project_sources'] and set(d['imported_project_sources']) <= imported,
                'Original F64device,scope,source closure,reference or work differs')
        seen = set()
        for row in d['rows']:
            key = row['identifier'],row['start']
            require(key in cases and key not in seen and row['actual_internal_advances'] == k
                    and row['graph_geometry_setup_seconds'] >= 0,'Original F64case/advance/setup differs')
            validate_row(dict(row,calibration=[]),cases[key],dict(K=k),dict(protocol,calibration_pairs=0),phase)
            seen.add(key)
            calls += len(row['block_ms'])
            outputs += len(row['block_ms'])*k
        require(seen == cases.keys(),'F64worker omitted a common case')
        for where in ('resources_before','resources_after'):
            if d[where]['other_compute_pids']:
                competing.append(dict(K=k,phase=phase,where=where,pids=d[where]['other_compute_pids']))
        reports[k][phase] = d
        evidence.append(binding(path))
    require(calls == protocol['expected_primary_calls'] == 5120
            and outputs == protocol['expected_primary_physical_outputs'] == 12800,'Complete actual F64work differs')
    require(len({str(d['original_parameter_inventory']) for pair in reports.values() for d in pair.values()}) == 1,
            'Kvariants/phases must retain the exact same original F64parameter inventory')
    results,lookup = [],{}
    for k,pair in sorted(reports.items()):
        require(set(pair) == {0,1},'Both independent F64phases required')
        left = {(r['identifier'],r['start']):r for r in pair[0]['rows']}
        right = {(r['identifier'],r['start']):r for r in pair[1]['rows']}
        pooled = []
        for key,row in left.items():
            other = right[key]
            require(row['output_hashes'] == other['output_hashes'] and row['metrics'] == other['metrics'],
                    'Independent F64phases changed physical outputs')
            pooled.append(dict(row,block_ms=row['block_ms']+other['block_ms'],per_output_ms=row['per_output_ms']+other['per_output_ms']))
        lookup[k] = left
        def groups(key):
            population = defaultdict(list)
            for row in pooled:
                population[key(row)].append(row)
            return {name:summarize(rows) for name,rows in sorted(population.items())}
        results.append(dict(K=k,assembly_execution_id=pair[0]['assembly_execution_id'],GPU=pair[0]['GPU'],
            device='CUDA',summary=summarize(pooled),pooled_rows=pooled,
            by_cells_per_metre=groups(lambda r:str(r['cells_per_metre'])),
            by_grid_dimensions=groups(lambda r:'x'.join(map(str,r['counts_xyz']))),
            by_density_and_grid=groups(lambda r:f'{r["cells_per_metre"]}cpm:'+ 'x'.join(map(str,r['counts_xyz']))),
            phase_summaries={str(p):summarize(d['rows']) for p,d in pair.items()}))
    for k,rows in lookup.items():
        for key,row in rows.items():
            require(row['output_hashes'] == lookup[4][key]['output_hashes'][:k],
                    'Sequential original N1prefix must agree across Koutput requests')
    result = dict(status='all_K1to4_F64_cell_same64_native_cases_reduced',created_unix=time.time(),
        execution='Original cell-coordinate native entry; sequential N1history branch; N8coordinate change unused.',
        cases_per_K=64,process_blocks=8,primary_calls=calls,primary_physical_outputs=outputs,
        models=results,CPU_neural_forwards=0,all_repeated_and_phase_output_hashes_exact=True,
        sequential_K_prefixes_exact=True,competing_GPU_observations=competing,
        temporal_matched_contract=protocol['unsupported_temporal'],full_portfolio_complete=False,
        timing_scope=protocol['timing'],exposure=protocol['reference_scope'],
        original_parameter_inventory=reports[1][0]['original_parameter_inventory'],
        evidence=[binding(PROTOCOL),binding(base/'status.json'),*evidence],producer=binding(Path(__file__)))
    OUTPUT.write_bytes(canonical(result)+b'\n')
    full_path = DOC/'reports/complete-generator-GPU-benchmark-01.json'
    full = read(full_path)
    require(full['status'] == 'all35_original_generators_common_GPU_primary_and_hook_calibration_reduced',
            'Complete original35generator reduction required')
    baselines = {m['K']:m for m in results}
    compared = []
    for model in full['models']:
        baseline = baselines[model['K']]
        require(model['GPU'] == baseline['GPU'],'Compared GPUhardware differs')
        compared.append(dict(fit_item_id=model['fit_item_id'],identifier=model['identifier'],K=model['K'],
            fit_summary=model['summary'],F64_summary=baseline['summary'],
            median_time_ratio_fit_over_F64=model['summary']['median_case_median_per_output_ms']/baseline['summary']['median_case_median_per_output_ms']))
    comparison = dict(status='all35_original_fit_summaries_compared_with_same_K_F64_native_baseline',created_unix=time.time(),
        rows=compared,cases_per_configuration=64,device='CUDA',
        matching='Identical original CPUstates/known scene controls/native targets,dt1/60,allKphysical outputs,per-lead field/equation scores and primary hooks-off CPUtoCPUtiming.',
        limits='Different execution backends,precision,weights and integration structure. Sequential graph baseline is not learned joint generation. Before/after resources cannot prove continuous exclusivity; all16scenes consumed by fitting ancestry. Original temporal fine-clock scene contract is unsupported here.',
        evidence=[binding(full_path),binding(OUTPUT)],full_portfolio_complete=False,producer=binding(Path(__file__)))
    COMPARISON.write_bytes(canonical(comparison)+b'\n')
    print(result['status'],'5120primary calls/12800physical outputs;35matchedfit comparisons retained.')


if __name__ == '__main__':
    run()
