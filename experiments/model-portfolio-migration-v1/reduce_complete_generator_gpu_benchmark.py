"""Check all70original results, pool both phases and preserve case/lead metrics.

This independently reduces measured scalars/work and original output hashes;
it does not claim independently rerun physics, continuous GPU exclusivity or
holdout generalization. Historical instrumented times remain separate.
"""
from collections import defaultdict
import math
from pathlib import Path
import statistics
import time

from reduce_joint_generator_gpu_benchmark import summarize
from review_water_learning_sequence import ROOT, binding, canonical, read, require

RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROTOCOL = RUN/'complete-generator-GPU-protocol-01.json'
OUTPUT = DOC/'reports/complete-generator-GPU-benchmark-01.json'
FIELDS = {'pressure','vx','vy','vz','tracer'}
METRICS = {name+suffix for name in FIELDS for suffix in ('_relative_l2','_absolute_rms')} | {
    'velocity_relative_l2','equation_relative_l2'}


def validate_row(row,case,model,protocol,phase):
    k = model['K']
    require(row['counts_xyz'] == case['scene']['counts'] and row['cells_per_metre'] == case['scene']['cells_per_metre']
            and row['K'] == k and row['output_dt_s'] == 1/60 and row['block_advance_s'] == k/60
            and row['repeat_and_ownership_checks_passed'], 'Native geometry,physical advance or ownership differs')
    require(len(row['block_ms']) == len(row['per_output_ms']) == protocol['samples']
            and len(row['warmup_block_ms']) == protocol['warmups']
            and len(row['calibration']) == protocol['calibration_pairs']
            and len(row['metrics']) == len(row['output_hashes']) == k,
            'Missing primary,warmup,calibration or physical lead')
    require(all(math.isfinite(v) and v > 0 for v in [row['first_call_block_ms'],*row['warmup_block_ms'],*row['block_ms']]),
            'Invalid measured block time')
    for block,step in zip(row['block_ms'],row['per_output_ms'],strict=True):
        require(step == block/k,'Divide block latency only by physical Koutputs')
    for pair,c in enumerate(row['calibration']):
        require(c['order'] == (['on','off'] if (pair+phase)%2 == 0 else ['off','on'])
                and all(math.isfinite(c[n]) and c[n] > 0 for n in ('hook_on_ms','hook_off_ms')),
                'Calibration order or timing differs')
    for lead,(metric,hashes) in enumerate(zip(row['metrics'],row['output_hashes'],strict=True),1):
        require(metric['lead'] == lead and metric['time_s'] == case['reference_leads'][lead-1]['time_s']
                and set(metric['values']) == METRICS
                and set(hashes) == FIELDS
                and all(len(v) == 64 and all(c in '0123456789abcdef' for c in v) for v in hashes.values())
                and all(v is None or (math.isfinite(v) and v >= 0) for v in metric['values'].values()),
                'Native physical timestamp,field hash or scalar differs')


def historical_results(protocol):
    result = {}
    for family in ('joint','sharing'):
        path = RUN/f'{family}-generator-GPU-benchmark-protocol-01.json'
        previous = read(path)
        for block in previous['process_blocks']:
            if block['phase'] != 0:
                continue
            path = ROOT/block['result']
            d = read(path)
            result[d['fit_item_id']] = dict(binding=binding(path),rows=d['rows'],hash_key='output_hashes',timing_instrumented=True)
    for model in read(RUN/'inherited-generator-CUDA-qualification-protocol-01.json')['models']:
        path = RUN/'inherited-generator-CUDA-qualification-01'/(model['identifier']+'.json')
        d = read(path)
        result[d['fit_item_id']] = dict(binding=binding(path),rows=d['rows'],hash_key='complete_output_hashes',timing_instrumented=None)
    require(len(result) == 26,'All26previously measured identities required')
    return result


def calibration(rows):
    pairs = [c for row in rows for c in row['calibration']]
    return dict(pairs=len(pairs),median_paired_hook_on_minus_off_block_ms=statistics.median(
        c['hook_on_ms']-c['hook_off_ms'] for c in pairs),
        median_hook_on_block_ms=statistics.median(c['hook_on_ms'] for c in pairs),
        median_hook_off_block_ms=statistics.median(c['hook_off_ms'] for c in pairs),
        scope='Alternating order same-case pairs; observed callback difference,not attribution of all historical speed changes.')


def run():
    require(not OUTPUT.exists(),'Preserve original independent complete reduction')
    protocol = read(PROTOCOL)
    base = ROOT/protocol['output']
    owner = read(base/'status.json')
    require(owner['status'] == 'all70_original_workers_complete_reduction_pending' and owner['active'] is None
            and owner['protocol'] == binding(PROTOCOL) and len(owner['completed']) == 70
            and not Path(f'/proc/{owner["pid"]}').exists(),'Original70workers must be terminal')
    for spec in protocol['dependencies']:
        require(binding(ROOT/spec['path']) == spec,'Admitted shared authority changed')
    prepared = read(ROOT/protocol['cases']['path'])
    cases = {(c['identifier'],c['start']):c for c in prepared['cases']}
    models = {m['fit_item_id']:m for m in protocol['models']}
    require(len(models) == 35 and len(cases) == 64,'Exact35fit/64casepopulation required')
    previous = historical_results(protocol)
    admitted = {(b['fit_item_id'],b['phase'],b['result']) for b in protocol['process_blocks']}
    require(admitted == {(b['fit_item_id'],b['phase'],b['result']) for b in owner['completed']},'Original worker membership differs')
    reports,evidence,competing = defaultdict(dict),[],[]
    primary,total,outputs,guarded = 0,0,0,0
    for completed in owner['completed']:
        path = ROOT/completed['result']
        require(binding(path) == completed['result_binding'],'Raw original worker changed')
        d = read(path)
        fit,phase = d['fit_item_id'],d['phase']
        model = models[fit]
        require(d['status'] == 'all64_original_cases_uninstrumented_primary_and_paired_hooks_measured'
                and d['protocol'] == binding(PROTOCOL) and d['K'] == model['K']
                and d['identifier'] == model['identifier'] and d['phase'] == completed['phase']
                and phase not in reports[fit] and len(d['rows']) == 64
                and d['CPU_neural_forwards'] == d['forbidden_forward_attempts'] == 0
                and not d['primary_validation_hook_active']
                and d['guarded_network_calls'] == 64*(1+protocol['warmups']+protocol['calibration_pairs'])
                and d['logical_primary_calls'] == 64*protocol['samples']
                and d['logical_calibration_off_calls'] == 64*protocol['calibration_pairs']
                and d['logical_complete_calls'] == 64*(1+protocol['warmups']+protocol['samples']+2*protocol['calibration_pairs'])
                and d['peak_CUDA_allocated_bytes'] <= protocol['max_CUDA_allocated_bytes']
                and d['original_reference_binary_sha256'] == protocol['reference_binary_sha256'],
                'Worker device,timing,constructor,reference or exact work differs')
        for spec in model['bound_inputs']:
            require(binding(ROOT/spec['path']) == spec,'Original model input changed')
        constructor = d['constructor']
        require(constructor['CPU_neural_forwards'] == 0 and constructor['scoped_hooks_removed'],
                'CUDA-only setup and hook removal required')
        if 'readiness' in model:
            require(constructor['original_final_tensor_inventory_exact'] and constructor['all_parameters_and_buffers_CUDA']
                    and constructor['neural_calls_by_phase']['construction']['FrozenSpatial'] == constructor['expected_constructor_FrozenSpatial_calls'],
                    'Original inherited CUDA setup work differs')
        else:
            require(constructor['constructor_neural_forwards'] == 0,'Plain constructor unexpectedly executed a neural forward')
        seen = set()
        for row in d['rows']:
            key = row['identifier'],row['start']
            require(key in cases and key not in seen,'Missing,duplicate or unknown physical case')
            validate_row(row,cases[key],model,protocol,phase)
            seen.add(key)
        require(seen == cases.keys(),'Original worker omitted a common case')
        for where in ('resources_before','resources_after'):
            if d[where]['other_compute_pids']:
                competing.append(dict(fit_item_id=fit,phase=phase,where=where,pids=d[where]['other_compute_pids']))
        primary += d['logical_primary_calls']
        total += d['logical_complete_calls']
        outputs += d['logical_primary_calls']*model['K']
        guarded += d['guarded_network_calls']
        reports[fit][phase] = d
        evidence.append(binding(path))
    require(primary == protocol['expected_primary_calls'] and total == protocol['expected_logical_complete_calls']
            and outputs == protocol['expected_primary_physical_outputs'],'Aggregate actual work differs')
    results = []
    for fit,pair in reports.items():
        require(set(pair) == {0,1},'Both independent process phases required')
        left = {(r['identifier'],r['start']):r for r in pair[0]['rows']}
        right = {(r['identifier'],r['start']):r for r in pair[1]['rows']}
        old = {(r['identifier'],r['start']):r for r in previous[fit]['rows']} if fit in previous else None
        pooled = []
        for key,row in left.items():
            other = right[key]
            require(row['output_hashes'] == other['output_hashes'] and row['metrics'] == other['metrics'],
                    'Independent phases changed numerical outputs')
            if old is not None:
                require(row['output_hashes'] == old[key][previous[fit]['hash_key']],
                        'New timing boundary changed previously qualified physical output bytes')
            pooled.append(dict(row,block_ms=row['block_ms']+other['block_ms'],
                per_output_ms=row['per_output_ms']+other['per_output_ms'],calibration=row['calibration']+other['calibration']))
        def groups(key):
            selected = defaultdict(list)
            for row in pooled:
                selected[key(row)].append(row)
            return {k:summarize(rows) for k,rows in sorted(selected.items())}
        results.append(dict(fit_item_id=fit,identifier=models[fit]['identifier'],K=models[fit]['K'],
            GPU=pair[0]['GPU'],device='CUDA',summary=summarize(pooled),calibration=calibration(pooled),
            by_cells_per_metre=groups(lambda r:str(r['cells_per_metre'])),
            by_grid_dimensions=groups(lambda r:'x'.join(map(str,r['counts_xyz']))),
            by_density_and_grid=groups(lambda r:f'{r["cells_per_metre"]}cpm:'+ 'x'.join(map(str,r['counts_xyz']))),
            phase_summaries={str(p):summarize(d['rows']) for p,d in pair.items()},
            prior_physical_output_hashes_exact=old is not None,
            prior_result=previous[fit]['binding'] if fit in previous else None,pooled_rows=pooled))
    result = dict(status='all35_original_generators_common_GPU_primary_and_hook_calibration_reduced',
        created_unix=time.time(),models_count=35,cases_per_model=64,process_blocks=70,primary_calls=primary,
        primary_physical_outputs=outputs,logical_complete_calls=total,observed_guarded_prediction_calls=guarded,
        models=sorted(results,key=lambda r:(r['K'],r['fit_item_id'])),original26_output_hashes_exact=True,
        CPU_neural_forwards=0,competing_GPU_observations=competing,
        timing_scope=protocol['timing'],resource_scope=protocol['contention'],exposure=protocol['exposure'],
        historical_instrumented_packets_preserved=True,full_portfolio_complete=False,
        evidence=[binding(PROTOCOL),binding(base/'status.json'),*evidence],producer=binding(Path(__file__)))
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print(result['status'],primary,'primary calls/',outputs,'physical outputs.')


if __name__ == '__main__':
    run()
