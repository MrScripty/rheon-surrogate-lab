"""Recompute native generator scalars directly from twenty original raw blocks.

This oracle does not import the packet producer or either campaign reducer.
It qualifies packaging and native queries, not independent physical reexecution.
"""
import base64
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PACKETS = RUN/'common-generator-GPU-measurements-01'
NATIVE = RUN/'sources/Tuldok-composition-source-artifacts'
REPORT = DOC/'reports/common-generator-GPU-measurements-qualification-01.json'
ERRORS = ('pressure_relative_l2','velocity_relative_l2','tracer_relative_l2',
          'equation_relative_l2','pressure_absolute_rms','tracer_absolute_rms')
TIMES = ('block_median_ms','per_output_median_ms','per_output_p95_ms','per_output_maximum_ms')


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(condition,message):
    if not condition:
        raise ValueError(message)


def same(left,right):
    return struct.pack('!d',left) == struct.pack('!d',right)


def raw_scalars(a,b,original,k):
    """Decide exact case/horizon/work and independently reduce both raw phases."""
    for row in (a,b):
        require(row['identifier'] == original['identifier'] and row['start'] == original['start']
                and row['counts_xyz'] == original['scene']['counts']
                and row['cells_per_metre'] == original['scene']['cells_per_metre']
                and row['K'] == k and row['output_dt_s'] == 1/60 and row['block_advance_s'] == k/60,
                'Raw physical case, density, dimensions or horizon differs')
        require(len(row['metrics']) == len(row['output_hashes']) == k
                and [m['lead'] for m in row['metrics']] == list(range(1,k+1)),
                'Raw block omits an ordered physical lead')
        require([m['time_s'] for m in row['metrics']] == [r['time_s'] for r in original['reference_leads'][:k]],
                'Raw output timestamps differ from original reference')
        require(len(row['block_ms']) == len(row['per_output_ms']) == 10
                and len(row['warmup_block_ms']) == 3 and row['all_repeat_and_ownership_checks_passed']
                and all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in row['block_ms'])
                and all(same(value,block/k) for block,value in zip(row['block_ms'],row['per_output_ms'],strict=True)),
                'Raw samples, ownership or per-output divisor differs')
    require(a['output_hashes'] == b['output_hashes'] and a['metrics'] == b['metrics'],
            'Original raw phases disagree on returned fields or scores')
    result = {}
    for metric in ERRORS:
        values = [lead['values'][metric] for lead in a['metrics']]
        require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in values),
                'Undefined lead cannot be silently omitted from block error')
        result[metric] = max(values)
    blocks = a['block_ms']+b['block_ms']
    samples = [v/k for v in blocks]
    ordered = sorted(samples)
    # With exactly twenty samples, P95 lies at rank18.05. Spell this independently
    # rather than importing the producer/reducer's quantile implementation.
    rank = .95*19
    p95 = ordered[18]+(ordered[19]-ordered[18])*(rank-18)
    result.update(block_median_ms=statistics.median(blocks),per_output_median_ms=statistics.median(samples),
        per_output_p95_ms=p95,per_output_maximum_ms=ordered[-1])
    return result


def qualify():
    require(not REPORT.exists(),'Preserve original independent table qualification')
    prep_path = PACKETS/'preparation.json'
    prep = read(prep_path)
    require(prep['exact_model_count'] == 10 and prep['measured_scalars'] == 6400
            and prep['original_raw_blocks'] == 20 and len(prep['packets']) == 4,
            'All ten models, four K scopes and 6400 scalar observations required')
    for spec in [prep['producer'],prep['native_table_owner'],prep['source_manifest'],
            *prep['accepted_reductions'],*prep['original_case_selections']]:
        require(binding(ROOT/spec['path']) == spec,'Prepared authority changed')
    populations = [read(ROOT/s['path'])['cases'] for s in prep['original_case_selections']]
    require(populations[0] == populations[1] and len(populations[0]) == 64,'Original common physical population differs')
    original_cases = {(c['identifier'],c['start']):c for c in populations[0]}
    sys.path.insert(0,str(NATIVE))
    from retained_measurements import MeasurementTable,MeasurementComparison
    tables = []
    raw_seen,subjects_seen = set(),set()
    total,queries = 0,[]
    for part in prep['packets']:
        path = PACKETS/part['path']
        raw = path.read_bytes()
        require(binding(path)['sha256'] == part['sha256'] and len(raw) == part['bytes'],'Native packet changed')
        packet = json.loads(raw)
        table = MeasurementTable(raw)
        decoded = {s['id']:json.loads(base64.b64decode(s['data_base64'])) for s in packet['sources']}
        context,evidence = decoded['context'],decoded['reduction']
        k = part['K']
        require(k in (1,2,3,4) and len(packet['subjects']) == (1 if k == 1 else 3)
                and len(packet['cases']) == 64 and context['scope']['physical_outputs'] == k
                and context['scope']['reference_binary_sha256'] == 'ecf8252c88c248603325bf36be19c4aae0633b62a71de60740ab9610117c2283'
                and context['scope']['population'] == 'consumed'
                and {m['id'] for m in packet['metrics']} == set(ERRORS+TIMES),
                'K scope, population, old reference or metric membership differs')
        require([c['original_case'] for c in context['cases']] == populations[0],
                'Packet changed original current fields, controls or reference identities')
        expected = {}
        pooled = {(r['subject_id'],r['case_id']):r for r in evidence['rows']}
        require(len(pooled) == len(evidence['rows']) == 64*len(packet['subjects']),'Exact pooled-row population required')
        for subject in packet['subjects']:
            name = subject['id']
            require(name not in subjects_seen and subject['dimensions']['inference_device'] == 'cuda'
                    and subject['dimensions']['fit_item_id'] == evidence['models'][name]['fit_item_id'],
                    'Original fit identity or GPU device differs')
            subjects_seen.add(name)
            phases = []
            for phase,spec in enumerate(evidence['raw_blocks'][name]):
                raw_id = name+f'-raw-{phase}'
                source = next(s for s in packet['sources'] if s['id'] == raw_id)
                original_path = ROOT/spec['path']
                require(binding(original_path) == spec and original_path.read_bytes() == base64.b64decode(source['data_base64'])
                        and spec['sha256'] not in raw_seen,'Original raw source changed or duplicated')
                raw_seen.add(spec['sha256'])
                d = decoded[raw_id]
                require(d['identifier'] == name and d['phase'] == phase and d['K'] == k
                        and d['fit_item_id'] == subject['dimensions']['fit_item_id']
                        and d['CPU_neural_forwards'] == d['forbidden_forward_attempts'] == 0
                        and d['CUDA_network_calls'] == 64*14 and len(d['rows']) == 64,
                        'Raw model/device/work qualification differs')
                keyed = {(r['identifier'],r['start']):r for r in d['rows']}
                require(len(keyed) == 64 and keyed.keys() == original_cases.keys(),'A raw physical case is missing')
                phases.append(keyed)
            require(len(phases) == 2,'Both original raw phases required')
            for case in context['cases']:
                original = case['original_case']
                key = (original['identifier'],original['start'])
                a,b = phases[0][key],phases[1][key]
                values = raw_scalars(a,b,original,k)
                row = pooled[(name,case['id'])]
                projected = dict(a,block_ms=a['block_ms']+b['block_ms'],per_output_ms=a['per_output_ms']+b['per_output_ms'])
                require(row['original_pooled_case'] == projected
                        and values.keys() == row['scalars'].keys()
                        and all(same(value,row['scalars'][m]) for m,value in values.items()),
                        'Retained pooled samples, lead fields or native block scalars differ from raw calls')
                expected[(name,case['id'])] = values
        case_map = {c['id']:c['dimensions'] for c in packet['cases']}
        for index,c in enumerate(context['cases']):
            dims = case_map[c['id']]
            original = c['original_case']
            require(dims == c['dimensions'] and [dims[n] for n in ('grid_x','grid_y','grid_z')] == original['scene']['counts']
                    and dims['cells_per_metre'] == original['scene']['cells_per_metre'] and dims['K'] == k
                    and dims['start_index'] == original['start'] and dims['trajectory_id'] == original['identifier']
                    and dims['split'] == 'train' and dims['output_dt_s'] == 1/60 and dims['block_duration_s'] == k/60,
                    'Case density, domain, physical cadence or exposure differs')
        positions = {(r['subject_id'],r['case_id']):i for i,r in enumerate(evidence['rows'])}
        for observation in packet['observations']:
            pair = (observation['subject_id'],observation['case_id'])
            metric = observation['metric_id']
            require(observation['state'] == 'measured' and observation['reason'] is None
                    and observation['dimensions'] == dict(samples=20,process_blocks=2)
                    and same(observation['value'],expected[pair][metric])
                    and observation['reference'] == dict(source_id='reduction',pointer=f'/rows/{positions[pair]}/scalars/{metric}'),
                    'Native scalar differs from independently recomputed raw observation')
            total += 1
        filters = [{},dict(cells_per_metre=8),dict(cells_per_metre=16),dict(grid_x=16),dict(grid_x=32),
            dict(start_index=0),dict(start_index=24),dict(start_index=64),dict(start_index=120),
            dict(cells_per_metre=16,grid_x=32,start_index=120)]
        for selection,metric in itertools.product(filters,ERRORS+TIMES):
            q = table.query(metric,filters=selection,limit=1)
            for summary in q['summary']:
                values = [v[metric] for (name,cid),v in expected.items() if name == summary['subject_id']
                    and all(case_map[cid][n] == value for n,value in selection.items())]
                require(summary['count'] == summary['states']['measured'] == len(values)
                        and all(v == 0 for n,v in summary['states'].items() if n != 'measured')
                        and ((not values and summary['maximum'] is None)
                            or (values and same(summary['maximum']['value'],max(values)))),
                        'Native density/grid/start query omits cases or changes extrema')
            queries.append(dict(K=k,metric=metric,filters=selection,selected_rows=q['total']))
        tables.append((dict(origin_catalog_id='a'*32,packet_sha256=table.sha256),table))
    incompatible = 0
    for pair in itertools.combinations(tables,2):
        try:
            MeasurementComparison(list(pair))
        except Exception as error:
            require(getattr(error,'code',None) == 'incompatible','Unexpected cross-K comparison diagnostic')
            incompatible += 1
        else:
            raise ValueError('Native comparison incorrectly pooled different physical horizons')
    require(total == 6400 and len(raw_seen) == 20 and len(subjects_seen) == 10 and incompatible == 6
            and 'numpy' not in sys.modules and 'torch' not in sys.modules,'Incomplete work or numerical runtime imported')
    result = dict(status='ten_complete_GPU_generators_all6400_native_scalars_and400_filter_queries_qualified',
        created_unix=time.time(),measured_scalars=total,original_raw_blocks_retained_once=len(raw_seen),
        exact_models=10,scope_count=4,all_original_physical_leads_retained=True,queries=queries,
        every_block_error_maximum_and_timing_statistic_recomputed_from_raw=True,
        incompatible_K_pair_rejections=incompatible,independent_new_physics_execution=False,
        new_inference=False,new_reference_generation=False,native_mutations=0,native_ingestion_pending=True,
        producer=binding(Path(__file__)),preparation=binding(prep_path),native_owner=binding(NATIVE/'retained_measurements.py'),
        packet_bindings=[binding(PACKETS/p['path']) for p in prep['packets']],full_portfolio_complete=False)
    REPORT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('Qualified 6400 scalars from all20 raw blocks,400 native filter queries,6 cross-K rejections. Native ingestion pending.')


if __name__ == '__main__':
    qualify()
