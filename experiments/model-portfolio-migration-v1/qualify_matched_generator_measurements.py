"""Independently recompute24960native scalars from all78original raw blocks."""
import base64
import itertools
import json
from pathlib import Path
import sys
import time

from qualify_common_generator_measurements import binding, read, require, same, raw_scalars, ERRORS, TIMES

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PACKETS = RUN/'matched-generator-GPU-measurements-01'
NATIVE = RUN/'sources/Tuldok-composition-source-artifacts'
REPORT = DOC/'reports/matched-generator-GPU-measurements-qualification-01.json'


def qualify():
    require(not REPORT.exists(),'Preserve original independent matched table qualification')
    prep_path = PACKETS/'preparation.json'
    prep = read(prep_path)
    require(prep['exact_fit_count'] == 35 and prep['F64_assembly_executions'] == 4
            and prep['exact_subject_count'] == 39 and prep['measured_scalars'] == 24960
            and prep['original_raw_blocks'] == 78 and len(prep['packets']) == 4,'Complete matched population required')
    for spec in [prep['producer'],prep['native_table_owner'],prep['source_manifest'],*prep['accepted_reductions']]:
        require(binding(ROOT/spec['path']) == spec,'Prepared matched authority changed')
    common = read(RUN/'complete-generator-GPU-protocol-01.json')
    originals = read(ROOT/common['cases']['path'])['cases']
    original_cases = {(c['identifier'],c['start']):c for c in originals}
    sys.path.insert(0,str(NATIVE))
    from retained_measurements import MeasurementTable,MeasurementComparison
    tables,queries = [],[]
    raw_seen,subjects_seen,fit_seen = set(),set(),set()
    total,assemblies = 0,0
    for part in prep['packets']:
        path = PACKETS/part['path']
        raw = path.read_bytes()
        require(binding(path)['sha256'] == part['sha256'] and len(raw) == part['bytes'],'Prepared matched packet changed')
        table = MeasurementTable(raw)
        packet = json.loads(raw)
        decoded = {s['id']:json.loads(base64.b64decode(s['data_base64'])) for s in packet['sources']}
        context,evidence = decoded['context'],decoded['reduction']
        k = part['K']
        require(len(packet['cases']) == 64 and context['scope']['physical_outputs'] == k
                and context['scope']['reference_binary_sha256'] == common['reference_binary_sha256']
                and context['scope']['population'] == 'consumed'
                and [c['original_case'] for c in context['cases']] == originals
                and {m['id'] for m in packet['metrics']} == set(ERRORS+TIMES),'Exact population,scope and metric membership required')
        pooled = {(r['subject_id'],r['case_id']):r for r in evidence['rows']}
        require(len(pooled) == len(evidence['rows']) == 64*len(packet['subjects']),'All subject/case pairs required')
        expected = {}
        for subject in packet['subjects']:
            identity = subject['id']
            dims = subject['dimensions']
            require(identity not in subjects_seen and identity == dims['fit_or_execution_id']
                    and dims['inference_device'] == 'cuda' and 'GPU' in subject['label'], 'Unique explicit GPU identity required')
            subjects_seen.add(identity)
            model = evidence['models'][identity]
            kind = model['origin_kind']
            require(kind == dims['origin_kind'] and kind in ('original_fit','assembly_execution'),'Scientific origin kind differs')
            if kind == 'original_fit':
                require(identity == model['original_model']['fit_item_id'],'Original fit identity changed')
                fit_seen.add(identity)
            else:
                require(not model['original_model']['original_fit_identity_invented'],'An assembly cannot invent a fit identity')
                assemblies += 1
            phases = []
            for phase,spec in enumerate(model['raw_process_blocks']):
                source = next(s for s in packet['sources'] if s['id'] == identity+f'-raw-{phase}')
                require(binding(ROOT/spec['path']) == spec and (ROOT/spec['path']).read_bytes() == base64.b64decode(source['data_base64'])
                        and spec['sha256'] not in raw_seen,'Original raw block changed or duplicated')
                raw_seen.add(spec['sha256'])
                d = decoded[source['id']]
                require(d['K'] == k and d['phase'] == phase and d['CPU_neural_forwards'] == d['forbidden_forward_attempts'] == 0
                        and not d['primary_validation_hook_active'] and len(d['rows']) == 64,'Original raw device/timing/work differs')
                if kind == 'original_fit':
                    require(d['fit_item_id'] == identity and d['status'] == 'all64_original_cases_uninstrumented_primary_and_paired_hooks_measured',
                            'Original fit result differs')
                else:
                    require(d['assembly_execution_id'] == model['original_model']['assembly_execution_id']
                            and d['status'] == 'all64_original_cases_F64_N1_GPU_measured','Original assembly result differs')
                keyed = {(r['identifier'],r['start']):r for r in d['rows']}
                require(len(keyed) == 64 and keyed.keys() == original_cases.keys(),'Raw physical case omitted')
                phases.append(keyed)
            require(len(phases) == 2,'Both original raw process phases required')
            for case in context['cases']:
                original = case['original_case']
                key = original['identifier'],original['start']
                a,b = phases[0][key],phases[1][key]
                # Only translate the renamed ownership field. Reuse the frozen
                # independent raw statistic oracle, not the packet producer.
                values = raw_scalars(dict(a,all_repeat_and_ownership_checks_passed=a['repeat_and_ownership_checks_passed']),
                    dict(b,all_repeat_and_ownership_checks_passed=b['repeat_and_ownership_checks_passed']),original,k)
                retained = pooled[(identity,case['id'])]
                merged = retained['original_pooled_case']
                require(merged['block_ms'] == a['block_ms']+b['block_ms']
                        and merged['per_output_ms'] == a['per_output_ms']+b['per_output_ms']
                        and merged['metrics'] == a['metrics'] and merged['output_hashes'] == a['output_hashes']
                        and values.keys() == retained['scalars'].keys()
                        and all(same(v,retained['scalars'][name]) for name,v in values.items()),
                        'Pooled original fields,samples or scalar differs from independent raw reduction')
                expected[(identity,case['id'])] = values
        case_map = {c['id']:c['dimensions'] for c in packet['cases']}
        require(case_map == {c['id']:c['dimensions'] for c in context['cases']},'Native case dimensions changed')
        positions = {(r['subject_id'],r['case_id']):i for i,r in enumerate(evidence['rows'])}
        for observation in packet['observations']:
            pair = observation['subject_id'],observation['case_id']
            metric = observation['metric_id']
            require(observation['state'] == 'measured' and observation['reason'] is None
                    and observation['dimensions'] == dict(samples=20,process_blocks=2)
                    and same(observation['value'],expected[pair][metric])
                    and observation['reference'] == dict(source_id='reduction',pointer=f'/rows/{positions[pair]}/scalars/{metric}'),
                    'Native scalar changed or lost exact source reference')
            total += 1
        grids = sorted({tuple(c['scene']['counts']) for c in originals})
        selections = [{},*[dict(cells_per_metre=n) for n in (8,16)],
            *[dict(zip(('grid_x','grid_y','grid_z'),g)) for g in grids],
            *[dict(zip(('grid_x','grid_y','grid_z'),g),cells_per_metre=n) for g,n in itertools.product(grids,(8,16))]]
        for filters,metric in itertools.product(selections,ERRORS+TIMES):
            q = table.query(metric,filters=filters,limit=1)
            for summary in q['summary']:
                values = [v[metric] for (name,cid),v in expected.items() if name == summary['subject_id']
                    and all(case_map[cid][n] == value for n,value in filters.items())]
                require(summary['count'] == summary['states']['measured'] == len(values)
                        and all(v == 0 for n,v in summary['states'].items() if n != 'measured')
                        and ((not values and summary['maximum'] is None)
                            or (values and same(summary['maximum']['value'],max(values)))), 'Native density/grid query changed cases/extrema')
            queries.append(dict(K=k,metric=metric,filters=filters,selected_rows=q['total']))
        tables.append((dict(origin_catalog_id='a'*32,packet_sha256=table.sha256),table))
    incompatible = 0
    for pair in itertools.combinations(tables,2):
        try:
            MeasurementComparison(list(pair))
        except Exception as error:
            require(getattr(error,'code',None) == 'incompatible','Unexpected cross-K diagnostic')
            incompatible += 1
        else:
            raise ValueError('Different physical horizons incorrectly pooled')
    require(total == 24960 and len(raw_seen) == 78 and len(subjects_seen) == 39 and len(fit_seen) == 35
            and assemblies == 4 and len(queries) == 600 and incompatible == 6
            and 'torch' not in sys.modules and 'numpy' not in sys.modules,'Incomplete metadata-only qualification')
    result = dict(status='all39matched_GPU_configurations_24960scalars600queries_and6cross_K_rejections_qualified',
        created_unix=time.time(),measured_scalars=total,exact_fits=35,assembly_executions=4,exact_subjects=39,
        original_raw_blocks_retained_once=78,queries=queries,incompatible_K_pair_rejections=6,
        producer=binding(Path(__file__)),oracle=binding(Path(__file__).with_name('qualify_common_generator_measurements.py')),
        preparation=binding(prep_path),native_owner=binding(NATIVE/'retained_measurements.py'),
        packet_bindings=[binding(PACKETS/p['path']) for p in prep['packets']],
        native_ingestion_pending=True,new_inference=False,new_reference_generation=False,native_mutations=0,full_portfolio_complete=False)
    REPORT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(result['status'])


if __name__ == '__main__':
    qualify()
