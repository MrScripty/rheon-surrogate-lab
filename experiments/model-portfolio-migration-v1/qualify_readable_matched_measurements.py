"""Recompute all readable-table scalars directly from original raw processes.

Presentation IDs are checked against captured names/schemas. Scientific identity
comes from original fit/execution dimensions and exact raw source bytes. No
formatter or packet producer supplies the numerical oracle.
"""
import base64
import itertools
import json
from pathlib import Path
import sys
import time

from qualify_common_generator_measurements import ROOT,RUN,DOC,binding,read,require,same,raw_scalars,ERRORS,TIMES

PACKETS = RUN/'matched-generator-GPU-measurements-02'
NATIVE = RUN/'sources/Tuldok-composition-source-artifacts'
REPORT = DOC/'reports/readable-matched-generator-GPU-measurements-qualification-01.json'


def run():
    require(not REPORT.exists(),'Preserve original readable-table qualification')
    prep_path = PACKETS/'preparation.json'
    prep = read(prep_path)
    for spec in [prep['producer'],prep['source_manifest'],prep['presentation_inverse'],prep['native_table_owner'],*prep['accepted_reductions']]:
        require(binding(ROOT/spec['path']) == spec,'Readable preparation authority changed')
    common = read(RUN/'complete-generator-GPU-protocol-01.json')
    originals = read(ROOT/common['cases']['path'])['cases']
    population = {(c['identifier'],c['start']):c for c in originals}
    sys.path.insert(0,str(NATIVE))
    from retained_measurements import MeasurementTable,MeasurementComparison
    tables,queries = [],[]
    raw_seen,fit_ids,aliases = set(),set(),set()
    total,assemblies = 0,0
    for part in prep['packets']:
        path = PACKETS/part['path']
        raw = path.read_bytes()
        require(binding(path)['sha256'] == part['sha256'] and len(raw) == part['bytes'],'Readable packet changed')
        table = MeasurementTable(raw)
        packet = json.loads(raw)
        decoded = {s['id']:json.loads(base64.b64decode(s['data_base64'])) for s in packet['sources']}
        context,evidence = decoded['context'],decoded['reduction']
        k = part['K']
        require([c['original_case'] for c in context['cases']] == originals
                and context['scope']['physical_outputs'] == k
                and context['scope']['reference_binary_sha256'] == common['reference_binary_sha256']
                and context['scope']['population'] == 'consumed'
                and packet['title'] == packet['scope']['label'] == f'Matched GPU generators and F64 baseline · K{k} · same 64 cases',
                'Readable title or original physical population changed')
        retained = {(r['subject_id'],r['case_id']):r for r in evidence['rows']}
        require(len(retained) == len(evidence['rows']) == 64*len(packet['subjects']),'Exact subject/case membership required')
        expected = {}
        for subject in packet['subjects']:
            alias = subject['id']
            dims = subject['dimensions']
            identity = dims['fit_or_execution_id']
            model = evidence['models'][alias]
            require(alias not in aliases and dims['inference_device'] == 'cuda'
                    and model['origin_kind'] == dims['origin_kind'], 'Unique readable GPU identity required')
            aliases.add(alias)
            original = model['original_model']
            if dims['origin_kind'] == 'original_fit':
                require(identity == original['fit_item_id'] and identity not in fit_ids,'Original fit identity differs')
                fit_ids.add(identity)
                expected_alias = original['identifier']
                if expected_alias == 'b1':
                    schema = read(ROOT/original['configuration']['path'])['schema']
                    require(schema in ('rheon-temporal-fit-v1','rheon-temporal-fit-v2'),'Original b1 schema differs')
                    expected_alias = {'rheon-temporal-fit-v1':'b1-v1','rheon-temporal-fit-v2':'b1-v2'}[schema]
            else:
                require(dims['origin_kind'] == 'assembly_execution' and not original['original_fit_identity_invented'],
                        'Assembly cannot invent a fit')
                assemblies += 1
                expected_alias = f'F64-N1-K{k}'
            require(alias == expected_alias and subject['label'] == alias+f' · GPU · full K{k} block'
                    and subject['reference'] == dict(source_id='reduction',pointer='/models/'+alias),
                    'Plot identity/label does not describe its original source')
            phases = []
            for phase,spec in enumerate(model['raw_process_blocks']):
                source = next(s for s in packet['sources'] if s['id'] == alias+f'-raw-{phase}')
                require(binding(ROOT/spec['path']) == spec and spec['sha256'] not in raw_seen
                        and (ROOT/spec['path']).read_bytes() == base64.b64decode(source['data_base64']),
                        'Raw process source changed or duplicated')
                raw_seen.add(spec['sha256'])
                d = decoded[source['id']]
                require(d['K'] == k and d['phase'] == phase and not d['primary_validation_hook_active']
                        and d['CPU_neural_forwards'] == d['forbidden_forward_attempts'] == 0,'Raw device or timing scope differs')
                if dims['origin_kind'] == 'original_fit':
                    require(d['fit_item_id'] == identity and d['status'] == 'all64_original_cases_uninstrumented_primary_and_paired_hooks_measured',
                            'Readable alias points to a different fit')
                else:
                    require(d['assembly_execution_id'] == original['assembly_execution_id']
                            and d['status'] == 'all64_original_cases_F64_N1_GPU_measured','Assembly identity differs')
                keyed = {(r['identifier'],r['start']):r for r in d['rows']}
                require(len(keyed) == len(d['rows']) == 64 and keyed.keys() == population.keys(),'Raw case omitted')
                phases.append(keyed)
            require(len(phases) == 2,'Both raw process phases required')
            for c in context['cases']:
                current = c['original_case']
                key = current['identifier'],current['start']
                a,b = phases[0][key],phases[1][key]
                values = raw_scalars(dict(a,all_repeat_and_ownership_checks_passed=a['repeat_and_ownership_checks_passed']),
                    dict(b,all_repeat_and_ownership_checks_passed=b['repeat_and_ownership_checks_passed']),current,k)
                row = retained[(alias,c['id'])]
                pooled = row['original_pooled_case']
                require(pooled['block_ms'] == a['block_ms']+b['block_ms'] and pooled['per_output_ms'] == a['per_output_ms']+b['per_output_ms']
                        and pooled['metrics'] == a['metrics'] and pooled['output_hashes'] == a['output_hashes']
                        and row['scalars'].keys() == values.keys() and all(same(v,row['scalars'][m]) for m,v in values.items()),
                        'Presentation changed retained scientific fields,samples or scalar')
                expected[(alias,c['id'])] = values
        case_map = {c['id']:c['dimensions'] for c in packet['cases']}
        require(case_map == {c['id']:c['dimensions'] for c in context['cases']},'Native case dimensions changed')
        positions = {(r['subject_id'],r['case_id']):i for i,r in enumerate(evidence['rows'])}
        for observation in packet['observations']:
            key = observation['subject_id'],observation['case_id']
            metric = observation['metric_id']
            require(observation['state'] == 'measured' and observation['reason'] is None
                    and observation['dimensions'] == dict(samples=20,process_blocks=2)
                    and same(observation['value'],expected[key][metric])
                    and observation['reference'] == dict(source_id='reduction',pointer=f'/rows/{positions[key]}/scalars/{metric}'),
                    'Readable observation lost an exact scientific scalar/source')
            total += 1
        grids = sorted({tuple(c['scene']['counts']) for c in originals})
        filters = [{},*[dict(cells_per_metre=n) for n in (8,16)],
            *[dict(zip(('grid_x','grid_y','grid_z'),g)) for g in grids],
            *[dict(zip(('grid_x','grid_y','grid_z'),g),cells_per_metre=n) for g,n in itertools.product(grids,(8,16))]]
        for selection,metric in itertools.product(filters,ERRORS+TIMES):
            query = table.query(metric,filters=selection,limit=1)
            for summary in query['summary']:
                values = [v[metric] for (alias,cid),v in expected.items() if alias == summary['subject_id']
                    and all(case_map[cid][n] == value for n,value in selection.items())]
                require(summary['count'] == summary['states']['measured'] == len(values)
                        and all(v == 0 for n,v in summary['states'].items() if n != 'measured')
                        and ((not values and summary['maximum'] is None) or (values and same(summary['maximum']['value'],max(values)))),
                        'Readable native density/grid query changes extrema')
            queries.append(dict(K=k,metric=metric,filters=selection,selected_rows=query['total']))
        tables.append((dict(origin_catalog_id='a'*32,packet_sha256=table.sha256),table))
    incompatible = 0
    for pair in itertools.combinations(tables,2):
        try:
            MeasurementComparison(list(pair))
        except Exception as error:
            require(getattr(error,'code',None) == 'incompatible','Unexpected cross-K diagnostic')
            incompatible += 1
        else:
            raise ValueError('Different physical horizons pooled')
    require(total == 24960 and len(raw_seen) == 78 and len(fit_ids) == 35 and assemblies == 4 and len(aliases) == 39
            and len(queries) == 600 and incompatible == 6 and 'numpy' not in sys.modules and 'torch' not in sys.modules,
            'Incomplete metadata-only readable table qualification')
    result = dict(status='readable_matched_tables_all_scientific_scalars_and_queries_qualified',created_unix=time.time(),
        all24960_scientific_observations_exact=True,measured_scalars=total,exact_fits=35,assembly_executions=4,
        exact_subjects=39,raw_process_blobs_exact=78,native_filter_queries=600,queries=queries,incompatible_K_pair_rejections=6,
        producer=binding(Path(__file__)),oracle=binding(Path(__file__).with_name('qualify_common_generator_measurements.py')),
        preparation=binding(prep_path),native_owner=binding(NATIVE/'retained_measurements.py'),
        packet_bindings=[binding(PACKETS/p['path']) for p in prep['packets']],
        native_ingestion_pending=True,native_UI_pending=True,new_neural_forwards=0,full_portfolio_complete=False)
    REPORT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(result['status'],'39 readable subjects; 24,960 scalars; 600 queries; no inference.')


if __name__ == '__main__':
    run()
