"""Check native queries against original scalar reports and timing samples."""
import base64
import itertools
from pathlib import Path
import struct
import sys
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require

PACKET = RUN/'native-RHS-component-measurements-01'
NATIVE = RUN/'sources/Tuldok-composition-source-artifacts'
REPORT = DOC/'reports/native-RHS-component-measurements-qualification-01.json'


def same(a,b):
    return struct.pack('!d',a) == struct.pack('!d',b)


def run():
    require(not REPORT.exists(), 'Preserve first native-RHS qualification')
    prep = read(PACKET/'preparation.json'); part, = prep['packets']
    for spec in [prep['producer'],prep['native_table_owner'],prep['source_manifest'],*prep['accepted_reductions']]:
        require(binding(ROOT/spec['path']) == spec, 'Prepared authority changed')
    path = PACKET/part['path']; raw = path.read_bytes()
    require(binding(path)['sha256'] == part['sha256'] and len(raw) == part['bytes'], 'Prepared packet changed')
    import json
    packet = json.loads(raw); decoded = {s['id']:json.loads(base64.b64decode(s['data_base64'])) for s in packet['sources']}
    for sid,spec in zip(('modal','shared'),prep['accepted_reductions'],strict=True):
        require(base64.b64decode(next(s['data_base64'] for s in packet['sources'] if s['id']==sid)) == (ROOT/spec['path']).read_bytes(),
                'Embedded scalar report is not its exact accepted original')
    sys.path.insert(0,str(NATIVE))
    from retained_measurements import MeasurementTable
    table = MeasurementTable(raw); expected = {}; median_checks = 0
    for subject in packet['subjects']:
        sid = subject['reference']['source_id']; model = decoded[sid]
        for component in subject['reference']['pointer'].split('/')[1:]:
            model = model[int(component)] if isinstance(model,list) else model[component]
        require(subject['dimensions']['fit_item_id'] == model['fit_item_id'] and subject['dimensions']['inference_device'] == 'cuda', 'Fit/device changed')
        for i,row in enumerate(model['rows']):
            case = packet['cases'][i]; identity = row['identity']; z,y,x = identity['counts_zyx']; d = case['dimensions']
            require([d['grid_x'],d['grid_y'],d['grid_z']] == [x,y,z] and d['cells_per_metre'] == identity['cells_per_metre']
                    and d['trajectory_id'] == identity['identifier'] and d['state_index'] == identity['state'], 'Case geometry/density/identity differs')
            samples = sorted(row['raw_wall_nanoseconds']); require(len(samples)==20 and all(type(v) is int and v>0 for v in samples), 'Original twenty positive wall samples required')
            # Independently spell the even-population median before ns -> ms.
            median = ((samples[9]+samples[10])/2)/1000000
            require(same(median,row['pooled_median_ms']), 'Original pooled timing does not match twenty samples'); median_checks += 1
            for metric in packet['metrics']:
                name = metric['id']; value = median if name=='pressure_call_median_ms' else row['metrics'][name]
                expected[(subject['id'],case['id'],name)] = value
    require(len(expected) == len(packet['observations']) == 4320, 'Six complete subjects ×144cases ×five metrics required')
    for observation in packet['observations']:
        key = observation['subject_id'],observation['case_id'],observation['metric_id']
        require(observation['state']=='measured' and same(observation['value'],expected[key]), 'Observed scalar differs from accepted original')
    subjects = [s['id'] for s in packet['subjects']]; cases = {c['id']:c['dimensions'] for c in packet['cases']}
    densities = sorted({d['cells_per_metre'] for d in cases.values()}); grids = sorted({(d['grid_x'],d['grid_y'],d['grid_z']) for d in cases.values()})
    filters = [{}]+[dict(cells_per_metre=v) for v in densities]+[dict(zip(('grid_x','grid_y','grid_z'),v)) for v in grids]
    filters += [dict(cells_per_metre=d,**dict(zip(('grid_x','grid_y','grid_z'),g))) for d,g in itertools.product(densities,grids)]
    queries, maxima_checks = 0,0
    for metric in packet['metrics']:
        name = metric['id']
        for chosen in filters:
            selected = [cid for cid,d in cases.items() if all(d[k]==v for k,v in chosen.items())]
            result = table.query(name,subject_ids=subjects,filters=chosen,limit=1)
            require(result['total']==6*len(selected), 'Native filter membership differs')
            for summary in result['summary']:
                values = [expected[(summary['subject_id'],cid,name)] for cid in selected]
                require(summary['count']==len(values) and summary['states']['measured']==len(values), 'Native summary count differs')
                require((summary['maximum'] is None) if not values else same(summary['maximum']['value'],max(values)), 'Native maximum differs')
                maxima_checks += 1
            queries += 1
    require(median_checks==864 and len(subjects)==6 and len({s['dimensions']['fit_item_id'] for s in packet['subjects']})==5, 'Five original fits/six policies required')
    save(REPORT,dict(status='native_RHS4320scalars_and_density_grid_queries_match_originals',created_unix=time.time(),
        fit_identities=5,execution_policies=6,cases_per_policy=144,metric_definitions=5,exact_scalar_checks=4320,
        independent_twenty_sample_median_checks=median_checks,native_queries=queries,native_maximum_checks=maxima_checks,
        density_filters=densities,grid_filters=[list(g) for g in grids],packet=binding(path),preparation=binding(PACKET/'preparation.json'),
        producer=binding(Path(__file__)),native_table_owner=binding(NATIVE/'retained_measurements.py'),
        source_reports_exact=True,native_project_mutations=0,new_model_forwards=0,prediction_fields_opened=0,
        native_import_pending=True,actual_UI_pending=True,full_source_closure_portable=False,full_portfolio_complete=False))
    print('Qualified4320scalars/864twenty-sample medians;',queries,'native queries;',maxima_checks,'exact maxima/count checks.',flush=True)


if __name__ == '__main__':
    run()
