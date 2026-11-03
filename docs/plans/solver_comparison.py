"""Separate learned proposals, numerical hybrids, and component diagnostics."""
import hashlib
import json
import os
from pathlib import Path
import statistics


def enrich_solver_comparison(data, base):
    base = Path(base)
    classification = {}
    sources = data.setdefault('history_sources', {})
    def read(path):
        raw = path.read_bytes()
        relative = str(path.relative_to(base)) if path.is_relative_to(base) else os.path.relpath(path,base)
        digest = hashlib.sha256(raw).hexdigest()
        if relative in sources and sources[relative] != digest:
            raise ValueError(f'Solver classification source changed: {relative}')
        sources[relative] = digest
        return json.loads(raw), relative
    for point in data['cohort_points']:
        kind, reason, evidence = 'full', 'Learned pressure proposal; no numerical inverse or Krylov solver stage', point['source']
        if point.get('replay_artifacts'):
            receipt, evidence = read(base/point['replay_artifacts']/'receipt.json')
            architecture = receipt['architecture']
            if architecture.get('schema') == 'rheon-multilevel-local-smoother-v1':
                kind, reason = 'component', 'Local network only; deployed with a numerical hierarchy and GCR, both excluded from this original diagnostic'
            elif architecture.get('solver_training'):
                kind, reason = 'hybrid', 'Captured proposal includes GCR iterations around learned corrections'
            else:
                method_path = base/point['replay_project']/'hybrid-method.json'
                if method_path.exists():
                    method, evidence = read(method_path)
                    if method['kind'] in ('numerical-coarse','sparse-anchors'):
                        kind, reason = 'hybrid', 'Captured proposal includes '+method['kind']+' numerical pressure correction'
                    elif method['kind'] not in ('learned-coarse','coarse-head-rate','v2-headbase','v2-headshape'):
                        raise ValueError(f'Unclassified pressure method: {method["kind"]}')
        elif int(point['id'][1:]) > 48:
            raise ValueError(f'Unclassified new cohort: {point["id"]}')
        classification[point['id']] = dict(kind=kind, reason=reason, source=evidence)

    known = {p['id']:p for p in data['cohort_points']}
    policies = []
    for row in data.get('methods_v5',{}).get('quiet',[]):
        if row['population'] != 'historical128' or not row['model'] or row.get('control_kind'):
            continue
        point = known[row['model']]
        record, source = read(base/row['source'])
        cases = record['rows']
        identity = lambda r: (r['trajectory_id'],r.get('native_frame',r['frame']),tuple(r['counts_xyz']),r.get('cell_size_m') or 1/r['cells_per_metre'])
        if (record['entry']['weights_sha256'] != point['weights_sha256'] or len(cases) != 128
                or sorted(map(identity,cases)) != sorted(map(identity,point['timing_cases']))
                or record['cpu'] != 23 or record['threads'] != 1 or record['repeats'] != 3):
            raise ValueError('Hybrid policy population, checkpoint or serial timing differs')
        if statistics.median(r['seconds'] for r in cases)*1000 != row['median_ms']:
            raise ValueError('Hybrid latency differs from its matched case timings')
        errors = [r['relative_residual'] for r in cases]
        if max(errors) != row['worst_relative_residual'] or sum(r['qualified'] for r in cases) != row['qualified']:
            raise ValueError('Hybrid policy summary differs from its returned-pressure cases')
        eligible = [r for r in cases if max(r['counts_xyz']) > data['best_error_policy']['axis_limit']]
        best = min(eligible,key=lambda r:r['relative_residual'])
        policies.append(dict(point, comparison_id=point['id']+':'+row['identifier'],
            name=point['name']+' · '+row['identifier'], policy_name=row['identifier'],
            measurement_scope='complete-hybrid-policy',source=source,timing_source=source,
            best_error=best['relative_residual'],best_error_raw=min(errors),median_error=statistics.median(errors),
            worst_error=max(errors),best_error_status='filtered',best_error_eligible_cases=len(eligible),
            best_error_excluded_cases=len(cases)-len(eligible),best_error_source=source,
            best_error_case=dict(trajectory_id=best['trajectory_id'],counts_xyz=best['counts_xyz'],frame=best['frame']),
            case_source=source,validation_summary_source=source,
            inference_median_ms=row['median_ms'],inference_best_ms=min(r['seconds']for r in cases)*1000,
            inference_worst_ms=max(r['seconds']for r in cases)*1000,qualified=row['qualified'],
            policy='Complete '+row['policy']+' pressure policy; includes numerical hierarchy, learned correction, adaptive iteration and common returned-pressure gate',
            timing_scope=record['timing_scope'],diagnostic_scope='Matched historical128 pressure requests; CPU 23, one thread, whole-population warmup and three repetitions per request. Excludes fluid integration and GUI.'))
    v6_policies = []
    population = None
    identities = None
    for row in data.get('methods_v6',{}).get('quiet',[]):
        if not row['model'] or row.get('control_kind'):
            continue
        point = known[row['model']]
        record, source = read(base/row['source'])
        cases = record['rows']
        identity = lambda r: (r['population'],r['trajectory_id'],r['frame'],tuple(r['counts_xyz']),r['cells_per_metre'])
        observed = sorted(map(identity,cases))
        if population is None:
            population, identities = record['population_sha256'], observed
        if (record['entry']['weights_sha256'] != point['weights_sha256'] or len(cases) != 20
                or record['population_sha256'] != population or observed != identities
                or record['cpu'] != 23 or record['threads'] != 1 or record['repeats'] != 3
                or statistics.median(r['seconds'] for r in cases)*1000 != row['median_ms']):
            raise ValueError('V6 complete-policy timing or population differs')
        errors = [r['relative_residual'] for r in cases]
        if max(errors) != row['worst_relative_residual'] or sum(r['qualified'] for r in cases) != row['qualified']:
            raise ValueError('V6 complete-policy summary differs')
        eligible = [r for r in cases if max(r['counts_xyz']) > data['best_error_policy']['axis_limit']]
        best = min(eligible,key=lambda r:r['relative_residual'])
        v6_policies.append(dict(point,comparison_id='v6:'+point['id']+':'+row['identifier'],
            name=point['name']+' · '+row['identifier'],policy_name=row['identifier'],comparison_population='v6-native20:'+population,
            measurement_scope='complete-hybrid-policy',source=source,timing_source=source,case_source=source,validation_summary_source=source,
            best_error=best['relative_residual'],best_error_raw=min(errors),median_error=statistics.median(errors),worst_error=max(errors),
            best_error_status='filtered',best_error_eligible_cases=len(eligible),best_error_excluded_cases=len(cases)-len(eligible),
            best_error_source=source,best_error_case=dict(trajectory_id=best['trajectory_id'],counts_xyz=best['counts_xyz'],frame=best['frame']),
            cases=20,timed_cases=20,qualified=row['qualified'],inference_median_ms=row['median_ms'],
            inference_best_ms=min(r['seconds'] for r in cases)*1000,inference_worst_ms=max(r['seconds'] for r in cases)*1000,
            timing_cases=[dict(trajectory_id=r['trajectory_id'],native_frame=r['frame'],frame=r['frame'],counts_xyz=r['counts_xyz'],
                cell_size_m=1/r['cells_per_metre'],split='consumed-native') for r in cases],
            policy='Complete '+row['policy']+' pressure policy; hierarchy, neural corrections, iteration and common returned-pressure gate included',
            timing_scope=record['timing_scope'],diagnostic_scope='Twenty consumed native requests on 32³–128³ and rectangular grids; CPU 23, one thread, warmup plus three repeats. No fluid integration or GUI.'))
    data['solver_comparison'] = dict(version=1,classification=classification,hybrid_policies=policies,
        definition='Full AI uses learned pressure corrections without a numerical inverse or Krylov solver. Partial AI includes numerical pressure correction or GCR. Residual checks, tensor transfers and learned coarse layers alone do not make a method partial AI.',
        complete_policy_population='V5 historical128; CPU 23, one thread, warmup plus three repeats; all numerical/neural pressure-policy work and common returned-pressure gate included',
        component_policy='Network-only diagnostics are retained separately and excluded from both solver charts and recommendations')
    if v6_policies:
        data['solver_comparison']['hybrid_v6_policies'] = v6_policies


if __name__ == '__main__':
    import re
    page = Path(__file__).with_name('cohort-progress.html')
    text = page.read_text()
    match = re.search(r'(<script id="chart-data" type="application/json">)(.*?)(</script>)',text,re.S)
    data = json.loads(match[2]);enrich_solver_comparison(data,page.parent)
    page.write_text(text[:match.start(2)]+json.dumps(data,separators=(',',':'),allow_nan=False)+text[match.end(2):])
    from collections import Counter
    print('Solver views:',dict(Counter(r['kind'] for r in data['solver_comparison']['classification'].values())),len(data['solver_comparison']['hybrid_policies']),'complete policies')
