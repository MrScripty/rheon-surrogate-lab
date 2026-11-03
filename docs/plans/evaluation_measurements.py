"""Build display measurements without altering the retained scientific records.

A ranking key includes task, device, population and timing capture. Historical
single-sweep measurements deliberately have no ranking key. Never infer GPU
timing from the device used to train a checkpoint.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json


def measurement_catalog(data, base):
    groups = defaultdict(list)
    cohorts = {p['id']: p for p in data['cohort_points']}
    for row in data.get('full_ai_study', {}).get('pressure', []):
        if row['model'] in cohorts:
            groups[row['timing_source'], row['regime']].append(row)
    benchmarks = []
    for (source, population), rows in sorted(groups.items()):
        path = Path(base) / source
        capture = json.loads(path.read_text())
        capture_id = path.parent.name
        key = capture_id + '/' + population
        # These are retained CPU pressure-invocation captures. Repeated warm
        # timing must be declared in the evidence before ranking is enabled.
        rankable = (capture.get('status') == 'completed'
                    and capture.get('threads') == 1
                    and bool(capture.get('warmup'))
                    and all(r.get('repeats', 0) >= 3 for r in capture.get('rows', []))
                    and bool(capture.get('rows')))
        records = []
        for index, row in enumerate(rows):
            p = cohorts[row['model']]
            records.append(dict(
                cohort=p['id'], execution=row['identifier'],
                comparison_id=key + '/' + str(index),
                comparison_population=key, ranking_eligible=rankable,
                worst_error=row['worst_error'], inference_median_ms=row['median_ms'],
                repeat_min_ms=row['min_repeat_ms'], repeat_max_ms=row['max_repeat_ms'],
                timing_source=source, source=row.get('source', source),
                qualification=row.get('qualification', ''),
                historical_population=population == 'historical_small',
            ))
        benchmarks.append(dict(
            id=key, label=capture_id.replace('pressure-', '').replace('-timing-01', '')
                + ' · ' + population.replace('_', ' '),
            task='pressure-proposal', device='cpu', population=population,
            ranking_eligible=rankable, source=source,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            warmup=capture.get('warmup', 'Not recorded'),
            timing_scope=capture.get('timing_scope',
                'Prepared pressure invocation; exact coverage and timing subsets in source'),
            threads=capture.get('threads'), cpu=capture.get('cpu'),
            batch_size=capture.get('batch_size'), measurements=records,
        ))
    preferred = 'pressure-compiled-timing-01/historical_small'
    available = {b['id'] for b in benchmarks}
    default = preferred if preferred in available else next(
        (b['id'] for b in benchmarks if b['ranking_eligible']), 'historical')
    return dict(version=1, default_benchmark=default, benchmarks=benchmarks,
                gpu_timings_available=False,
                reference_frontier='Common repeated original pressure benchmark; complete hybrid policy frontiers remain separate')
