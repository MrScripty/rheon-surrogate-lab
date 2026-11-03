"""Derive a comparable best-case score without discarding raw measurements."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

TINY_GRID_AXIS_LIMIT = 6


def enrich_best_error(data, base):
    """Exclude grids fitting wholly inside the six-node numerical solve.

    The rule is identical for every solver, depends only on grid dimensions,
    and affects only the reported best-case score. Historical records without
    per-case grid dimensions retain an explicitly labelled original minimum.
    """
    base = Path(base)
    domains = {r['weights_sha256']: r for r in data.get('resolution_evaluations', [])}
    sources = data.setdefault('history_sources', {})
    filtered, historical = [], []
    for point in data['cohort_points']:
        original = point.get('best_error_raw', point['best_error'])
        point['best_error_raw'] = original
        detail = domains.get(point.get('weights_sha256'))
        source = detail['source'] if detail else point.get('case_source', point['source'])
        path = base / source
        if not path.exists() and point.get('timing_source'):
            root = (base / point['timing_source']).resolve().parents[3]
            path = next((p for folder in ['evaluation-water', 'validation-evaluation', 'raw-validation-native']
                         if (p := root / folder / 'cases.jsonl').exists()), path)
            if path.exists():
                source = os.path.relpath(path, base)
                summary = json.loads(path.with_name('evaluation.json').read_text())
                if summary['weights_sha256'] != point['weights_sha256']:
                    raise ValueError('Best-case checkpoint differs from its evaluation')
        if not path.exists():
            raise ValueError(f'Best-case source unavailable: {point["id"]}')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if source in sources and sources[source] != digest:
            raise ValueError(f'Best-case measurement source changed: {source}')
        sources[source] = digest
        if path.suffix == '.jsonl':
            cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            cases = json.loads(raw)['cases']
        errors = [(c['raw_learned'].get('quality') or c['raw_learned'])['relative_residual'] for c in cases]
        if (len(cases) != point['cases'] or any(not math.isfinite(v) or v < 0 for v in errors)
                or not math.isclose(min(errors), original, rel_tol=1e-12)
                or not math.isclose(max(errors), point['worst_error'], rel_tol=1e-12)
                or not math.isclose(statistics.median(errors), point['median_error'], rel_tol=1e-12)):
            raise ValueError(f'Best-case population differs from original measurements: {point["id"]}')
        if not all(c.get('counts_xyz') for c in cases):
            point.update(best_error=original, best_error_status='historical_unfiltered',
                         best_error_eligible_cases=None, best_error_excluded_cases=None,
                         best_error_source=source)
            historical.append(point['id'])
            continue
        for case in cases:
            counts = case['counts_xyz']
            if len(counts) != 3 or any(type(n) is not int or n <= 0 for n in counts):
                raise ValueError('Best-case grid dimensions must be positive integers')
        eligible = [(c, error) for c, error in zip(cases, errors)
                    if max(c['counts_xyz']) > TINY_GRID_AXIS_LIMIT]
        best = min(eligible, key=lambda row: row[1]) if eligible else None
        point.update(best_error=best[1] if best else None,
                     best_error_status='filtered' if best else 'no_eligible_cases',
                     best_error_eligible_cases=len(eligible), best_error_excluded_cases=len(cases)-len(eligible),
                     best_error_source=source,
                     best_error_case=({k:best[0][k] for k in ['trajectory_id','counts_xyz']}
                                      | {'frame':best[0].get('native_frame',best[0].get('frame'))}) if best else None)
        filtered.append(point['id'])
    data['best_error_policy'] = dict(id='exclude-grids-with-all-axes-at-most-six-v1',
        axis_limit=TINY_GRID_AXIS_LIMIT, eligible='At least one grid dimension exceeds 6 cells',
        reason='Exclude domains wholly covered by the six-node-per-axis numerical solve from best-case summaries',
        applies_to='Every cohort with per-case grid dimensions; no error-value cutoff',
        filtered_cohorts=filtered, historical_unfiltered_cohorts=historical,
        raw_minimum_field='best_error_raw', other_metrics='Median, worst error and timing retain their original populations')
