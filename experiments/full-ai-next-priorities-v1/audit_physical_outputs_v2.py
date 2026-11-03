"""Check raw physical-output records for an explicitly chosen evaluation directory."""
import argparse
import math
from study_support import ROOT, BASE, read, write, sha


def run(identifier, output, directory=None):
    if output.exists():
        raise ValueError('Fresh audit output required')
    directory = BASE / 'fixed-viscosity-replication-evaluation-01' / identifier if directory is None else directory
    summary_path = directory / 'summary.json'
    summary = read(summary_path)
    norm_path = BASE / 'temporal-training-view-02/normalization.json'
    norm = read(norm_path)
    if summary['status'] != 'completed' or len(summary['scenes']) != 8:
        raise ValueError('Complete eight-scene evaluation required')
    fields = ('pressure', 'equation', 'velocity', 'tracer')
    worst = {field: 0.0 for field in fields}
    sources = [summary_path, norm_path]
    failures = []
    total = 0
    complete = True
    k = summary['model']['horizons']
    for scene in summary['scenes']:
        path = directory / (scene['identifier'] + '.json')
        raw = read(path)
        sources.append(path)
        rows = raw['rows']
        single = [r for r in rows if r['mode'] == 'single_block']
        own = [r for r in rows if r['mode'] == 'own_history']
        if len(single) != 121*k or {(r['start'], r['horizon']) for r in single} != {(s, h) for s in range(121) for h in range(1, k+1)}:
            raise ValueError('Missing intermediate output')
        limit = 240 if raw['first_unavailable_step'] is None else raw['first_unavailable_step'] - 1
        if [r['step'] for r in own] != list(range(1, limit+1)):
            raise ValueError('Own-history coverage differs')
        complete &= limit == 240
        for row in rows:
            metric = row['metrics']
            if any(not math.isfinite(metric['relative'][f]) for f in fields):
                raise ValueError('Nonfinite observed error')
            expected = {'equation': metric['relative']['equation'] <= .004}
            for field in ('pressure', 'velocity', 'tracer'):
                expected[field] = (metric['absolute_rms'][field] <= norm['degenerate_absolute_rms_tolerance'][field]
                    if metric['degenerate_reference'][field] else metric['relative'][field] <= .004)
            if metric['criterion_pass'] != expected or metric['all_accuracy_criteria_pass'] != all(expected.values()):
                raise ValueError('Raw criterion differs from unchanged registered limits')
            for field in fields:
                worst[field] = max(worst[field], metric['relative'][field])
            if not all(expected.values()):
                failures.append(dict(scene=scene['identifier'], mode=row['mode'], step=row['step'],
                    failed_fields=[f for f in fields if not expected[f]], relative=metric['relative']))
            total += 1
        if scene['summaries'] != raw['summaries']:
            raise ValueError('Scene summaries differ from raw trace')
        for interval in raw['summaries']:
            selected = [r for r in rows if r['mode'] == interval['mode'] and r['step'] <= interval['interval_steps']]
            extrema = {f: max(r['metrics']['relative'][f] for r in selected) for f in fields}
            passed = all(r['metrics']['all_accuracy_criteria_pass'] for r in selected)
            if interval['outputs'] != len(selected) or interval['worst'] != extrema or interval['all_accuracy_criteria_pass'] != passed:
                raise ValueError('Interval counts, extrema or pass decisions differ')
        for field, levels in raw['diagnostic_drift_thresholds'].items():
            for level, observed in levels.items():
                first = next((r for r in own if r['metrics']['relative'][field] > float(level)), None)
                expected = None if first is None else dict(step=first['step'], time_s=first['step']/60)
                if observed != expected:
                    raise ValueError('Diagnostic drift crossing differs')
    value = dict(status='completed', record_consistency_passed=True, accuracy_pass=complete and not failures,
        model=identifier, evaluation_directory=str(directory.relative_to(ROOT)), evaluated_model=summary['model'],
        physical_outputs_verified=total, scenes=8, worst=worst, failures=failures,
        complete_own240=complete, source_evidence={str(p.relative_to(ROOT)): sha(p) for p in sources},
        auditor_sha256=sha(__file__), primary_selection_changed=False, CPU_latency_claim=False)
    write(output, value)
    print(identifier, 'outputs', total, 'accuracy_pass', value['accuracy_pass'], 'worst', worst, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identifier', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--evaluation-directory')
    args = parser.parse_args()
    run(args.identifier, ROOT / args.output, ROOT / args.evaluation_directory if args.evaluation_directory else None)
