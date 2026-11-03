"""Reconcile pressure-trajectory outcomes and measured diagnostic-loop costs."""
from pathlib import Path
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha


def report():
    reports = ROOT/'docs/plans/pressure-methods-v1'
    runs = ROOT/'runs/pressure-methods-v1'
    execution = read(reports/'followup-execution.json')
    if execution['status'] != 'completed' or not all(w['reaped'] and w['returncode'] == 0 for w in execution['workers']):
        raise ValueError('Completed and drained follow-up matrix required')
    plan = read(reports/'followup-plan.json')
    rows = []
    evidence = {}
    for entry in plan['rows']:
        directory = runs/'trajectories'/entry['id']
        result_path = directory/'results.json'
        result = read(result_path)
        if not result['native_reaped'] or result['kind'] != entry['kind'] or result['scene']['steps'] != entry['steps']:
            raise ValueError('Trajectory identity/lifecycle mismatch')
        if sha(directory/'native/frames.jsonl') != result['frames_sha256']:
            raise ValueError('Native trajectory fields changed')
        evidence[str(result_path.relative_to(ROOT))] = sha(result_path)
        points = result['points']
        frames = [json.loads(line) for line in (directory/'native/frames.jsonl').read_text().splitlines()]
        selected = [f for f in frames if f.get('ai_quality')]
        if len(selected) != len(points) or any(not f['reference_quality']['qualified'] for f in frames):
            raise ValueError('Incomplete native reference evidence')
        if result['status'] == 'completed' and len(points) != entry['steps']-result['scene']['warmup_steps']:
            raise ValueError('Missing accepted simulation steps')
        rows.append(dict(**entry, status=result['status'], accepted_steps=len(points),
            final_time_s=points[-1]['time_s'] if points else None,
            worst_pressure_trajectory_relative_l2=max((p['pressure_trajectory_relative_l2'] for p in points), default=None),
            worst_equation_residual=max((p['equation_residual'] for p in points), default=None),
            pressure_crossings={str(limit):next((dict(step=p['step'], time_s=p['time_s']) for p in points
                    if p['pressure_trajectory_relative_l2'] > limit), None) for limit in plan['diagnostic_thresholds']},
            native_loop_s=result['native_elapsed_s'], adapter_loading_s=result['adapter_loading_s'],
            callback_s=result['summed_callback_s'],
            native_ai_track_s=sum(f['ai_seconds'] for f in selected),
            network_calls=result['network_calls'], numerical_corrections=result['numerical_corrections']))
    repeated = []
    for kind in ['native-control', 'full-numerical', 'neural']:
        group = [r for r in rows if r['kind'] == kind]
        if len(group) != 3:
            raise ValueError('Three paired timing repetitions required')
        repeated.append(dict(kind=kind, completed=sum(r['status'] == 'completed' for r in group),
            repetitions=3, median_native_loop_s=statistics.median(r['native_loop_s'] for r in group),
            median_ai_track_s=statistics.median(r['native_ai_track_s'] for r in group),
            median_callback_s=statistics.median(r['callback_s'] for r in group)))
    result = dict(status='pressure_followups_verified', rows=rows, repeated_timing=repeated, evidence=evidence,
        source_sha256=sha(__file__), plan_sha256=sha(reports/'followup-plan.json'),
        timing_scope='Serial native diagnostic loop, with both independently evolved reference and candidate tracks and current reference diagnostics. This is not a single-track production benchmark.',
        velocity_tracer_drift_verified=False, deployment_promoted=False, all_eight_cohorts_complete=False)
    write(reports/'followup-results.json', result)
    lines = ['# Native pressure follow-up and periodic numerical correction', '',
        'All planned workers exited successfully and reaped their native children.',
        'A worker may report a scientific pressure rejection; that is kept separately',
        'from infrastructure failure and is not counted as a complete trajectory.', '',
        '| Policy | Complete repeats | Median diagnostic loop s | Median AI track s |',
        '| --- | ---: | ---: | ---: |']
    for row in repeated:
        lines.append(f"| {row['kind']} | {row['completed']}/3 | {row['median_native_loop_s']:.3f} | {row['median_ai_track_s']:.3f} |")
    lines += ['', '| Policy | Outcome | Accepted AI steps | Worst pressure trajectory error |',
              '| --- | --- | ---: | ---: |']
    for row in rows:
        if 'repeat' not in row:
            lines.append(f"| {row['id']} | {row['status']} | {row['accepted_steps']} | {row['worst_pressure_trajectory_relative_l2']} |")
    lines += ['', 'Timing includes the instrumented dual-track diagnostic loop and IPC. Repeated',
        'comparisons use the same scene and instrumentation in a predeclared shuffled',
        'order. The report retains reference/candidate work separately. This does not',
        'establish a single-track production speedup.', '',
        'Pressure drift thresholds of 0.1% and 1% are diagnostics, not user-specified',
        'application limits. The present wire does not expose velocity/tracer drift.',
        'Those measurements and the remaining trained cohorts are still required.', '',
        '[Per-run results and evidence hashes](followup-results.json).', '']
    (reports/'followup-results.md').write_text('\n'.join(lines))
    print('Verified', len(rows), 'native pressure trajectories', flush=True)


if __name__ == '__main__':
    report()
