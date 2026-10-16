"""Retain a complete regression decision, including an unused final population."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'configs', 'reports'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    root, reports = LAB/protocol['root'], args.reports
    assert not (reports/'summary.json').exists()
    regression = read(reports/'regression-summary.json')
    timing = read(reports/'serial-timing.json')
    assert timing['state'] == 'completed'
    phase = read(root/'development/report.json')
    source = read(reports/'numerical-source-admission.json')
    assert all(digest(LAB/name) == sha for name, sha in source['canonical_runtime_source_sha256'].items())
    eligible = regression['consumed_acceptance_passed']
    fresh_summary = reports/'fresh-validation-summary.json'
    if eligible:
        assert fresh_summary.exists(), 'eligible regression still requires the independent validation measurement'
        fresh = read(fresh_summary)
        fresh_state = 'completed'
        last = read(root/'validation/report.json')['finished_at_utc']
    else:
        assert not (root/'validation').exists(), 'do not silently consume a final population after regression rejection'
        fresh = None
        fresh_state = 'deferred; consumed replacement gates failed'
        last = timing['finished_at_utc']
    first = read(root/'qualification/fixture/report.json')['started_at_utc']
    elapsed = (datetime.fromisoformat(last)-datetime.fromisoformat(first)).total_seconds()
    native, controllers, peaks = [], [], []
    for path in sorted(root.rglob('receipt.json')):
        if 'reference-links' in path.parts:
            continue
        record = read(path)
        if 'native_process' not in record:
            continue
        assert record['native_process']['reaped'] and not record['cuda_initialized']
        native.append({'path': str(path.relative_to(LAB)), 'sha256': digest(path),
                       'status': record['status'], 'native_process': record['native_process']})
    for path in root.rglob('*-timing.json'):
        record = read(path)
        assert record['child_reaped']
        controllers.append({'path': str(path.relative_to(LAB)), 'status': record['status'],
                            'elapsed_seconds': record['elapsed_seconds'], 'child_reaped': True})
    for path in root.rglob('diagnostics.json'):
        record = read(path)
        if 'peak_rss_bytes' in record:
            peaks.append(record['peak_rss_bytes'])
    result = {'schema': 'rheon-pressure-bound-broad-completed-v1', 'state': 'completed',
        'regression': regression['replacement_groups'], 'consumed_acceptance_passed': eligible,
        'fresh_validation_state': fresh_state, 'fresh_validation': fresh,
        'reserved_fresh_scenes': len(read(args.configs/'fresh-scenes.json')['scenes']),
        'fresh_validation_consumed': eligible, 'candidate_promoted': False, 'deployment_changed': False,
        'optimizer_updates': 0, 'archived_reference_cases_reused': 74,
        'independent_pressure_bound_checks': regression['independent_pressure_bound_checks'],
        'regression_elapsed_seconds': phase['elapsed_seconds'], 'serial_timing_elapsed_seconds': timing['elapsed_seconds'],
        'first_fixture_through_terminal_measurement_seconds': elapsed,
        'elapsed_scope': 'first fixture start through serial timing or independent validation finish, including preparation/analysis gaps; excludes later documentation',
        'serial_timing_summaries': timing['summaries'], 'serial_timing_all_qualified': timing['all_qualified'],
        'new_native_receipts': native, 'all_new_native_receipts_reaped': True,
        'controller_timing_receipts': controllers, 'all_recorded_controller_children_reaped': True,
        'maximum_completed_worker_peak_rss_bytes': max(peaks, default=None),
        'memory_scope': 'largest individual process high-water mark; not simultaneous system memory',
        'local_retained_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        'canonical_runtime_sources_unchanged': True, 'source_sha256': digest(__file__),
        'analysis_sha256': digest(reports/'regression-summary.json'),
        'timing_sha256': digest(reports/'serial-timing.json'),
        'limitations': ['All74 development scenes are consumed; independent validation is conditional and cannot be replaced by development success.',
            'Pressure-error bound applies to the sealed cubic constant-density same-RHS operator, not accumulated autonomous trajectory error.',
            'Archived reference timings are not current latency evidence; current serial samples are three finite workloads with desktop services running.',
            'Reference sparse frequency/gradient diagnostics are unavailable for archived states; no zero-error baseline is invented.',
            'No new CUDA training, model parameters, inference inputs, native refinement, fallback or deployment activation.']}
    write(reports/'summary.json', result)
    write(reports/'regression-episodes.json', phase)
    print(json.dumps({k: result[k] for k in ('state', 'consumed_acceptance_passed', 'fresh_validation_state',
        'independent_pressure_bound_checks', 'first_fixture_through_terminal_measurement_seconds')}))


if __name__ == '__main__':
    main()
