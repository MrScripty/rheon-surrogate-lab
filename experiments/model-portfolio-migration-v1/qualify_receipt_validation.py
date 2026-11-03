"""Bind the receipt-decoding proposal to existing native checks and phase timings.

No model, project, SQL service or additional performance run is started here.
The full Open/UI acceptance is deliberately outside this source qualification.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/model-portfolio-migration-v1'
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1/reports'
SOURCE = RUN / 'sources/Tuldok-candidate-receipt-validation'
BASELINE = RUN / 'sources/Tuldok-composition-source-artifacts'
OUTPUT = DOC / 'candidate-receipt-validation-source-qualification-01.json'
PRODUCTION = ('project_files.py', 'project_sources.py', 'project_measurements.py', 'project_projection.py')
FIXTURE = 'tests/test_project_receipt_validation.py'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    require(not OUTPUT.exists(), 'Preserve original qualification')
    evidence = []
    identifiers = set()
    groups = []
    for name, count in [('candidate-receipt-validation-SQL-01', 137),
                        ('candidate-receipt-validation-integration-01', 161)]:
        report_path = DOC / (name + '.json')
        report = read(report_path)
        log_path = RUN / name / 'tests.log'
        log = log_path.read_text()
        require(report['status'] == 'passed_disposable_native_project_SQL_fixture'
                and report['test_returncode'] == 0 and report['postgres_stopped']
                and report['source_stable_through_tests']
                and sha(log_path) == report['tests_log_sha256'], 'Native fixture/log differs')
        # Real HTTP/capture tests may emit diagnostics, and unittest puts a
        # test docstring on a separate line before "... ok". Bind every start
        # identity and require the unskipped terminal suite verdict separately.
        passed = re.findall(r'^test_\S+ \(([^\n]+)\)', log, re.M)
        require(len(passed) == len(set(passed)) == count
                and not identifiers.intersection(passed)
                and re.search(rf'Ran {count} tests in [\d.]+s\n\nOK\s*$', log),
                'Complete distinct passing test population required')
        identifiers.update(passed)
        for name, digest in report['source_sha256'].items():
            require(sha(SOURCE / name) == digest, 'Qualified source changed: ' + name)
        require(all(name in report['source_sha256'] for name in PRODUCTION), 'Missing changed production binding')
        groups.append(dict(report=str(report_path.relative_to(ROOT)), checks=count))
        evidence.extend([report_path, log_path])
    require(len(identifiers) == 298, 'Complete unique native checks required')
    before_path = DOC / 'receipt-validation-pinned-baseline-01.json'
    after_path = DOC / 'candidate-receipt-validation-profile-01.json'
    before, after = read(before_path), read(after_path)
    for profile in (before, after):
        require(profile['status'] == 'fixed_record_validation_completed'
                and profile['manifest_unchanged'] and profile['complete_tracked_source_unchanged']
                and profile['source_unchanged'] and not profile['neural_runtime_imported']
                and not profile['SQL_opened'] and not profile['Store_or_Session_constructed']
                and not profile['artifact_bodies_available'] and profile['models_executed'] == 0,
                'Full bounded metadata-only validation required')
    require(all(before[k] == after[k] for k in ('checkpoint', 'records', 'canonical_header_sha256',
            'producer_sha256', 'wall_budget_seconds', 'memory_limit_bytes')),
            'Compared metadata/profiler/budget differs')
    for old, new in [('validate_file_result', 'file_receipt_validator'),
                     ('validate_source_result', 'source_receipt_validator'),
                     ('validate_measurement_result', 'measurement_receipt_validator')]:
        owned = after['owned_receipt_validator_totals'][new]
        require(owned['candidate_decodes'] == 1 and owned['receipt_calls']
                == before['operation_validator_totals'][old]['calls'], 'A receipt was omitted or decoded repeatedly')
    for profile, owner in [(before, BASELINE), (after, SOURCE)]:
        for name, digest in profile['source_sha256'].items():
            require(sha(owner / name) == digest, 'Measured source changed')
    evidence.extend([before_path, after_path])
    head = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == before['source_commit'] == after['source_commit'], 'Proposal must descend from measured baseline')
    changes = subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--name-only'], text=True).splitlines()
    untracked = subprocess.check_output(['git', '-C', str(SOURCE), 'ls-files', '--others', '--exclude-standard'], text=True).splitlines()
    require(set(changes) == set(PRODUCTION) and untracked == [FIXTURE], 'Proposal write set differs')
    require(read(DOC / 'candidate-receipt-validation-SQL-01.json')['source_sha256'][FIXTURE] == sha(SOURCE / FIXTURE),
            'New regression fixture changed after checks')
    value = dict(status='candidate_receipt_ownership_source_qualified_not_installed', created_unix=time.time(),
        base_commit=head, source=str(SOURCE.relative_to(ROOT)), source_sha256={n: sha(SOURCE/n) for n in (*PRODUCTION, FIXTURE)},
        check_groups=groups, unique_selected_checks=298, every_record_and_receipt_retained=True,
        metadata_records=after['records'], checkpoint=after['checkpoint'],
        phase_CPU_seconds=dict(baseline=before['codec_cpu_seconds'], candidate=after['codec_cpu_seconds']),
        phase_wall_seconds=dict(baseline=before['codec_wall_seconds'], candidate=after['codec_wall_seconds']),
        phase_CPU_reduction_fraction=1-after['codec_cpu_seconds']/before['codec_cpu_seconds'],
        profile_launch_cpu_affinity=[8], affinity_evidence='Both original launches used taskset -c 8; profiler does not self-record affinity.',
        repetitions_per_version=1, statistical_speedup_established=False,
        process_peak_RSS_KiB=dict(baseline=before['peak_RSS_KiB'], candidate=after['peak_RSS_KiB']),
        regression_reproduced_before_fix=True, fixed_candidate_owned_values=True,
        global_cache_added=False, artifact_custody_and_fresh_candidate_negatives_passed=True,
        installed=False, actual_Open_qualified=False, actual_UI_qualified=False, near_instant_Open_achieved=False,
        scope='Complete metadata-only codec phase and affected native/SQL/HTTP/artifact checks. Excludes whole Open, SQL construction, real project artifact bodies and UI latency.',
        source_worktree_disposition='retained-protected', worktree_owner='/root',
        worktree_purpose='Qualified private proposal awaiting clean live-owner stop, fast-forward and actual Open/UI acceptance.',
        next_disposition='Retire clean worktree only after installed source/ref equality and actual UI acceptance.',
        full_portfolio_complete=False, evidence=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in evidence],
        producer_sha256=sha(Path(__file__)))
    OUTPUT.write_text(json.dumps(value, indent=2) + '\n')
    print(value['status'], len(identifiers), 'native checks', flush=True)


if __name__ == '__main__':
    main()
