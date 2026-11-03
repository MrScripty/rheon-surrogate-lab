"""Qualify the narrow measurement read proposal, preserving the failed fixture."""
import json
from pathlib import Path
import re
import subprocess
import time

from import_common_generator_measurements import ROOT, DOC, RUN, read, save, binding, require
from component_material import observed_file


SOURCE = RUN/'sources/Tuldok-measurement-source-reads'


def verdicts(log):
    starts = list(re.finditer(r'^test_\S+ \(([^\n]+)\)', log, re.M))
    result = {}
    for index, match in enumerate(starts):
        end = starts[index+1].start() if index+1 < len(starts) else len(log)
        outcomes = re.findall(r'\.\.\. (ok|FAIL|ERROR)\s*$', log[match.start():end], re.M)
        require(len(outcomes) == 1 and match[1] not in result, 'Each selected test needs one exact terminal verdict')
        result[match[1]] = outcomes[0]
    return result


def run():
    output = DOC/'reports/measurement-source-reads-source-qualification-01.json'
    require(not output.exists(), 'Preserve original source qualification')
    groups, evidence = [], []
    for suffix in ('01','02'):
        path = DOC/f'reports/measurement-source-reads-SQL-{suffix}.json'
        report = read(path)
        logpath = RUN/f'measurement-source-reads-SQL-{suffix}/tests.log'
        log = logpath.read_text()
        require(report['postgres_stopped'] and binding(logpath)['sha256'] == report['tests_log_sha256'],
                'Original fixture must be terminal and its exact log retained')
        for name, sha in report['source_sha256'].items():
            observed_file(ROOT, str((SOURCE/name).relative_to(ROOT)), dict(sha256=sha))
        group = verdicts(log)
        require(re.search(rf'Ran {len(group)} tests in [\d.]+s', log), 'Terminal group count differs')
        groups.append((report, group))
        evidence.extend([binding(path), binding(logpath)])
    first, second = groups
    failed = [name for name, verdict in first[1].items() if verdict != 'ok']
    require(len(first[1]) == 122 and len(failed) == 1
            and failed[0].endswith('.test_mixed_cold_projection_preserves_notes_and_definitions_without_worker_constructors')
            and first[0]['status'] == 'failed_or_incomplete_fixture'
            and first[0]['test_returncode'] == 1
            and 'Distinct-role cold database is a required fixture for this selection.' in
                (RUN/'measurement-source-reads-SQL-01/tests.log').read_text(),
            'Preserve and identify the original missing-role fixture failure')
    require(second[0]['status'] == 'passed_disposable_native_project_SQL_fixture'
            and second[0]['test_returncode'] == 0 and second[0]['source_stable_through_tests']
            and second[0]['cold_role_provisioned'] and second[0]['cold_role_source_CONNECT_denied']
            and all(value == 'ok' for value in second[1].values()) and failed[0] in second[1],
            'The corrected cold-role group must actually pass on the unchanged candidate source')
    passed = {name for name, verdict in first[1].items() if verdict == 'ok'} | set(second[1])
    require(set(first[1]) <= passed and len(passed) == 122, 'All 122 distinct affected tests must pass across retained fixtures')
    baseline_path = DOC/'reports/measurement-source-lookup-baseline-01.json'
    candidate_path = DOC/'reports/measurement-source-lookup-candidate-01.json'
    baseline, candidate = read(baseline_path), read(candidate_path)
    for profile in (baseline, candidate):
        require(profile['status'] == 'fixed_measurement_source_lookup_completed' and profile['source_unchanged']
                and profile['canonical_header_unchanged'] and profile['exact_literal_documents']
                and not profile['freshness_IO_in_timing'] and not profile['SQL_opened']
                and profile['model_forwards'] == profile['artifact_bodies_read'] == 0
                and not profile['native_endpoint_latency_qualified'] and not profile['whole_Open_qualified'],
                'Complete fixed-record mechanism measurements required, without stronger latency claims')
    for key in ('producer_sha256','checkpoint','canonical_header_sha256','records','canonical_record_bytes',
                'expected_documents_sha256','measurement_documents','CPU_affinity','memory_limit_bytes','python'):
        require(baseline[key] == candidate[key], 'Paired fixed-record workload/environment differs: '+key)
    require(len(baseline['samples']) == len(candidate['samples']) == 2
            and baseline['records'] == 53976 and baseline['measurement_documents'] == 31,
            'Exact current representative checkpoint and two sequential observations required')
    require(all(row['full_graph_copies'] == row['material_decodes'] == 1 for row in baseline['samples'])
            and all(row['full_graph_copies'] == row['material_decodes'] == 0 for row in candidate['samples']),
            'The actual unused copies and material redecodes must disappear')
    red_path = DOC/'reports/measurement-read-copy-red-01.json'
    green_path = DOC/'reports/measurement-read-copy-green-01.json'
    red, green = read(red_path), read(green_path)
    require(red['status'] == 'failed' and red['unrelated_graph_copies'] == 1 and green['status'] == 'passed'
            and green['unrelated_graph_copies'] == 0
            and all(item['literal_documents_equal'] and item['caller_mutation_isolated'] for item in (red,green)),
            'Original real-owner regression must be red before and green after the fix')
    for name, sha in green['source_files'].items():
        observed_file(ROOT, str((SOURCE/name).relative_to(ROOT)), dict(sha256=sha))
    changed = subprocess.check_output(['git','-C',str(SOURCE),'diff','--name-only'],text=True).splitlines()
    new = subprocess.check_output(['git','-C',str(SOURCE),'ls-files','--others','--exclude-standard'],text=True).splitlines()
    require(changed == ['project_measurements.py'] and new == ['tests/test_project_measurement_reads.py'],
            'Exact admitted production/fixture write set required')
    evidence.extend(binding(p) for p in (baseline_path,candidate_path,red_path,green_path))
    result = dict(status='measurement_source_read_proposal_source_qualified_not_installed', created_unix=time.time(),
        producer=binding(Path(__file__)), source=str(SOURCE.relative_to(ROOT)),
        base_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        production_files=[binding(SOURCE/'project_measurements.py')],
        fixture=binding(SOURCE/'tests/test_project_measurement_reads.py'), distinct_passed_checks=122,
        first_fixture_passes=121, corrected_cold_role_group_passes=len(second[1]),
        original_fixture_failure_preserved=True, both_disposable_PostgreSQL_owners_stopped=True,
        exact_canonical_measurement_documents=31, metadata_records=53976,
        fixed_record_CPU_seconds=dict(baseline=[r['cpu_seconds'] for r in baseline['samples']],
                                      candidate=[r['cpu_seconds'] for r in candidate['samples']]),
        process_peak_RSS_KiB=dict(baseline=baseline['peak_RSS_KiB'],candidate=candidate['peak_RSS_KiB']),
        repetitions_per_version=2, statistical_native_latency_speedup_established=False,
        original_candidate_and_packet_validation_unchanged=True, global_cache_added=False,
        whole_Open_qualified=False, native_endpoint_latency_qualified=False, actual_UI_qualified=False,
        installed=False, worktree_disposition='retained-protected pending serial native acceptance or rejection',
        scope='Allocation/codec component and affected native SQL/filesystem/client semantics. '
              'Excludes actual endpoint speedup, canonical freshness I/O timing, browser rendering and whole Open.',
        evidence=evidence, full_portfolio_complete=False)
    save(output,result)
    print(result['status'], result['distinct_passed_checks'], 'distinct passing checks', flush=True)


if __name__ == '__main__':
    run()
