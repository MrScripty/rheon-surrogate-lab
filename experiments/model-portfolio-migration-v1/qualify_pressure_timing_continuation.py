"""Independently bind a retained timing prefix and known contention exclusion."""
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'
OUTPUT = ROOT / 'docs/plans/model-portfolio-migration-v1/reports/pressure-timing-contention-reconciliation-01.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    require(not OUTPUT.exists(), 'Preserve original independent reconciliation')
    continuation_path = BASE / 'pressure-portfolio-GPU-timing-continuation-protocol-01.json'
    continuation = read(continuation_path)
    original_path = ROOT / continuation['original_protocol']['path']
    require(sha(original_path) == continuation['original_protocol']['sha256'], 'Original protocol differs')
    original = read(original_path)
    cases_path = ROOT / original['frozen_cases']['path']
    require(sha(cases_path) == original['frozen_cases']['sha256'], 'Frozen cases differ')
    frozen = read(cases_path)['cases']
    require(len(frozen) == 128, 'All original native requests required')
    sources = []
    qualified_calls = 0
    prefix = continuation['accepted_prefix']
    excluded = continuation['excluded_contended_block']
    selected = [dict(result=entry['result'], sha256=entry['sha256'], identifier=entry['identifier'],
                     original_block=entry['block'], qualified=True) for entry in prefix]
    selected.append(dict(result=excluded['path'], sha256=excluded['sha256'],
                         identifier='C102', original_block=99, qualified=False))
    for entry in selected:
        result_path = ROOT / entry['result']
        require(sha(result_path) == entry['sha256'], 'Retained original result bytes differ')
        result = read(result_path)
        require(result['identifier'] == entry['identifier']
                and result['protocol_sha256'] == sha(original_path), 'Wrong original model/protocol')
        reference = original['accuracy'][entry['identifier']]
        reference_path = ROOT / reference['path']
        require(sha(reference_path) == reference['sha256'], 'Original scored accuracy capture differs')
        reference_rows = read(reference_path)['rows']
        require(len(result['rows']) == len(reference_rows) == 128, 'A native case was omitted')
        require(result['host_affinity'] == [4] and result['torch_threads'] == 1
                and result['interop_threads'] == 1 and not any(result['tf32'].values())
                and result['deterministic_algorithms'] and not result['cudnn_benchmark']
                and not result['device_qualification']['forbidden_module_calls'], 'Hardware/precision scope differs')
        for actual, expected, case in zip(result['rows'], reference_rows, frozen, strict=True):
            require(actual['case_id'] == case['id']
                    and actual['input_sha256'] == expected['input_sha256'] == case['original_case']['input_sha256']
                    and actual['output_sha256'] == expected['output_sha256'], 'Original exact request/output differs')
            samples = actual['raw_wall_nanoseconds']
            require(len(samples) == 10 and all(type(value) is int and value > 0 for value in samples),
                    'Exactly ten positive raw wall samples required')
            require(actual['median_ms'] == statistics.median(samples) / 1e6
                    and actual['minimum_ms'] == min(samples) / 1e6
                    and actual['maximum_ms'] == max(samples) / 1e6,
                    'Stored worker statistics do not reduce from original raw samples')
        if entry['qualified']:
            require(result['status'] == 'complete_pressure_GPU_timing_pilot'
                    and result['architecture_unchanged'] and result['learned_parameters_unchanged']
                    and not any(r['other_compute_pids'] for r in result['resource_observations']),
                    'Completed prefix contains a contended/incomplete execution')
            qualified_calls += 1280
        else:
            require(result['status'] == 'failed_timing_pilot'
                    and any(r['other_compute_pids'] for r in result['resource_observations']),
                    'Exact known contention exclusion required')
        sources.append(dict(**entry, reference=reference, exact_cases=128))
    require(len(prefix) == 98 and qualified_calls == 125440, 'Exact qualified prefix denominator differs')
    report = dict(status='98_original_exact_blocks_retained_C102_contention_excluded', created_unix=time.time(),
        accepted_prefix_blocks=98, accepted_prefix_timed_calls=qualified_calls,
        excluded_contended_blocks=1, excluded_timed_calls=1280,
        all99_original_native_input_and_output_hashes_exact=True,
        every_raw_statistic_independently_reduced=True, source_bindings=sources,
        continuation_protocol=dict(path=str(continuation_path.relative_to(ROOT)), sha256=sha(continuation_path)),
        remaining_blocks=156, remaining_timed_calls=199680,
        no_completed_uncontended_block_replayed=True, CPU_neural_forwards=0, new_parameter_fits=0,
        all127_pooled_timing_complete=False, full_portfolio_complete=False,
        producer_sha256=sha(Path(__file__)))
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
