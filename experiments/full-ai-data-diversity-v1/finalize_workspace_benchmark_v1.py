"""Finalize already recorded GPU timings after a NumPy-boolean JSON error.

Standard library only: no GPU access, learned inference or new timing samples.
Every recorded call/output and full finite population is revalidated before
native Python summary values are written to a separate immutable artifact.
"""
import argparse
import collections
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile95(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * .95
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    # Linear interpolation, matching the registered NumPy default definition.
    return (ordered[lower] + (ordered[upper] - ordered[lower]) * fraction if fraction < .5
            else ordered[upper] - (ordered[upper] - ordered[lower]) * (1 - fraction))


def run(args):
    if args.output.exists():
        raise ValueError('Fresh offline finalization artifact required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen finalization evidence changed: ' + name)
    protocol = read(ROOT / reg['measurement_protocol'])
    raw = read(ROOT / reg['raw_results'])
    if raw['protocol_sha256'] != digest(ROOT / reg['measurement_protocol']):
        raise ValueError('Raw measurements do not bind the registered study')
    for name, expected in protocol['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Original measurement source/evidence changed: ' + name)
    baseline = read(ROOT / protocol['baseline_outputs'])
    if not baseline['all_accuracy_pass'] or len(baseline['rows']) != 960:
        raise ValueError('Complete native-qualified baseline required')
    expected_outputs = {(r['identifier'], r['start'], r['K'], r['lead']): r['output_sha256']
                        for r in baseline['rows']}
    blocks = {(r['identifier'], r['start'], r['K']) for r in baseline['rows']}
    required_calls = {(*block, repeat, mode) for block in blocks
                      for repeat in range(protocol['repetitions']) for mode in protocol['modes']}
    observed_calls = set()
    groups = collections.defaultdict(list)
    for row in raw['rows']:
        key = row['identifier'], row['start'], row['K'], row['repeat'], row['mode']
        if key in observed_calls:
            raise ValueError('Duplicate timed call')
        observed_calls.add(key)
        expected = [expected_outputs[(row['identifier'], row['start'], row['K'], lead)]
                    for lead in range(1, row['K'] + 1)]
        if row['actual_complete_output_sha256'] != expected:
            raise ValueError('Actual recorded timed output differs from native-qualified baseline')
        if (not math.isfinite(row['full_block_ms']) or row['full_block_ms'] < 0
                or row['ms_per_generated_step'] != row['full_block_ms'] / row['K']
                or row['internal_advances_per_block'] != row['K'] * row['internal_substeps']):
            raise ValueError('Invalid recorded complete-call timing or denominator')
        groups[(row['mode'], row['K'], row['cells_per_metre'], tuple(row['counts_xyz']), row['internal_substeps'])].append(row)
    cold_keys = {(r['identifier'], r['start'], r['K'], r['mode']) for r in raw['first_pre_warmup_calls']}
    required_cold = {(*block, mode) for block in blocks for mode in protocol['modes']}
    if (observed_calls != required_calls or len(raw['rows']) != protocol['expected_timed_calls']
            or raw['timed_complete_outputs'] != protocol['expected_timed_outputs']
            or sum(r['K'] for r in raw['rows']) != protocol['expected_timed_outputs']
            or raw['warmup_complete_outputs'] != protocol['expected_warmup_outputs']
            or cold_keys != required_cold or len(raw['first_pre_warmup_calls']) != len(required_cold)):
        raise ValueError('Complete predeclared measurement/warmup population not recoverable')
    summaries = []
    for (mode, k, density, counts, n), rows in sorted(groups.items()):
        values = [r['ms_per_generated_step'] for r in rows]
        ceiling = 1.2 * max(1., math.prod(counts) / (32**3))
        summaries.append(dict(mode=mode, K=k, cells_per_metre=density, counts_xyz=list(counts), internal_substeps=n,
            measured_calls=len(rows), median_ms_per_generated_step=statistics.median(values),
            p95_ms_per_generated_step=percentile95(values), maximum_ms_per_generated_step=max(values),
            voxel_budget_ms_per_generated_step=ceiling, observed_latency_within_voxel_budget=max(values) <= ceiling))
    result = dict(raw)
    result.update(status='completed_balanced_workspace_gpu_entry_benchmark_finalized_offline', active=None,
                  summaries=summaries, original_raw_status=raw['status'], original_tool_exit_code=1,
                  raw_measurement_file=reg['raw_results'], raw_measurement_sha256=digest(ROOT / reg['raw_results']),
                  offline_finalization_only=True, new_inference_or_timing_samples_executed=False,
                  every_actual_timed_output_hash_verified_offline=True,
                  warmup_verification='Original harness verifies each actual warmup output before incrementing its counter. Full11520counter and all768first-call records are present in the last per-scene checkpoint.',
                  original_terminal_error='NumPy boolean in final summary was not JSON serializable; all per-scene raw timing rows were saved before summary construction.',
                  missing_original_final_only_metadata=['graph_setup_observations', 'peak_gpu_allocated_bytes', 'elapsed_diagnostic_seconds'],
                  setup_accounting='Constructor,geometry and workspace allocations remain recorded in raw data. Per-graph setup observations and peak GPU allocation were added only to the failed final serialization and cannot be reconstructed from timing rows; not inferred or substituted.',
                  finalization_created_unix=time.time(), finalization_protocol_sha256=digest(args.protocol))
    # Force strict JSON serialization here; no NumPy objects enter this program.
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Finalized original raw timing study:', len(raw['rows']), 'calls;',
          raw['timed_complete_outputs'], 'timed outputs; no new timing samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
