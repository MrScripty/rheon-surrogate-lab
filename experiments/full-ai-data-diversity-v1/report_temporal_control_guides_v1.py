"""Summarize matched guide errors and complete GPU timings without inference."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


NAMES = ('equation', 'pressure', 'velocity', 'tracer')


def summary(rows):
    return dict(blocks=len(rows), physical_outputs=sum(r['K'] for r in rows),
        accuracy_pass_blocks=sum(r['all_accuracy_pass'] for r in rows),
        latency_pass_blocks=sum(r['all_latency_samples_pass'] for r in rows),
        worst_relative={n: max(m['relative'][n] for r in rows for m in r['native_metrics_per_output']) for n in NAMES},
        equal_block_and_1overK_head_weighted_squared_relative={n:
            math.fsum(math.fsum(m['relative'][n] ** 2 for m in r['native_metrics_per_output']) / r['K']
                      for r in rows) / len(rows) for n in NAMES},
        median_of_block_median_GPU_ms_per_physical_output=statistics.median(r['median_ms_per_physical_output'] for r in rows),
        maximum_GPU_ms_per_physical_output=max(r['maximum_ms_per_physical_output'] for r in rows))


def run(args):
    if args.output.exists():
        raise ValueError('Fresh guide summary required')
    data = json.loads(args.results.read_text())
    if data['status'] != 'completed_broad_training_control_guide_ablation':
        raise ValueError('Terminal complete guide experiment required')
    grouped = defaultdict(list)
    for row in data['rows']:
        grouped[row['mode']].append(row)
    baseline = {(r['identifier'], r['start_fine_step'], r['output_steps_per_s'], r['K']): r
                for r in grouped['original']}
    result = dict(status='completed_matched_training_control_guide_summary', created_unix=time.time(),
        evidence=str(args.results), evidence_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(),
        source_sha256={str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        inference_device='GPU (CUDA)', gpu_name=data['gpu_name'],
        new_parameter_fits=0, cpu_model_forward=False, reserved52_references_opened=0,
        independent_confirmation=False, model_promotion=False, full_generator_target_pass=False,
        original_wrapper_parity_blocks=data['original_wrapper_parity_blocks'], modes={})
    for mode, rows in grouped.items():
        matched = {(r['identifier'], r['start_fine_step'], r['output_steps_per_s'], r['K']) for r in rows}
        if matched != set(baseline) or len(rows) != len(baseline):
            raise ValueError('Guide populations do not match the frozen baseline')
        item = summary(rows)
        item['paired_physical_output_changes'] = {}
        for name in NAMES:
            counts = dict(improved=0, unchanged=0, worsened=0)
            for row in rows:
                key = (row['identifier'], row['start_fine_step'], row['output_steps_per_s'], row['K'])
                for a, b in zip(row['native_metrics_per_output'], baseline[key]['native_metrics_per_output']):
                    x, y = a['relative'][name], b['relative'][name]
                    counts['improved' if x < y else 'worsened' if x > y else 'unchanged'] += 1
            item['paired_physical_output_changes'][name] = counts
        item['stratified'] = {}
        for group, key in [('density', lambda r: r['cells_per_metre']),
                           ('counts_xyz', lambda r: tuple(r['counts_xyz'])),
                           ('force_profile', lambda r: r['force_profile']),
                           ('force_motion', lambda r: r['force_motion']),
                           ('cadence_K', lambda r: (r['output_steps_per_s'], r['K']))]:
            groups = defaultdict(list)
            for row in rows:
                groups[str(key(row))].append(row)
            item['stratified'][group] = {key: summary(rs) for key, rs in groups.items()}
        result['modes'][mode] = item
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({m: {k: v for k, v in s.items() if k != 'stratified'}
                      for m, s in result['modes'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
