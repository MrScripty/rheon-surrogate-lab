"""Report exact-case reproduction of the two cited frozen GPU implementations.

Compare fresh evaluated outputs and metrics with their original captures. Timing
is deliberately absent: these are accuracy replays on already consumed cases,
not independent scene confirmation or the complete portfolio benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'runs/full-ai-data-diversity-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(original, replay, temporal=False):
    old, new = (json.loads(p.read_text()) for p in (original, replay))
    fields = ('identifier', 'start_fine_step', 'output_steps_per_s', 'K', 'lead', 'mode') if temporal else (
        'identifier', 'start', 'K', 'lead')
    def indexed(rows):
        result = {tuple(r[k] for k in fields): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Repeated output key in accuracy capture')
        return result
    before, after = indexed(old['rows']), indexed(new['rows'])
    shared = before.keys() & after.keys()
    hashes = [list(k) for k in shared if before[k]['output_sha256'] != after[k]['output_sha256']]
    metrics = [list(k) for k in shared if before[k]['metrics'] != after[k]['metrics']]
    rows = [list(k) for k in shared if before[k] != after[k]]
    expected_status = ('completed_explicit_time_GPU_graph_and_grouped_controls' if temporal
                       else 'completed_cell_coordinate_native_scored_blocks')
    ok = (new['status'] == expected_status and before.keys() == after.keys()
          and not hashes and not metrics and not rows
          and old['source_sha256'] == new['source_sha256'])
    quoted = [r for r in new['rows'] if
              (temporal and r['identifier'] == 'long-v1-continuous_varying-oscillating-r0-cpm50'
               and r['start_fine_step'] == 0 and r['output_steps_per_s'] == 120
               and r['K'] == 4 and r['mode'] == 'fine_reference') or
              (not temporal and r['identifier'] == 'diversity-development-g025-cpm100'
               and r['start'] == 192 and r['K'] == 4)]
    if len(quoted) != 4:
        raise ValueError('Exact four-output cited case missing')
    modes = sorted({r.get('mode', 'original_N1_N8') for r in new['rows']})
    groups = []
    for mode in modes:
        selected = [r for r in new['rows'] if r.get('mode', 'original_N1_N8') == mode]
        groups.append(dict(mode=mode, outputs=len(selected),
                           diagnostic_failures=sum(not r['all_accuracy_pass'] for r in selected),
                           worst_relative={f: max(r['metrics']['relative'][f] for r in selected)
                                           for f in ('equation', 'pressure', 'velocity', 'tracer')}))
    block_fields = tuple(k for k in fields if k != 'lead')
    return dict(status='exact_reproduction' if ok else 'differences_require_review',
                original=dict(path=str(original.relative_to(ROOT)), sha256=sha(original)),
                replay=dict(path=str(replay.relative_to(ROOT)), sha256=sha(replay)),
                replay_status=new['status'], scored_outputs=len(after),
                explicit_prediction_calls=len({tuple(r[k] for k in block_fields) for r in new['rows']}),
                same_case_keys=before.keys() == after.keys(), output_hash_mismatches=hashes,
                metric_mismatches=metrics, full_accuracy_row_mismatches=rows,
                same_frozen_source_bindings=old['source_sha256'] == new['source_sha256'],
                quoted_case_worst_relative={f: max(r['metrics']['relative'][f] for r in quoted)
                                           for f in ('equation', 'pressure', 'velocity', 'tracer')},
                quoted_case_output_sha256=[r['output_sha256'] for r in sorted(quoted, key=lambda r: r['lead'])],
                summaries=groups, GPU_model=new['gpu_model'], latency_claim=False,
                cpu_model_forward=new['cpu_model_forward'], new_parameter_fits=0)


def main(cell, temporal, output):
    if output.exists():
        raise ValueError('Fresh comparison report required')
    reports = [compare(OLD / 'cell-coordinate-blocks-01/results.json', cell),
               compare(OLD / 'temporal-frozen-backbone-qualification-03/results.json', temporal, True)]
    result = dict(schema='rheon.cited-anchor-accuracy-reproduction.v1', created_unix=time.time(),
                  status='both_cited_implementations_reproduced_exactly' if all(
                      r['status'] == 'exact_reproduction' for r in reports) else 'requires_review',
                  comparisons=reports, scored_outputs=sum(r['scored_outputs'] for r in reports),
                  explicit_prediction_calls=sum(r['explicit_prediction_calls'] for r in reports),
                  source_sha256=sha(Path(__file__)), latency_claim=False,
                  new_parameter_fits=0, independent_confirmation=False,
                  full_portfolio_benchmark_complete=False,
                  interpretation=[
                      'The two frozen implementations retain their historical accuracy on these exact consumed cases.',
                      'The original cell-coordinate test has mixed N1/N8 work; its cited g025 case is N1.',
                      'Fine-reference temporal output accuracy and reduced-pass schedule accuracy are different comparison groups.',
                      'This replay does not establish current speed or equal performance for different models, cases or integration schedules.'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], scored_outputs=result['scored_outputs'],
                         explicit_prediction_calls=result['explicit_prediction_calls'],
                         cited_pressure_errors=[r['quoted_case_worst_relative']['pressure'] for r in reports],
                         latency_claim=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell-coordinate', type=Path, required=True)
    parser.add_argument('--temporal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.cell_coordinate.resolve(), args.temporal.resolve(), args.output)
