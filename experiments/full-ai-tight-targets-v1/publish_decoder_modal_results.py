"""Publish the managed decoder-label refit and frozen independent confirmation."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/plans/full-ai-tight-targets-v1'
BASE = ROOT / 'runs/full-ai-tight-targets-v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run():
    report = DOCS / 'decoder-compatible-pressure-results-09.md'
    summary_path = DOCS / 'decoder-compatible-pressure-summary-09.json'
    state_path = DOCS / 'work-state-08.json'
    if any(path.exists() for path in (report, summary_path, state_path)):
        raise ValueError('Fresh managed decoder-refit publication required')
    paths = dict(pair=BASE / 'decoder-modal-pair-09/receipt.json',
        cpu_restore=BASE / 'decoder-modal-pair-09/restore/receipt.json',
        training=BASE / 'decoder-modal-pair-09/train/receipt.json',
        history=BASE / 'decoder-modal-pair-09/train/history.json',
        qualification=BASE / 'decoder-modal-graph-qualification-09.json',
        development=BASE / 'decoder-modal-all-horizon-development-09/results.json',
        component=BASE / 'decoder-modal-component-09/results.json',
        fresh_bank=BASE / 'decoder-modal-fresh-reference-09/bank.json',
        fresh_blocks=BASE / 'decoder-modal-fresh-blocks-09/results.json',
        fresh_rollout=BASE / 'decoder-modal-fresh-rollout-09/results.json')
    data = {key: read(path) for key, path in paths.items()}
    pair = data['pair']
    if pair['status'] != 'completed' or pair['optimizer_updates'] != 80 or not all(pair['artifact_parity'].values()):
        raise ValueError('Exact bounded managed fit/restore required')
    if data['training']['execution']['device'] != 'cuda' or data['cpu_restore']['optimizer_updates'] != 0 or not data['cpu_restore']['full_cpu_probe_bitwise_verified']:
        raise ValueError('Actual CUDA learning and typed CPU probe required')
    for key in ('training_run_id', 'restore_run_id'):
        path = BASE / 'decoder-modal-pair-09' / ('terminal-' + pair[key] + '.json')
        owner = read(path)
        if owner['state'] != 'completed' or owner['custody_phase'] != 'drained':
            raise ValueError('Both managed owners must drain')
        paths[key] = path
    if data['qualification']['complete_states'] != 64:
        raise ValueError('Complete trained execution qualification missing')
    for key, count in [('development', 7680), ('fresh_blocks', 320), ('fresh_rollout', 15360)]:
        if len(data[key]['rows']) != count or not data[key]['every_intermediate_state_scored']:
            raise ValueError('Complete all-horizon physical population missing: ' + key)
    bank = data['fresh_bank']
    if bank['status'] != 'completed' or len(bank['trajectories']) != 16:
        raise ValueError('Independent capture bank incomplete')
    for row in bank['trajectories']:
        receipt = read(ROOT / row['receipt'])
        owner = receipt['native_process']
        if len(receipt['states']) != 241 or not owner['reaped'] or owner['returncode'] != 0 or not owner['graceful_protocol_complete']:
            raise ValueError('Independent native reference did not complete and reap')
    costs = {}
    for label in ('p1', 'p2', 'p4'):
        path = BASE / f'decoder-modal-{label}-fresh-cost-09.json'
        value = read(path)
        if value['status'] != 'completed_paired_frozen_sampler_comparison' or len(value['rows']) != 128 or not value['all_intermediate_outputs_bitwise_equal']:
            raise ValueError('Complete exact equivalent CPU costs missing')
        paths[label + '_cost'] = path
        costs[label] = value
    accuracy = {str(k): all(row['all_fields_pass'] for key in ('fresh_blocks', 'fresh_rollout')
                           for row in data[key]['summaries'] if row['horizons'] == k)
                for k in (1, 2, 3, 4)}
    latency = {str(k): any(row['maximum_ms_per_state'][variant] <= 1.2
                          for value in costs.values() for row in value['summaries']
                          if row['regime'] == '32cubed' and row['horizons'] == k
                          for variant in ('baseline', 'frozen_v7')) for k in (1, 2, 3, 4)}
    # These timings do not independently prove a different deployment's full
    # own-history accuracy. Preserve that requirement even if a clock passes.
    combined = {str(k): False for k in (1, 2, 3, 4)}
    cross_horizon = {}
    for key in ('development', 'fresh_rollout'):
        initial = {(row['identifier'], row['step']): row['output_identity']
                   for row in data[key]['rows'] if row['horizons'] == 1}
        cross_horizon[key] = all(row['output_identity'] == initial[row['identifier'], row['step']]
                                 for row in data[key]['rows'])
    cache, checks = {}, []
    for path in sorted(DOCS.glob('*.json')):
        document = read(path)
        for field in ('source_sha256', 'source_evidence'):
            sources = document.get(field)
            if not isinstance(sources, dict):
                continue
            count = 0
            for name, expected in sources.items():
                if not isinstance(expected, str) or len(expected) != 64:
                    continue
                if name not in cache:
                    cache[name] = digest(ROOT / name)
                if cache[name] != expected:
                    raise ValueError('Historical or registered source changed: ' + str(path) + ' ' + name)
                count += 1
            checks.append(dict(registration=str(path.relative_to(ROOT)), field=field, sources_verified=count))
    evidence = {str(path.relative_to(ROOT)): digest(path) for path in paths.values()}
    summary = dict(status='completed_decoder_modal_independent_confirmation_joint_target_unmet',
                   updated_unix=time.time(), managed_optimizer_calls=80, total_learned_parameters=40017,
                   training_run_id=pair['training_run_id'], restore_run_id=pair['restore_run_id'],
                   exact_managed_artifact_parity=pair['artifact_parity'],
                   original_training_scenes=16, native_training_states=144,
                   independent_scenes=16, independent_short_states=320,
                   independent_own_history_states=15360, own_history_steps=240,
                   independent_accuracy_criteria_pass=accuracy, observed_latency_criteria_pass=latency,
                   full_generator_targets_verified=combined, cross_horizon_output_bytes_equal=cross_horizon,
                   development=data['development']['summaries'], component=data['component']['summaries'],
                   independent_short=data['fresh_blocks']['summaries'], independent_own240=data['fresh_rollout']['summaries'],
                   complete_costs={label: value['summaries'] for label, value in costs.items()},
                   maximum_allowed_physical_cores=4, single_core_is_primary_efficiency_metric=True,
                   higher_density_phase_started=False, phase600_started=False,
                   all_managed_and_native_owners_drained=True, evidence=evidence,
                   source_registrations_verified=checks, distinct_artifacts_verified=len(cache),
                   publisher_sha256=digest(__file__), report=str(report.relative_to(ROOT)))
    opening = ('All four horizons independently pass the 1e-5 complete-state accuracy criteria through 240 own-history steps.'
               if all(accuracy.values()) else 'Independent accuracy confirmation completed; one or more 1e-5 criteria remain unmet.')
    lines = ['# Decoder-compatible pressure refit: independent results', '', opening + ' The combined 1.2 ms / 1e-5 goal remains unachieved; higher-density and 600-step stages have not started.', '',
             '## Managed training and candidate scope', '',
             'One managed CUDA refinement used 80 LBFGS optimizer calls on the same 144 native pressure/RHS pairs from 16 original training scenes. The 38,673 double learned pressure parameters started from the previous exact managed checkpoint. A separate managed CPU Run restored all five artifacts bitwise and reproduced the double physical-pressure probe. Both owners completed and drained. The complete generator retains 40,017 learned parameters and predicts pressure, three staggered velocities and tracer.', '',
             'Only the training label coordinates and fixed input projection changed, followed by additional managed optimization. Pressure labels were encoded using the inverse of the small fixed cosine-coordinate matrices offline. This forms no physical pressure inverse, coefficient ratio or analytical gain target. Candidate inference remains the existing learned modal multiplier, learned local operators and learned sampling head. No native pressure solver or numerical correction enters the candidate.', '',
             'This experiment changes both label coordinates and continued optimization; it does not separately establish their causal contributions. CPU/CUDA objective gradients and typed serialization qualified without updates before the fit. The existing 2 MiB normalization declarations passed the actual publication verifier before admission.', '',
             '## Independent complete-state accuracy', '',
             'The checkpoint and inference sources were frozen after all K1–4 development rollouts passed. Sixteen previously unused whole scenes were then captured and scored. They cover the four supported geometries and densities 8/16 cells per metre, with new seeds and forcing parameters. This establishes scene generalization within those geometries, not unseen-shape or higher-density generalization.', '',
             '| Interval | Domain group | Worst equation | Worst pressure | Worst velocity | Worst tracer |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for key, label in [('fresh_blocks', 'Short K1–4 blocks'), ('fresh_rollout', '240 own-history steps, K1–4')]:
        for regime in ('small', '32cubed'):
            chosen = [row for row in data[key]['summaries'] if row['regime'] == regime]
            values = [max(row['worst'][metric] for row in chosen) for metric in ('equation', 'pressure', 'velocity', 'tracer')]
            lines.append('| ' + label + ' | ' + regime + ' | ' + ' | '.join(f'{v:.6g}' for v in values) + ' |')
    lines += ['', 'Every actual intermediate state was typed, checked for finiteness, hashed and scored without teacher resets. The equation criterion includes accumulated trajectory differences against the independent native RHS. The accuracy execution uses the frozen contiguous v3 implementation and single-thread FFTW pressure on CPU 23. Timing variants have separately verified complete output equivalence; the final deployment still needs its complete own-history verification under its final timing policy.', '',
              '## Complete CPU costs on independent current-state inputs', '',
              'Performance-core allocations are CPU 6, CPUs 6/4 and CPUs 6/4/5/7. The one-core baseline is serial v5; parallel v6 is the baseline for two/four cores. Each is paired with frozen-point v7, sharing identical learned weights and pressure-plan owners. Costs include validation, conditioning, every neural stage and every intermediate physical output. Output hashing is outside the clock. Full-block and raw observations are retained; per-step values are full-block time divided by K.', '',
              '| Cores | Steps/pass | Baseline median ms/step | Frozen-point median | Frozen-point p95 | Frozen-point maximum |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for label in ('p1', 'p2', 'p4'):
        for row in costs[label]['summaries']:
            if row['regime'] != '32cubed':
                continue
            lines.append(f"| {len(costs[label]['cores'])} | {row['horizons']} | {row['median_ms_per_state']['baseline']:.3f} | {row['median_ms_per_state']['frozen_v7']:.3f} | {row['p95_ms_per_state']['frozen_v7']:.3f} | {row['maximum_ms_per_state']['frozen_v7']:.3f} |")
    lines += ['', 'All 3,840 timed blocks across the three allocations were byte-identical within their paired execution comparisons. Single-core performance remains a primary efficiency metric. More cores provide an optional latency tradeoff, without a linear-scaling assumption. No timing outliers were removed.', '',
              '## Disposition', '',
              'Retain this managed checkpoint and its full independent evidence. The same-request pressure component, all development trajectories and all three preregistered K1 pressure-execution controls are recorded separately from independent confirmation. Multi-step execution still uses internal recurrence and exposes every intermediate state; no endpoint-only shortcut is claimed.', '',
              'Next work is to reduce complete execution cost while preserving this checkpoint and its numerical behavior, then verify the selected final CPU deployment under the full accuracy and latency contract. Higher-density training remains behind all K1–4 joint gates, followed by the 600-step 60/30 Hz objective. These later objectives remain active and incomplete.', '',
              '[Machine-readable evidence](decoder-compatible-pressure-summary-09.json) · [Frozen selection](decoder-modal-frozen-selection-09.json) · [CPU resource policy](cpu-resource-policy-04.md)', '']
    if not all(accuracy.values()):
        lines += ['Independent accuracy failures must also be resolved. These 16 scenes are now consumed confirmation evidence; any learned successor requires new scene reservations before it can receive independent confirmation.', '']
    report.write_text('\n'.join(lines))
    write(summary_path, summary)
    write(state_path, dict(status='active_phase1_decoder_modal_independent_confirmation_complete_joint_target_unmet',
          updated_unix=time.time(), previous_goal_turn_classification='progress', current_live_processes=[],
          new_managed_optimizer_calls=80, maximum_allowed_physical_cores=4,
          single_core_is_primary_efficiency_metric=True, independent_accuracy_criteria_pass=accuracy,
          observed_latency_criteria_pass=latency, full_generator_targets_verified=combined,
          fresh_precision_reserved_scenes=16, fresh_precision_scenes_consumed=16,
          higher_density_phase_started=False, phase600_started=False,
          report=str(report.relative_to(ROOT)), source_registrations_verified=checks,
          original_completed_study_evidence_unchanged=True,
          next=([] if all(accuracy.values()) else ['Diagnose independent accuracy failures; reserve new confirmation scenes before successor learning']) + ['Preserve the exact managed decoder-compatible checkpoint and independent results',
                'Reduce complete transport, conditioning, pressure and allocation costs without changing learned values',
                'Verify final CPU deployment own-history behavior and maximum block/K latency for all horizons',
                'Open higher-density and 600-step phases only after every joint gate passes']))
    print('Published managed refit and complete independent confirmation; joint staged target remains active', flush=True)


if __name__ == '__main__':
    run()
