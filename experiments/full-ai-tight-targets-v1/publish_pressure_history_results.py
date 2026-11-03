"""Publish managed mixed-query training, causal pressure reuse and measured CPU cost."""
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


def relative(path):
    return str(path.relative_to(ROOT))


def checked_evaluation(path, count, independent):
    data = read(path)
    if len(data['rows']) != count or data['fresh_holdout_consumed'] != independent:
        raise ValueError('Actual complete evaluation population required: ' + relative(path))
    if (data['optimizer_updates'] != 0 or data['cuda_initialized'] or
            not data['all_plan_owners_closed'] or not data['every_intermediate_state_scored'] or
            not data['truth_used_only_after_causal_prediction']):
        raise ValueError('Causal zero-update complete evaluation required')
    return data


def horizon_identity(data):
    states = {}
    for row in data['rows']:
        key = row['identifier'], row['start'], row['step']
        if key in states and states[key] != row['output_identity']:
            raise ValueError('Horizon changed a corresponding physical state')
        states[key] = row['output_identity']
    return len(states)


def main():
    report = DOCS / 'pressure-history-and-mixed-query-results-12.md'
    summary_path = DOCS / 'pressure-history-and-mixed-query-summary-12.json'
    state_path = DOCS / 'work-state-10.json'
    if any(p.exists() for p in (report, summary_path, state_path)):
        raise ValueError('Fresh pressure-history publication required')
    pair_path = BASE / 'mixed-modal-pair-11/receipt.json'
    pair = read(pair_path)
    if pair['status'] != 'completed' or not all(pair['artifact_parity'].values()):
        raise ValueError('Managed train/restore artifact parity required')
    condition_path = BASE / 'mixed-modal-view-11/conditioning-diagnostic.json'
    condition = read(condition_path)
    if condition['development_or_reserved_used'] or not all(
            d['original_pairs_bitwise_retained'] for d in condition['diagnostics']):
        raise ValueError('Train-only augmentation and all original pairs required')
    consumed_path = BASE / 'pressure-history-all-horizon-consumed-12/results.json'
    consumed = checked_evaluation(consumed_path, 38400, False)
    horizon_identity(consumed)
    fresh_path = BASE / 'pressure-history-fresh-rollout-12/results.json'
    short_path = BASE / 'pressure-history-fresh-blocks-12/results.json'
    fresh = checked_evaluation(fresh_path, 15360, True)
    short = checked_evaluation(short_path, 320, True)
    horizon_identity(fresh)
    reservation_path = DOCS / 'pressure-successor-fresh-scene-reservation-10.json'
    selection_path = DOCS / 'pressure-history-frozen-selection-12.json'
    bank_path = BASE / 'pressure-history-fresh-reference-12/bank.json'
    reserved = {r['identifier']: r for r in read(reservation_path)['rows']}
    bank = read(bank_path)
    if (bank['status'] != 'completed' or len(bank['trajectories']) != 16 or
            bank['frozen_selection_sha256'] != digest(selection_path) or
            {r['identifier'] for r in bank['trajectories']} != set(reserved)):
        raise ValueError('Actual complete reserved capture and pre-capture selection required')
    for trajectory in bank['trajectories']:
        path = ROOT / trajectory['receipt']
        receipt = read(path)
        scene = {k: v for k, v in reserved[trajectory['identifier']].items()
                 if k not in ('identifier', 'split')}
        if (digest(path) != trajectory['receipt_sha256'] or receipt['scene'] != scene or
                len(receipt['states']) != 241 or not receipt['native_process']['reaped'] or
                receipt['native_process']['returncode'] != 0):
            raise ValueError('Reserved scene or reference custody changed')
    for data in (fresh, short):
        if ({r['identifier'] for r in data['rows']} != set(reserved) or
                data['model_identity'] != consumed['model_identity']):
            raise ValueError('Independent evaluation changed selected model or scene population')
    for data in (consumed, fresh):
        if not data['own_history_without_teacher_resets'] or data['rollout_steps'] != 240:
            raise ValueError('Every own-history rollout must advance all 240 actual states')
    controls = []
    evidence_paths = [pair_path, condition_path, consumed_path, fresh_path, short_path,
                      reservation_path, selection_path, bank_path]
    for mode in ('direct', 'reuse'):
        for role in ('learned', 'OFFLINE'):
            path = BASE / f'pressure-history-{mode}-{role}-12/results.json'
            data = checked_evaluation(path, 720, False)
            controls.append(dict(mode=mode, role=role, independent=False,
                                 numerical_oracle_excluded_from_candidates=role == 'OFFLINE',
                                 summaries=data['summaries']))
            evidence_paths.append(path)
    costs = {}
    for label in ('p1', 'p2', 'p4'):
        path = BASE / f'sampler-minmax-{label}-cost-12.json'
        data = read(path)
        if data['status'] != 'completed_paired_minmax_sampler_comparison' or not data['all_intermediate_outputs_bitwise_equal']:
            raise ValueError('Matched complete costs with exact outputs required')
        costs[label] = dict(cores=data['cores'], summaries=data['summaries'],
                            clock_scope=data['clock_scope'], execution_team=data['execution_team'])
        evidence_paths.append(path)
    scaling = []
    for k in (1, 2, 3, 4):
        def values(label):
            return next(s for s in costs[label]['summaries'] if s['regime'] == '32cubed' and s['horizons'] == k)
        one = values('p1')['median_ms_per_state']['minmax_v12']
        for label in ('p1', 'p2', 'p4'):
            s = values(label)
            n = len(costs[label]['cores'])
            med = s['median_ms_per_state']['minmax_v12']
            scaling.append(dict(horizons=k, cores=n, median_ms_per_step=med,
                p95_ms_per_step=s['p95_ms_per_state']['minmax_v12'],
                maximum_ms_per_step=s['maximum_ms_per_state']['minmax_v12'],
                median_block_ms=s['median_block_ms']['minmax_v12'],
                p95_block_ms=s['p95_block_ms']['minmax_v12'],
                maximum_block_ms=s['maximum_block_ms']['minmax_v12'],
                speedup_over_matching_single_core=one/med,
                core_normalized_efficiency=one/(n*med),
                speedup_over_same_core_v7=s['median_speedup']))
    cache, checks = {}, []
    for path in sorted(DOCS.glob('*.json')):
        document = read(path)
        for field in ('source_sha256', 'source_evidence', 'execution_assets_sha256'):
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
                    raise ValueError('Registered evidence changed: ' + relative(path) + ' ' + name)
                count += 1
            checks.append(dict(registration=relative(path), field=field, sources_verified=count))
    fresh_pass = {str(k): all(s['all_fields_pass'] for s in fresh['summaries'] + short['summaries']
                             if s['horizons'] == k) for k in (1, 2, 3, 4)}
    failed_scenes = sorted({row['identifier'] for row in fresh['rows'] + short['rows']
                            if not row['metrics']['all_accuracy_criteria_pass']})
    gates = {str(k): False for k in (1, 2, 3, 4)}
    evidence = {relative(p): digest(p) for p in evidence_paths}
    summary = dict(status='completed_pressure_history_independent_check_latency_targets_unmet',
        updated_unix=time.time(), managed_pair=pair, new_managed_optimizer_calls=80,
        new_optimizer_calls_in_composition_and_execution=0,
        consumed_development_states=38400, consumed_summaries=consumed['summaries'],
        independently_reserved_scenes=16, independent_own_history_states=15360,
        independent_short_block_states=320, independent_summaries=fresh['summaries'],
        independent_short_summaries=short['summaries'], independent_accuracy_pass=fresh_pass,
        independent_failed_scenes=failed_scenes, pressure_history_controls=controls,
        corresponding_K_outputs_byte_exact=True, cpu_costs=costs, cpu_scaling=scaling,
        maximum_allowed_physical_cores=4, single_core_is_primary_efficiency_metric=True,
        full_generator_targets_verified=gates, higher_density_phase_started=False,
        phase600_started=False, source_registrations_verified=checks,
        distinct_artifacts_verified=len(cache), evidence=evidence,
        publisher_sha256=digest(__file__), report=relative(report))
    lines = ['# Learned pressure reuse, mixed-query refinement and CPU efficiency', '',
        'The learned causal pressure-reuse prototype passes all 40 consumed development scenes for K1–4. '
        f'Its frozen independent check passes {16-len(failed_scenes)} of 16 new whole scenes. '
        'The maximum 1.2ms per generated step target remains unmet on one, two and four cores. '
        'No joint target or later-phase gate is declared complete.', '',
        '## Managed training and data', '',
        'All144 original native training pairs were retained byte for byte, with 128 mixed-mode '
        'reference queries added across the same four geometries. These inputs improve modal signal '
        'coverage using only training data. Numerical pressure references produce offline labels only; '
        'they never enter candidate inference. The four minimum modal RMS/global ratios rose from '
        'approximately 4.69e-6,4.57e-5,1.06e-3 and 1.28e-7 to 0.405,0.449,0.471 and0.353.', '',
        f"Managed CUDA training Run `{pair['training_run_id']}` performed 80 LBFGS calls. "
        f"Exact typed CPU restore Run `{pair['restore_run_id']}` reproduced all five archived artifacts. "
        'The resulting weighted objective was 2.26634e-25. The full generator retains 40,017 learned '
        'parameters; all new parameter fitting ran on managed CUDA. Raw checkpoints are unchanged. '
        'The inference head uniformly rounds its 48 managed-restored parameters to 2^-40.', '',
        'The initial recipe declared 64 MiB per input, above the actual 40 MiB contract. The largest '
        'shard is 34,418,718 bytes. An actual decoder red/green check isolated that declaration; '
        'the version 11b project changes only the declared cap to 40 MiB. Both training and restore '
        'recipes passed before admission. Earlier evidence remains preserved.', '',
        '## Why pressure history mattered', '',
        'Direct learned pressure still failed two of40 consumed scenes. Even a numerical pressure '
        'oracle in that direct formulation failed two scenes. The native reference uses the '
        'supplied current pressure as a starting point. The new candidate expresses that causal '
        'skip path using only existing learned flux, tied learned momentum and learned pressure '
        'modules: form the learned residual from the current state, predict its pressure correction, '
        'add current pressure and regauge. It performs no numerical pressure solve or analytical '
        'inverse, makes no teacher resets and sees no future labels. It required no new parameter fit.', '',
        '| Three-scene control | Pressure role | Small-grid worst equation |32³ worst equation |',
        '| --- | --- | ---: | ---: |']
    for c in controls:
        s = {r['regime']: r for r in c['summaries']}
        lines.append(f"| {c['mode']} | {c['role']} | {s['small']['worst']['equation']:.6g} | {s['32cubed']['worst']['equation']:.6g} |")
    lines += ['', 'OFFLINE rows are diagnostic oracles, excluded from AI candidates, latency and frontier claims.', '',
        '## Frozen independent accuracy', '',
        'The selected model, inference sources, raw managed evidence and equivalent execution '
        'variants were frozen before generating the 16 previously reserved reference trajectories. '
        'Every complete pressure, velocity and tracer state is scored, including every intermediate '
        'K2–4 output. Own-history tests run 240 actual simulation steps from the initial state.', '',
        '| K | Grid regime | States | Worst equation | Worst pressure | Worst velocity | Worst tracer | All pass |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for s in fresh['summaries']:
        w = s['worst']
        lines.append(f"| {s['horizons']} | {s['regime']} | {s['complete_states']} | {w['equation']:.6g} | {w['pressure']:.6g} | {w['velocity']:.6g} | {w['tracer']:.6g} | {s['all_fields_pass']} |")
    lines += ['', f"Failed independent scenes: {', '.join(failed_scenes) if failed_scenes else 'none'}. "
        'These results describe the frozen candidate; any successor design based on them must '
        'treat the scenes as consumed. Corresponding own-history physical outputs match byte for '
        'byte across K1–4. K here packages recurrent generation and does not establish that '
        'multiple future states are computed in parallel.', '',
        '## Single-core efficiency and measured scaling', '',
        'Finite donor min/max selection instructions replace equivalent comparisons and selects. '
        'Signed-zero operand order is preserved. For each allocation, 336 component cases and 64 '
        'complete states matched the qualified serial implementation byte for byte before timing. '
        'Alternating v7/minmax measurements share learned tensors and FFTW owners. The clock '
        'includes input checks, forcing, all learned stages and every intermediate output. '
        'One core is the primary efficiency metric; multicore results remain separate measured options.', '',
        '| K | Physical cores | Median ms/step | P95 ms/step | Maximum ms/step | Median block ms | Speedup vs1 core | Core efficiency |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in scaling:
        lines.append(f"| {s['horizons']} | {s['cores']} | {s['median_ms_per_step']:.3f} | {s['p95_ms_per_step']:.3f} | {s['maximum_ms_per_step']:.3f} | {s['median_block_ms']:.3f} | {s['speedup_over_matching_single_core']:.3f} | {s['core_normalized_efficiency']:.3f} |")
    lines += ['', 'These are 32³ consumed-development measurements on CPU 6, CPUs 6/4 and CPUs 6/4/5/7, '
        'with verified placement. Larger allocations are not consistently faster. In this sample, '
        'two cores have the lowest median for each K and lower maximum latency than four cores. '
        'The maximum, rather than the median, must meet the declared 1.2ms target. Historical '
        'CPU 23 efficiency-core results are separate. This experiment does not remeasure a matched '
        'Rheon application comparison.', '',
        'The build-only registration retains inactive CPU, pair and old implementation-description '
        'fields inherited from its earlier template. Actual qualification and timing use their '
        'explicit resource protocols and the frozen pressure-reuse factory. '
        '[The scope clarification](minmax-execution-metadata-clarification-12.json) records which '
        'fields the builder consumes; all bound files remain unchanged.', '',
        '## Next work and preserved gates', '',
        'The next speed experiment should fuse construction of learned pressure features into '
        'compiled execution to remove repeated tensor indexing and temporary arrays, followed by '
        'complete-block fusion to reduce Python dispatch and thread synchronization. Coefficients '
        'must remain inherited learned tensors. Require exact agreement with this frozen graph '
        'before measuring cost. Keep single-core cost and measured two/four-core scaling visible. '
        'Do not add local fitting, numerical pressure solvers or future reference features.', '',
        'The 1.2 ms and 1e-5 joint gates remain open for every K. Densities 50/100, random-aspect larger '
        'domains and 600-step 30/60 Hz drift optimization remain behind their ordered gates. '
        'Registered sources and original evidence were audited without modifying bound files.', '',
        '[Machine-readable results](pressure-history-and-mixed-query-summary-12.json) · '
        '[Frozen selection](pressure-history-frozen-selection-12.json) · '
        '[CPU resource policy](cpu-resource-policy-04.md)', '']
    report.write_text('\n'.join(lines))
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    state = dict(status='active_phase1_pressure_reuse_independent_check_complete_latency_unmet',
        updated_unix=time.time(), previous_goal_turn_classification='progress', current_live_processes=[],
        new_managed_optimizer_calls_this_goal_turn=80, new_optimizer_calls_in_diagnostics=0,
        maximum_allowed_physical_cores=4, single_core_is_primary_efficiency_metric=True,
        consumed_scene_development_accuracy_pass=True, independent_accuracy_pass=fresh_pass,
        fresh_successor_scenes_consumed=16, full_generator_targets_verified=gates,
        higher_density_phase_started=False, phase600_started=False, reports=[relative(report)],
        source_registrations_verified=checks, original_completed_study_evidence_unchanged=True,
        next=['Compile the frozen learned pressure feature composition and qualify byte-exact outputs',
              'Fuse complete learned execution to reduce allocations, dispatch and OpenMP synchronization',
              'Measure one-core efficiency and separate two/four-core full-block latency',
              'Reserve new whole-scene confirmation before any successor selection using fresh results',
              'Keep higher-density and600-step phases gated on all joint targets'])
    state_path.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    print('Published pressure-history accuracy and CPU efficiency; phase1 remains active', flush=True)


if __name__ == '__main__':
    main()
