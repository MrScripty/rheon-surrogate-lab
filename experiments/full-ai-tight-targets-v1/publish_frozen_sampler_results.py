"""Publish exact learned-point specialization evidence and retain target failures."""
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


def run():
    report = DOCS / 'frozen-point-execution-results-07.md'
    summary_path = DOCS / 'frozen-point-execution-summary-07.json'
    state_path = DOCS / 'work-state-06.json'
    if any(path.exists() for path in (report, summary_path, state_path)):
        raise ValueError('Fresh specialization publication required')
    evidence, costs, profiles = {}, {}, {}
    for label in ('p1', 'p4'):
        for kind in ('transport', 'graph', 'cost'):
            path = BASE / f'frozen-sampler-{label}-{kind}-07.json'
            data = read(path)
            evidence[str(path.relative_to(ROOT))] = digest(path)
            if kind == 'transport':
                if data['status'] != 'passed_bitwise_sampler_address_qualification' or len(data['rows']) != 336:
                    raise ValueError('Incomplete address qualification')
            elif kind == 'graph':
                if data['status'] != 'passed_parallel_complete_graph_execution' or data['complete_states'] != 64 or not data['all_physical_output_bytes_exact']:
                    raise ValueError('Incomplete exact complete graph qualification')
            else:
                if data['status'] != 'completed_paired_frozen_sampler_comparison' or len(data['rows']) != 64 or not data['all_intermediate_outputs_bitwise_equal']:
                    raise ValueError('Incomplete paired complete-block evidence')
                costs[label] = data
        path = BASE / f'sampler-cores-{label}-profile-07.json'
        profiles[label] = read(path)
        if not profiles[label]['predictions_bitwise_unchanged']:
            raise ValueError('Profile instrumentation changed outputs')
        evidence[str(path.relative_to(ROOT))] = digest(path)
    for relative in ('frozen-sampler-guards-07.json',
                     'frozen-point-specialization-07/receipt.json',
                     'frozen-point-specialization-07/frozen_point.hpp',
                     'recurrent-sampler-frozen-build-07/receipt.json',
                     'recurrent-sampler-frozen-build-07/librecurrent_sampler.so'):
        path = BASE / relative
        evidence[str(path.relative_to(ROOT))] = digest(path)
    guards = read(BASE / 'frozen-sampler-guards-07.json')
    if guards['status'] != 'passed_frozen_coefficient_guards' or len(guards['rows']) != 4:
        raise ValueError('Frozen coefficient mismatch guards unqualified')
    cache, checks = {}, []
    for path in sorted(DOCS.glob('*.json')):
        value = read(path)
        for field in ('source_sha256', 'source_evidence'):
            sources = value.get(field)
            if not isinstance(sources, dict):
                continue
            count = 0
            for name, expected in sources.items():
                if not isinstance(expected, str) or len(expected) != 64:
                    continue
                if name not in cache:
                    cache[name] = digest(ROOT / name)
                if cache[name] != expected:
                    raise ValueError('Historical/registered evidence changed: ' + str(path) + ' ' + name)
                count += 1
            checks.append(dict(registration=str(path.relative_to(ROOT)), field=field,
                               sources_verified=count))
    lines = ['# Frozen learned point execution: controlled results', '',
             'Compiling the existing learned point projection into specialized code reduces single-core latency without changing the checked physical output bytes. Four-core results are mixed. The complete 1.2 ms maximum and 1e-5 own-history targets remain unmet.', '',
             '## Mechanism and provenance', '',
             'The generator reads the exact inherited managed CPU checkpoint through the existing uniform inference quantization and CSR ordering. It emits the 33 nonzero point coefficients as exact hexadecimal double literals, preserving term order. Carrier indices become compile-time constants. Query gates and mixing remain the original 48 managed-trained double parameters. Viscosity, flux, pressure, momentum, forcing and per-voxel arithmetic are unchanged. No analytical stencil or new weight is assigned, and no learning occurs.', '',
             'The compiled kernel checks all incoming point CSR rows, channels, offsets and weights against its generated constants before use. Separate controls reject a changed member of each array and an undeclared three-worker configuration. The generated header, source checkpoint provenance, strict compiler flags, binary and all registrations are retained.', '',
             '## Correctness', '',
             'One and four performance-core configurations each passed 336 boundary, extreme and subnormal donor-address controls plus 64 complete four-step state comparisons against the declared tensor graph. All complete fields were also byte-identical to the previous serial implementation. Every output in 1,280 alternating timed complete blocks matched the baseline bytes. No independent scene reservation was consumed. Long-history accuracy was not rerun for this execution-only change; the previous measured accuracy failures remain active.', '',
             '## Paired complete latency at 32 cubed', '',
             '| Cores | Steps/pass | Baseline median ms/step | Specialized median ms/step | Baseline/specialized | Specialized p95 | Specialized maximum |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for label in ('p1', 'p4'):
        for row in costs[label]['summaries']:
            if row['regime'] != '32cubed':
                continue
            lines.append(f"| {len(costs[label]['cores'])} | {row['horizons']} | {row['median_ms_per_state']['baseline']:.3f} | {row['median_ms_per_state']['frozen_v7']:.3f} | {row['median_speedup']:.3f} | {row['p95_ms_per_state']['frozen_v7']:.3f} | {row['maximum_ms_per_state']['frozen_v7']:.3f} |")
    lines += ['', 'The one-core comparator is the faster serial v5 implementation on CPU 6. The four-core comparator is v6 on CPUs 6/4/5/7. Each pair shares the same learned model and pressure-plan owners, with alternating execution order and five repeats per scene/start/horizon. Input checks, forcing, every model stage, complete intermediate outputs and synchronization inside inference count. Output hashing occurs outside the clock. All raw observations, including timing tails, remain available.', '',
              'Single-core median latency falls about 9–10% at 32 cubed. Four-core K1–3 medians improve modestly, while K4 is about 1.6% slower in this run. Keep the specialization as a single-core optimization and retain the preceding parallel path for comparison. These are development timings and do not establish independent timing qualification or a maximum-latency guarantee. Compare paired ratios rather than interpreting different absolute timings across sessions as additional optimization gains.', '',
              '## Profile informing this experiment', '',
              '| CPU allocation | Median complete profiled ms | Transport | Learned viscosity | Learned pressure | Forcing |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label in ('p1', 'p4'):
        row = next(v for v in profiles[label]['summaries'] if v['regime'] == '32cubed')
        stages = row['stage_median_ms']
        lines.append(f"| {len(profiles[label]['cores'])} | {row['whole_median_ms']:.3f} | {stages['transport']:.3f} | {stages['spatial0']:.3f} | {stages['pressure']:.3f} | {stages['forcing']:.3f} |")
    lines += ['', 'These profiles describe the preceding serial/parallel implementations, with instrumentation overhead; they are diagnostic costs, not the specialized model benchmark. Transport remains the largest measured stage. Even removing one stage would leave substantial complete work, so no extrapolated target pass is claimed.', '',
              '## Next work', '',
              'Retain single-core efficiency as a primary metric and preserve the two/four-core deployment alternatives. Further speed work must reduce complete work rather than assume additional cores scale linearly. The remaining 32-cubed own-history equation difference also needs resolution before the reference target can pass. Freeze a selected successor and execute independent confirmation before opening higher-density or 600-step stages.', '',
              '[Machine-readable evidence](frozen-point-execution-summary-07.json) · [CPU resource policy](cpu-resource-policy-04.md) · [Previous complete prototype and CPU results](recurrent-sampler-followup-results-02.md)', '']
    summary = dict(status='completed_frozen_point_execution_single_core_improved_targets_unmet',
                   updated_unix=time.time(), optimizer_updates=0, generated_learned_point_nonzero=33,
                   new_analytical_coefficients_assigned=False, fresh_holdout_consumed=False,
                   all_observed_complete_outputs_bitwise_equal=True,
                   complete_address_controls=672, complete_graph_states=128,
                   paired_timed_complete_blocks=1280,
                   costs={label: data['summaries'] for label, data in costs.items()},
                   baseline_stage_profiles={label: data['summaries'] for label, data in profiles.items()},
                   source_registrations_verified=checks, distinct_artifacts_verified=len(cache),
                   evidence=evidence, publisher_sha256=digest(__file__),
                   maximum_allowed_physical_cores=4, single_core_is_primary_efficiency_metric=True,
                   full_generator_targets_verified={str(k): False for k in (1, 2, 3, 4)},
                   higher_density_phase_started=False, phase600_started=False,
                   report=str(report.relative_to(ROOT)))
    report.write_text('\n'.join(lines))
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    state = dict(status='active_phase1_single_core_execution_improved_targets_unmet',
                 previous_goal_turn_classification='progress', updated_unix=time.time(),
                 current_live_processes=[], source_registrations_verified=checks,
                 maximum_allowed_physical_cores=4, single_core_is_primary_efficiency_metric=True,
                 new_optimizer_updates=0, fresh_precision_reserved_scenes=16,
                 fresh_precision_scenes_consumed=0, full_generator_targets_verified=summary['full_generator_targets_verified'],
                 higher_density_phase_started=False, phase600_started=False,
                 report=summary['report'], original_completed_study_evidence_unchanged=True,
                 next=['Preserve verified single-core specialization and explicit parallel alternatives',
                       'Reduce transport/conditioning/pressure complete work while retaining precision',
                       'Narrow 32-cubed pressure/trajectory rounding with bounded learned-operator controls',
                       'Freeze selected successor before independent confirmation and ordered later phases'])
    state_path.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    print('Published exact learned-point specialization; staged objective remains active', flush=True)


if __name__ == '__main__':
    run()
