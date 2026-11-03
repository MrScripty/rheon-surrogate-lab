"""Publish serial efficiency and bounded core scaling without changing target gates."""
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
    destinations = [DOCS / 'recurrent-sampler-followup-results-02.md',
                    DOCS / 'recurrent-sampler-followup-summary-02.json',
                    DOCS / 'work-state-05.json']
    if any(path.exists() for path in destinations):
        raise ValueError('Fresh follow-up publication required')
    costs, apps, evidence = {}, {}, {}
    for label in ('e1', 'p1', 'p2', 'p4'):
        for kind in ('transport', 'graph', 'cost'):
            path = BASE / f'sampler-cores-{label}-{kind}-06.json'
            data = read(path)
            evidence[str(path.relative_to(ROOT))] = digest(path)
            if kind == 'transport':
                if data['status'] != 'passed_bitwise_sampler_address_qualification' or len(data['rows']) != 336:
                    raise ValueError('Incomplete donor-address qualification')
            elif kind == 'graph':
                if data['status'] != 'passed_parallel_complete_graph_execution' or data['complete_states'] != 64:
                    raise ValueError('Incomplete actual graph qualification')
            else:
                if data['status'] != 'completed_paired_parallel_sampler_comparison' or len(data['rows']) != 64 or not data['all_intermediate_outputs_bitwise_equal']:
                    raise ValueError('Incomplete paired timing/equivalence evidence')
                costs[label] = data
    for label in ('p1', 'p2', 'p4'):
        path = BASE / f'sampler-cores-{label}-application-06/results.json'
        data = read(path)
        timed = [row for row in data['rows'] if not row['warmup']]
        if data['status'] != 'completed_matched_application_comparison' or len(timed) != 36 or len(data['rows']) != 48:
            raise ValueError('Incomplete matched application population')
        if any(not row['result']['completed_physical_interval'] for row in data['rows']):
            raise ValueError('Native or candidate interval failed')
        for row in data['rows']:
            for owner in row['result']['native_processes']:
                if not owner['reaped'] or owner['returncode'] != 0 or not owner['graceful_protocol_complete']:
                    raise ValueError('Undrained native application owner')
                resource = owner['resource_evidence']
                if resource['cores'] != data['cores'] or any(set(v) != set(data['cores']) for v in resource['native_task_affinities'].values()):
                    raise ValueError('Native controls escaped their core allocation')
        apps[label] = data
        evidence[str(path.relative_to(ROOT))] = digest(path)
    for relative in (
        'modal-precision-pair-02/receipt.json',
        'modal-precision-component-evaluation-02/results.json',
        'refined-pressure-sampler-development-rollout-02/results.json',
        'sampler-precision-tensor-dct-rollout-01/results.json',
        'sampler-precision-quantized-locals-rollout-01/results.json',
        'sampler-precision-tensor-dct-quantized-locals-rollout-01/results.json',
        'sampler-offline-pressure-floor-01/results.json',
        'trained-sampler-physical-rollout-04/results.json',
        'paired-sampler-load-cost-03.json',
        'paired-sampler-call-boundary-05.json',
    ):
        path = BASE / relative
        evidence[str(path.relative_to(ROOT))] = digest(path)
    # Verify all prospective and historical registrations, hashing each distinct
    # artifact only once. Earlier evidence remains immutable.
    checks, cache = [], {}
    for path in sorted(DOCS.glob('*.json')):
        value = read(path)
        for field in ('source_sha256', 'source_evidence'):
            sources = value.get(field)
            if not isinstance(sources, dict):
                continue
            verified = 0
            for name, expected in sources.items():
                if not isinstance(expected, str) or len(expected) != 64:
                    continue
                if name not in cache:
                    cache[name] = digest(ROOT / name)
                if cache[name] != expected:
                    raise ValueError('Registered source/evidence changed: ' + str(path) + ' ' + name)
                verified += 1
            checks.append(dict(registration=str(path.relative_to(ROOT)), field=field,
                               sources_verified=verified))
    by_k = {label: {row['horizons']: row for row in data['summaries']
                    if row['regime'] == '32cubed'} for label, data in costs.items()}
    table = []
    for k in (1, 2, 3, 4):
        table.append(dict(horizons=k,
                          single_p_core_serial_ms_per_state=by_k['p1'][k]['median_ms_per_state']['serial_v5'],
                          two_p_core_ms_per_state=by_k['p2'][k]['median_ms_per_state']['parallel_v6'],
                          four_p_core_ms_per_state=by_k['p4'][k]['median_ms_per_state']['parallel_v6'],
                          four_p_core_p95_ms_per_state=by_k['p4'][k]['p95_ms_per_state']['parallel_v6'],
                          four_p_core_maximum_ms_per_state=by_k['p4'][k]['maximum_ms_per_state']['parallel_v6'],
                          paired_two_core_speedup=by_k['p2'][k]['median_speedup'],
                          paired_four_core_speedup=by_k['p4'][k]['median_speedup'],
                          paired_two_core_speedup_per_allocated_core=by_k['p2'][k]['median_speedup'] / 2,
                          paired_four_core_speedup_per_allocated_core=by_k['p4'][k]['median_speedup'] / 4))
    summary = dict(status='completed_cpu_efficiency_and_scaling_targets_unmet',
                   updated_unix=time.time(), resource_policy='docs/plans/full-ai-tight-targets-v1/cpu-resource-policy-04.md',
                   maximum_allowed_physical_cores=4, single_core_is_primary_efficiency_metric=True,
                   complete_32cubed_costs=table,
                   all_regimes_and_horizons={label: data['summaries'] for label, data in costs.items()},
                   matched_applications={label: data['summaries'] for label, data in apps.items()},
                   qualification_component_cases=1344, qualification_complete_states=256,
                   paired_timed_complete_blocks=2560, timed_matched_application_episodes=108,
                   warmup_matched_application_episodes=36, cpu_experiment_optimizer_updates=0,
                   fresh_holdout_consumed=False, all_native_processes_reaped=True,
                   source_registrations_verified=checks, distinct_registered_artifacts_verified=len(cache),
                   evidence=evidence, publisher_sha256=digest(__file__),
                   full_generator_targets_verified={str(k): False for k in (1, 2, 3, 4)},
                   higher_density_phase_started=False, phase600_started=False)
    lines = ['# Recurrent sampler follow-up: CPU efficiency and precision', '',
             'The owner permits up to four physical CPU cores while retaining single-core latency as a primary efficiency metric. Additional cores are optional; every measured regression remains visible. The 1.2 ms maximum per generated step and 1e-5 worst-error gates remain unmet.', '',
             '## Controlled CPU scaling', '',
             'The existing managed-trained and exactly CPU-restored model is unchanged. A new strict OpenMP implementation divides independent rows in transport and sparse learned stages; pressure transforms and prescribed forcing remain serial. It preserves per-voxel operation order, guards and every intermediate physical output. No training occurred in this CPU experiment.', '',
             'Performance-core configurations use CPU 6, CPUs 6/4 and CPUs 6/4/5/7. Historical efficiency-core CPU 23 is measured separately. Worker placement is verified in fresh processes; Torch and BLAS remain single-threaded. Passive workers prevent an idle worker pool from intentionally spinning during serial controls. The comparison includes checks, conditioning, all model stages and every complete intermediate output.', '',
             'Each resource allocation passed 336 donor-address controls and 64 complete four-step state comparisons. Across all four allocations, 2,560 alternating timed complete blocks were byte-identical to the serial implementation. These are development measurements, without new held-out scene consumption.', '',
             '| Steps/pass | One P core, serial ms/step | Two P cores ms/step | Four P cores ms/step | Four-core p95 | Four-core maximum |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in table:
        values = [row[key] for key in ('single_p_core_serial_ms_per_state', 'two_p_core_ms_per_state',
                  'four_p_core_ms_per_state', 'four_p_core_p95_ms_per_state', 'four_p_core_maximum_ms_per_state')]
        lines.append('| ' + str(row['horizons']) + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += ['', 'All values are for 32 cubed. The first three columns are medians. For K steps, multiply a per-step value by K to obtain full-block latency; the machine-readable evidence also retains full-block median, p95 and maximum observations.', '',
              'Within the alternating measurements, two cores provide about 1.56–1.59x speedup over the paired serial path and four provide about 1.67–2.03x. Speedup divided by allocated cores is about 0.78–0.79 and 0.42–0.51 respectively; this is a resource-efficiency proxy, not a CPU utilization measurement. These experiments do not isolate memory bandwidth, cache sharing, scheduling or synchronization as the cause of sublinear scaling.', '',
              'The OpenMP implementation is slower when constrained to one core, so the original serial path remains the single-core implementation. CPU 23 measurements retain their separate hardware label. Small-domain timing tails also worsen with extra cores, including one roughly 30 ms/step four-core observation. Extra cores improve some medians but do not establish a maximum-latency guarantee. No timing outliers were removed.', '',
              '## Matched Rheon application controls', '',
              'Both 32-cubed development scenes, densities 8 and 16 cells per metre, steady forcing, 24 accepted steps, three timed repeats plus a separate 14-step warmup. Each configuration compares K1–4 generators with native PCG and the spectral reference control. Native processes and Rayon workers receive the same physical core allowance as the generator. Parent pressure-exchange transforms remain single-threaded; native process affinities are checked. The one-core generator uses the faster original serial implementation.', '',
              '| Core allowance | Method | Completed timed episodes | Median application ms/accepted step |',
              '| --- | --- | ---: | ---: |']
    for label in ('p1', 'p2', 'p4'):
        for row in apps[label]['summaries']:
            lines.append(f"| {len(apps[label]['cores'])} | {row['method']} | {row['completed']}/{row['episodes']} | {row['median_application_ms_per_accepted_step']:.3f} |")
    lines += ['', 'The application clock includes prediction or numerical advance, physical checks, state exchange and final verification/close; setup and pure offline exports are separate. All 108 timed and 36 warmup episodes completed, with every native owner reaped. Neural consumers perform zero native numerical advances. Native control physical gates remain unchanged. Physical outputs are retained for deferred accuracy analysis; this timing experiment does not newly certify long-history accuracy or replace the original independent 240-step evaluation.', '',
              '## Pressure precision investigation retained from this follow-up', '',
              'A bounded managed CUDA refinement of the existing 38,673 learned modal gains completed 80 LBFGS calls and an exact typed managed CPU restore. The original attempt completed its updates but failed publication because its 1,218,858-byte normalization artifact exceeded a declared 1 MiB output cap. A separately registered attempt corrected only that declaration to 2 MiB and completed train/restore custody. Both attempts and the red/green publication regression remain preserved.', '',
              'On 1,920 native-RHS development requests, refined learned pressure reaches approximately 1.23e-12 worst equation residual, versus roughly 1.07e-9 previously. However, the complete generator still reaches about 1.07e-4 reference-equation difference over 240 own-history steps at 32 cubed. The same-request pressure improvement does not establish an improved trajectory. All K1–4 complete own-history paths were evaluated; no target pass is claimed.', '',
              'Changing only the fixed transform implementation to explicit tensor cosine transforms reduces the small-domain own-history equation difference to about 1.18e-13 with zero velocity difference, but the 32-cubed difference remains about 1.06e-4. Uniform local/head coefficient rounding and the physical-coordinate arithmetic control do not resolve it. An explicitly offline numerical pressure oracle was a diagnostic only and is excluded from candidate/frontier results. No analytical inverse was inserted into candidate inference.', '',
              'The refined pressure checkpoint is a development result. The 16 newly reserved precision-confirmation scenes remain unconsumed. The original fully independently tested prototype remains available.', '',
              '## Next work', '',
              'Keep the faster serial path, treat two/four cores as explicit deployment options, and reduce complete work and timing tails before claiming a speed pass. Continue narrowing the remaining 32-cubed trajectory differences with bounded controls. Freeze any selected successor before independent confirmation. Higher-density training and the 600-step phase remain behind the existing ordered gates.', '',
              '[CPU resource policy](cpu-resource-policy-04.md) · [Machine-readable results](recurrent-sampler-followup-summary-02.json) · [Original independent prototype results](recurrent-sampler-results-01.md)', '']
    destinations[0].write_text('\n'.join(lines))
    destinations[1].write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    state = dict(status='active_phase1_cpu_scaling_and_pressure_refinement_complete_targets_unmet',
                 updated_unix=time.time(), current_live_processes=[], previous_goal_turn_classification='progress',
                 maximum_allowed_physical_cores=4, single_core_is_primary_efficiency_metric=True,
                 resource_policy=summary['resource_policy'], report=str(destinations[0].relative_to(ROOT)),
                 cpu_experiment_optimizer_updates=0,
                 pressure_refinement_successful_optimizer_calls=80,
                 pressure_refinement_failed_publication_optimizer_calls=80,
                 fresh_precision_reserved_scenes=16, fresh_precision_scenes_consumed=0,
                 full_generator_targets_verified=summary['full_generator_targets_verified'],
                 higher_density_phase_started=False, phase600_started=False,
                 source_registrations_verified=checks,
                 original_completed_study_evidence_unchanged=True, evidence=evidence,
                 next=['Preserve best serial efficiency and explicit multi-core deployment results',
                       'Reduce complete inference work and timing tails; resolve 32-cubed own-history equation drift',
                       'Freeze selected successor before independent confirmation; retain ordered later-phase gates'])
    destinations[2].write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    print('Published CPU efficiency/scaling; complete target program remains active', flush=True)


if __name__ == '__main__':
    run()
