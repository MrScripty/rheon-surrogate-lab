"""Publish consumed-scene failure isolation with numerical oracles kept separate."""
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
    report = DOCS / 'pressure-floor-and-head-precision-results-10.md'
    summary_path = DOCS / 'pressure-floor-and-head-precision-summary-10.json'
    state_path = DOCS / 'work-state-09.json'
    if any(path.exists() for path in (report, summary_path, state_path)):
        raise ValueError('Fresh precision-diagnostic publication required')
    baseline_path = BASE / 'decoder-modal-fresh-rollout-09/results.json'
    baseline = read(baseline_path)
    bank = read(BASE / 'decoder-modal-fresh-reference-09/bank.json')
    selected = ('modal-precision-fresh-03', 'modal-precision-fresh-15')
    paths = {'frozen_learned_baseline': baseline_path,
             'OFFLINE_numerical_pressure': BASE / 'decoder-modal-OFFLINE-pressure-floor-09/results.json'}
    for force in ('gaussian', 'scalar'):
        for pressure in ('learned', 'OFFLINE'):
            paths[f'forcing_{force}_{pressure}'] = BASE / f'conditioning-{force}-{pressure}-10/results.json'
    for bits in (30, 40):
        for pressure in ('learned', 'OFFLINE'):
            paths[f'head_m{bits}_{pressure}'] = BASE / f'decoder-head-m{bits}-{pressure}-10/results.json'
    for pressure in ('tensor-multiplier', 'tensor-division', 'fftw-division'):
        paths['head_m40_' + pressure] = BASE / f'decoder-pressure-{pressure}-10/results.json'
    results = []
    for name, path in paths.items():
        data = read(path)
        if data['optimizer_updates'] != 0 or not data['all_plan_owners_closed']:
            raise ValueError('Every diagnostic must be zero-update and close its owners')
        rows = [row for row in data['rows'] if row['horizons'] == 1 and row['identifier'] in selected]
        if len(rows) != 480:
            raise ValueError('Every matched diagnostic needs both complete own240 scenes')
        values = {}
        for identifier in selected:
            chosen = [row for row in rows if row['identifier'] == identifier]
            values[identifier] = dict(worst={key: max(row['metrics']['relative'][key] for row in chosen)
                                      for key in ('equation', 'pressure', 'velocity', 'tracer')},
                                      all_fields_pass=all(row['metrics']['all_accuracy_criteria_pass'] for row in chosen))
        results.append(dict(name=name, pressure_role='OFFLINE numerical oracle' if 'OFFLINE' in name else 'managed learned pressure',
                            scenes=values, no_new_fit=True, independent_confirmation=False))
    failing = []
    for trajectory in bank['trajectories']:
        rows = [row for row in baseline['rows'] if row['horizons'] == 1 and row['identifier'] == trajectory['identifier']]
        failed = [row for row in rows if not row['metrics']['all_accuracy_criteria_pass']]
        if not failed:
            continue
        receipt = read(ROOT / trajectory['receipt'])
        first = next((row['step'] for row in rows if any(
            row['output_identity'][name]['sha256'] != receipt['states'][row['step']]['arrays'][name]['sha256']
            for name in ('vx', 'vy', 'vz', 'tracer'))), None)
        failing.append(dict(identifier=trajectory['identifier'], first_carrier_byte_difference_step=first,
                            first_accuracy_failure_step=failed[0]['step'],
                            worst_equation=max(row['metrics']['relative']['equation'] for row in rows)))
    if {row['identifier'] for row in failing} != set(selected):
        raise ValueError('Frozen independent failure population changed')
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
                    raise ValueError('Registered source/evidence changed: ' + str(path) + ' ' + name)
                count += 1
            checks.append(dict(registration=str(path.relative_to(ROOT)), field=field, sources_verified=count))
    reservation = DOCS / 'pressure-successor-fresh-scene-reservation-10.json'
    reserved = read(reservation)
    if len(reserved['rows']) != 16 or reserved['reserved_references_captured']:
        raise ValueError('New unconsumed confirmation reservation required')
    evidence = {str(path.relative_to(ROOT)): digest(path) for path in paths.values()}
    evidence[str(reservation.relative_to(ROOT))] = digest(reservation)
    summary = dict(status='completed_consumed_scene_precision_isolation_targets_unmet',
                   updated_unix=time.time(), new_optimizer_updates=0,
                   additional_causal_diagnostic_states=9120,
                   frozen_candidate_failure_scenes=failing, matched_control_results=results,
                   independent_candidate_scenes_passing=14, independent_candidate_scenes_total=16,
                   head_precision_repair_scope='K1, two consumed scenes only; no independent successor claim',
                   new_confirmation_scenes_reserved=16, new_confirmation_references_captured=0,
                   full_generator_targets_verified={str(k): False for k in (1, 2, 3, 4)},
                   maximum_allowed_physical_cores=4, single_core_is_primary_efficiency_metric=True,
                   higher_density_phase_started=False, phase600_started=False,
                   source_registrations_verified=checks, distinct_artifacts_verified=len(cache),
                   evidence=evidence, publisher_sha256=digest(__file__), report=str(report.relative_to(ROOT)))
    lines = ['# Pressure-floor and sampling-head precision diagnosis', '',
             'The frozen decoder-compatible candidate passes all short blocks and 14 of 16 independent own240 scenes. Its two equation-criterion failures remain real target failures. Consumed-scene controls identify one repair and narrow the remaining pressure issue; none is a new independently qualified candidate.', '',
             '## Failure timing', '',
             '| Consumed scene | First carrier byte difference | First 1e-5 criterion failure | Worst equation difference |',
             '| --- | ---: | ---: | ---: |']
    for row in failing:
        lines.append(f"| {row['identifier']} | {row['first_carrier_byte_difference_step']} | {row['first_accuracy_failure_step']} | {row['worst_equation']:.6g} |")
    lines += ['', 'Pressure, velocity and tracer errors remain below 1e-5 in both failures. The equation criterion includes accumulated differences from the independent native trajectory. Tiny carrier changes can grow into a failing equation difference, so isolated pressure residuals and training loss cannot replace own-history confirmation.', '',
              '## Matched controls on the two consumed scenes', '',
              '| Control | Pressure role | 16-cubed worst equation | 32-cubed worst equation |',
              '| --- | --- | ---: | ---: |']
    for row in results:
        lines.append(f"| {row['name']} | {row['pressure_role']} | {row['scenes'][selected[0]]['worst']['equation']:.6g} | {row['scenes'][selected[1]]['worst']['equation']:.6g} |")
    lines += ['', 'Every control predicts causally through all 240 steps from the initial physical state. Future labels enter only offline scoring. Oracle rows deliberately replace learned pressure with the numerical reference and are excluded from AI candidates, frontier claims and latency results. All other pressure rows use only exact managed learned parameters or declared arithmetic/precision transforms of those parameters. No new parameter fitting occurred in these diagnostics.', '',
              'The numerical pressure reference removes the 32-cubed failure while leaving the original 16-cubed failure. Replacing factorized Gaussian forcing with a scalar Gaussian, then also using scalar sine metadata, changes neither failure. These forcing changes are not adopted.', '',
              'Uniform rounding of all 48 restored query-head parameters to either 2^-30 or 2^-40 repairs the 16-cubed learned-pressure trajectory: worst equation difference becomes about 8.33e-14 and velocity difference is zero. The raw managed checkpoint is preserved, and no algorithmic coefficient is assigned. The head controls keep the original forcing implementation; an auxiliary execution clarification records an unused inherited forcing-policy field in their registrations.', '',
              'After head rounding, explicit cosine multiplication, explicit cosine division and FFTW division by derived learned values all leave the 32-cubed failure near 6.75e-5. These tests narrow the remaining issue to learned-pressure behavior in this scene; they do not establish whether supervision conditioning, parameter representation or another pressure detail is the cause.', '',
              '## Next bounded experiment', '',
              'Keep the uniform head-precision repair as a candidate inference policy, then improve pressure supervision before spending another independent bank. The next preparation should audit modal signal conditioning and add bounded mixed-mode pressure-query supervision from the offline reference, preserving the original native training pairs. A directly learned double divisor is a separate representation to compare with log/exp gains. It must be fitted on managed CUDA from data; no analytical divisor or inverse-coefficient labels may be inserted.', '',
              'Before either fit: qualify the actual objective on CPU and CUDA, freeze its finite update budget and sources, and require exact managed CPU restoration. Sixteen new whole scenes have now been reserved, with seeds disjoint from all listed prior reservations. Their references remain uncaptured. Full K1–4 development checks and candidate/source freezing must precede any new confirmation.', '',
              'The speed requirement also remains unmet. Preserve the independently measured one/two/four-core costs and the single-core efficiency metric. Reduce complete execution work while keeping raw checkpoints and numerical checks intact. Densities 50/100 and 600-step drift testing remain behind the ordered joint gates.', '',
              'A setup path typo in the conditioning build registration was corrected before compilation or numerical execution; its observation is retained. All completed controls closed their owners, and the registered source audit passed.', '',
              '[Machine-readable controls](pressure-floor-and-head-precision-summary-10.json) · [Frozen candidate independent results](decoder-compatible-pressure-results-09.md) · [New confirmation reservation](pressure-successor-fresh-scene-reservation-10.json)', '']
    report.write_text('\n'.join(lines))
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    state = dict(status='active_phase1_independent_failures_isolated_head_precision_repair_pressure_and_latency_unmet',
                 updated_unix=time.time(), previous_goal_turn_classification='progress', current_live_processes=[],
                 new_managed_optimizer_calls_this_goal_turn=80, new_optimizer_calls_in_diagnostics=0,
                 maximum_allowed_physical_cores=4, single_core_is_primary_efficiency_metric=True,
                 fresh_precision_scenes_consumed=16, successor_confirmation_scenes_reserved=16,
                 successor_confirmation_references_captured=0,
                 full_generator_targets_verified=summary['full_generator_targets_verified'],
                 higher_density_phase_started=False, phase600_started=False,
                 reports=['docs/plans/full-ai-tight-targets-v1/decoder-compatible-pressure-results-09.md', summary['report']],
                 source_registrations_verified=checks, original_completed_study_evidence_unchanged=True,
                 next=['Retain uniform 2^-40 query-head inference precision as a development repair; verify all horizons before confirmation',
                       'Audit modal signal conditioning; prepare bounded mixed-mode reference pressure supervision with original native pairs retained',
                       'Compare data-fitted direct double divisors with log/exp learned gains under managed CUDA and exact CPU restore',
                       'Register and qualify each actual objective before learning; freeze successor before the new reserved bank is captured',
                       'Reduce complete execution cost; retain one-core efficiency and explicit two/four-core measurements',
                       'Do not open higher-density or 600-step phases until every joint target passes'])
    state_path.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    print('Published pressure/local precision isolation; staged objective remains active', flush=True)


if __name__ == '__main__':
    run()
