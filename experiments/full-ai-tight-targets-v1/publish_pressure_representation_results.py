"""Publish bounded learned-pressure controls without promoting a target miss."""
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
    report = DOCS / 'learned-pressure-representation-results-08.md'
    summary_path = DOCS / 'learned-pressure-representation-summary-08.json'
    state_path = DOCS / 'work-state-07.json'
    if any(path.exists() for path in (report, summary_path, state_path)):
        raise ValueError('Fresh learned-pressure representation publication required')
    original = BASE / 'sampler-precision-tensor-dct-rollout-01/results.json'
    baseline = read(original)
    evidence = {str(original.relative_to(ROOT)): digest(original)}
    controls = {}
    for policy in ('reciprocal-division', 'negative-log-division'):
        path = BASE / f'learned-pressure-{policy}-rollout-08/results.json'
        data = read(path)
        if len(data['rows']) != 1920 or data['learned_inventory'] != baseline['learned_inventory'] or data['optimizer_updates'] != 0 or data['fresh_holdout_consumed']:
            raise ValueError('Exact unchanged managed model and full diagnostic population required')
        if not data['own_history_without_teacher_resets'] or data['rollout_steps'] != 240 or data['independent_target_pass']:
            raise ValueError('Registered development-only own240 scope changed')
        controls[policy] = data
        evidence[str(path.relative_to(ROOT))] = digest(path)
    path = BASE / 'modal-label-encoding-diagnostic-08.json'
    labels = read(path)
    if labels['status'] != 'completed_train_only_label_encoding_diagnostic' or len(labels['rows']) != 144 or labels['development_or_reserved_used'] or labels['optimizer_updates'] != 0:
        raise ValueError('Complete train-only label diagnostic required')
    evidence[str(path.relative_to(ROOT))] = digest(path)
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
                    raise ValueError('Registered source/evidence changed: ' + str(path) + ' ' + name)
                count += 1
            checks.append(dict(registration=str(path.relative_to(ROOT)), field=field, sources_verified=count))
    lines = ['# Learned pressure representation controls', '',
             'Two predeclared execution controls reduce the worst 32-cubed reference-equation difference from about 1.06e-4 to 6.67e-5 over 240 own-history steps. The 1e-5 criterion still fails. No new fitting, native pressure solver, analytical eigenvalue or pressure inverse enters candidate inference.', '',
             '## Complete own-history results', '',
             'The fixed explicit cosine transform and every raw managed-restored parameter remain unchanged. The baseline multiplies modal inputs by exp(log_gain). One control divides by the reciprocal of that learned gain; the other divides by exp(-log_gain). Gauge mode handling remains explicit. Derived constants come only from existing learned parameters, and their double inventories are recorded.', '',
             'Each control scores all pressure, velocity and tracer fields at all 240 steps on the existing eight development scenes: 1,920 states per policy, including 480 at 32 cubed. This is K1 development diagnosis, with no new independent scene consumption, timing qualification, K2–4 result or promotion claim.', '',
             '| Policy, 32 cubed | Worst reference equation | Worst pressure | Worst velocity | Worst tracer |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, data in [('Learned gain multiplication', baseline), *controls.items()]:
        row = next(v for v in data['summaries'] if v['regime'] == '32cubed')
        lines.append('| ' + name + ' | ' + ' | '.join(f"{row['worst'][key]:.6g}" for key in ('equation', 'pressure', 'velocity', 'tracer')) + ' |')
    lines += ['', 'Both division controls preserve zero velocity difference on the smaller supported grids, with worst equation difference about 1.18e-13. At 32 cubed, 311 of 480 states satisfy all 1e-5 criteria; the first failing state is step 42 in the density-16 development scene. The worst equation difference is about 37% lower than the multiplication control, but remains about 6.7 times the limit. This metric compares pressure with the independently evolved reference RHS and therefore includes accumulated trajectory differences; it is not an isolated pressure solve residual.', '',
              'The two controls have matching reported carrier errors and almost identical pressure/equation errors. Their benefit is evidence of numerical sensitivity in this learned operator, not proof that either is independently qualified or fastest. Every raw learned parameter inventory matches the multiplication baseline.', '',
              '## Training label encoding diagnosis', '',
              'A separate offline test uses all 144 original pressure/RHS pairs from the 16 training scenes. It compares the original projected pressure labels, explicit cosine projection, and inversion of only the small fixed cosine-coordinate basis for label encoding. The latter is a data-coordinate round-trip control: it constructs no physical pressure inverse, gain ratio or inverse-operator target and fits no model.', '',
              '| Geometry, z/y/x | Native teacher worst equation | Original label round trip | Explicit projection round trip | Fixed-basis inverse label round trip |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in labels['summaries']:
        values = [row['worst_native_equation'], *[row['controls'][key]['reference_equation_relative_l2'] for key in ('original_projected', 'explicit_projected', 'fixed_basis_inverse_encoded')]]
        lines.append('| ' + row['key'] + ' | ' + ' | '.join(f'{v:.6g}' for v in values) + ' |')
    lines += ['', 'At 32 cubed, the coordinate control reduces the pressure round-trip equation difference from 7.54e-13 to 2.48e-13. This supports testing decoder-compatible supervision in a separately registered managed CUDA fit. It does not prove that label encoding is the sole cause of rollout drift or that retraining will improve it. Merely changing projection to explicit einsum did not improve this round-trip control.', '',
              '## Disposition and next experiment', '',
              'Preserve both execution controls as development evidence. Before the next pressure fit, prepare decoder-compatible labels from the same training-only physical data, qualify the actual double objective on CPU/CUDA without updates, and register a bounded managed CUDA refinement plus exact CPU restore. Retain the old labels and models as paired controls. A direct learned-divisor parameterization is a separate possible experiment if encoding alone is insufficient; no analytical divisor table may be inserted.', '',
              'Single-core execution specialization and the existing parallel implementations remain separately available. The complete generator must still satisfy the latency and all-field own-history criteria, then independent confirmation, for K1–4. The 16 reserved precision-confirmation scenes remain unconsumed. Densities 50/100 and the 600-step drift stage remain gated.', '',
              '[Machine-readable evidence](learned-pressure-representation-summary-08.json) · [Single-core execution improvement](frozen-point-execution-results-07.md) · [CPU resource policy](cpu-resource-policy-04.md)', '']
    summary = dict(status='completed_learned_pressure_controls_accuracy_improved_target_unmet',
                   updated_unix=time.time(), new_optimizer_updates=0,
                   complete_diagnostic_own_history_states=3840, horizons=[1], own_history_steps=240,
                   exact_raw_learned_inventories_preserved=True,
                   results={key: data['summaries'] for key, data in controls.items()},
                   multiplication_baseline=baseline['summaries'], training_label_coordinate_controls=labels['summaries'],
                   diagnostic_training_states=144, new_independent_scenes_consumed=0,
                   full_generator_targets_verified={str(k): False for k in (1, 2, 3, 4)},
                   higher_density_phase_started=False, phase600_started=False,
                   source_registrations_verified=checks, distinct_artifacts_verified=len(cache),
                   evidence=evidence, publisher_sha256=digest(__file__), report=str(report.relative_to(ROOT)))
    report.write_text('\n'.join(lines))
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    state = dict(status='active_phase1_execution_and_pressure_controls_improved_targets_unmet',
                 previous_goal_turn_classification='progress', updated_unix=time.time(), current_live_processes=[],
                 maximum_allowed_physical_cores=4, single_core_is_primary_efficiency_metric=True,
                 new_optimizer_updates=0, fresh_precision_reserved_scenes=16, fresh_precision_scenes_consumed=0,
                 full_generator_targets_verified=summary['full_generator_targets_verified'],
                 higher_density_phase_started=False, phase600_started=False,
                 reports=[str((DOCS / 'frozen-point-execution-results-07.md').relative_to(ROOT)), summary['report']],
                 source_registrations_verified=checks, original_completed_study_evidence_unchanged=True,
                 next=['Register decoder-compatible native pressure label view and bounded managed CUDA precision fit',
                       'Qualify actual objective/device/typed restoration before learning; preserve raw data and prior models',
                       'Continue complete-work speed optimization with verified single-core and explicit parallel paths',
                       'Freeze selected K1-4 successor before independent confirmation; retain ordered later-phase gates'])
    state_path.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    print('Published bounded pressure precision progress; staged objective remains active', flush=True)


if __name__ == '__main__':
    run()
