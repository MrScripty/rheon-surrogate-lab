# Spec-only WIP integration review

Subject: `integration/current-dependencies-20261016`, base `d6146a20a083ed73a8b59fb1d54b14b8ac706945`, HEAD `d6146a20a083ed73a8b59fb1d54b14b8ac706945` plus tracked working changes and the explicit new-file inventory below. Review is source inspection, not a new acceptance execution. Standards are a separate review.

**Incorrect behavior — P2, resolved after independent recheck:** The original runtime receipt key mismatch is fixed: `qualify_cohort.py:13–16,34` now consumes the producer's single `all_original_fields_and_asset_bytes_identical` key (`dependency_runtime.py:40`). The regression invokes the actual producer with a tiny runtime owner, admits its receipt, rejects an alternate-key-only receipt and rejects failed second verification. Independent command `../app-env/bin/python -B -m unittest discover -s experiments/dependency-migration-v1 -p test_qualify.py -v` passed all eight tests in 0.006 s. Requirements: [plan.md:70](../plan.md#decisions), [acceptance-protocol.md:24](../acceptance-protocol.md#prerequisites-and-evidence-storage) and [acceptance-protocol.md:258](../acceptance-protocol.md#q7--74-scene-regression-and-timing). No unresolved source finding remains from this review. Historical campaign runner bytes remain separately retained at `runs/dependency-migration-v1/numerical-qualification-v1/cohort-runner-source-01.py`; the fix does not rewrite their receipt origin or transfer numerical acceptance to unexecuted code.

**Missing/partial requirements:** Q2 catalog preservation/cold restore, full new74/timing and post-switch/live/typed smoke remain acceptance gates, not source defects ([plan.md:182–192](../plan.md#acceptance-claims)). Do not infer cutover or final migration acceptance from the passing first-frame/lifecycle fixtures. The plan labels these gates pending.

**Unrequested scope:** none identified. Indexed endpoints/public diagnostic ownership, runtime link topology, bounded historical references and pressure-only live observation have explicit decisions ([implementation-decisions.md:3–17,19–35,44–65,83–110](implementation-decisions.md)). Reviewed current paths preserve separate teacher/trajectory meaning, stop-on-reject, unchanged weights/preprocessing/stopping limits, CPU inference and current-only application contracts.

This is a WIP review; pin and recheck the final committed range. No source, receipt, service or scientific artifact was changed.

Explicit new-file inventory at review start:

```text
docs/plans/rheon-tuldok-migration-v1/reports/implementation-decisions.md
docs/plans/rheon-tuldok-migration-v1/reports/q4-scientific-parity-v1.json
docs/plans/rheon-tuldok-migration-v1/reports/q4-scientific-parity-v1.md
docs/plans/rheon-tuldok-migration-v1/reports/q7-old-reproduction-v1.json
experiments/dependency-migration-v1/README.md
experiments/dependency-migration-v1/config.py
experiments/dependency-migration-v1/config.template.json
experiments/dependency-migration-v1/dependency-original-project.json
experiments/dependency-migration-v1/dependency-reference-bootstrap.py
experiments/dependency-migration-v1/dependency-reference-project.py
experiments/dependency-migration-v1/dependency_config.py
experiments/dependency-migration-v1/dependency_project.py
experiments/dependency-migration-v1/dependency_runtime.py
experiments/dependency-migration-v1/dependency_tests.py
experiments/dependency-migration-v1/dependency_training.py
experiments/dependency-migration-v1/dependency_training_check.py
experiments/dependency-migration-v1/dependency_training_entry.py
experiments/dependency-migration-v1/live_migration.md
experiments/dependency-migration-v1/live_migration.py
experiments/dependency-migration-v1/model_adapter.md
experiments/dependency-migration-v1/model_adapter.py
experiments/dependency-migration-v1/model_adapter_numerics.py
experiments/dependency-migration-v1/model_adapter_policy.py
experiments/dependency-migration-v1/package_model_adapter.py
experiments/dependency-migration-v1/qualify.py
experiments/dependency-migration-v1/qualify_aa82.py
experiments/dependency-migration-v1/qualify_adapter.py
experiments/dependency-migration-v1/qualify_baseline.py
experiments/dependency-migration-v1/qualify_candidate.py
experiments/dependency-migration-v1/qualify_cohort.py
experiments/dependency-migration-v1/qualify_operator.py
experiments/dependency-migration-v1/qualify_timing.py
experiments/dependency-migration-v1/qualify_typed_pressure.md
experiments/dependency-migration-v1/qualify_typed_pressure.py
experiments/dependency-migration-v1/stage_tuldok.md
experiments/dependency-migration-v1/stage_tuldok.py
experiments/dependency-migration-v1/test_live_migration.py
experiments/dependency-migration-v1/test_model_adapter.py
experiments/dependency-migration-v1/test_qualify.py
experiments/dependency-migration-v1/test_qualify_typed_pressure.py
experiments/dependency-migration-v1/test_stage_tuldok.py
experiments/dependency-migration-v1/test_typed_pressure_project.py
experiments/dependency-migration-v1/test_typed_pressure_workflow.py
experiments/dependency-migration-v1/typed_pressure_evidence.py
experiments/dependency-migration-v1/typed_pressure_project.py
experiments/dependency-migration-v1/typed_pressure_workflow.md
experiments/dependency-migration-v1/typed_pressure_workflow.py
```
