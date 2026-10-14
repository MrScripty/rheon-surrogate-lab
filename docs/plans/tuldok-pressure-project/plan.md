# Tuldok pressure project

Plan status: Verifying. Current phase: project contract implemented; CPU system receipts pending.
Next slice: clean-source build receipt, actual project assembly and CPU parity evaluation.
Acceptance status: pending. [Ledger](execution-ledger.md), [issues](issues.md),
[baseline and contract](reports/baseline.md).

## Objective and scope

Create a lab-owned Tuldok project with versioned configurations, declarative
external recipes, checkpoint/evaluation references, a clean-run procedure and
existing timeline-viewer integration. First establish unchanged-checkpoint
accuracy and complete-rollout CPU timing. Approximately eight times is a
historical measurement, never an acceptance threshold or promise.

Write set: `tuldok/`, `probes/tuldok-pressure/`,
`scripts/test_tuldok_pressure.py`, `docs/plans/tuldok-pressure-project/`,
`README.md`, `THIRD_PARTY.md`, `viewer/viewer.js`, `.gitignore`. Generated native/checkpoint/data/runtime/project
and full evaluation artifacts stay under ignored `target/`, `.venv/`, `runs/`.
Existing Rust sources, model/experiment scripts, freezes and checkpoints
remain unchanged. Viewer runtime label reads optional replay metadata, preserving
the historical default. No Tuldok edits, release tags or solver/aircraft/submarine
changes. Passenger-jet and forward-plus-ascent submarine domains await solver
validation and a separately admitted physics/data contract.

## Acceptance

| ID | Observable claim | Kind/environment/mode | Status |
| --- | --- | --- | --- |
| A1 | Original checkpoint/source identities retained; new native uses exact tested Rheon revision | contract / real Git+Cargo / automated | pending |
| A2 | Project/recipe JSON accepted by current generic Tuldok readers; source/data/artifact hashes, closed config and no overwrite | contract / representative Linux / automated | pending |
| A3 | CPU restored-model full trajectories pass existing physics gates and full-field parity against both references; three balanced timing repetitions with honest runtime/scope | system / cloud CPU / automated | pending |
| A4 | Clean CPU training path retains whole-trajectory splits, actual fitting and restored-checkpoint equality; no test-driven fitting | system / cloud CPU / automated | pending |
| A5 | Existing viewer consumes newly generated CPU replay; generic result/progress formats and lab UI path documented | user-workflow / local browser / automated | pending |
| A6 | Focused failures and existing affected tests pass; feature branch committed/pushed and draft PR opened | focused + release artifact / GitHub / automated | pending |

A Tuldok managed Run/UI end-to-end acceptance depends on its separate execution
owner and configured catalog/runtime/approval; direct recipe execution does not
claim that acceptance. Cloud CPU timings are new measurements, not the original
hardware/runtime reproduction. Cold startup and resident throughput are separate.

## Decisions and owners

- Lab owns physics, configuration, split semantics, model, accuracy and timing.
- Tuldok owns project/recipe interpretation, captured CPU runtime, immutable
  input/output custody, Run lifecycle, progress and generic result display.
- Reuse `external_recipe_v1`, project schema1, `tuldok_progress_v1` and
  `tuldok_result_summary_v1`; no new executor or Tuldok physics registration.
- New native manifest pins Rheon `2209012bc5f902693d0f695ec1c2d854126af317`
  and parallel feature, reusing unchanged trajectory Rust source. Old sibling-path
  manifests remain runnable. New lockfile/toolchain own dependency resolution.
- CPython3.12, NumPy2.3.5, Torch2.8.0+cpu are isolated fresh dependencies.
  Historical Torch2.14.0+cu132 and its records retain their original identity.
- Recipe assembly only inspects/copies bytes. Execution is explicit; no packages
  or sources are downloaded by a recipe. Native loader policy uses existing
  Tuldok contract; process descendants remain owned by Runs.

## Simplicity and ownership review

Applicable: independently evolving lab and executor contracts are composed.
1. Numerical source/model is the existing lab; project assembly packages its
   immutable references; local orchestration connects fixed configurations to
   existing numerical functions. Tuldok supervises execution and publishes bytes.
2. Git/build/checkpoint hashes identify bytes; schemas identify representations;
   frame time is physical1/60s; wall time is measured separately. Split policy is
   lab-owned and never inferred from file order. No shared mutable model registry.
3. Callers supply measured native/build/runtime identities, immutable inputs and
   output directory. They need no solver/model algorithm knowledge.
4. Physics/model change touches lab+requalification; runtime/custody change touches
   Tuldok; new experiment config touches lab config+new evaluation receipt.
5. Existing project/recipe/progress/result formats are stable interfaces; original
   script global paths are confined to orchestration hooks, not hidden in Tuldok.
6. Inspection works without Torch; model and native remain independently tested;
   failure produces partial files without a success receipt. Original paths remain.
7. Deleting assembly pushes measured source/data inventory into users; deleting
   orchestration pushes captured-path/provenance mapping into each recipe caller.
   No new scheduler, runtime installer, service, data parser or generic registry.
8. Required complexity is captured-source/native identity and scientific gates,
   confined to lab orchestration. Reuse existing numerical routines and generic
   executor semantics; bulk artifacts are external references, never Git payloads.

## Milestone and evidence plan

One slice: project assembly, CPU train/restore/compare and viewer packaging.
Allowed writes are above. Gate: A1–A6 with receipt links, numeric acceptance by
unchanged native gates and full-field comparison; negative fixture failures must
reach the intended config/hash/output boundary. Supporting checks: original Python
suite, root Rust tests/fmt/clippy and new probe locked build. Inspect exact staged
diff and cumulative branch before push. No new runtime/GPU/remote-worker claim.

## Coordination and blockers

Parent coordinates compatible checkpoint and merge. This task owns feature branch
`feat/tuldok-pressure-project` from `e8a90bfef6b2955c502008ccc62443e29dd9f2ec`,
target main; retain pending draft review, retire after integration by parent.
No worktree created. Paired Tuldok owner: `01a13906-0d11-7681-87b7-439a7097c039`.
No available cross-cloud-thread message tool; collaboration cannot resolve parent
thread ID. Prompt/status and durable contract are the available handoff.
Cloud initially lacked Cargo and Torch; isolated provisioning succeeded. No
managed Tuldok catalog/runtime is assumed. Replan for missing generic contract,
changed solver pin, breached memory/time budget or parity failure; preserve prior
reproduction while diagnosing. No tuning on exposed test seeds.
