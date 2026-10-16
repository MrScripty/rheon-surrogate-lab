# Migration acceptance procedures

This protocol is the binding evidence procedure delegated by [plan.md](plan.md).
The plan owns scope, one-time conversion policy, gates and lifecycle. All procedures
below are **planned, not executed**. Updating this document cannot make a pending
claim pass. Owners T/R are defined in the plan. Save command, executable/source
and input hashes, environment qualification, elapsed time, exit status and small
receipts under a new attempt in `runs/dependency-migration-v1/`; retain scalar
results here. Never overwrite a prior attempt, dataset, weight or receipt.

## Prerequisites and evidence storage

Use the plan's pinned upstream targets and the lab water lineage at 1548266.
Read [baseline-bindings.json](reports/baseline-bindings.json) before comparison.
It binds hashes, not portable absolute paths: the new reusable experiment
configuration supplies actual roots and verifies the selected bytes. Do not
rewrite an old recipe to bless a relocation. Capture the candidate source/build
and installed-runtime inventory before its first measured comparison. Recheck
the identities at completion. Missing or changed material fails admission.

The baseline freeze was assembled from existing bytes/records without executing
models. It is not a fresh validation of every installed Torch file. Full runtime
verification remains mandatory before Q3/Q4. The old manifest is reference evidence;
the new runtime owner emits only its current descriptor, binding the same installed
numerical assets. Keep the same measured Python 3.14,
Torch 2.14.0+cu132, NumPy 2.5.3 numerical installation for old/new comparison; app
hosting uses a separately provisioned application environment. Do not add GPU
libraries to the captured-CPU profile or mutate the live app environment.

Rehearsal isolation: use a separate PostgreSQL database and absent destination
directory under the migration root. Configure explicit loopback endpoint/root
bindings and local credentials without logging secrets. A restored dataset ID
and restored credentials can match production, so inspect endpoint, DSN target,
process source and root path as well as catalog ID. Never select production by
an ambient default. Raw backups/model files remain private and outside Git.

## Q1 — retained-state backup, upgrade and restore (T; A1)

1. Stop the original app and every Run/provider/artifact writer. Verify cessation,
   no active/unknown attempt and available disk before backup. Capture actual
   current row/file inventories, schema/signatures, sequences, dataset identity
   and root bindings. The assessed 120 Projects/160 Runs are historical counts,
   not a license to discard subsequent legitimate additions.
2. With the configured candidate application Python and source, use the upstream
   tools below. Each restore uses an absent destination in the isolated database;
   every backup destination is new. Pin their resulting source/schema identities.

   With the source database explicitly selected in this process environment:

   ```sh
   "$MIGRATION_APP_PYTHON" "$TULDOK_SOURCE/backup_metadata.py" backup "$SOURCE_CATALOG" "$BACKUP_PATH" --quiesced
   ```

   In a separate process environment selecting the isolated destination database:

   ```sh
   "$MIGRATION_APP_PYTHON" "$TULDOK_SOURCE/backup_metadata.py" restore "$BACKUP_PATH" "$RESTORED_CATALOG"
   "$MIGRATION_APP_PYTHON" "$TULDOK_SOURCE/upgrade_metadata.py" "$RESTORED_CATALOG" --quiesced
   "$MIGRATION_APP_PYTHON" "$TULDOK_SOURCE/upgrade_metadata.py" "$RESTORED_CATALOG" --quiesced
   ```

   Select the source database for backup and the separate destination database
   for restore/upgrade explicitly through each process's private configuration.
   Do not run this block with one inherited database selection. A backup command
   cannot prove that external lab directories have been copied: inventory and
   preserve those separately as the plan requires.
3. Require first upgrade to recognize exact8→9→10 and second to report verified
   `already_current`. At this schema-only stage compare every source domain row, managed file
   hash, sequence state and identity against the stopped bookend. New schema 9/10
   tables start empty. Complete schema 10 backup/restore into another absent
   destination must reproduce the upgraded state. No refreshed signature, row
   rewrite or mutable cached source is acceptable evidence.
4. Use existing `tests/test_workspace_metadata.py` and
   `tests/test_experiment_lineage.py` for transactional failure, re-entry and
   retained restore contracts. Then perform Q2's separate one-time domain
   conversion and inspect its current-format outputs. An unknown
   upgrade commit keeps writers stopped; re-enter only the same upstream
   verifying upgrade command. Never repeat Run admission as recovery.

After the rehearsal backup the old service may resume against its unchanged
schema 8 store. A new source bookend/backup is required for Q8. If new writes add
an unhandled source representation, extend and requalify the converter before
cutover; do not add that representation to normal app readers.

## Q2 — one-time conversion and current-format inspection (T; A1/A9)

The converter runs offline against the stopped restored copy, with explicit
source/target identities and an independently verified archive of original rows,
files and metadata. Enumerate every Project, Run, material/runtime form, terminal
state and external reference before conversion. Produce an accounting manifest
with source IDs/hashes, disposition, destination IDs/hashes, preserved facts and
any representation changes. Every source record must be accounted for exactly
once; target counts/IDs may differ, but no source result or artifact may disappear.

Convert projects/activity, runnable recipes and model descriptors to their chosen
current forms. Rebuild current execution material from current source/runtime;
old approvals do not authorize it. Do not rewrite an old success receipt to claim
the old job ran under the new host. Records already valid under current semantics
can stay unchanged. Where no truthful current Run representation exists, retain
the exact original and expose its results/artifacts through existing current
Project evidence/data inspection, visibly marked historical evidence. Do not
create a new history framework, obsolete material decoder or fake current Run.

Stage conversion outputs privately, validate them through current owners, then
publish the coordinated row/reference changes transactionally while writers
remain stopped. Keep archived originals recoverable before replacing active rows.
Compare model weights, normalization values and scientific results exactly.
Validate source→destination relationships and all converted references independently
of the converter's success message. A failure rolls back publication; an unknown
commit keeps writers stopped and is reconciled by checking the exact conversion
receipt and inventories, never by guessing or automatically rerunning conversion.
Repeat invocation against a completed conversion verifies it or explicitly
refuses; it must not create duplicates. Back up/restore the completed current state
and confirm inspection works with the converter/source directories unavailable.

Implement bounded conversion fixtures under `tools/migrations/rheon-lab-once/**`
and, if needed, `tests/test_rheon_lab_migration.py`. Test original/converted facts,
all source dispositions, duplicate/lost-ack handling, failure before publication,
missing/corrupt inputs and normal-app rejection of obsolete forms. Run that test
file plus affected current Projects, recipe/runtime, public API/client and UI
tests from candidate Tuldok with explicit isolated PostgreSQL configuration and
`PYTHONDONTWRITEBYTECODE=1`, using
`python -B -m unittest discover -s tests -p FILE -v` for each recorded selector.
The migration test file is planned, not an existing pass. Require positive test
discovery. No frozen old discovery digest, old-client matrix or legacy loader
test is a migration acceptance requirement. Remove superseded application paths
and update their callers/fixtures in the same deployment.

The converter is not imported by the app or invoked at startup. After successful
cutover retire it from the deployed artifact and ongoing compatibility CI; retain
its frozen source and receipts as historical migration evidence only. Current
app cold reads must succeed without it, without the old runtime installation
where reading results needs no execution, and without parsing obsolete records.

## Q3 — managed CUDA/CPU success and failure lifecycle (T; A2/A8)

Through the candidate's real approval/admission/Run/custody owner, using a
disposable schema 10 fixture first and then the rehearsal installation:

The success case uses the pressure trainer and its original train-only packet,
with one declared batch/update in a current qualification recipe and the ordinary
fresh-process restore stage. Numerical model/loss semantics stay unchanged;
runtime orchestration and recipe/custody glue migrate to the current contract.
Record their changed source identity. Do not replace this
pipeline claim with a toy CUDA tensor check, train on a regression/holdout frame,
or modify any historical recipe/checkpoint. Failure-injection cases may use
small dedicated fixtures because their claim is custody/lifecycle, not learning.

| Case | Trigger and required observation |
| --- | --- |
| Success | One real float32 CUDA optimizer update with device/dtype/gradient evidence; completed/drained Run. Fresh managed CPU restore matches all weight artifact hashes, produces finite expected output, initializes no CUDA and performs zero optimizer updates |
| Hash cancellation | Cancel after inventory verification starts but before model entry; cancelled/drained, no optimizer entry or launched orphan; retain actual supervisor observations |
| Training cancellation | Cancel after a real CUDA-context/optimizer-entry barrier, before completion; cancelled/drained, at most one optimizer update, no published success artifact, no live custodied child. Partial evidence remains explicitly partial |
| Startup/runtime failure | Inject missing/changed inventoried asset and a separately bounded entrypoint exception. Fail before unapproved execution or terminate honestly; never retry on CPU or submit another Run |
| Deadline | An explicit fixture allocates/uses CUDA then waits at a bounded barrier beyond its declared child deadline; deadline outcome/drainage is observed. Expiring during import cannot substitute for this GPU-active case |
| Shutdown/reopen | Stop the candidate while its own CUDA fixture Run is active; verify known cessation or the existing unknown-custody fence. Cold reopen shows the same Run/material/grant and cannot duplicate dispatch. Explicit fresh retry, if tested, gets a new attempt after known cessation |
| Ambiguous outcome | Reuse existing real-helper lost-GO/admission-ack tests: unknown state fences work, consumes no second grant, and reconciles the original receipt. A stopped-helper fixture proves the owner state machine, not real GPU cessation |

Use the current 600-second child ceiling for success/cancel fixtures, and a
predeclared short fixture deadline for the deadline case; copy the exact
readiness/hash/cancel/drain budgets from the existing profile into its admission
receipt before execution. Bound GPU/RAM occupancy to the small fixture; no
parallel optimizers or full fitting campaign. Report actual updates even after
cancellation. At most one success update plus one cancelled-case update per
qualification environment; no optimizer work in the other failure cases.

Extend existing tests, including `test_torch_runtime.py`, `test_recipe_process.py`,
`test_external_runs.py` and `test_runs.py`; add
`test_torch_runtime_lifecycle.py` for the missing profile-specific cases if needed.
Use `python -B -m unittest discover -s tests -p FILE -v` for each explicit file.
Require real GPU-case evidence in addition to controlled supervisor fixtures.
Repeat changed runtime/custody boundary cases on the restored candidate; do not
claim real lifecycle acceptance from the old nine negotiation tests. Respect
unknown cessation rather than manufacturing a hard shutdown guarantee.

## Q4 — first recorded frame and exact model parity (R; A3)

Select scene 0, first AI frame (physical step 16), capture SHA256
`4525a244c864ab70f55d40e54ad7928df0778aabc74aeaf209e247df2e762b5f`,
from `runs/water-early-pressure-v1/regression/development/scene-00-control-0/state-0016.npz`.
The archived
`regression-episodes.json` and baseline bindings identify the matching model,
scene and native receipt. Require that exact capture; do not silently select an
easier frame if unavailable. Pin its raw input/output hashes before executing
the new bridge. Run one frame first; a failure stops expansion to Q7.

Require exact shapes, ordering, cubic spacing, origin, density/dt/viscosity and
solved prior lineage. Feed identical little-endian input values through old/new
preprocessing on the same one-thread CPU runtime. The one-channel physical
residual model's q=(rho/h)r, frozen scales, float32 input/output, pressure gauge,
full float64 reconstruction and fixed correction/call policy must be bit-identical
for the same input. Zero residual returns the supplied base exactly. Validate
the pressure operator independently with the existing analytical two-cell stencil
and lab neighbor oracle; retain their original tolerances. Exactness applies to
numerical arrays, not timestamps, Run IDs, archive encoding or elapsed time.

Before cutover, use the original aa82 bundle and retained predictions as an
offline numerical reference for its newly packaged current adapter. The original
seven-channel preprocessing is distinct from the control's one-channel model.
The converted descriptor has a new hash; weights and predicted pressure must
match the original reference exactly. The reference tool/bundle is not shipped
as an alternative loader or decoder in the migrated app.
Reject unsupported physical/history contexts before invoking inference. No
non-bit-identical replay result passes merely because its difference is explained;
replan any necessary arithmetic/runtime change separately.

## Q5 — Rheon process and recording lifecycle (R; A4/A6 rehearsal)

First run existing public pressure-context/configured-session/recording contracts
and affected current `rheon-ai-compare` tests on the selected release build:

```sh
cargo test --offline --locked --features desktop --bin rheon-ai-compare \
  configurable_cpu_adapter_cancels_after_predictions_and_restarts -- --nocapture
```

This existing analytic adapter test is supporting evidence only. Extend the
existing worker tests with the real current-format package containing the same
aa82 weights, using a new evidence directory and no old deployment loader.
Record the exact trained-package test selector and assert it actually executed.
Require consumed predictions before cancellation, child termination/reaping,
new run identity on restart and exactly one prediction for the 16-step restart
after 15 actual native seed steps. Test configured generic adapter startup,
failure/blocked I/O, cancel, shutdown and saved cold inspection through the
existing Worker/ModelProcess owner; package-hidden cold reading must execute
no model. Record accepted native fallback, raw rejection and dispatch facts
separately. The original model's known numerical failure must remain visible.

## Q6 — rehearsal user paths (T/R; A5/A6 rehearsal)

For Tuldok, run its actual `tests/browser_workspace_project.cjs --lab-root`
fixture against the migration lab source, then use the rehearsal endpoint to
open the converted project through the current interface and a new live project.
Inspect mapped historical evidence, import a small declared scientific reference, inspect
old evidence and update one disposable live activity file. Observe visible
refresh without navigating to a private collection. Confirm source changes do
not fabricate Run custody or numerical acceptance.

For Rheon, use a real egui desktop window on the intended local display, select
native/AI pressure and signed difference voxels, verify the legend/units and a
numeric probe against the recorded difference, then save/close/reopen. Observe
force settings, actual warm-up and truthful failure/fallback presentation. Record
display/session/graphics capabilities, process exit, source/binary/package and
screen evidence. Headless egui/software-display tests are supporting evidence
unless those are the actual target workflow; do not transfer their qualification
to the user's desktop or copy test-only sandbox flags into normal startup.

## Q7 — 74-scene regression and timing (R; A7)

Freeze one fresh attempt configuration containing the old/new source and binary
hashes, fixed control package/runtime/policy, exact ordered scenes digest and
group labels from the bindings. Reuse the existing bounded rollout/oracle
owners through the qualified adapter; implement the migration selector at that
boundary rather than another simulation integrator.

First reproduce the control with the bound old binary in the same environment;
require its archived pass/completeness, exact call-sequence hashes and archived
per-case maximum paired pressure errors. If that cannot be reproduced, retain
the failure and resolve the baseline before measuring migration acceptance.
That reproduced cohort is the old side; do not run a duplicate old-side cohort.
Run the matched new-side native controls and 74 control-model trajectories,
keeping the original 58/amended 8/recent 8 groups and all horizons/force schedules.

The following are required together, with no additional comparison tolerance:

- Every case completes its full horizon and all expected AI steps after 15 native
  steps; every native and AI physics check passes; all children are reaped.
- Stopping remains relative residual 1e-6, absolute residual 1e-12, predicted
  divergence 1e-7 and mean-free same-RHS pressure-error bound 1e-6, with at most 32
  actual single-network calls and zero native pressure refinement/fallback.
  These are the existing conjunctive policy semantics, not a new approximate
  formula. Keep actual divergence, walls, flux and Courant gates from the bound
  old engine unchanged: actual divergence at most 1e-5, Courant at most 0.5,
  boundary-normal velocities exactly zero and absolute net flux at most
  max(1e-12,1e-8×sum(abs(divergence))×cell_volume). Native reference solves remain
  at their existing 1e-8 relative residual policy. Do not count an accepted native
  fallback as AI.
- Each migrated case's maximum paired trajectory pressure relative L2 error is
  no greater than both its archived value and the matched reproduced old value.
  Each case's mean/max network calls is no greater than the old case; retain
  every step's actual calls. All three group maxima/means must also be no worse.
  Zero added pressure-error/call allowance is intentional for an unchanged-model
  migration. Numerical changes needing a tolerance require a separate replan
  before another qualification, not an explanation attached to a pass.
- At first AI, every 64th step and final capture, reuse the independent same-RHS
  oracle after child reaping and its existing pressure-bound coverage checks.
  Preserve all failures. Do not compare same-request oracle error to independently
  advanced native-trajectory drift as if they were the same metric.

Use at most two one-thread workers on the previously measured P-core affinity,
8 GiB available-memory reserve, 120-second native child deadline and 1 GiB raw
phase budget. Exact rows/commands, cache status and timeouts are retained.
Separately time the existing three representative cases: original scene 0,
24×16×12 long scene and 40³ short scene. Old/new order reverses over two serial
repetitions on the same CPU/core/thread count, with first AI call separated from
steady-state samples. Record full-step/native/model/startup/hash/restore clocks,
sample counts and spread, RSS/CUDA peak and workload. Performance is advisory:
no speedup or performance acceptance is inferred from one sample. No untouched
eight-scene data, training update or checkpoint selection is allowed in Q7.

## Q8 — Tuldok-only cutover and post-switch smoke (T; A5 live/A10)

Admission requires satisfied A1/A2/A5-rehearsal/A8/A9 for the candidate composition.
Stop writers again, capture the current source bookend and new complete verified
backup, and check its source states against the qualified conversion inventory.
Run the same explicit schema upgrade and one-time domain conversion against the
original store, verify complete accounting/current reads, and start only the
new service with unchanged intended endpoint/catalog identity and explicit roots.
Rheon binary/package enablement remains gated separately. Converted Tuldok
records do not require an old Rheon client, manifest or protocol to remain supported.

Keep general Run submissions closed while checking current history/artifact
reads and the live project refresh/import/evidence path, then admit only one
explicit one-update CUDA smoke and its fresh CPU restore through normal grants.
Require completed/drained artifacts plus browser/API readback. Only then resume
ordinary submissions. On failure stop new writers, preserve all new receipts,
and keep service unavailable while the verified recovery choice is made. Do not
blindly restore over a schema 10 store with smoke writes; a new-destination
restore of the stopped pre-upgrade backup and old binary is the rollback path.

## Q9 — Rheon enablement and post-switch smoke (R; A6 live)

Admission requires satisfied A3/A4/A6-rehearsal/A7 for the exact candidate. Preserve
old binary/package selectors as offline rollback evidence. Select the qualified
new binary and converted current-format package with the original model weights,
perform one 16-step desktop comparison, inspect pressure
and signed difference/legend and cold-open its recording. Keep numerical
rejection truthful. A regression-control evaluation cannot implicitly deploy
that control in place of the original model. On runtime/UI failure restore the
old selectors and retain the failed output; no Tuldok database downgrade occurs.

## Q10 — typed pressure-model development workflow (T/R; A11)

After M3, publish a new project snapshot through the selected project interface,
with explicit pressure task/reader/model/trainer/evaluator declarations and exact
field units/geometry/gauge/source/runtime identities. Bind new Study/Trial intent
and finite budgets prospectively; old Runs retain unknown historical ancestry.
Generic scalar or CPU-only evidence profiles cannot stand in for this pressure
task's managed-CUDA contract. Unsupported combinations remain unavailable until
their existing owners explicitly implement and qualify them.

Use one finite one-update diagnostic Trial through the real selected operations,
then fresh CPU restore/fixed-probe reconstruction and field inspection. Verify
the same immutable model/material/Run/artifact references through API, CLI and
browser; no forged checkpoint/evaluator acceptance or automatic promotion.
Register and test any new negotiated contracts, with corresponding evidence
context and permission negatives, before starting this Trial. Existing
`tests/test_experiments.py`, `test_experiment_lineage.py` and the official browser
model-development journey provide workflow regressions, not pressure accuracy.
This bounded workflow qualification is separate from any subsequent model fit.
