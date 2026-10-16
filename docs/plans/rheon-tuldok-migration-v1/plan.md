# Migrate the lab to current Rheon and Tuldok

Status: Planned. Current phase: source assessment complete; runtime migration has
not started. Acceptance: partial (A0 satisfied; migration claims pending).
Owner: lab migration task. The user
requested GitHub updates, a change assessment and migration plan, and explicitly
authorized improvements to Tuldok, including CUDA support, where needed.

Next slice: M1, integrate and qualify the existing managed Torch and live-project
capabilities on a new Tuldok branch based on the fetched main, including the
Tuldok runtime failure cases in A8. Keep the running schema-8 service in place.
This revision addresses R1–R5 from the retained [standards review](reports/standards-review.md).

## Outcome and boundaries

Use the new upstream solver and model-development capabilities while retaining
CUDA training, CPU float32 model inference, cubic cells, existing physical gates,
the original catalog, live visibility, and all historical evidence. Dependency
migration is separate from improving or promoting a pressure model. No retraining
campaign, model architecture change or fresh holdout consumption is part of this
migration. A bounded optimizer smoke qualifies execution only.

The fetched targets are Rheon
`bc2f472787060d9a69cb619ed2c2fea10fd754b6` and Tuldok
`300f12df6f48e028fd9379911703f7c89337952a`. These are immutable assessment
subjects, not claims that later GitHub commits have been qualified. See
[assessment](reports/assessment.md), [version inventory](reports/versions.json)
and [upstream history/path inventory](reports/upstream-changes.json).

The lab currently starts from main `03c05e82383ce0a18ded838803ce1f8f4ea1cd09`.
The latest local training lineage is separately retained at
`experiment/water-pressure-recovery-v1`,
`154826620d26a7dd6c785d36eb7fa33b76463b41`. Main is its ancestor. The
recommended lab implementation base is that complete water lineage, incorporating
this assessment through a normal merge, with integration into main reviewed
separately. Do not migrate only main's older GroupNorm code and describe that as
an upgrade of the current water model. Do not rewrite either lineage.

## Decisions

1. The existing Tuldok catalog remains authoritative. It was observed at schema 8,
   with 120 Projects and 160 Runs. New main requires schema 10. Stop all writers,
   take and verify a complete backup, rehearse restore and explicit 8→9→10 upgrade,
   then perform a controlled cutover. Never point the new app at the live store
   to discover whether startup migrates it; startup explicitly refuses old schemas.
2. Port required local capabilities into current Tuldok owners. Main currently
   rejects the managed CUDA and CPU recipes and the live activity manifest.
   Preserve the existing managed runtime's historical descriptor/custody meaning;
   qualify new execution against the new host source. The compatibility table
   below owns the retained-read/new-execution distinction. Do not relabel a CUDA
   installation as `captured_cpu_python_v1`, route GPU work through an unqualified
   system-Python escape, or introduce a second runner/catalog. CPU-only captured
   snapshots may be adopted later through their own supported runtime contract.
3. Keep live-project data import and evidence inspection working before the
   Tuldok cutover. Preserve the closed upstream v1 contract; read the existing
   local activity-bearing v1 snapshots through an explicitly selected interface.
   New live projects use ordinary v1 evidence references for activity. New
   v2 extension snapshots currently have different action support. Adopt typed
   Model/Study/Trial lineage and field evidence in a subsequent vertical slice,
   using the actual pressure task, preprocessing, output and evaluator contracts.
   Upstream scalar-A examples are workflow examples, not our pressure model.
4. Update the lab's dependency selection through a new reusable migration
   configuration: separately select source roots/revisions, build artifacts,
   runtime profiles and their measured identities. Derive fresh build/source
   witnesses from that selection. Keep historical Cargo pins, dependency
   manifests, patches, captured recipes, checkpoints and results unchanged.
   Each source, runtime, model and catalog version keeps its own authority.
5. Replace the probe's dependence on private desktop-engine representation at
   its owning boundary. Reuse Rheon's typed pressure context, existing numerical
   and process owners, and supported diagnostic/session entry points. First
   establish which minimal public/exported entry point meets the diagnostic
   requirement; add one upstream if necessary. Do not copy the new desktop
   engine and process stack into the lab or teach each caller its internals.
6. Preserve two distinct experiments: teacher/same-request pressure diagnostics
   and autonomous trajectories. Configured accepted Sessions may fall back to
   native pressure; every such fallback must remain visible and excluded from
   a claim of successful AI solving. The diagnostic stop-on-reject path retains
   raw failures. Equal physical time alone does not establish equal pressure RHS.
7. Freeze the same physics and numerical policies for before/after comparisons:
   sealed full water, rho=1000 kg/m³, mu=0.001 Pa·s, dt=1/60 s, zero origin, cubic
   h with domain extent counts×h, existing force schedules, actual accepted
   history and 15 native warm-up steps. First inference is step 16. Increasing
   upstream density/count edit limits does not expand model qualification.
8. Keep existing weights, normalization, channel mapping, float32 neural compute,
   pressure gauge and correction reconstruction unchanged for migration parity.
   Previous pressure is added exactly once. Current iterative correction and
   stopping policies stay explicit; they are not supplied automatically by a
   generic one-call adapter. The older aa82 deployment and the later residual
   models are separate source/model contracts and need separate adapters/tests.

## Fixed comparison and compatibility policy

The control for migration regression is the existing harder-data checkpoint
`26b66232928ce81d428dbdb1c8cc750a0cf5c0bc5f678977efc111a0a56d2d9a`,
normalization `f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1`.
Its train/restore Runs are e20126d732a24013aafa6cc09ebbc0a5 and
56adc4f5651d495489650f0ed5355a38. This is a fixed diagnostic comparator, not
a replacement for the desktop's aa82 deployment or a model promotion. No other
checkpoint may be chosen after observing migration results. The exact source,
runtime, old binary, policy, 74 scene identities, group membership and archived
case metrics are bound in [baseline-bindings.json](reports/baseline-bindings.json).

[acceptance-protocol.md](acceptance-protocol.md) owns the executable procedures,
exact versus numerical comparisons, failure handling and measured timing scope
delegated by this plan. Its rules are binding: saved-input model parity is exact;
each of the 74 trajectories must qualify with unchanged stopping/physics limits,
zero native refinement/fallback and no per-case pressure-error or call regression.
The protocol states the precise comparisons; an explanation alone cannot waive
a mismatch. Timing is recorded as advisory evidence, with no speedup claim.
Freeze candidate build/source/runtime hashes before comparisons. If the existing
installed numerical runtime cannot be retained, replan before selecting another.

| Contract / owner | Retained reads | New writes or execution / rejection policy |
| --- | --- | --- |
| PostgreSQL schema 8 / Tuldok Metadata | Exact source definition and historical rows/files retained in complete stopped backup | Only ordered explicit 8→9→10 upgrade; no domain-row rewriting, automatic startup migration or downgrade. Unknown/corrupt source refuses. Rehearsal uses an absent destination in a separate PostgreSQL database |
| Ordinary upstream project v1 and extension project v2 / Projects | Existing supported snapshots remain byte-identical and readable through their supported projections | Existing closed forms and operation versions stay unchanged. V2 still does not acquire v1 actions by implication |
| Local project v1 with bounded `activity_path` / Projects | Recognize this exact persisted local variant, with its original strict field/path validation, in the selected snapshot reader; keep original manifest/ID/hash | No new creation of the local variant. Add `project_snapshots.get/2` for explicit inspection of ordinary v1/v2 plus the retained local variant; `/1` and legacy `projects.*` retain their published forms and return typed unsupported for the local variant. New v1 projects declare activity JSON as an ordinary evidence reference, read by a new selected `projects.activity/1` operation. UI and caller migrate together; historical snapshot bytes are never stripped or rehashed |
| `external_recipe_v1` with `managed_torch_python_v1`, explicit cpu/cuda and its material/spec/runtime descriptors / recipe_contract, Runs, TorchRuntimes | Preserve exact current local managed-profile meanings, manifest/bootstrap hashes and captured host closure; cold reads must not compare retained host files against current installation | Newly prepared material uses the same managed profile only after its implementation passes A2/A8/A9; current host/runtime files are remeasured and approval is fresh. No execution replay of historical material, reused approval or silent runtime substitution |
| Existing `system_python`, `captured_cpu_python_v1`, project/recipe/material and public operation contracts / existing Tuldok owners | All actual retained forms keep their present semantics and cold-read support | Preserve the frozen 56-operation legacy discovery bytes. A changed public request/result shape requires an explicit new negotiated version and coordinated producer/UI/client/fixture update. Unknown versions remain unsupported; malformed supported forms remain invalid. Unchanged operation bodies may keep their existing selected versions |
| Rheon legacy deployment v2 versus generic adapter v1 / ModelProcess and lab adapters | Preserve legacy descriptor/checkpoint and recorded frame meanings; inspect historical archives without loading a model | Validate exact selected package/role/context. Legacy full pressure must not have prior added again. Generic correction adds prior once. Unsupported physical/history settings reject before inference; numerical rejection remains visible, with accepted native fallback separately reported |

These compatibility obligations cover actual retained material, not arbitrary
old variants. A9 binds the real snapshot/material inventory and representative
cold reads to each row. Material/source/schema changes with different meanings
require a separately versioned contract rather than widening an old decoder.

## Milestones and allowed implementation writes

| Milestone | Coherent outcome and write set | Gate | State |
| --- | --- | --- | --- |
| M0 | Fetch all three remotes; fast-forward clean primary Tuldok; create isolated fetched Rheon; write only this planning directory in the lab | Immutable version, dirty-state, compatibility and catalog observations retained | Accepted |
| M1 | New Tuldok branch based on 300f12d. Runtime: `torch_runtime.py`, `torch_runtimes.py`, `recipe_contract.py`, `runs.py`; live inspection: `project_activity.py`, `projects.py`, `app.py`, `agent_api.py`, `project_cli.py`, `static/workspace-project*`, `static/workspaces*`; corresponding contracts/tests, including new `tests/test_torch_runtime_lifecycle.py` and `tests/test_managed_torch_retention.py` if needed for these missing claims. New application environment under the migration run root | A2/A8/A9 on a disposable schema 10 fixture; source/contract and browser live-path checks in protocol. Keep production untouched | Planned |
| M2-T | Tuldok retained-state rehearsal: existing backup/restore/upgrade tools, isolated database/destination, root bindings and reports; no changes to domain rows or historical files | A1 and A5-rehearsal, repeating A2/A8/A9 through the restored candidate as specified. Requires M1 | Planned |
| M2-R | Lab branch from 1548266 plus this assessment: new `experiments/dependency-migration-v1/**`, build/adapter callers under `experiments/local-pressure-v1`, `experiments/water-residual-learning-v1`, `experiments/live-water-comparison-v1`, `probes/residual-rollout-v1`, live manifest and reports. A necessary upstream public seam gets its own narrow branch, contract tests and docs | A3 first (one frame), then A4/A6-rehearsal/A7. No new numerical-runtime selection or model change. Can proceed independently of M2-T/M3 | Planned |
| M3 | Tuldok service cutover only: stopped-writer backup/upgrade/service launch/root bindings and small receipts | Requires M1/M2-T and satisfied A1/A2/A5-rehearsal/A8/A9. Then A10 post-cutover smoke before ordinary submissions resume. Does not enable a new Rheon binary or model | Planned |
| M4 | Rheon enablement and separately qualified typed project workflow: new deployment selection/project snapshot, owned task/model/evaluator/field-evidence contributions and reports through existing owners | Rheon switch requires M2-R with satisfied A3/A4/A6-rehearsal/A7; then A6-live smoke. Typed workflow A11 additionally requires M3; it cannot weaken the runtime, pressure or provenance gates | Planned |

M1 uses a separately identified disposable upstream fixture. M2-T uses a restored
copy of the actual stopped catalog in a separate database, not a second production
collection. Keep its endpoint, paths and DSN distinct; a restored catalog ID alone
cannot distinguish it from production. Never run two versions against the original.
The old service may resume on schema 8 after the rehearsal backup; M3 must take a
new stopped bookend/backup and validate its current retained-state inventory.
Before expanding a milestone's concrete write set, inspect its affected owners;
replan if that changes contracts, data custody, process ownership or physics.

## Acceptance claims

Owner T is this task acting as Tuldok integration/operations owner; owner R is
this task acting as lab/Rheon integration owner. Procedure IDs refer to
[the acceptance protocol](acceptance-protocol.md). All claims are pending except
A0. A5/A6 have distinct rehearsal and live receipts; a rehearsal cannot satisfy
their live portions. Existing upstream tests are supporting evidence, not local
migration acceptance. Required-real means the named hardware/app/store is actually
used; substitute fixture-only runs establish only their declared contracts.

| ID | Observable criterion | Owner / procedure / evidence / environment / mode / status |
| --- | --- | --- |
| A0 | Targets, local deviations, live source, schema and incompatible consumers identified | T/R; existing [assessment](reports/assessment.md), [probes](reports/contract-probes.json), [nine checks](reports/negotiation-qualified-env.json); contract/integration; required-real local sources/PostgreSQL; automated; satisfied |
| A1 | Every retained domain row/file matches before/after schema upgrade and restore, with only the declared DDL/version-ledger changes; cold reads cover each actual project/runtime/material form and terminal Run state | T; Q1/Q2; contract/system; required-real quiesced catalog copy/PostgreSQL; automated; pending |
| A2 | Actual CUDA float32 optimizer update and fresh CPU restore complete/drain with matching artifact hashes; CPU hides CUDA and has zero optimizer updates | T; Q3; system; required-real RTX5090/Pumas runtime/candidate Tuldok; automated; pending |
| A3 | Fixed saved-input preprocessing, tensor/output reconstruction and zero-residual bypass match exactly; independent operator/gauge check passes | R; Q4 and fixed bindings; contract/integration; required-real CPU/native/checkpoint; automated; pending |
| A4 | Rheon real model startup/failure/cancel/reap/restart/shutdown/cold inspection pass; 16-step restart has exactly one prediction after 15 native steps | R; Q5; system/user-workflow; required-real CPU model child and egui desktop; automated plus manual recorded desktop interaction; pending |
| A5 | Original live v1 data import/evidence/progress/refresh path works on candidate before cutover and on original catalog after cutover | T; Q6 rehearsal and Q8 live; user-workflow; required-real browser/API; automated plus manual recorded interaction; pending |
| A6 | Live/saved native–AI signed voxel difference and legend work, with honest rejection/fallback and distinct same-request/trajectory metrics | R; Q5/Q6 rehearsal and Q9 live; user-workflow/integration; required-real Rheon desktop/CPU package; automated plus manual recorded interaction; pending |
| A7 | All 74 cases satisfy fixed quality, per-case pressure and call non-regression rules; no native refinement/fallback or heldout use | R; Q7; system; required-real bounded CPU/native builds/checkpoint; automated; pending |
| A8 | Tuldok CUDA cancellation during hashing/training, startup/deadline failure, drainage and cold interrupted/unknown recovery are qualified without duplicate work or CPU fallback | T; Q3 lifecycle cases; system/contract; required-real GPU plus isolated catalog; automated; pending |
| A9 | Compatibility table passes actual retained reads, new-profile preparation, unsupported/invalid negatives, unchanged legacy discovery and negotiated new-operation tests | T; Q2; contract/system; required-real candidate app/PostgreSQL and retained fixtures; automated; pending |
| A10 | Tuldok cutover bookend, identity/root/history and one bounded CUDA→CPU live smoke pass before general submissions resume | T; Q8; system/user-workflow; required-real original catalog/GPU/browser; automated plus manual inspection; pending |
| A11 | New task/model/Study/Trial and field evidence use exact supported pressure semantics and runtime identity; no invented historical lineage or acceptance | T/R; Q10; contract/user-workflow; required-real candidate service/browser and bounded real GPU/CPU execution; automated plus manual inspection; pending |

The older aa82 model has retained accuracy failures. Its migration lifecycle and
byte-parity checks must reproduce honest operation; they cannot turn those
failures into passes. A7 uses only the preselected 26b control and exact policy
bound above. No checkpoint selection or promotion is authorized by migration.

## Composition review

Applicable: the runtime and protocol integrations cross independently owned
source, persistence and process boundaries.

1. Rheon owns numerical advancement/context/history and desktop model processes;
   the lab owns pressure-model preprocessing/reconstruction and scientific
   qualification; Tuldok owns snapshots, Runs, admission, custody and catalog
   migration. Configuration selects these independent artifacts explicitly.
2. Necessary interleavings are native prefix→history→residual→model→quality gate,
   and stop writers→backup→upgrade→verify→restart. A changing model does not
   require a catalog migration or a new host lifecycle implementation.
3. Callers supply validated physical context, exact model/runtime selections and
   finite resource limits. They do not own process reaping, decode arbitrary
   recipe internals or infer pressure units from tensor shape.
4. A checkpoint change touches the lab package and new model references; a
   CUDA runtime change touches Tuldok runtime selection and qualification; an
   operator change touches Rheon plus scientific parity; a catalog change
   touches the existing storage upgrade/backup owners.
5. Typed context/provider and negotiated operation contracts are the seams.
   Private `engine.rs` imports are current accidental coupling to replace;
   retained historical imports remain historical evidence.
6. Tuldok cutover requires its retained-state/runtime/live-path rehearsal; Rheon
   enablement separately requires pressure parity, lifecycle and all 74 cases.
   One component's pass cannot authorize the other. Both stop ordinary work
   until their bounded post-switch smoke succeeds.
7. Reusing the existing Runtime/Runs and Worker/ModelProcess owners avoids new
   queues, catalogs and lifecycle registries. Deleting the small pressure adapter
   would move required tensor/physics mapping into its callers; keep that mapping
   together. Removing the selected retained-project reader would make the actual
   local-v1 snapshots unreadable; removing the evidence-based activity operation
   would lose live browser observation. Both remain inside Projects, with no new
   source store or runner. A second universal training framework adds no capability.
8. Retained machinery is one explicit dependency selection, the existing runtime
   support port, one model adapter per actually supported contract and existing
   catalog upgrade tools, plus explicit Projects projections for the retained
   local variant and new evidence-based activity. Each uses the existing operation
   registry and bounded project reader. Add no scheduler, automatic retry, shadow
   catalog or permissive schema fallback.

## Operations, retention and reconsideration

Use one GPU optimizer and at most two bounded one-thread CPU simulation workers,
with the previously measured affinity and 8 GiB available-memory reserve. Limit
the first qualification to one frame before full rollouts. Record timing and
memory throughout; no assumption that using 24 threads is faster. Large geometry
settings need allocation preflight before fields or preview histories exist.

Keep the existing 8091 service untouched except the bounded stopped-backup window
in M2-T, then the explicit M3 cutover. Stop every app/provider/artifact
writer before backup/upgrade. Complete backup covers SQL, managed files, runtime
snapshots and receipts; external lab runs, model bundles and project-root bindings
need separate preserved inventory because Tuldok Project backup does not copy
external paths. Verify backup by restore to an absent destination/schema. Use
upstream `upgrade_metadata.py CATALOG --quiesced`; on unknown commit keep writers
stopped and re-enter the same verifying command. Do not manually bless signatures.
Rollback uses the stopped pre-upgrade backup in a new destination and the old
matching binary. Post-cutover writes require a separate retained-state recovery
decision; an old backup cannot represent them.

Current blockers: none prevent M1 source work. No runtime, service or solver
switch is currently qualified: A1–A11 are pending. This is an unsatisfied gate,
not an assertion that hardware/authority is missing. Required legacy/source
artifacts have local bindings; full runtime revalidation is a Q3 precondition.
If a prerequisite is unavailable during execution, record the affected claim as
blocked and continue only independent work. Do not repeat unknown admissions.

Source isolation is deliberate: primary Rheon and the old live viewer have
uncommitted user work. The task's detached
`runs/dependency-migration-v1/Rheon` worktree is retained-protected at bc2f472,
reachable through primary Rheon's fetched `origin/main`. Owner: migration task;
consumer: this assessment and subsequent migration qualification; synchronize
only by explicitly selecting and requalifying a new revision. Retire after
accepted migration or explicit abandonment, checking cleanliness and protected
reachability before removal. Do not stash, reset or sweep other worktrees.

The assessment branch `plan/rheon-tuldok-migration-20261016` is a local retained
proposal based on 03c05e8. Target: lab integration review, ultimately main. Serial
integration owner: migration task. Retain until its assessment is integrated or
explicitly superseded; no push, history rewrite or automatic cleanup here.

Replan for a changed upstream contract, unavailable historical bytes, catalog
drift, CUDA custody incompatibility, new physical behavior, changed fallback
semantics, unexplained numerical differences or a need for new mutable owners.
Current findings and dispositions are in [issues](issues.md); performed actions
and verification are in [the ledger](execution-ledger.md). Standards were routed
through the coding-standards MCP at the snapshot in the version inventory.
The revision's [finding dispositions](reports/standards-review-resolution.md)
retain the original review and current MCP snapshot without overwriting history.
