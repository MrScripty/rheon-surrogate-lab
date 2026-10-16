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
The user's subsequent direction supersedes the previous compatibility design:
this is a one-time coordinated migration with no backward-compatibility support.

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
2. Implement required CUDA/CPU execution and live activity in current Tuldok
   owners. Select one current representation for each concern and convert this
   project's consumers and data to it. Old runtime descriptors, recipes and
   captured host metadata are converter inputs or archived evidence, not formats
   the migrated runtime must support. Qualify fresh material against the new
   host; never attribute old execution to that new material. Do not relabel a CUDA
   installation as `captured_cpu_python_v1`, route GPU work through an unqualified
   system-Python escape, or introduce a second runner/catalog. CPU-only captured
   snapshots may be adopted later through their own supported runtime contract.
3. Keep live-project data import, evidence inspection and activity working before
   cutover through the chosen current project contract. The one-time converter
   consumes the local `activity_path` variant and emits the current form; no
   historical snapshot reader or compatibility-only API is added. Update the UI,
   CLI, API schemas, recipes and callers together. Adopt typed
   Model/Study/Trial lineage and field evidence in a subsequent vertical slice,
   using the actual pressure task, preprocessing, output and evaluator contracts.
   Upstream scalar-A examples are workflow examples, not our pressure model.
4. Update the lab's dependency selection through a new reusable migration
   configuration: separately select source roots/revisions, build artifacts,
   runtime profiles and their measured identities. Derive fresh build/source
   witnesses from that selection. Preserve original Cargo pins, manifests,
   patches, captured recipes and results in the historical evidence/archive;
   active configuration and representations may change in the one-time migration.
   Keep model weight/normalization values unchanged. Converted content gets its
   actual new identity and an explicit old→new provenance mapping.
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
   generic one-call adapter. The older aa82 model and later residual model keep
   their distinct numerical mappings, packaged under the current adapter contract.
   Old bundle/recording formats are read only by the one-time converter or
   pre-cutover reference tools, never by a compatibility branch in production.

## Fixed comparison and one-time migration policy

The control for migration regression is the existing harder-data checkpoint
`26b66232928ce81d428dbdb1c8cc750a0cf5c0bc5f678977efc111a0a56d2d9a`,
normalization `f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1`.
Its train/restore Runs are e20126d732a24013aafa6cc09ebbc0a5 and
56adc4f5651d495489650f0ed5355a38. This is a fixed diagnostic comparator, not
a replacement for the desktop's aa82 weights or a model promotion. No other
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

| State / owner | One-time treatment | Migrated application |
| --- | --- | --- |
| Source schema 8/catalog / Metadata and migration owner | Stop writers, retain verified complete source backup, rehearse the explicit 8→9→10 upgrade and a separate one-time domain conversion. Verify a source→destination accounting manifest before publication | Only current schema and current records; no startup conversion, downgrade path or mixed-version writers |
| Projects, local `activity_path`, recipes, CLI/UI/API callers / current owners | Convert to the chosen current project/activity and execution definitions. Replace all affected references together; retain original bytes in archive and map changed IDs/hashes | Current producers/readers only. No `project_snapshots.get/2` solely for old snapshots, frozen legacy discovery requirement, aliases, dual reads/writes or old-client overlap window |
| Historical Run/material/approval/descriptor records / migration owner | Preserve exact originals and artifacts in verified archive. Retain records already valid under current semantics; otherwise convert truthfully into a current history/evidence representation. If a Run cannot become a current Run without changing its execution meaning, expose it through existing Project evidence/data inspection as a historical import, not a fabricated new Run | Current result/evidence readers only; original results/source identity remain inspectable as generic archived evidence. No old material validator/bootstrap/launch path. New execution always gets fresh current material and approval |
| Model packages and scientific recordings / lab/Rheon owners | Repackage unchanged weights and preprocessing into the current generic adapter contract; convert recordings needed for inspection or expose them as archived evidence. Retain package/source hashes and old→new descriptor/record identities | One current wire/package/record contract per concern. Preserve pressure reconstruction exactly, adding prior only once; no old deployment loader or recording-format branch |

One temporary offline converter owns source-format decoding. It is explicitly
invoked with exact source/target identities while writers are stopped, is excluded
from app startup/normal execution and retired from deployed code after verified
cutover. Its frozen source/receipts may remain in Git as migration evidence; it
does not become an ongoing compatibility suite. Every source record must have
exactly one disposition: already-current, converted-current or archived-visible
evidence. Preserve original numerical values, timestamps, failures and custody
claims; never invent acceptance, ancestry or execution under the new runtime.
Any unaccounted or unconvertible record blocks publication. Verify before/after
backups and the mapping; keep originals recoverable even where active rows/IDs
are replaced. The production service must not parse the obsolete source formats.
Unrelated upstream surfaces outside this migration are not a repo-wide cleanup
task; they create no compatibility obligation for this project.

## Milestones and allowed implementation writes

| Milestone | Coherent outcome and write set | Gate | State |
| --- | --- | --- | --- |
| M0 | Fetch all three remotes; fast-forward clean primary Tuldok; create isolated fetched Rheon; write only this planning directory in the lab | Immutable version, dirty-state, compatibility and catalog observations retained | Accepted |
| M1 | New Tuldok branch based on 300f12d. Current `torch_runtime.py`, `torch_runtimes.py`, `recipe_contract.py`, `recipe_process.py`, `runs.py`, `projects.py`, `project_activity.py`, `agent_api.py`, `project_cli.py`, `app.py`, `static/workspace-project*`, `static/workspaces*`, affected current tests/contracts and new application environment. Temporary converter under `tools/migrations/rheon-lab-once/**` and its conversion fixtures/tests use existing Metadata/Projects/data/evidence publication owners | A2/A8 on disposable schema 10 fixture; A9 current-contract and converter-source tests. No backward-compatible paths. Keep production untouched | Planned |
| M2-T | Rehearse schema upgrade plus one-time domain conversion on isolated restored database/destination; preserve originals, validate identity mapping and converted evidence/history through current readers | A1/A5-rehearsal/A9; repeat changed A2/A8 boundaries. Requires M1. Domain representations/IDs may change only with verified source→destination mapping | Planned |
| M2-R | Lab branch from 1548266 plus this assessment: new `experiments/dependency-migration-v1/**`, build/adapter callers under `experiments/local-pressure-v1`, `experiments/water-residual-learning-v1`, `experiments/live-water-comparison-v1`, `probes/residual-rollout-v1`, live manifest and reports. Rheon current public seam/package/process changes and obsolete project-path removal are isolated to their owners, `src/bin/desktop/ai_comparison/{model_process,worker,adapter}.rs` and directly affected current contracts/tests/docs, on a separate branch | A3 first (one frame), then A4/A6-rehearsal/A7. No new numerical-runtime selection or model change. Can proceed independently of M2-T/M3 | Planned |
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
| A1 | All source records/artifacts are accounted for in verified converted-current or archived-visible evidence, preserving original scientific facts and exact archived bytes; no unaccounted loss | T; Q1/Q2; contract/system; required-real quiesced catalog copy/PostgreSQL; automated; pending |
| A2 | Actual CUDA float32 optimizer update and fresh CPU restore complete/drain with matching artifact hashes; CPU hides CUDA and has zero optimizer updates | T; Q3; system; required-real RTX5090/Pumas runtime/candidate Tuldok; automated; pending |
| A3 | Fixed saved-input preprocessing, tensor/output reconstruction and zero-residual bypass match exactly; independent operator/gauge check passes | R; Q4 and fixed bindings; contract/integration; required-real CPU/native/checkpoint; automated; pending |
| A4 | Rheon real model startup/failure/cancel/reap/restart/shutdown/cold inspection pass; 16-step restart has exactly one prediction after 15 native steps | R; Q5; system/user-workflow; required-real CPU model child and egui desktop; automated plus manual recorded desktop interaction; pending |
| A5 | Original live v1 data import/evidence/progress/refresh path works on candidate before cutover and on original catalog after cutover | T; Q6 rehearsal and Q8 live; user-workflow; required-real browser/API; automated plus manual recorded interaction; pending |
| A6 | Live/saved native–AI signed voxel difference and legend work, with honest rejection/fallback and distinct same-request/trajectory metrics | R; Q5/Q6 rehearsal and Q9 live; user-workflow/integration; required-real Rheon desktop/CPU package; automated plus manual recorded interaction; pending |
| A7 | All 74 cases satisfy fixed quality, per-case pressure and call non-regression rules; no native refinement/fallback or heldout use | R; Q7; system; required-real bounded CPU/native builds/checkpoint; automated; pending |
| A8 | Tuldok CUDA cancellation during hashing/training, startup/deadline failure, drainage and cold interrupted/unknown recovery are qualified without duplicate work or CPU fallback | T; Q3 lifecycle cases; system/contract; required-real GPU plus isolated catalog; automated; pending |
| A9 | One-time conversion mapping and current producer/consumer path pass; obsolete formats are rejected by normal app routes, converter is unreachable there and superseded application paths are removed | T; Q2; contract/system; required-real candidate app/PostgreSQL and conversion fixtures; automated; pending |
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
5. Typed context/provider and the current operation contracts are the seams.
   Private `engine.rs` imports are current accidental coupling to replace;
   retained historical imports remain historical evidence.
6. Tuldok cutover requires its retained-state/runtime/live-path rehearsal; Rheon
   enablement separately requires pressure parity, lifecycle and all 74 cases.
   One component's pass cannot authorize the other. Both stop ordinary work
   until their bounded post-switch smoke succeeds.
7. Reusing the existing Runtime/Runs and Worker/ModelProcess owners avoids new
   queues, catalogs and lifecycle registries. Deleting the small pressure adapter
   would move required tensor/physics mapping into its callers; keep that mapping
   together. The temporary converter contains unavoidable one-time source decoding;
   removing it after conversion leaves no missing runtime capability. Permanent
   old-format readers/API branches add no requested value and are not retained.
8. Retained machinery is one explicit dependency selection, the existing runtime
   support port, one model adapter per actually supported contract and existing
   catalog upgrade tools and current project/activity interfaces. Temporary
   conversion uses those owners plus a frozen accounting manifest and archive.
   No permanent compatibility registry, alternate history reader, scheduler,
   automatic retry, shadow catalog or permissive schema fallback is added.

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
not an assertion that hardware/authority is missing. Required reference/source
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
The earlier [review dispositions](reports/standards-review-resolution.md) remain
historical. The user's [one-time migration decision](reports/one-time-migration-decision.md)
supersedes their ongoing-compatibility design; numerical, lifecycle and cutover
gates remain required.
