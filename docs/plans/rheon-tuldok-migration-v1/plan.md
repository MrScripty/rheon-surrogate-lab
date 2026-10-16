# Migrate the lab to current Rheon and Tuldok

Status: Planned. Current phase: source assessment complete; runtime migration has
not started. Migration acceptance: pending. Owner: lab migration task. The user
requested GitHub updates, a change assessment and migration plan, and explicitly
authorized improvements to Tuldok, including CUDA support, where needed.

Next slice: M1, integrate and qualify the existing managed Torch and live-project
capabilities on a new Tuldok branch based on the fetched main. Keep the running
schema-8 service in place during that work.

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
   qualify new execution against the new host source. Do not relabel a CUDA
   installation as `captured_cpu_python_v1`, route GPU work through an unqualified
   system-Python escape, or introduce a second runner/catalog. CPU-only captured
   snapshots may be adopted later through their own supported runtime contract.
3. Keep v1 live-project data import and evidence inspection working first. New
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

## Milestones and allowed implementation writes

| Milestone | Coherent outcome and write set | Gate | State |
| --- | --- | --- | --- |
| M0 | Fetch all three remotes; fast-forward clean primary Tuldok; create isolated fetched Rheon; write only this planning directory in the lab | Immutable version, dirty-state, compatibility and catalog observations retained | Accepted |
| M1 | New Tuldok branch based on 300f12d. Runtime owners: `torch_runtime.py`, `torch_runtimes.py`, `recipe_contract.py`, `runs.py`, their focused contracts/tests. Live project owners: `project_activity.py`, `projects.py`, `app.py`, corresponding `static/workspace-project*`, `static/workspaces*`, project/runtime docs and tests. New runtime application environment under the migration run root; no alteration of the running environment | Managed CUDA and CPU definitions admitted; captured-CPU and legacy contracts retained; historical material/project reads supported; actual CUDA update/fresh CPU restore on a nonproduction qualification fixture on a schema-10 qualification fixture | Planned |
| M2 | Lab integration branch from 1548266 plus this assessment. New `experiments/dependency-migration-v1/**`, matching reports, and narrowly affected build/adapter callers under `experiments/local-pressure-v1`, `experiments/water-residual-learning-v1`, `experiments/live-water-comparison-v1`, `probes/residual-rollout-v1`. Update live manifest only after the migrated root is usable. Any necessary Rheon public diagnostic seam has its own upstream branch/tests/docs | One-frame operator/input/output parity first; then real model process lifecycle and new diagnostic/session recording semantics | Planned |
| M3 | Existing Tuldok backup/restore/upgrade tools and service launch configuration; migration receipts here, large backups in a new ignored/external destination with sufficient space. No manual SQL patching or historical JSON rewriting | Rehearsed schema 8 restore→10 upgrade, idempotent re-entry and unchanged historical rows/artifact bytes; actual controlled original-catalog cutover and smoke | Planned |
| M4 | New lab project snapshot, owned pressure task/model/evaluator/field-evidence contributions and migration reports; use Tuldok's existing Model/Study/Trial/Run owners | Live project and managed CUDA→CPU path visible in Tuldok; Rheon desktop comparison/difference legend and cold reopen work with actual model; all consumed 74-scene regressions retain thresholds and provenance | Planned |

M1 source integration and M2 source qualification can precede live service
cutover. M1's real managed execution needs an isolated migration rehearsal
catalog restored from a quiesced backup, or an explicitly disposable upstream
qualification fixture with its own identity. Neither becomes a production data
collection. Never run two application versions against the original catalog.
Before expanding a milestone's concrete write set, inspect its affected owners;
replan if that changes contracts, data custody, process ownership or physics.

## Acceptance claims

All are pending except assessment A0. Existing upstream tests and receipts identify
what to reuse; they do not establish local migration acceptance.

| ID | Observable criterion | Evidence / environment / mode |
| --- | --- | --- |
| A0 | Targets, local deviations, live source, schema and incompatible consumers identified | Satisfied: linked assessment, inert probes, local catalog read and nine negotiation checks; contract/integration; required-real local sources/PostgreSQL; automated |
| A1 | Historical Project snapshots and completed/failed/interrupted Run documents, source/runtime descriptors and artifact bytes survive cold reopen and backup/restore | Contract/system; actual quiesced catalog copy and PostgreSQL; automated; exact counts/digests with complete source/destination identity |
| A2 | Actual managed CUDA optimizer update, GPU float32 weights/grads and fresh-process CPU checkpoint restore; both completed/drained and artifacts matched | System; RTX5090/Pumas runtime and new Tuldok host; automated; record launch, hashing, optimizer, restore, wall/RSS/CUDA memory and child cessation separately |
| A3 | New Rheon sees the same saved one-frame grid, RHS, residual, prior, channels, normalization and output reconstruction; oracle/gauge checks and zero-residual bypass remain correct | Contract/integration; real CPU/native artifact and unchanged checkpoint; automated; compare before/after arrays and independent equation checks, with any non-bit-identical difference explicitly explained |
| A4 | Start, failure, in-flight cancel, termination/reaping, restart, shutdown and cold saved inspection retain truthful runtime/fallback facts | System/user-workflow; actual CPU model child and egui desktop; automated plus recorded desktop interaction. Use the upstream real-deployment lifecycle test; its 16-step restart must observe exactly one inference after 15 native steps |
| A5 | Existing catalog/project and generated fields/progress are visible through Tuldok; new typed Trial lineage is truthful; no fabricated retroactive checkpoint/evaluator evidence | User-workflow/contract; actual browser/API and existing catalog after cutover; automated plus recorded UI inspection |
| A6 | Native/AI pressure and signed voxel difference with legend load live and from saved recordings; same-request metrics stay separate from trajectory drift; fallbacks cannot masquerade as AI passes | User-workflow/integration; actual Rheon desktop/CPU checkpoint; either |
| A7 | All74 already-consumed regression scenes run with the frozen baseline model/policy under old/new runtimes; report per-scene pressure/residual/divergence, actual network calls/fallbacks, full-step and inference time, RSS and failures | System; bounded local CPU and real native builds; automated. Keep original 58/amended 8/recent 8 groups distinct; no use of the untouched eight scenes |

The older aa82 model has retained accuracy failures. Its migration lifecycle and
byte-parity checks must reproduce honest operation; they cannot turn those
failures into passes. Select the later model/policy for A7 from its existing
retained acceptance record, not from migration test outcomes. No new checkpoint
selection or promotion is authorized by a successful migration.

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
6. Source integration/builds and model parity can fail independently. Production
   cutover is gated on their qualification and verified retained-state migration.
7. Reusing the existing Runtime/Runs and Worker/ModelProcess owners avoids new
   queues, catalogs and lifecycle registries. Deleting the small pressure adapter
   would move required tensor/physics mapping into its callers; keep that mapping
   together. A second universal training framework adds no deciding capability.
8. Retained machinery is one explicit dependency selection, the existing runtime
   support port, one model adapter per actually supported contract and existing
   catalog upgrade tools. Add no scheduler, automatic retry, shadow catalog or
   permissive schema fallback.

## Operations, retention and reconsideration

Use one GPU optimizer and at most two bounded one-thread CPU simulation workers,
with the previously measured affinity and 8 GiB available-memory reserve. Limit
the first qualification to one frame before full rollouts. Record timing and
memory throughout; no assumption that using 24 threads is faster. Large geometry
settings need allocation preflight before fields or preview histories exist.

Keep the existing8091 service untouched until M3. Stop every app/provider/artifact
writer before backup/upgrade. Complete backup covers SQL, managed files, runtime
snapshots and receipts; external lab runs, model bundles and project-root bindings
need separate preserved inventory because Tuldok Project backup does not copy
external paths. Verify backup by restore to an absent destination/schema. Use
upstream `upgrade_metadata.py CATALOG --quiesced`; on unknown commit keep writers
stopped and re-enter the same verifying command. Do not manually bless signatures.
Rollback uses the stopped pre-upgrade backup in a new destination and the old
matching binary. Post-cutover writes require a separate retained-state recovery
decision; an old backup cannot represent them.

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
