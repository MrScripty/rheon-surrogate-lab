# Reusable local CPU pressure experiment

**Status:** Active. **Acceptance:** pending.
**Current phase:** verify fresh packing and managed CPU bootstrap.
**Next slice:** freeze tested source and observe dry preflight against the actual captured runtime.

## Objective and authority

The owner requested a fresh local CPU pipeline through capture, packing, managed
Tuldok training, independent zero-update restore and native physics evaluation,
with reusable experiment configuration and commits on a new branch. This request
authorizes the new development lineage; historical action limits and identities
remain historical. No GPU or final-suite work belongs to this experiment.

Branch `feat/reusable-local-cpu-pipeline` started at `f772a2d` and fast-forwarded
to merged main `03c05e8` at the owner's request. The root agent owns
implementation, verification and commits; the branch remains a local review
proposal based on `feat/groupnorm-native-development-evaluation`. No merge, push,
history rewrite or cleanup of other workspaces is included.

## Decisions and write set

- `experiments/local-pressure-v1/` owns reusable configuration, lineage, packing,
  train/restore recipes and bootstrap. Paths live in separate host configuration;
  scientific configuration contains no host paths or historical dataset hashes.
- `probes/local-pressure-v1/` owns capture with configurable development
  reservations. Native physics remains the pinned Rheon dependency. The existing
  independent face-incidence verifier and native solver evaluator are reused.
- `probes/local-pressure-evaluation-v1/` retains native diagnostic arithmetic
  byte-for-byte. Its build binding qualifies latest fetched Rheon `43ed26b6`
  with the unchanged reviewed pressure patch. `dependency.py` owns selected
  source preparation; `native.py` owns both build identities and bounded calls.
- The first profile retains the existing GN architecture, preprocessing and loss.
  Portability and fresh data are distinct from a model change. No new model is
  proposed in this implementation.
- Only new development seeds are used; parents stay whole across splits. Final
  reservations are excluded. A one-frame fixture must qualify before full capture
  and cannot be packed as a training dataset.
- Generic Tuldok at `f7543fb` owns runtime capture, catalog, approvals, custody and
  Runs. New source/packet/runtime identities are measured and frozen, not invented
  or substituted into historical validators.
- Additional writes: this plan directory, the current local handoff's stale
  status, a root README entry and `.github/workflows/local-pressure.yml`. Historical
  executable files, reservations, results and inventory remain unchanged. Bulk
  material and private owner state remain ignored under `runs/` and `.venv/`.

## Acceptance

| Claim | Evidence / environment / mode | Status |
| --- | --- | --- |
| Configuration can select two fresh lineages without source edits; invalid splits/paths fail | focused contract tests / representative Linux / automated | pending |
| One-frame train fixture passes independent native physics before full capture | real producer and independent verifier / local CPU / automated | pending |
| Complete new development packet binds capture/native residual; preprocessing uses train only | real capture/reader plus leakage tests / local CPU / automated | pending |
| Dry preflight creates no training Run; real runtime/source/project identities are bound | generic Tuldok / local PostgreSQL and captured CPU / automated | pending |
| Train and independent restore complete/drain; history, state and predictions agree exactly | generic Tuldok Runs / captured CPU / automated | pending |
| Native evaluation retains original gates, baseline and all warm/fallback work | real native operator / local CPU / automated | pending |
| Historical executable/results bytes remain unchanged and work is committed | baseline diff and staged review / repository / automated | pending |

## Sequence and design review

1. Portable capture/configuration and fixture qualification.
2. Fresh packet, train-only preprocessing and zero-update qualification/budget.
3. Runtime/catalog/project bootstrap and dry preflight.
4. Managed training/restore, native development evaluation and retained evidence.

Each milestone includes new source and focused tests in the declared write set.
Lifecycle transitions require the corresponding acceptance evidence. Keep
portability commits separate from any later model-change commits.

Composed-design review is applicable: experiment policy belongs to configuration;
numeric lineage belongs to the packet reader; physics belongs to Rheon and its
independent oracle; execution/custody belongs to Tuldok. The caller supplies two
configuration files and a fresh destination. It does not select hashes or fabricate
custody. Model changes affect the model profile; host changes affect bootstrap;
data changes affect reservations and measured receipts. Invalid identity stops
before use; numerical failures retain fallback and costs; process failures retain
incomplete evidence. Removing the configuration owner would spread policy across
capture, packing and training; no new generic executor or runtime registry is added.

## Standards, blockers and replanning

Adopted standards `188beda`: Core/Router, Implementation, Verification/Oracles,
Development Proportionality, Planning, Commit, Documentation, Build, Contracts,
Architecture/Code Design, IPC/Persistence, Dependencies, Performance and Security.
The written plan records sequencing and independent scientific/execution owners;
routine reversible implementation does not require another approval ceremony.

Replan if local runtime capture cannot qualify, generic custody is unavailable,
the fixture fails unchanged physics gates, resource budgets are exceeded, or a
model/physics/final-suite change is needed. Do not relax numerical tolerances to
complete the pipeline. Runtime inspection initially rejected Ubuntu's external
sitecustomize symlink and compiler-only python.o. The bootstrap materializes
installed Python bytes and excludes link-time objects/archives before the
unchanged runtime inspector. Inspection now qualifies22,501 files/825,750,560bytes;
actual captured execution remains pending.

[Ledger](execution-ledger.md) · [Issues](issues.md).
