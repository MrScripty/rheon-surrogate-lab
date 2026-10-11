# Pressure data and first CPU experiment

Plan status: Blocked. Current phase: publication; bounded local diagnostics accepted.
Next slice: publish the committed local experiment after the user creates the exact authorized public repository.
Acceptance status: blocked (publication); previous local generator/model acceptance satisfied; diagnostic acceptance satisfied.
Author/owner: Puma; implementation and integration owned by the task agent.

[Ledger](execution-ledger.md) · [Issues](issues.md) ·
[ADR](../../decisions/0001-pinned-reference.md) · [Review](reports/review.md)

## Objective and scope

Create public MrScripty/rheon-surrogate-lab and publish a reproducible small
pressure-surrogate experiment. First deliver qualified reference data and tests,
then a bounded CPU train/evaluate path if practical. User explicitly authorized
creation, implementation, commits and push. No Rheon-main edits, alternate
repository, paid service, large downloads, long training, external deployment,
existing-PR merge or immediate integration dependency is in scope.

## Acceptance claims

| ID | Observable criterion | Kind / environment / mode | Status |
| --- | --- | --- | --- |
| A1 | Exact pinned public Rheon APIs generate and physically qualify 3D targets | integration / representative Linux CPU / automated | satisfied: 16 tests |
| A2 | Persisted data roundtrips; deterministic bytes; sampled corruptions rejected at named boundaries | contract / representative Linux CPU / automated | satisfied: 16 tests |
| A3 | A clean committed CLI generates 18 cases and a new process verifies them | system / representative Linux CPU / automated | satisfied: reports/generator-smoke.json |
| A4 | Independent numerical/source review; fmt, tests and Clippy pass | focused / representative Linux CPU / automated and manual review | satisfied: reports/review.md |
| A5 | Requested public GitHub URL and pushed commit verified | release-artifact / required-real GitHub / automated | blocked |
| A6 | Bounded CPU supervised CNN path consumes train-only normalization and reports held-out raw prediction physics | system / representative CPU / automated | satisfied: reports/tiny-cpu-smoke.json and reports/heldout-cpu-smoke.json |
| A7 | Development-only baselines and unchanged-checkpoint diagnostics identify qualification failures, preserve test isolation and report bounded compute | system / representative Linux CPU / automated | satisfied: reports/qualification-diagnosis.md and reports/development-diagnostic.json |

## Decisions, constraints and observations

ADR 0001 owns the pinned-reference/data decision. Schema v1 is fixed and strict;
no compatibility with an unimplemented format is promised. Git pin, lockfile and
Rust toolchain own source resolution. Raw arrays are unnormalized. Runtime
normalization belongs to the training consumer. No upstream implementation is
copied. Target qualification owns full residual and rounded f32 divergence.

Rheon pin and source reviewed: 34235d33c886aa05b5bdf95d920b863d1fb2ad6f.
No AGENTS.md or .agents exists in that pinned tree. Coding-Standards reference:
188beda1fa477d21d576c233dd8c7f4c4c267d23; core, router, planning, implementation,
verification/oracles, commit, architecture, Rust and licensing guidance consulted.
Fresh cloud workspace only: /workspace/rheon-surrogate-lab. Separate read-only
Rheon reference checkout; existing /workspace/Rheon untouched.

GitHub get-repository returned 404; authenticated create returned GraphQL
Resource not accessible by integration (createRepository). This blocks A5 only.
The present public API always zero-starts; initial guesses are a deferred API
milestone, not a hidden imitation in this repo. Python NumPy exists, Torch does
not; a tiny NumPy CNN avoids downloading a framework/model.

## Evidence and oracle boundaries

Independent hand/dyadic 3D fixtures assemble the finite-volume stencil and
velocity gradient only in tests. They check signs, gauge, spacing and all axes
against both preconditioners. This is not another production solver. Production
verification proves conformance to pinned Rheon on this contract, not an
independent general CFD/PDE accuracy claim. Hash equality proves repeatability
and corruption detection, not authenticity. Negative tests start from valid data
and isolate hash, residual, gauge, finiteness, shape, split, completion, metadata,
receipt or boundary failure. Cross-platform bitwise equality is unqualified.

## Composed-design review: applicable

- Independent concepts: deterministic preprojection scenarios/qualification,
  durable array exchange, and later supervised model training. Each has one owner.
- State/identity/value/time/policy: source commits and case seeds identify data;
  dt is physical time, not a trajectory; immutable complete datasets are read-only.
  Schema v1 owns exchange guarantees and references Rheon solver authority.
  No schema overlap/migration or dynamic registry is supported.
- Caller knowledge: CLI names output directory/count; dataset module supplies
  arrays/metadata; model reads explicit shapes, units, splits and verifies first.
- Change paths: new physical domain changes experiment and schema/consumer;
  optimizer changes model only; publication credentials change no numeric code.
- Stable interfaces: Rheon public operator/workspace; JSON/raw array contract.
  No private solver state, hidden warm-start dependency or inherited UI state.
- Independent failure: verification rejects incomplete/invalid data; model can
  fail without changing data; solver pin can change only under requalification.
- Deletion: removing the data boundary would move shape/hash/split knowledge to
  every consumer; removing the model leaves a useful reference-data tool. No
  speculative Tuldok or GUI adapter is retained.
- Necessary complexity: numerical acceptance and durable metadata are contained
  in two modules; one CLI composes them. No service, registry or orchestration.

## Milestones and allowed write sets

1. Generator/verifier: src/, tests/, Cargo.toml/lock, build.rs, rust-toolchain.toml,
   .gitignore, README.md, THIRD_PARTY.md, docs/, optional .github/workflows/. Gate:
   A1-A4. Status: Accepted. Preservation: independent repo, exact upstream pin.
2. CPU smoke: scripts/, requirements.txt, docs/ and README. Gate: train-only
   stats, gradient regression, tiny-overfit attempt and held-out numerical report.
   Status: Accepted as a runnable smoke path, not model quality. Keep data/model checkpoints ignored.
3. Publish: existing repo Git configuration and exact requested remote only.
   Gate: A5. Status: Blocked. No unrelated repository changes.
4. Bounded diagnostics: Accepted. Allowed write set: src/diagnostics.rs,
   src/dataset.rs, src/main.rs, src/lib.rs, tests/contracts.rs, scripts/, README
   and docs/. Gate A7: fixed baselines, no new optimizer steps, train/validation
   comparisons, independent review and an untouched future test reservation.
   Parent explicitly authorized local continuation while publication is blocked.
5. Controlled model improvement: Deferred; proposed one global-context, zero-response
   candidate bounded to 500 CPU steps and one validation comparison, with explicit
   pressure/full-residual loss and fixed baseline gates (see diagnostic report).
   No extra model was trained in this slice. One-case loss reduction succeeded but
   held-out physical qualification failed for all predictions. Broader receptive
   fields and physics-aware validation are candidates; select using training and
   validation only, preserve the test set. Checked warm-start API and matched-quality
   comparison follow separately after useful initial-guess behavior is established.
   UI numerical/model/error slices and Tuldok manifest
   adapter, then Pantograph/Pumas integration are subsequent deferred work.

## Blockers and re-plan triggers

Publication is paused until the user creates the exact empty public repo and
write access exists. Parent instructed no creation retries, alternate destination
or upstream publication. Preserve original source and recovery bundle. Project license selection belongs to the owner;
no guessed license is added. Re-plan if fixed-grid targets fail actual divergence,
public API contract changes, dependencies require large provisioning, or data
leakage/ownership findings invalidate the admission. Ordinary test fixes remain
within the slice. Independent reviewer has no write set and returns findings.

The fresh local main branch is the retained experiment source pending push;
no temporary worktree or shared-history rewrite is used.

The development protocol and future test reservation are owned by
[ADR 0002](../../decisions/0002-development-evaluation.md). Existing checkpoints
are inspected without updates. Optional upstream initial-guess patch preparation
is not needed to answer this diagnostic question and remains deferred.
