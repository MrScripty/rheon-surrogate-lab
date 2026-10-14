# Native warm start and deterministic development capture

**Plan status:** `Accepted`

**Current phase:** Accepted slice1 native API and development reference capture

**Next slice:** Parent review; separately authorize model and Tuldok recipe smoke slice

**Acceptance status:** `satisfied`

**Execution ledger:** [execution-ledger.md](execution-ledger.md)

**Issues:** [issues.md](issues.md)

## Objective

Qualify an isolated sealed-domain initial-pressure API at Rheon
`2209012bc5f902693d0f695ec1c2d854126af317`, generate deterministic development-only
reference trajectories and independently verify capture semantics and physics.
This is slice1 of the [reviewed design](../global-pressure-correction-v1/plan.md),
newly authorized by parent after acceptance. No training, model implementation,
final-test generation, tolerance relaxation or upstream Rheon/Tuldok write.

## Objective Acceptance

| ID | Criterion | Kind | Environment | Mode | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| N1 | Lab-owned immutable patch at exact native base admits sealed guesses and preserves old zero-start, full original-b residual, cancellation/reuse for both PCGs | integration | required-real pinned native CPU | automated | satisfied | [Native checks](reports/native.md), [receipt](reports/native.json) |
| N2 | Native and independent hand-stencil/gradient/boundary oracles agree for anisotropic rectangles | focused | representative Linux CPU | automated | satisfied | [Capture](reports/capture.md), [receipt](reports/capture.json) |
| N3 | Only smoke train/validation trajectories generated deterministically; complete manifest/arrays roundtrip through fresh independent verifier, corruption fails at intended boundary | system | required-real fresh CPU processes | automated | satisfied | [Capture](reports/capture.md), [receipt](reports/capture.json) |
| N4 | Actual reference gates, elapsed generation/verification and RAM/payload feasibility reported; failure stops preset without dropping a case | system | required-real CPU | automated | satisfied | [Capture](reports/capture.md), [receipt](reports/capture.json) |
| N5 | Independent numerical and standards source/evidence reviews closed, affected tests/format/lints pass | integration | representative independent GPT6.1 Sol High reviewers | either | satisfied | [Review](reports/review.md) |
| N6 | Exact scoped branch pushed, draft implementation PR verified; no training/final arrays/upstream edits | release-artifact | required-real GitHub | automated | satisfied | [Publication](reports/verification.md) |

## Binding Decisions

Allowed tracked writes: `experiments/global-pressure-v1/` (dependency patch/manifest,
prepare helper, capture contract/config/README and dedicated fixture tests),
`probes/global-pressure-v1/` (new isolated Cargo manifest/lock/build source/Rust
producer and tests), `docs/plans/global-pressure-native-capture-v1/` and
`.github/workflows/global-pressure-native.yml` (new supporting native/capture CI).
No root manifests, old probe sources, old reproduction/design/negative-result
receipts, reservation.json, Tuldok or upstream Rheon files change. Generated
source copies/toolchain/data/target/run evidence live only in task-created ignored
`runs/` or /tmp/workspace tooling, with provenance recorded.

Native implementation owns the new exact API/patch provenance; capture implementation
owns the actual trajectory producer and verifier; main agent owns integration,
plan/evidence/CI and Git lifecycle. Review agents have no write set. Use one
lab-owned content-hashed patch and prepared dependency source closure, never an
ambient sibling checkout. No remote dependency fork/publication is created.

Capture follows [data/model contract](../global-pressure-correction-v1/data-and-model.md)
and [strict common physics](../global-pressure-correction-v1/acceptance.md).
The reservation remains immutable; only existing smoke train/validation entries
are admissible. Reject test/full-generation requests at the CLI before any output
creation. Hash/provenance/shape validity and native physics are distinct claims.
Stop for parent review on unsupported API or reference-physics failure.

Toolchain Rust1.92.0/Cargo and explicit pinned dependency closure; provision only
an isolated workspace toolchain if absent. CPU native one thread for feasibility;
no numerical training/package/driver downloads. A local toolchain is not a runtime
qualification for Tuldok. Cargo fmt/Clippy/test/build and Python helper/verification
checks target the new source; old source/tests are supporting CI, not retuned.

## Simplicity And Ownership Review

**Applicability:** `applicable`

- Independent concepts and dimensions: Immutable native patch, trajectory capture,
  independent verification and generic execution each retain their own owner.
- State, identity, value, time, policy, and mechanism: Commit/patch/tree identity
  selects exact code; trajectory/reservation identity selects development eligibility;
  dt denotes physical time; elapsed wall denotes measured cost. No shared model state.
- Caller and composition-root knowledge: Caller selects actual source/dependency
  receipt, admissible preset and fresh output. It need not know private Krylov state.
- Representative change paths and forced owners: API change requalifies native;
  scenario/schema change requalifies capture/verifier; custody changes belong to Tuldok.
- Stable Interfaces versus hidden knowledge: Public operator/correction functions,
  explicit full RHS/guess API and named arrays/manifest are visible contracts.
- Independent evolution, testing, failure, and replacement: Dependency preparation
  refuses drift; invalid capture has no success manifest; verification never repairs
  input; independent oracles do not import the generator's physics implementation.
- Necessary complexity and containment: Provenance, original-RHS/gauge contracts,
  dev-only admission and strict f32 gates are necessary and confined to this namespace.
- Deletion and cumulative machinery result: Without the patch no initial guess is
  available; without capture/verifier each consumer would repeat array/physics
  knowledge. No scheduler, model registry, installer or Tuldok executor is added.

## Milestones

| Unit | Write owner/set | Gate | State |
| --- | --- | --- | --- |
| Native | Worker: new experiment dependency files/tests only; prepared source under ignored runs | N1 | Complete |
| Capture/verifier | Worker: new probe source/manifest/lock and new experiment contract/config/verifier tests; no dependency-file overlap | N2–N4 | Complete |
| Review/integration | Main: plan/reports/new CI; reviewers read-only | N5/N6 | Complete |

Main serializes shared authority, builds/final runs and Git publication. Workers
must send API/capture contract early; no commits or upstream writes by workers.
Material protocol findings are collected before replan; no test exposure.

## Blockers

No numerical or implementation blocker remains. Independent source and complete
evidence reviews are clear; affected local checks and source-head remote CI pass.
PR3 remains draft for parent review; that review and separate authorization govern
further model/Tuldok work. Final publication checks the documentation-only head.

## Re-Plan Triggers

Unsupported API, reference-physics failure, drifted source/patch, accidental final
exposure, resource-cap breach, independently observed oracle defect or materially
broader source footprint stops affected work. Preserve evidence and report exact
blocker; no case deletion/retry/tolerance change to make results pass.

## Repository Isolation

DesignPR2 passed final CI38540675841 at5a6374a and was marked ready, then merged via
normal merge at4743a16a083040fa32da24903484705c72bd2ce9. Current implementation branch
`feat/global-pressure-native-capture-v1` derives from that main. Main agent owns
writes/push/draftPR; parent owns review/integration. Public authorized lab only.
Retain while draft implementation review is pending, parent retires after disposition.
No task-created worktree or history rewrite. Existing design branch remains intact.

## Final Acceptance

- Acceptance status: `satisfied`
- Final status: `Accepted`
- Deferred follow-ups: parent implementation review/merge and separately authorized model/Tuldok smoke slice
