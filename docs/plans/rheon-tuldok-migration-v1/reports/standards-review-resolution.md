# Migration plan review dispositions

Scope: user-authorized revision of the plan reviewed at 58d37a1, following the
five findings retained in [standards-review.md](standards-review.md) at 0502ce4.
The original plan/review remain in Git history; the review files are unchanged.
These dispositions mean **addressed in the planning specification**, not runtime
acceptance or independent review of an implementation.

Current coding standards MCP snapshot:
`snapshot:v1:29a39beb-fa6b-469b-9dfa-453e02f37fa0`, 25 selected policies, zero
unresolved routing fact categories. Routing is not a conformance attestation.
Applied Planning acceptance/replanning, Contracts evolution, Persistence,
Verification/concurrency, composition review and scoped commit guidance.

| Finding | Disposition in the revised canonical plan/protocol | Evidence boundary |
| --- | --- | --- |
| R1 — numerical acceptance | Fixed 26b control, normalization, captured source, runtime manifest, old binary, policy, ordered 74 scenes, group/case metrics and first-frame capture in [baseline-bindings.json](baseline-bindings.json). Q4 requires exact same-input model arrays; Q7 requires every case's physical gates, complete horizon, no native fallback/refinement, no per-case pressure-error or call regression and group non-regression. No added numeric comparison allowance; a mismatch fails/replans. Performance remains advisory with paired timing/spread | Hashes were checked against local files and immutable 1548266 Git records without inference/training. Candidate build hashes and actual acceptance still pending |
| R2 — cutover gates | M2-T and M2-R are explicit independent prerequisite streams. M3 switches Tuldok only after A1/A2/A5-rehearsal/A8/A9; A10 gates ordinary submissions. M4 enables Rheon only after A3/A4/A6-rehearsal/A7 and then requires A6-live. Q8/Q9 distinguish rollback and retain new failed writes. All 74 cases precede Rheon enablement; they do not block an independent Tuldok-only upgrade | No source/service/catalog cutover executed |
| R3 — CUDA failure lifecycle | New A8/Q3 specifies hashing/training cancellation, GPU-active deadline, startup/asset failure, shutdown/reopen and ambiguous-custody fences. Existing Runs/process/runtime tests plus profile-specific real GPU cases are named, with positive discovery and update/resource limits. A4 remains separately Rheon-owned | New lifecycle tests are planned, not claimed to exist or pass |
| R4 — compatibility policy | Canonical table classifies retained local-v1 activity, ordinary v1/v2 Projects, managed recipe/material/runtime closures, public operations and Rheon packages. Historical bytes stay unchanged. New live projects use evidence references and an explicit activity operation; selected snapshot inspection supports retained local variants. No silent legacy response widening. Cold reads validate retained host identity; new execution requires current measured material and fresh approval. Q2 requires complete actual-variant inventory and cold/public-negative tests | Selected project/runtime APIs require implementation and qualification; unknown actual variants trigger replan |
| R5 — claim metadata | Acceptance is partial while A0 is satisfied. A0–A11 have owners, named procedures, evidence kind, required-real environment, mode and status. A5/A6 distinguish rehearsal/live evidence. Current blockers are explicitly none for M1 source work, while deployment gates remain pending | No execution claim advanced to satisfied in this revision |

## Revision verification

Checks performed for this document-only revision:

- Baseline control weights/normalization/source-witness/runtime manifest and
  binary hashes match their retained records; first-frame SHA matches captures.
- Exactly 74 ordered scenes match the archived control execution configuration;
  all archived control cases are complete/qualified and group counts are 58/8/8.
- Existing pressure policy values and original divergence/flux/Courant limits
  were read from their source owners; no acceptance limit was relaxed.
- Prior review files, historical manifests/recipes/results and unrelated user
  chart/generated files are unchanged.
- JSON parsing, local document links, procedure/claim references, staged scope
  and whitespace are checked before commit. No new model tests or simulations
  are needed to verify a plan revision; Q1–Q10 remain future execution.

The small [revision verification receipt](plan-revision-verification.json)
binds the revised documents and the completed artifact/reference checks.

The acceptance protocol is a written procedure, not a new runner, scheduler,
schema-validation framework or evidence service. Existing production/test
owners supply the actual observations. Named planned tests require positive
discovery before they can contribute evidence.
