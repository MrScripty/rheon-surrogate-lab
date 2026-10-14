# Execute the frozen GroupNorm development diagnostic

**Plan status:** `Verifying`

**Current phase:** First native action failed stop after one case due to a receiver arithmetic defect; corrective source conformance and renewed review are in progress.

**Next slice:** Freeze and independently review the exact RHS-norm correction, then perform at most one corrective full32 pass under the owned transition.

**Acceptance status:** Initial conformance/reviews passed; real first-case response and failed source guard are preserved. Corrected conformance review and complete C6/C7 outcomes remain pending.

**Ledger:** [execution-ledger.md](execution-ledger.md). **Findings:** [issues.md](issues.md).

## Objective and authorization

Implement and execute the previously reviewed [PR7 bounded protocol](../global-pressure-groupnorm-native-dev-v1/protocol.md), using final200 cached predictions and unchanged native6. The owner explicitly approved implementation, independent conformance review and one fixed32 local development pass on2026-10-14. This supersedes the historical planning packet's pending implementation/execution labels, while preserving its scientific contract and reviewed bytes. PR6 was separately approved and merged as `28c65ac868678e496603d248c5c1e350ffc0fab3`; its three main CI workflows passed. PR7 remains draft/unmerged; its original commits are preserved in this branch through normal merge `2f66709`.

Exactly24train/eightvalidation teacher snapshots from16whole parents atframes0/23 remain development data. Use previous-pressure-only full-budget Jacobi (the existing temporal_full_budget method), distinct zero-start Jacobi, and every prescribed raw/short-temporal diagnostic. No source/math gate, fallback limit, data reservation, candidate, model or original negative result changes. No new training, inference, final-suite arrays, state advance, rollout, GPU, credential renewal, grant or managed Tuldok Run is authorized.

## Acceptance claims

| Claim | Required evidence and boundary | Status |
| --- | --- | --- |
| C1 All retained inputs/native/source/reservations and old tracked bytes unchanged | Real local read-only identity inventory; [input preservation](reports/input-preservation.md) and [input freeze](execution-freeze.json) | satisfied |
| C2 Distinct closed typed GN/artifact admission and whole-parent case/raw/native residual correspondence | Focused fixture tests plus actual offline CLI verify; no model/native call | verifying |
| C3 Exactzero/base-retention, fresh nonzero-base reset, all original quality/work semantics and honest unknown iteration handling | Literal unchanged-native composition, source-bound native regressions and focused independent fixtures/review | verifying |
| C4 Bounded process streams/watchdog/RSS/owned descendant drainage and failure publication | Real Linux synthetic process/system tests and durable publication failure regressions | verifying |
| C5 Independent exact source/input/resource review before scientific launch | [Numerical review](reports/numerical-review.md) and [standards review](reports/standards-review.md), exact committed source and execution freeze | satisfied |
| C6 One complete fixed32 observation or honest fail-stop partial evidence, all original methods/gates and full available costs | Authorized real native6 local action; raw journals, aggregate and terminal measurement | pending |
| C7 Reviewable publication and exact-head CI | New source/tests/results committed/pushed, draftPR, clean scope and terminal GitHub CI | pending |

## Binding decisions and owners

Main owns serial admission, execution, report composition and repository publication. Admission owns GN identities, immutable-byte decoding and packet/capture/operator correspondence. Process owns one Linux native child/descendants, bounded IO and terminal drain. Native6 remains the sole projection/PCG/physics owner; no native rebuild or API expansion. Tuldok is historical train/restore custody only.

The selected runtime is existing local Python3.12.14/NumPy2.3.5 with one-thread environment. Runtime qualification bytes are separately identified from256MiB scientific admission reads; current runtime observations are reported independently of old captured train/restore runtime. Scientific repeated reads and new source/freeze bytes share the256MiB evaluation read cap. Limits remain verify60s/evaluate600s, perchild10s identity/60s case, observed4GiB process high-water, request1MiB/stdout512KiB/stderr64KiB, output32MiB. One enclosing SIGALRM watchdog includes source freeze/admission and leaves1s for failure reporting. It starts after interpreter/stdlib bootstrap, which is explicitly outside the reported local workflow clock.

Fresh direct/tmp private outputs have one writer and no concurrent path mutation or readers before terminal completion. Cases are fsynced before their outcome is consumed; malformed output and terminal numerical/resource failure stop with retained evidence, no scientific retry. Final publication stages bytes and revokes successful filename on post-write failure. Pending/incomplete final files are non-authoritative and remain within the16MiB report-family cap. Terminal summary supplies measured elapsed through publication; artifact records its before-publication scope. No candidate-specific standalone wall/speedup is measurable with native6.

The new source freeze binds the committed implementation, its bytes, [execution-freeze.json](execution-freeze.json) identity, explicit local paths and fixed limits. Review reports bind that exact material before main runs the already authorized pass. Existing native identity is verified before/after the32cases; all declared source/input bytes are rechecked before successful publication. No ambient mutable source or Python bytecode cache supplies module authority.

## Composed-design review

**Applicability:** `applicable`. Preserve PR7's reviewed composition.

1. Independent concerns are candidate/data identity, native numerical policy and process/publication lifetime; each has one owner.
2. Candidate update200/source/bytes belong to historical training/restore. Snapshot originalb/base/fallback belongs to native/protocol. Time/output/child state belongs to adapter.
3. Caller supplies exact admitted freeze and local roots. Admission alone knows GN schema/array correspondence; caller never relabels old receipts or imitates physics.
4. Candidate/schema changes affect admission; native gates/solver affect native closure and require a new scope; IO/cost/report changes affect process/orchestration.
5. Stable interfaces are unchanged native6 request/response and pinned existing reader primitives. Exact-byte compiled new modules prevent a second cached execution authority.
6. Identity failure precedes native work; numerical rejection keeps declared fallback; process/publication failure produces incomplete evidence. Their causes remain distinct.
7. Deleting the admission owner leaves no qualified GN receipt consumer; deleting the process owner loses stream/drain bounds; deleting speculative registries/executors changes nothing, so none is added.
8. Necessary machinery is one fixed scientific adapter in three ownership modules, fixture tests and bounded durable reports. It adds no training system, solver abstraction, service or model registry.

## Corrective transition

The actual first action preserved fail-stop after one native case; no automatic retry occurred. The RHS-norm receiver correction now revalidates that saved response offline without physics changes. [Owned corrective admission](reports/corrective-pass-admission.md) selects at most one renewed independently reviewed full32 action to complete the already authorized objective, with the extra first observation and all costs disclosed separately. No further scientific retry is admitted after that correction. The initial exact-source reviews remain historical evidence and do not qualify changed source.

## Milestones and write set

M1 adapter/conformance: new `experiments/global-pressure-v1/groupnorm-native-dev-v1/**`, this new plan directory, and `.github/workflows/groupnorm-native-development.yml` only. Gate C1–C5 plus bounded real offline admission; `Accepted`;60fixtures, [actual offline verification](reports/offline-verification.json) and both independent exact-source reviews pass.

M2 one authorized local pass: one predeclared fresh `/tmp` output and ignored task-local evidence below `runs/global-pressure-groupnorm-native-dev-run-v1/**`; public measured reports only in this new plan directory. Gate C6 unchanged original quality/failure/work/cost semantics; `Verifying`; exact source freeze and independent review gates passed, one action still unperformed.

M3 draft publication: commit/push only declared new source/tests/docs/workflow plus preserved originalPR7 history; gate C7 and no secrets/bulkarrays; `Planned`. Main target is verified PR6 merge main; branch `feat/groupnorm-native-development-evaluation` remains a review proposal until separately approved integration. No history rewriting, cleanup, tags or outreach.

## Blockers and replan triggers

Source findings are tracked in [issues](issues.md) and must be resolved before C5. The18:06 executor transport disconnect blocked a read before process creation; a needed successful read at18:07:32 and22fixture tests at18:17:29 established resumed access. No scientific attempt existed to repeat.

Stop on new training/access decision, input/native/candidate drift, missing archive, unavailable closure/zero correspondence, exhausted bounds, incomplete drain/publication or changing scientific scope. Do not substitute another model, binary, subset or tolerance. A numerical failed complete-reset is retained fail-stop evidence, never a qualified solver claim. Autonomous rollout, final trajectory/geometry/regime generalization, tracer/momentum/energy conservation and matched cold/resident acceleration remain outside this task.

## Standards

Adopted pin `188beda1fa477d21d576c233dd8c7f4c4c267d23`: Core/Router, Implementation, Verification/Oracles, Development Proportionality, Planning, Commit, Documentation, Contracts/Protocols/Evolution, IPC/Persistence, Performance, Security and code ownership. The scientific protocol is already independently reviewed; current reviews decide executable conformance. No applicable AGENTS/SKILL file was found in checkout or parent workspace; no new approval rule is inferred from a skill. All workers are GPT-6.1 Sol High with disjoint writes; reviewers are independent of authors.
