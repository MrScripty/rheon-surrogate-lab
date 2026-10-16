# Migration assessment ledger

## 2026-10-16 — source acquisition and bounded compatibility assessment

User requested latest GitHub Rheon/Tuldok sources, change assessment and migration
planning, then explicitly allowed required Tuldok improvements such as CUDA.
Scope performed: fetch/pull, isolated source checkout, read-only inspection and
small contract probes, planning records. No runtime migration or model training.

- Inspected all three primary repos and actual live source checkouts before
  mutation. No applicable AGENTS.md was found at primary roots/ancestors or in
  the lab file inventory. Unrelated `.generated/` and user chart tools/data
  remain untracked and outside the write set.
- `git fetch origin` succeeded for Rheon, Tuldok and lab. Primary Tuldok was
  clean and fast-forwarded with `git pull --ff-only origin main`, f58880df→300f12df.
  It remains clean. Primary Rheon was dirty; left it at 43ed26b6 and created the
  detached target worktree under `runs/dependency-migration-v1/Rheon` at bc2f4727.
- Created `plan/rheon-tuldok-migration-20261016` from lab main03c05e8. The later
  water-model branch 1548266 and all its results remain untouched. No rewrite,
  merge, model selection or upstream push was performed.
- Inspected fetched first-parent histories and affected contracts/consumers.
  Retained full changed-path inventories, exact revisions and old live commits
  in [versions](reports/versions.json) and [changes](reports/upstream-changes.json).
- Identified live service source using selected process paths only; read catalog
  schema and aggregate counts through its old Metadata owner using the existing
  service configuration. The initial direct Metadata read had no database URL
  in the shell and failed before connecting; the subsequent existing-configuration
  read succeeded. No credentials, process environment or auth tokens were saved.
- Inert new-reader probes reject the real live manifest and CUDA/CPU recipe
  definitions; managed profile alone also rejects. Legacy discovery remains byte
  equal. [Receipt](reports/contract-probes.json) scopes these claims to syntax
  and registry compatibility; it does not establish execution or retained reads.
- Upstream nine ContractSelectionTests first failed in the old app environment
  (missing new jsonschema; sandboxed loopback fixture restrictions). Retained
  [failure receipt](reports/negotiation-existing-env.json) and log. Repeated the
  same selected tests using existing lab env with jsonschema4.26.0 and host-local
  fixture sockets:9/9 passed, 23.572s wall. [Passing receipt](reports/negotiation-qualified-env.json)
  and log. No dependency install, service request or catalog write in those tests.
- Old initial-pressure patch `git apply --check` succeeded on fetched Rheon;
  no patch application, compilation, full runtime inventory rehash or model
  execution. Retained selected aa82 bundle hashes and runtime-presence observation
  in [availability receipt](reports/measured-bundle-availability.json).
- Coding standards were routed through MCP:22 selected, zero unresolved fact
  categories, snapshot d1f2bb97-7c56-430d-8687-e5dd5f78642c. Planning, contract
  evolution, persistence, lifecycle ownership and scoped commit guidance informed
  the migration sequence. This is serial bounded assessment, with no delegated
  implementation or new orchestration service.

Assessment complete. Migration acceptance remains pending. Next bounded work is
M1 in [plan.md](plan.md). New worktree is explicitly retained-protected for that
work with reachability and disposition documented in the plan/version record.

Final document checks: JSON decoding and local Markdown links passed. The first
staged whitespace check caught a trailing space emitted by unittest; only trailing
whitespace in the retained display log was normalized, with outcomes unchanged.
No model/data/credential payloads were staged. Review scope is the complete
assessment branch from 03c05e8; no preceding history was changed.

## 2026-10-16 — requested coding standards review

Reviewed the committed plan at 58d37a1 against fresh coding standards MCP
snapshot e1406d64-d247-449d-9410-3875bd1a6e67. The routed 25-policy selection has
zero unresolved fact categories; that is routing coverage, not a conformance
certificate. The [review](reports/standards-review.md) records two high, two
medium and one low finding: numerical acceptance, component cutover ordering,
CUDA lifecycle evidence, persisted contract policy and acceptance metadata.
The review recommends revisions before execution. The plan remains unchanged;
no migration, new scientific tests, catalog writes or service changes occurred.

## 2026-10-16 — revise the plan against the complete review

User requested revision. The new plan supersedes the original M2–M4 sequencing
and unbound comparison criteria at 58d37a1; the original commit and review remain
unchanged historical evidence. M2-T/M2-R now qualify independent component paths,
M3 is Tuldok-only cutover and M4 separately gates Rheon enablement and typed
workflow adoption. No new implementation was admitted or started by this edit.

R1–R5 are addressed in the [resolution](reports/standards-review-resolution.md):
fixed 26b comparator and zero added error/call allowance; per-component prerequisite
claims and post-switch smoke; real CUDA failure/custody evidence; explicit
retained-read/new-write/new-execution/public-version policy; partial acceptance
and fully owned procedures/status. The composition probe was updated for the
explicit Projects projections. MCP snapshot 29a39beb-fa6b-469b-9dfa-453e02f37fa0
returned 25 policies and zero unresolved routing fact categories.

Read source/retained artifacts to bind 74 scenes, original 58/amended 8/recent 8 group
metrics, first-frame capture, exact checkpoint/normalization, old native binary,
runtime manifest and train/restore identities. No raw tensors or model weights
were copied into Git, and no numerical/library/runtime change was made. Planned
Q1–Q10 actions remain unexecuted. Current next slice is M1; deployment gates
remain pending. Document/binding verification is retained with the resolution.
