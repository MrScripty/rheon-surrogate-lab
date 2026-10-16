# Coding standards review of the migration plan

Reviewed subject: `docs/plans/rheon-tuldok-migration-v1/plan.md` at
`58d37a1bdfef2805dbff2cdabf12bb8607eec40f`.

Result: **revisions needed before migration execution**. The isolation,
preservation and ownership direction is sound, but the acceptance and cutover
contracts are not yet sufficiently precise. This report records recommendations;
it does not revise the plan or admit implementation.

Method: routed actual planning, verification, IPC, persistence, architecture,
concurrency and performance facts through the coding standards MCP. Snapshot:
`snapshot:v1:e1406d64-d247-449d-9410-3875bd1a6e67`. Routing selected 25 policies,
with zero unresolved fact categories. Commit guidance was additionally read
from the same snapshot to retain this report. Routing completeness is not a
conformance certificate: the findings below are the reviewer's analysis of the
returned standards and the committed plan. [Machine review context](standards-review.json).

## R1 — High: numerical comparison lacks a decisive acceptance rule

Location: plan lines 113 and 117–123, claims A3 and A7.

A3 permits a non-bit-identical difference when it is explained. A7 requires
running 74 scenes and reporting metrics, but does not say which deviations fail
the migration. The later checkpoint/policy is left to selection from a retained
record rather than identifying that record and binding the selected material
before comparison. Two implementers could collect the same regression and reach
different acceptance decisions, including accepting a worse pressure solution.

Applicable requirements: `workflow.planning` / Acceptance Claims;
`workflow.verification` / Quality Gates And Execution Location;
`workflow.verification.oracles` / Property And Differential Evidence;
`topic.performance` / Measurement Contract.

Correction: freeze the baseline checkpoint, normalization, source/project,
stopping policy, existing numerical thresholds, scene membership and old/new
build/runtime selections before measurements. Name the existing record supplying
each identity. State which quantities require exact equality and predeclare the
per-metric acceptance bounds for quantities allowed to vary. Keep numerical
quality gates separate from performance measurements; record timing variability
without silently choosing a speed regression budget after seeing results.
Failure must preserve the comparison and trigger the existing replan path.

## R2 — High: cutover prerequisites are not explicitly bound to each component

Location: plan lines 90–99, 147–148 and 166–175.

M3 includes the original-catalog service cutover, while M4 contains live project
visibility, desktop workflow and 74-scene validation. The general sentence that
cutover is gated on qualification does not identify which claims must already
pass. It is unclear whether M3 switches only Tuldok or also enables the migrated
Rheon/model path. This can expose a new runtime before a relevant regression or
workflow failure is discovered, making rollback harder once new writes exist.

Applicable requirements: `workflow.planning` / Milestones And Slices;
`workflow.verification` / Acceptance Paths And Boundaries;
`profile.boundary.persistence` / Migration Execution Contract.

Correction: state separate Tuldok-service and Rheon/model enablement boundaries.
Bind each to explicit claim IDs and successful rehearsal evidence. Require the
retained-state, CUDA/restore and existing live-project path to work on a rehearsal
installation before Tuldok cutover; require numerical/lifecycle regression before
enabling the new Rheon/model path. The 74-scene numerical gate need not block an
independent Tuldok-only upgrade. Preserve a bounded post-cutover smoke and a
defined failure disposition before ordinary new writes resume.

## R3 — Medium: CUDA execution failure paths have no explicit gate

Location: plan lines 90 and 112–114, M1 and claims A2/A4.

A2 establishes one successful optimizer update and CPU restore. A4 covers
Rheon's CPU model process. Neither explicitly requires Tuldok's managed CUDA
integration to survive cancellation during inventory verification or training,
deadline/startup failure, child drainage and cold reopening of interrupted work.
The port changes runtime dispatch and custody boundaries, so CPU model-process
evidence cannot establish those CUDA Run outcomes.

Applicable requirements: Core / Lifecycle And Concurrency;
`workflow.verification` / Async Completion And Failure Evidence;
`topic.concurrency` / Verification.

Correction: add a Tuldok-owned runtime lifecycle claim selecting existing Runs
and runtime tests plus bounded real CUDA cases where necessary. Observe terminal
state, cessation/drainage, retained artifacts and ambiguous-outcome handling;
assert no silent CPU fallback or automatic resubmission. Reuse existing custody
owners. The local `tests/test_torch_runtime.py` covers inventory/device/material
checks but is not a substitute for the missing real lifecycle qualification.

## R4 — Medium: persisted recipe/project compatibility needs an explicit policy

Location: plan lines 43–54 and 90, decisions 2/3 and M1.

The plan correctly identifies the incompatible local managed recipes and live
manifests, but "port" and "preserve" do not select the supported representations
or their evolution policy. In particular, old material inspection, new execution
and independently negotiated public operations have different obligations. A
source port could admit new recipes while leaving the 120 saved Project snapshots
or completed Run material unreadable, or could silently widen a closed public
contract while retaining its old advertised identity.

Applicable requirements: `topic.contracts.evolution` / Contract Classes and
Persisted Evolution; `topic.contracts.schemas` / Version Scope And Invalidation.

Correction: identify the actual supported source project/recipe/material forms,
their owners and consumers. Distinguish retained-read support from eligibility
for new execution; select an explicit supported representation/version for each
new producer path. Bind producer, reader and negotiated-schema changes together
where coordinated, while retaining the historical bytes that must remain
inspectable. Add corresponding cold-read and unsupported-version fixtures.
This does not require supporting hypothetical historical formats or every
pairwise version combination.

## R5 — Low: acceptance metadata contradicts itself

Location: plan line 4 and 105–117.

The plan-level acceptance status is `pending` while required claim A0 is
`Satisfied`. Under Planning's table this is `partial`. A1–A7 also need the
required environment classification and an identifiable executable procedure
and evidence owner, beyond naming hardware and generic "automated" execution.
Record current blockers explicitly, distinguishing future unsatisfied gates from
a condition actually blocking the next slice.

Applicable requirements: `workflow.planning` / Required Active-Plan Fields and
Acceptance Claims; `workflow.verification` / Reporting.

Correction: use `partial` while A0 is in the required claim set, or move A0 to
assessment evidence and keep migration-only acceptance pending. Complete the
claim metadata and direct evidence links; do not require additional user approval
for already authorized routine migration work.

## Direction retained by the review

- Historical source/model/data identities and the original catalog are preserved.
- Dirty primary Rheon work is isolated without stash/reset or history rewrite.
- Runtime/portability migration remains separate from model improvement.
- Float32 CUDA training, CPU inference, cubic cells and real warm-up are explicit.
- Teacher diagnostics, autonomous trajectories and accepted native fallback are
  distinguished; a native fallback cannot become a learned-model success.
- Existing numerical, process, custody and storage owners are reused.
- Backup/restore, stopped writers and unknown upgrade outcomes are addressed.
- Resource limits and timing/memory recording reflect the prior OOM experience.

No new training, simulations, catalog writes, app restarts, source port or plan
edits were performed in this review. The earlier nine negotiation checks remain
evidence of their named contract scope, not proof of migration readiness. The
next action is to revise the plan using this complete finding set before starting
its implementation sequence.
