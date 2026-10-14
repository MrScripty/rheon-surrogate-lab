# Global pressure correction: next Tuldok experiment

**Plan status:** `Accepted`

**Current phase:** Reviewed design delivered in draft lab PR2

**Next slice:** none

**Acceptance status:** `satisfied`

**Execution ledger:** [execution-ledger.md](execution-ledger.md)

**Issues:** [issues.md](issues.md)

This lifecycle covers the requested **design deliverable**. The proposed numerical
implementation is not approved or started by this document. D1–D6 close this task;
E1–E9 are prospective experiments with no measured result.

## Objective

Can a modest trained multiscale pressure correction reduce **total matched-quality
rollout cost**, relative to a previous-pressure warm start and an analytic cosine
solver, on genuinely new trajectories, sealed rectangular geometries and forcing
regimes? A residual-controlled native solve remains the acceptance authority.
This is a testable proposal; neither useful warm starts nor acceleration is assumed.

The meaningful change from existing work is learning across physical box aspect
ratios and forcing histories, with autonomous rollouts and a competitive temporal
warm-start baseline. The historical modal model fits one operator. The analytic
cosine inverse already works for each proposed box, so it is a mandatory baseline
and a plausible winner. A failure to beat it is a useful negative result; do not
expand to obstacles or suppress that result to rescue the narrative.

Deliverable audience: a reproducible evidence package for Atomic Machines,
a documented Tuldok workflow, a candidate case study for Training Your Own Models,
and a reviewable BC AI collaboration task. Those uses depend on evidence gates;
this design establishes none of their scientific or commercial claims.

Current write set: **only `docs/plans/global-pressure-correction-v1/`**. Preserve
all earlier source, checkpoints, freezes, reservations, data, receipts and negative
results. Do not edit Tuldok, the prior reproduction directory or existing experiments.
No expensive training, release tags, merge, publication outside the authorized lab,
or public outreach. The requested lab branch/commit/push/draft PR is in scope.

## Objective Acceptance

Exact design criteria are owned by [acceptance.md](acceptance.md).

| ID | Criterion | Kind | Environment | Mode | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | Inspected source and historical negative evidence | release-artifact | required-real GitHub/clone | either | satisfied | [Inspection](inspection.md) |
| D2 | Sourced bounded model decision | focused | not-applicable | manual | satisfied | [Research](research.md) |
| D3 | Split/restore/physics/rollout/cost protocol | contract | not-applicable | manual | satisfied | [Data/model](data-and-model.md), [criteria](acceptance.md) |
| D4 | Existing generic paths and separate full GPU scope | contract | required-real merged source | manual | satisfied | [Handoff](execution-and-handoff.md), [GPU](local-gpu.md) |
| D5 | Independent numerical/standards design review | focused | representative independent reviewer | manual | satisfied | [Independent review](reports/design-review.md) |
| D6 | Scoped committed/pushed draft PR, no training | release-artifact | required-real Git/GitHub | automated | satisfied | [Publication verification](reports/verification.md) |

## Evidence and model decision

[inspection.md](inspection.md) separates the radius-one CNN failure, earlier
335-gain global candidate and reproduced 32³ modal baseline. The CNN performed
200 Adam updates and exact restore, but qualified 0/16 native cases. It remains a
training/restore diagnostic, never a qualified accelerated solver. Its old test
families/seeds and the consumed 9000–9015 reservation cannot be new holdouts.

Three practical choices are compared in [research.md](research.md):

| Choice | Practical role | Decision |
| --- | --- | --- |
| Conditioned global cosine/modal gains | Tiny CPU control; strong exact box prior; analytic counterpart is cheaper to establish | Preserve existing result; no repeat as the next flagship model |
| Small multiscale 3D encoder/decoder with global bottleneck | Standard CPU Torch operations; context spans the box; geometry conditioning; simple native residual loss | **One preregistered candidate** |
| Small Fourier neural operator | Global mixing, but periodic spectral assumptions, complex weights and boundary treatment add qualification work | Deferred; no sweep in this experiment |

The selected model is an experiment-specific U-Net-style correction, not a claim
that segmentation architecture solves Poisson systems. [data-and-model.md](data-and-model.md)
fixes the geometry, data, grouping, architecture and loss. This target keeps the
existing sealed operator and physical stages, avoiding unverified aircraft,
submarine, moving-boundary, free-surface or variable-coefficient physics.

## Binding Decisions

The lab owns generator/verifier, units, split grouping, operator identity, model,
training, native correction, rollout metrics and timing. Tuldok owns immutable
project/source/runtime/input/output custody, grants, Run lifecycle and display.
Use project schema 1, `external_recipe_v1`, `captured_cpu_python_v1`,
`tuldok_progress_v1` and `tuldok_result_summary_v1`. No generic physics registry,
new executor, new runtime or GPU-support claim belongs in this experiment.

Candidate base: lab `20dd90f6c28a7e3be98e95bc7d99e64e48a7da9c`, Tuldok
`f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6`, native trajectory reference Rheon
`2209012bc5f902693d0f695ec1c2d854126af317`. The old root pressure experiment
continues to pin `34235d33c886aa05b5bdf95d920b863d1fb2ad6f`.

The existing sealed warm-start patch applies to the older Rheon pin. **Its
availability at the trajectory pin is unverified.** First implementation must
qualify an isolated lab-owned native dependency/patch against the trajectory pin,
with the old zero-start semantics preserved. No upstream write is authorized by
this plan. If the necessary API cannot be admitted, stop model implementation and
record a blocker; do not substitute teacher pressure or a different hidden solver.

## Simplicity And Ownership Review

**Applicability:** `applicable`

Applicable: separately owned numerical and generic execution contracts are composed.

- Independent concepts and dimensions: Generator and verifier are lab-owned; training consumes immutable shards;
   evaluation restores weights then calls native operators; Tuldok supervises bytes.
- State, identity, value, time, policy, and mechanism: A trajectory ID identifies a complete physical history. Geometry IDs identify
   extents/counts/boundaries; seeds identify forcing. Physical dt differs from wall
   time. A persisted partition owns eligibility; hashes identify bytes, not quality.
- Caller and composition-root knowledge: Callers select a preset and explicit input hashes. They need no hidden sibling
   checkout, private solver state, mutable global model or implicit catalog context.
- Representative change paths and forced owners: Model changes touch only the new lab experiment and its qualification; operator
   changes invalidate all data/gates. Custody/runtime changes belong to Tuldok.
- Stable Interfaces versus hidden knowledge: Stable interfaces are explicit MAC arrays, native full operator/correction,
   lab manifests and existing recipe tokens. Input schemas stay lab-specific.
- Independent evolution, testing, failure, and replacement: Training can fail without changing the immutable data. Bad predictions take
   a bounded native fallback. Failed fallback aborts the step; no invalid state advances.
- Deletion and cumulative machinery result: Removing the model leaves reference data and baselines. Removing the wrapper
   would push split/provenance checks into every caller; it has a concrete purpose.
- Necessary complexity and containment: Necessary complexity is grouping, restored-state validation and common physics
   gates, confined to the lab. No scheduler, service, runtime installer or registry.

## Milestones

| Slice | Outcome | Prospective implementation write set | Exit gate | Lifecycle |
| --- | --- | --- | --- | --- |
| 0 (this PR) | Independently reviewed design and handoff | New design directory only | D1–D6; no experimental claims | Accepted |
| 1 | Admitted native warm start, deterministic generator and data verifier | New lab `experiments/global-pressure-v1/`, `probes/global-pressure-v1/`, dedicated tests and dependency manifests; exact scope fixed in implementation PR | E1–E2, adversarial fixtures, no training | Deferred; separately admitted implementation |
| 2 | One CPU 200-update smoke via generic recipes and fresh restore | Same new experiment, new test files, ignored `runs/global-pressure-v1/` | E3–E6; native quality and failures retained | Deferred; separately admitted implementation |
| 3 | Freeze checkpoint/protocol, generate new test suites and evaluate | New experiment freeze/evidence only; no numerical changes after access | E7–E9, independent pre-freeze review | Deferred; separately admitted implementation |
| 4 (conditional) | Separately reviewed local-GPU full run | Separate full preset/protocol/checkpoint provenance | All full gates; no inference from smoke | Deferred; separately admitted implementation |

[execution-and-handoff.md](execution-and-handoff.md) gives the Tuldok CLI/UI paths,
resource limits, implementation order and collaboration packet.
[local-gpu.md](local-gpu.md) describes full training separately. GPU execution is
outside the qualified captured CPU runtime; do not route it through that profile.

## Blockers

None for the design deliverable: D1–D6 are satisfied. Independent review passed
and [draft PR2](https://github.com/MrScripty/rheon-surrogate-lab/pull/2) is open.
Numerical implementation must resolve the native
warm-start API before training. Managed catalog/runtime/grant and local GPU hardware
are future environment requirements, not claimed present.

## Re-Plan Triggers

Replan triggers: unsupported native API, reference physics failure, data/input cap
breach, CPU time/memory budget breach, leakage, or inability to retain exact trained
weights. Fix correctness before freezing. After any final-test access, log findings
and finish the unchanged protocol; fixes require a new experiment ID and fresh
holdout reservation. No tolerance change, extra training or favorable-only reporting.

## Repository Isolation

Branch `design/global-pressure-correction-v1` isolates the requested draft design
from main. Task agent owns writes, commit and push; parent owns acceptance/integration.
Base is verified lab PR1 merge20dd90f6. Public visibility is the authorized lab only.
Keep branch while the draft proposal is open; parent retires it after integration
or records rejection. No worktree was created, no unrelated checkout is edited,
and no branch deletion/history rewrite is part of this task.

## Final Acceptance

- Acceptance status: `satisfied`
- Final status: `Accepted`
- Deferred follow-ups: separately authorized numerical implementation and local GPU full training
