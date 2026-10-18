# Pressure learning across grid shapes: gradient diagnosis and later decay

Status: Active. Canonical operation: start this plan, then continue its slices.
Jeremy authorized the recorded next proposal by asking to continue improving the
model after the completed C64–C66 study. This plan owns the new sequence; the
previous completed plan and its results remain unchanged. Coding-standards MCP
snapshot: `snapshot:v1:ffcfc3f8-b52a-434e-b57f-d259a0355711`.

Objective: measure per-shape loss-gradient relationships and test whether delaying
learning-rate decay improves shared four-pass fitting. C53 remains deployed.
Scientific success is distinct from accepting a correctly executed experiment.
Current acceptance: pending gradient measurements, qualified CUDA training,
fresh CPU restore and numerical checks. Blockers: none.

Binding design: same captured C66 model, normalization, cubic-grid physics,
float32 parameters, float64 physical residual/reconstruction, four shared training
passes, canonical/early feedback, zero teacher/companion loss, C53 initialization,
seeds and 200 balanced exposures per case. No architecture, input channels,
physical thresholds, simulation captures, untouched holdouts or promotion changes.

1. CPU gradients at C53 and C66: use the captured C66 four-pass objective for both
   states, on its same eight X fundamentals. Record raw/clipped norms, losses,
   parameter ordering, dot/cosine matrices, first-case relationships, provenance,
   elapsed time and peak RSS. Check finite nonzero gradients and exactly unchanged
   weight/buffer inventory. No optimizer or perturbation. Dot products accumulate
   in float64 from float32 gradients. Negative cosines describe local Euclidean
   gradient conflict, not proof about Adam dynamics or the cause of training error.
2. C67: same eight cases and 1,600 updates as retained C66. First 150 exposures per
   case at 1e-4, then 25 at 1e-5, 15 at 1e-6, 10 at 1e-7. Change rates after full
   cycles: updates 1201, 1401, 1521. The LR schedule is the only training-policy
   change. Exact model sources, normalized inputs, initialization, RNG order,
   sampling, physical scalars, exposure and optimizer work must match C66.
3. Qualify before admission: reused full-bank gradient/operator proof plus explicit
   matched-control and schedule checks. CUDA via existing visible Tuldok Project,
   catalog and Pumas runtime; journaled one-shot admission, bounded wall time,
   fresh CPU restore and exact published artifact parity. No automatic replay.
4. Restored CPU gate: every fitted case <=1e-6 at four/eight passes, and no case
   worse than C53 at one/two. Also record comparisons to C66, the exact 128 consumed
   real requests and 240 manufactured regressions. A specialist is not promoted.
5. Only if C67 passes that full gate, prepare C68 on the same 24 cases as C64, with
   the same revised per-case LR phases and 4,800 updates. Require the completed
   C67 decision and all evidence hashes before preparation. Qualify/run/restore/
   evaluate through the same owners. If C67 fails, do not start C68, lengthen the
   fit or change objectives. Finish with evidence and a next proposal.

Resource contract: max three independent CPU workers, one thread and distinct
logical CPUs; reserve 12 GiB plus 2 GiB per worker. Two initial gradient workers
run on logical CPUs 21/22; serial per-checkpoint snapshot timings use the existing
CPU owner. Gradient vectors <=16 MiB; fitting tensors <=8 MiB; CUDA trainer wall
cap 1,800 seconds and existing scheduler owns one GPU slot. CUDA float32 training,
CPU-only autograd/inference. Record actual work, time, memory, all failures and
Run lifecycle. Never stop user applications or read private Run output storage.

Composed-design review: not applicable to a new product architecture. This study
reuses existing Project assembly, qualification, managed lifecycle, CPU inference
and operator owners. New owners are experiment settings/conditional admission,
CPU gradient measurement, comparison evidence and its offline visualization.
No service, persistence schema, deployment or physical/model contract changes.

Milestones: gradient diagnosis (Active); C67 qualified comparison (Planned);
conditional C68 (Planned, gated); evidence/report closure (Planned). Replan if
control identity cannot be reproduced, labels/operator disagree, CUDA/restore
custody is unresolved, nonfinite values occur, or resource caps cannot be met.
Stop at that boundary and preserve evidence rather than weakening a gate.

Exact write set: `experiments/pressure-shape-decay-v1/` and matching docs directory;
only model-ID reservation and report-link functions in `docs/plans/four_pass_study.py`
and their owned link block in `cohort-progress.html`. Runtime outputs remain under
`runs/pressure-shape-decay-v1/`, ignored by Git. Necessary execution repairs must
be bounded, recorded and committed separately from training changes. Preserve all
unrelated dirty work. No weights, full gradient vectors, datasets or secrets in Git.

Branch `experiment/pressure-shape-decay-20261017`, based on
`90bf4e12789afed7a8166206cae65140ec580ce3`, is a private experiment proposal for
Jeremy's review, with the preceding experiment branch as its integration base.
Jeremy owns integration; no automatic merge/rebase/push. Retain for review with
no automatic synchronization; retire after integration or explicit disposition.
No worktree is created. Keep method, timings, immutable results and source identities
for review. Ledger: `execution-ledger.md`; issues: `issues.md`; evidence: `reports/`.

Current phase: gradient diagnosis. Exactly one next slice: implement and run the
two CPU gradient measurements, then qualify the matched C67 project.
