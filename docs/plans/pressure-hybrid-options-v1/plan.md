# Next pressure-solver comparisons

Status: proposed on 2026-10-18. This document adds no training, solver measurements,
or model promotion. The nine previously requested fits are complete; their
[outcome](../flow-reasoning-assessment/outcome.md) retains C53.

Start with a bounded **numerical correction plus frozen model** comparison using
C53, C44 and C37. It changes the information supplied to inference without
confounding that change with another fit. If numerical correction reduces the
required neural work at a lower total cost, use that evidence to design a trained
hybrid. Separately test whether finer teacher timesteps improve the numerical
reference before creating another training dataset.

## Evidence behind the ordering

The nine-fit detached-correction screen did not meet the four-pass capacity gate
from any starting checkpoint. It supports changing a mechanism before extending
those runs; it does not establish an architectural limit.

C53's [training-only frequency diagnosis](../pressure-four-pass-distillation-v1/C53-diagnosis.json)
attributes about 99.8% of remaining **pressure-error energy** after four passes to
spatial modes with each axis index at most three. This is an aggregate over 32
captured training requests, not a claim that 99.8% of the equation residual or
every future request lies in that band. Low-mode numerical correction is a
testable response to that observation.

A learned coarse branch was already trained to completion and
[failed pressure qualification](../../../runs/dependency-migration-v1/lab/docs/plans/pressure-coarse-full-fit-v1/results.md).
It learned the coarse mapping; it did not perform a numerical coarse solve.
The proposed experiment tests that distinction. The supporting literature,
operator algebra and limitations are in [research.md](research.md).

## Options and decision order

| Order | Change | Question answered before larger investment |
| --- | --- | --- |
| 1 | One numerical low-mode correction with frozen C53/C44/C37 | Can numerical information replace enough neural work to pay for itself? |
| 2 | Native teacher at `dt`, `dt/2`, `dt/4` | Does smaller timestep improve the reference trajectory at matched physical times? |
| 3, conditional | Consistent coarse numerical solve plus learned fine correction | Does training on the remaining fine residual improve the promising hybrid? |
| Later | Sparse numerical anchors plus learned infill | Can anchors be obtained cheaply without solving the full coupled problem? |
| Later | Larger grids; then mixed spacing and an explicit spacing feature | Is the error sensitive to domain extent, spatial refinement, or conditioning? |
| Later | Revised fully learned coarse/fine architecture | What specific mechanism differs from the learned hierarchy already tested? |

These are distinct comparisons. Changing grid counts, spacing, timestep and
architecture together would obscure which change moved the frontier.

### Accepted dataset expansion

The user subsequently requested adding 32-cell domains and larger domains now,
while deferring full-size training to keep iteration fast. Prepare **32³ and
64³** as the first new training tiers, retaining the existing 5–16-cell grids and
both 8 and 16 cells per metre. This work can proceed independently of the hybrid
screen; it is no longer deferred until that screen succeeds. Deployment targets
are 256³, 512³ and 1024³; training on smaller domains does not qualify those sizes.

The captured scene generator explicitly requires `counts_range=[5,16]`; its
reproducible reservation, native producer, capture verifier and packet consumer
must receive a versioned extension. Do not mutate historical reservations or
claim that changing a configuration alone added data. The current 65,536-cell
policy permits 32³ by cell count but excludes 64³ (262,144 cells). Qualify the
larger tier's resource and shape limits as part of this extension, preserving
the physical gates. Existing 30 MiB packet budgets require deliberate sharding
for a mixed larger-grid population rather than unbounded array loading.

Start with one training-only native capture at each new cube size and each
existing spacing: four pilot trajectories. Keep `dt=1/60`, 64 native steps,
snapshots at indices 15 and 63, water properties and pressure qualification
unchanged. These pilots establish numerical validity and actual time/memory/
storage cost before choosing the full trajectory counts. They are not validation
or test scenes. Record their source, geometry, forcing, split and array hashes.
Follow with separate whole-scene validation reservations, and odd/rectangular
neighbours within each supported tier. Do not silently reduce physical gates,
truncate trajectories or replace expensive examples with synthetic labels.

The controlled training comparison is the existing small-grid bank versus that
same bank plus the new tiers, with the starting checkpoint and objective fixed.
Keep dataset-size effects separate from optimizer-work effects: record updates,
case presentations, forward cells, time and peak memory. Balance sampling by
spatial tier so that larger arrays do not silently dominate the objective; report
the actual schedule and per-tier results. Select the schedule from training-only
cost/qualification pilots, before examining new validation outcomes. A fit count
and schedule remain to be specified after those measurements; the earlier nine
fits are not authorization to rerun them unchanged.

Status: expansion requirements recorded; no new native captures or larger-grid
training have yet executed.

## First comparison: no training

Freeze source, normalization, data and weights from the original C53, C44 and C37
[initialization identities](../flow-reasoning-assessment/initializations.json).
That file is a historical pre-execution manifest; the completed fits are recorded
in the linked outcome, and are not the frozen controls for this new comparison.
Use identical incoming pressure, RHS, physical parameters, gauge and fine-grid
operator across policies. No optimizer updates or remote training are needed.

For a residual `r=b-Ap`, form orthonormal cell-centred Neumann cosine columns `P`,
with axis indices `0..min(3,n-1)`, excluding the all-zero mode. Apply one correction:

`p <- p + P (P^T A P)^(-1) P^T r`.

Recompute the fine-grid residual afterward. This is a small Galerkin subspace
solve on the existing operator. It is an initial diagnostic for the numerical
idea, not a literal lower-resolution simulation or evidence for a trained infill
network. Use separable transforms rather than materializing a dense fine-grid
basis. Keep rank fixed; do not choose it using the consumed regression requests.

Predeclare these policies for **each** checkpoint:

- Neural corrections only: `K=1,2,4,8`.
- One numerical correction, then `K=1,2,4` neural corrections.
- `K=1,2,4` neural corrections, then one numerical correction.

This gives 30 policies. Add numerical low-mode correction alone and a full
numerical spectral solve: **32 policies** in the initial screen. Both hybrid
orderings address the concrete possibility that a neural correction reintroduces
low-mode error. Record every intermediate residual; do not conceal amplification
behind the final correction. Do not add adaptive ordering or rank selection.

The full numerical control matters: the current sealed, constant-coefficient box
admits a spectral solve, and the repo already has an
[oracle control](../../../experiments/pressure-pass-sweep-v1/oracle_control.py).
A hybrid win over an iterative baseline alone would leave this alternative open.
Keep independent accuracy checking outside the numerical solver implementation.

### Numerical qualification and populations

Before comparison, check eigenpair action against the actual fine operator,
nullspace compatibility, boundary conditions and gauge. Cover zero RHS, odd and
thin grids, and mixed low/high modes. A one-cell axis contributes only its constant
factor; test that mathematical edge case without extending the model's supported
deployment shapes. Reject incompatible RHS under existing rules rather than
silently changing the equation. Pure modes inside the solved subspace are
implementation checks, since success there follows by construction.

Use newly specified mixed-mode cases on existing training shapes for development;
freeze their generation rule before inspection. Report the existing 128 consumed
real requests and 240 manufactured cases as separate regression populations. These
are already inspected data, not fresh holdouts. Keep their case identities,
normalization, tolerances and current early/eight-pass retention checks fixed.
Do not use unused scenes to select numerical rank, order or call budget.

### Accuracy, cost and the existing chart

For every request report fine-grid relative equation residual, same-RHS mean-free
pressure error, pressure-gradient error, and intermediate amplification. State the
normalization for zero or near-zero reference fields. Report median and worst
errors; the latter matches the user's frontier concern.

Measure total policy CPU time including restriction/projection, numerical solve,
reconstruction, residual assembly, neural calls, conversion and common acceptance
checks. Keep cold setup and cached geometry reuse separate. Record actual neural
calls, operator applications and numerical work. Use a single CPU thread,
isolated timing, warmups and balanced repeated runs; keep oracle diagnostics
outside timed inference for every policy. A Python screen can establish a signal;
native integration and whole-simulation timing are needed for a deployment claim.

The existing [HTML chart](../cohort-progress.html) contains 70 matched checkpoints
and eight raw frontier points in the [recorded snapshot](frontier-reference.json).
Its timing is the original pressure proposal, not automatically the full policy
above. Remeasure those frontier controls under the new timing scope before saying
the frontier moved. C37 and C44 are already included; the other six original
frontier proposals bring the comparison to at most 38 policies. Bind each by
weights SHA and original policy, since chart cohort numbers can change as records
are added. C53's retained qualification status is separate from raw Pareto status.

Advance only a fixed policy that provides a reproducible accuracy/cost improvement
on the common population. A deployment candidate must additionally pass all 128
requests at the existing `1e-6` target and all existing physical, early-pass,
eight-pass and manufactured retention gates before independent scene evaluation.
Do not relax a gate to make a plotted improvement qualify. If the full numerical
solver dominates, report that result. If the low-mode intervention has no useful
cost/error tradeoff, stop this branch before training it.

## Second comparison: finer teacher time sampling

The user's `0.02 s` example means two `0.01 s` teacher steps per model step. Keep
the model interval fixed. Existing captured studies use `dt=1/60 s`; a pilot at
that interval could compare 120, 240 and 480 native steps over the same two
physical seconds. First inspect or expose the native generator's actual timestep
control and record its emitted times; changing metadata alone does not implement
substepping. Hold grid, initial data, pressure tolerance, precision and continuous
forcing fixed, applying forcing at the correct substep times.

Compare coarse/fine and fine/finer trajectories at matching times, including
velocity and tracer errors. Smaller differences suggest temporal convergence;
they do not prove exact-solution accuracy. Extend the pilot only if the comparison
shows a useful reference improvement.

Two fine steps also change advection and forcing. Their final pressure field is
therefore not automatically the correct label for the original coarse pressure
request. For pressure-only training, generate labels by solving the **learner's
actual pressure equation** from the selected input state. Fine trajectories can
supply better state samples. Learning the entire fine-step endpoint would change
the model target and needs a separate specification. Preserve the original
coarse-step equation and physical gates in a pressure-only comparison.

## Drift needs its own experiment

There is no measured universal number of safe AI steps. After request-level gates
pass, advance matched native and AI trajectories, recording physical time,
velocity, tracer, gauge-consistent pressure error and divergence at each step.
For any declared application tolerance report the first crossing in **seconds and
steps**, alongside the full curve. Thresholds are application decisions; the 1%
diagnostics in the existing
[one-minute precision study](../pressure-precision-drift-v1/one-minute/plan.json)
are not general accuracy limits.

That study compares numerical storage precision, not temporal refinement or AI
drift. A later accurate pressure projection also does not undo earlier velocity
and transport errors. Periodic native correction must be evaluated as a complete
policy with its accumulated error and cost, rather than assumed to reset drift.

## Spatial sampling and later architecture work

Use the [shared terminology](../../../CONTEXT.md): grid dimensions count cells;
grid resolution is cells per metre; physical extent is measured in metres.
For example, separate these experiments:

| Cubic grid | Cells per metre | Side length | Change relative to first row |
| --- | ---: | ---: | --- |
| `16^3` | 16 | 1 m | Reference |
| `32^3` | 16 | 2 m | Larger physical domain |
| `32^3` | 32 | 1 m | Finer sampling of the same extent |

Current pressure preprocessing already uses spacing: `A=(h/rho)L` and
`q=(rho/h)r`, so the network approximates `L delta_p=q` under the present uniform
sealed-box contract. Raw spacing is not a missing scalar in that exact inverse.
Before adding a feature, test scale consistency at fixed counts by rescaling RHS
to keep `q` fixed. A new conditioning feature needs an identified benefit beyond
that scalar cancellation, or a broader physical/operator contract.

`128^3`, `256^3` and `512^3` contain 2,097,152, 16,777,216 and 134,217,728 cells,
respectively. Each exceeds the current 65,536-cell cap. Treat these as resource
and supported-contract changes after the small-grid hypothesis is tested; larger
examples alone do not establish a cure for the present small odd-grid errors.

For a later literal coarse-grid hybrid, define conservative transfers, coarse
geometry/operator, nullspace and fine residual checks before fitting the fine
correction. Odd edge aggregates require correct volumes. For sparse anchors,
specify how the values are obtained cheaply: exact anchors from a full solve can
test information sufficiency, but cannot support a speedup claim. Local solves
need interface conditions because pressure is globally coupled. Keep those
studies separate from the first frozen-model comparison.
