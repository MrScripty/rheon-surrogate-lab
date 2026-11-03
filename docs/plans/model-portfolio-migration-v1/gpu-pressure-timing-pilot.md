# Complete GPU pressure-call timing pilot

Owner: the model portfolio migration task. Canonical plan:
[plan.md](plan.md), operation `continue`. This is a pressure-lane pilot within
P4; full-state generation and the remaining portfolio stay in scope separately.

Measure C106 and C107 on all 128 frozen old-Rheon pressure requests. These
models already have exact retained GPU accuracy results. Use their unchanged
captured model, normalization, weights and pressure policy. Loading is isolated
in one process per checkpoint because the existing captured module owner has
one global configuration. Execute four blocks in C106/C107/C107/C106 order.

The primary interval begins with independently owned native CPU NumPy arrays
for base pressure and input residual, plus the original cell spacing, fluid
density and timestep scalars. It includes input validation, contiguous views or
packing, host tensor/scalar construction, H2D, the captured GPU pressure graph,
reconstruction, allocation, D2H, the returned CPU NumPy view and synchronization.
The returned view owns its storage through an exclusive new CPU tensor and
must not alias the input arrays. Loading recorded files and model setup are
outside that interval.

Report milliseconds per **pressure request**, never per simulation step or
divided by K. These results cannot replace the full-state 0.22 ms anchor or
the complete K-output timing lane. Existing scientific metrics and old/new
Rheon reference epochs remain distinct.

Device hooks qualify the first call and five warmups for each case. Record the
instrumented first-call duration separately; earlier cases may already have
warmed the same geometry, so this is not a process-cold performance claim.
Remove hooks for 20 measured calls on the same deterministic inputs and CUDA
parameters. These modest pilot counts retain within-case variation and two
balanced blocks per model without committing to a production tail estimate.
Report all raw samples, empirical case medians and extrema; claim no percentile
guarantee or universal speed threshold from this pilot.

Every qualification, warmup and timed output must exactly match the retained
accuracy output hash. Check original input hashes, input immutability, output
ownership, captured architecture and parameter inventory. No reference field
or RHS label enters inference. No CPU neural forward, parameter update,
reference generation or numerical fallback is permitted.

Pin the host to logical CPU4, matching the existing pressure accuracy setup,
with one Torch/intra-op thread and one inter-op thread. Record GPU, driver,
runtime, precision, TF32, deterministic settings, host affinity, clocks,
temperature, power, load and other compute processes. Stop on observed unrelated
GPU compute; preserve the interrupted result. Resource observations are sampled,
not proof of uninterrupted exclusive hardware ownership. Ordinary desktop
graphics remain part of the representative environment. Run after the current
native card browser is terminal; no training or reference capture overlaps.

Write set: `pressure_gpu_timing.py`, one task-owned serial runner/protocol,
compact timing results, and the canonical plan/ledger/state. Reuse the existing
checkpoint loader and independently frozen packet reader. Do not change the
captured model or live Tuldok source. Composed-design review is not applicable:
this adds one offline measurement adapter and a serial lifecycle owner without
changing the product architecture or persisted project contracts.

Acceptance: all four original blocks terminate successfully, every output/input
binding and CUDA-only qualification passes, raw timing distributions and resource
observations are retained, and the model comparison uses identical cases and
call boundaries. Inspect paired variability before expanding to all available
pressure checkpoints. An adapter or hash mismatch remains a visible failure;
it does not justify a tolerance, a different model, or deleting its evidence.
