# Concurrent execution and pressure feedback

Status: Complete. Operation: close, authorized by the user's “yes. do that. continue
with the sequence”. Owner: root. Branch: experiment/pressure-throughput-feedback-20261017,
base 376fd7444dd64b9684a30f4f1a2077d1d9f9f29b. Tuldok changes use
performance/concurrent-cpu-cuda-20261017, base 8af5a6a1e288e5b2f86a75b53cab93ca71222a04;
the existing PR branch remains unchanged. Retain both branches for review and
subsequent integration; no rewrite, merge, cleanup or model promotion. Qualified application deployment is authorized.

Objective: shorten the complete model experiment cycle by overlapping independent
CPU work and one CUDA fit, then improve low-frequency pressure accuracy at reduced
network work. Preserve cubic cells, water physics, valid history/warm-up, float32
network arithmetic, existing pressure/physical gates and the 32-call maximum.

M1: Add resource-aware concurrent dispatch to Tuldok's existing Runs owner.
Keep individual process custody, exact job authorization, publication, queued
cancellation, restart recovery and unknown-state fences. The scheduler owns all
executor futures through shutdown. One CUDA slot; configurable CPU slots; explicit
per-device memory reservations and a host-available-memory reserve. Reservations
are admission accounting, not an OS memory sandbox. No speculative extra GPU fits.
First prove overlapping independent CPU processes, CPU/CUDA classification,
same-device exclusion, queued and running cancellation, shutdown drainage,
lost database and unknown-child fencing on an isolated PostgreSQL catalog.
Do not alter the live catalog or current deployment before qualification.

M2: Benchmark retained development work through the lab's existing episode owner,
increasing independent one-thread worker counts, with bounded batches, memory
checks and exact numerical comparisons. Choose concurrency using complete elapsed
time, not utilization alone. Record startup/verification/queue/compute scopes and
peak memory. Exclude untouched validation scenes. Reconfigure live execution only
after source and lifecycle qualification; preserve the existing catalog and UI.

M3: Diagnose errors of the retained accurate and efficient checkpoints on identical
consumed inputs at fixed call counts, including spatial bands and amplitude.
Freeze a matched CUDA control and band-feedback challenger using identical data,
initialization, sampling and update budget; change only feedback. Run CPU analysis
and preparation alongside GPU training. Mixed-amplitude training data is a later
separate experiment if this comparison does not resolve leakage.

M4: Fresh CPU checkpoint restore, all 74 development trajectories and independent
pressure checks; compare each population with both archived baselines. Select only
if physical gates, worst pressure and mean/maximum calls do not regress. Evaluate
the eight unused whole-scene holdouts only after freezing an eligible candidate.
Record timing and retain failures and rejected candidates. No automatic promotion.

Write set: this plan directory; experiments/pressure-throughput-feedback-v1/**;
existing lab episode/preparation owners only for measured configurable resource
selection; Tuldok runs.py, a scheduler module, affected lifecycle tests, contracts
and a matching plan/report directory. Trainer objective changes are a separate
commit after scheduling acceptance. No weights, simulation arrays, credentials,
unrelated diagnostics, historical receipts or previous plans are overwritten.

Acceptance: real process concurrency and complete drainage, bounded scheduling,
numerically unchanged CPU scheduling results, measured complete pipeline times,
matched training and per-case scientific records. Measurement completion does not
require a model improvement. Current phase: M1/M2/M3 accepted; M4 scientific
measurement and repeated serial timing complete. Both matched5000-update CUDA fits
and CPU restores passed. Control reproduction is exact. Challenger74/74physical
qualification and233independent checks passed in723.304s, all children reaped.
Population promotion gates reject it: worst pressure regresses versus the prior
accurate model, and one population maximum rises11to12calls versus control.
No candidate promoted and no unused holdouts consumed. Control evidence reuse is
explicitly bound to current-package equivalence; it is not another74executions.
The initial600s control custody timeout remains retained; both fresh fits used
the qualified1800s CUDA bound. Physics, data, seeds and update budgets stayed fixed.
Replan on lifecycle/authority interference, unavailable memory accounting, changed
numerical semantics or evidence that resource admission cannot meet its contract.


## Terminal decision

M1–M4 measurement sequence complete. Scheduling accepted and deployed; model
challenger rejected for promotion by unchanged population gates. Serial CPU timing
completed four drained batches with exact repeated physics/calls, all sixteen
records independently reviewed. No unused holdouts consumed. Final observed state
published to the same Tuldok/Rheon paths and exact owned observer stopped; original
applications/catalog remain available. Isolated fixture PostgreSQL is stopped.
No further training, promotion, push, merge or runtime-prototype deployment.
See results.md, reports/qualification-decision.json and reports/serial-timing.json.
Subsequent smaller-feedback ablation is a proposed new slice, not active work.
