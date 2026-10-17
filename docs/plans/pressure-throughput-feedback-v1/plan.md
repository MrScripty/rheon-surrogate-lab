# Concurrent execution and pressure feedback

Status: Active. Operation: start, authorized by the user's “yes. do that. continue
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
require a model improvement. Current phase: M1 accepted; M2 repeated throughput measurement complete.
Exactly one next slice: execute the frozen matched CUDA fits for M3 and qualify
their fresh CPU artifacts. The first full control attempt hit its600s custody
deadline after reporting4500updates; it is retained as infrastructure failure.
A separate tested/reviewed/deployed CUDA-only1800s bound now supports a fresh
from-initialization attempt; data/seeds/update budgets and all physics gates
remain fixed. No prior mutation is replayed. The 16-worker setting
is measured on shortened development episodes; full-regression throughput remains
to be measured. Spectral feedback source has five passing unit checks, but no
full scientific CUDA fit has started.
Replan on lifecycle/authority interference, unavailable memory accounting, changed
numerical semantics or evidence that resource admission cannot meet its contract.
