# Pressure accuracy versus correction work — results

Completed. No optimizer updates, checkpoint changes, inference deployment or
fresh-final consumption. Retain the existing learned control. The diagnosis
supports training late smooth-error directions at comparable work; a stricter
stopping policy is an accuracy/work tradeoff, not a newly improved checkpoint.

[Method and environment](reports/methods.md) · [Plan](plan.md) ·
[Execution ledger](execution-ledger.md) · [Proposed next training comparison](next-training-protocol.md).

## Same-input diagnosis

The qualified first post-warm-up input reproduces the pressure-error regression
without any earlier AI trajectory drift. Retained: ten calls / 1.6497162e-7
pressure relative L2 against the independent exact solve. Targeted: eight calls
/ 1.1411137e-5. Targeted strict-8 uses 12 calls / 1.2096470e-7, passing the same-input
no-pressure-regression repro. The original repro exits 1 (expected diagnostic
red); strict-8 exits 0. [Exact commands and outputs](reports/fixture-repro.json).

Both checkpoints are replayed on 18 captured states, 15 unique arrays, at every
call 0..32. The first state in each scene is duplicated between the two capture
trajectories because both have the same 15 native warm-up steps. The oracle's
worst independently enumerated-neighbor defect is 2.2216384e-13. Negative
no-correction fields are detected as substantially wrong. Results are sampled
consumed-case evidence, not a universal grid/solver guarantee.

The targeted checkpoint's original-policy remaining mean-free pressure error
has at least 99.669% of its energy in the first three cosine modes per axis;
median 99.979%. The dominant mode is usually the long X-axis fundamental.
Mean-free and gradient errors also regress, so a pressure gauge offset alone
cannot explain the result. Additional learned corrections reduce same-input
error; fixed 32-call pressure error is below 1e-15 on these inputs through the
**existing float64 physical accumulation**, with float32 network evaluations.
This does not promise full-pressure float32 storage to that accuracy, change
precision or restore a 1e-8 acceptance target.

| Same-input model/policy | Mean calls | Worst pressure relative L2 | Worst gradient relative L2 |
|---|---:|---:|---:|
| retained / original | 9.000 | 1.99265e-06 | 1.01536e-06 |
| retained / strict-7 | 10.889 | 2.27148e-07 | 1.35724e-07 |
| retained / strict-8 | 12.889 | 1.12973e-08 | 8.21256e-09 |
| targeted / original | 7.444 | 1.7339e-05 | 5.44322e-06 |
| targeted / strict-7 | 9.444 | 1.7855e-06 | 5.60447e-07 |
| targeted / strict-8 | 11.444 | 1.67979e-07 | 5.77172e-08 |

[Replay identities, modes and selected stages](reports/replays.json) ·
[Every call frontier](reports/frontiers.csv).

## Complete native paired trajectories

Three consumed scenes × two checkpoints × three stopping policies = 18 completed
simulations, 4,146 qualified AI steps. Every native reference/AI physical check
passes, with no cap hit, native cleanup or fallback counted as success. Maximum
observed calls 14, below the unchanged 32-call cap. The three targeted original
trajectories reproduce every historical pressure-difference metric exactly.

| Model/policy | Qualified simulations | AI steps | Total calls | Mean calls | Max calls | Worst paired pressure relative L2 |
|---|---:|---:|---:|---:|---:|---:|
| retained / original | 3/3 | 691 | 6108 | 8.839 | 10 | 6.13957e-06 |
| retained / strict-7 | 3/3 | 691 | 7454 | 10.787 | 12 | 1.70966e-06 |
| retained / strict-8 | 3/3 | 691 | 8825 | 12.771 | 14 | 1.16815e-06 |
| targeted / original | 3/3 | 691 | 5110 | 7.395 | 9 | 2.58927e-05 |
| targeted / strict-7 | 3/3 | 691 | 6278 | 9.085 | 11 | 3.1329e-06 |
| targeted / strict-8 | 3/3 | 691 | 7454 | 10.787 | 13 | 1.11323e-06 |

Targeted strict-8 reduces its worst paired pressure difference by
23.26× relative to targeted original, using
45.9% more mean calls. Against retained original it uses
22.0% more mean calls and increases maximum calls from 10 to 13.
Its aggregate worst pressure difference is smaller, but improvement is not
uniform scene by scene: 31×11×7 strict-8 peaks 6.26987e-7 versus retained original
4.81652e-7. No policy or checkpoint is promoted from this small diagnostic.

Tighter retained stopping also does not consistently improve every trajectory:
31×15×23 retained strict-7 peaks 1.70966e-6 versus original 1.28231e-6. Isolated
solve error shrinks greatly while independent simulation-state/reference
numerical differences remain. The observations do not uniquely identify their
cause, prove runaway instability, or establish a linear/exponential error law.
No native reference pressure arrays were captured, so do not interpret the
paired difference as an exact same-RHS error decomposition.

[All trajectories with first/peak/last, process receipts and timings](reports/trajectories.json) ·
[Every qualified AI step](reports/trajectory-steps.csv).

## Time and resources

Intel Core Ultra 9 275HX, 24 online affinity cores; Pumas Python 3.14.7,
PyTorch 2.14.0+cu132. CPU inference only; zero optimizer updates and no CUDA
context. Neural parameters, preprocessing and physical limits remain unchanged.

| Boundary | Elapsed |
|---|---:|
| Qualified one-frame capture command | 2.065 s |
| Paired fixture replay commands | 5.330 s |
| Initial 18-way native trajectory batch, including two timeouts | 125.413 s |
| 36 replay commands, 24 workers | 43.503 s |
| Two fresh long-case recoveries, 2 workers | 66.484 s |
| Qualified fixture start→last numerical completion, including gaps | 359.293 s |

All 24 cores execute replay work, one thread per child. Eighteen trajectories
run concurrently; two strict-8 long cases hit the 120s deadline at frames 423/490.
Both are terminated/reaped and retained as resource-limit failures, then
completed in fresh 2-way attempts with unchanged inputs and limits. Their lower
contention differs from the initial batch. Durations and callback times are
recorded cost, not an isolated speedup versus the native algorithm. Parallel
command durations do not add to elapsed makespan.

[All command/phase times](reports/timings.json) · [Environment](reports/environment.json).

## Retention, failures and visibility

The native inventory is **19 successful children**: 18 complete trajectories and
one qualified fixture. Three failed native children (initial fixture and two
timeouts) are terminated/reaped. All delegated commands and test children are
terminal. Two real-child lifecycle tests verify normal completion and SIGINT
cancellation/reaping with queued work cancelled. The retained controller also
leases cores to avoid overlapping assignments; the exact initial executed
controller and its hash are archived separately. Numerical owners are unchanged.

The record also retains the initial unavailable shell alias, relative-path
capture failure, summarizer timing-file selection error, and Tuldok's 400/409
rejections with their bounded recoveries. None are hidden as physical successes.
[Failure/recovery evidence](reports/reconciliations.json) · [Lifecycle tests](reports/lifecycle-tests.json).

Tuldok project snapshot 03cb4be737a1441d9f53a7e73c4173e2, **Rheon Surrogate Lab ·
pressure accuracy**, uses the existing catalog a7da4a8de53541a2a0e2aad08005bb71.
Tuldok correctly prohibits replacing an existing project snapshot ID; previous
projects remain registered. Its 64-reference group limit is handled through a
[historical evidence index](../experiment-history.md) preserving all 64 original
links. Current plan/results and live activity are directly visible in the new
snapshot. Existing datasets and all 784 future pressure frames are preserved.

Raw arrays, checkpoints and full traces stay in ignored local runs; only small
source/method/JSON/CSV receipts are committed. [Summary and exact bindings](reports/summary.json).
The previous 127×5×5 failure remains unresolved/out of scope. Application GPU
inference, arbitrary-grid accuracy, other fluids/boundaries and deployment
migration are not qualified here.

The next training test is the separately controlled early-versus-late residual
comparison in [next-training-protocol.md](next-training-protocol.md), keeping CUDA
training and CPU inference, with larger real-data expansion as a separate test.
