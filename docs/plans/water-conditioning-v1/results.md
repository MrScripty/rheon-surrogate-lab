# Pressure conditioning results

All 25 managed stages completed and drained in the existing Tuldok catalog.
Five 10,000-update known-frame fitting controls, a two-update CUDA/CPU smoke,
and four 500-update execution controls completed, totaling 52,002 CUDA updates.
Fresh CPU restores/inference succeeded. Model accuracy remains unqualified.
Historical sources, captures, results and deployed comparison model are preserved.

| One-frame trial | Training elapsed seconds | Fitted-frame relative residual | Median validation residual | Worst validation residual | Qualified validation |
|---|---:|---:|---:|---:|---:|
| constant-low | 234.94 | 0.001037400 | 0.070593372 | 0.284879817 | 0/128 |
| constant-high | 205.43 | 0.001524644 | 0.068144767 | 0.273329378 | 0/128 |
| delayed-decay | 205.75 | 0.002405627 | 0.069702583 | 0.274532582 | 0/128 |
| scalar-units | 229.40 | 0.001321878 | 0.132672495 | 0.440350046 | 0/128 |
| no-groupnorm | 129.91 | 0.000398222 | 0.082815901 | 0.293090166 | 0/128 |

The historical fit was 0.0017782435 on the same frame. Removing GroupNorm with
scalar-unit conditioning reaches 0.0003982222: approximately 4.47 times lower,
but still 398 times the unchanged 1e-6 equation target. Every new fitted proposal
fails; every validation population qualifies 0/128 proposals. These models were
trained on one frame to diagnose fitting, so their validation behavior does not
establish a better general-purpose model than the previous 1,024-parent model.
Removing GroupNorm helps fitting but does not identify it as the only cause.
All preceding native pressure context was supplied; these are not cold starts.
Native fallback success is not AI success and no solver acceleration is proven.

The reference correction rounded through actual float32 reconstruction reaches
9.73352e-8 on the fitted frame and passes every native physical gate under the
existing model profile. All 1,024 actual training states and 128 validation states
also pass their equation-only rounded-reference controls. Full native physical
qualification of those rounded states is not claimed from the equation check.
Captured residuals agree with the operator and the loss directional derivative
checks agree. Higher-precision model parameters and relaxed physical gates were
not used to explain away the remaining learning gap.

## CUDA execution controls

All use the same amplitude-conditioned seven-channel architecture, seed, frame
and Adam configuration for 500 updates. Rates measure the retained training
interval, including optimizer/cache setup and a short profiler window. Each is
one bounded measurement, not a statistical confidence claim or full-workflow
speedup. Tuldok verification/startup often dominates these short jobs.

| Execution | Updates/second | Samples/second | Batch size | Exact baseline checkpoints at 0/2/500 |
|---|---:|---:|---:|---|
| gpu-baseline | 31.19 | 31.19 | 1 | Yes |
| gpu-resident | 42.10 | 42.10 | 1 | Yes |
| gpu-batch4 | 41.32 | 165.29 | 4 | Not required; batch reductions differ |
| gpu-prepared | 52.18 | 52.18 | 1 | Yes |

The prepared path is 1.67 times as fast as the original measured training path,
with every retained parameter bit identical through update 500. Fresh CPU
restores preserve the same checkpoint and CPU prediction witnesses. The batch-4
fixture repeats a frame and changes sample exposure; its throughput is not an
accuracy gain or evidence of four independent simulations. Stacking heterogeneous
grids is rejected rather than silently padding their physical boundaries.

In ten profiled updates, baseline versus resident execution records 1,960 versus
400 CUDA stream synchronizations. Scalar extraction calls (CPU and CUDA combined)
fall from 2,520 to 1,030, then 860 with prepared fields. Tensor copy/conversion
calls fall from 100 to 30, then 20. The original artifacts label the latter count
host_device_copies; that includes dtype conversions and is not a host-transfer
measurement. They are preserved unchanged; future profile schema v2 corrects
the label and explicitly records synchronization counts. Absence of an operation
from the retained top-20 list does not mean zero calls.

Aggregate GPU utilization remains below saturation in the measured windows:
resident mean 26.8%, batch-4 mean 30.7%, including open apps and setup/profiling.
The baseline monitor started too late for a controlled utilization comparison.
This small model still launches many small operations. Throughput improvements
are demonstrated without claiming that the GPU is fully occupied. The current
Tuldok supervisor executes one managed Run at a time; additional CUDA submissions
queue. Concurrent managed training requires a separate scheduler/lifecycle change.
No private optimizer or second catalog bypassed that owner.

## CPU preparation and next model work

The 23-worker audit used every assigned core 1–23 on all 1,024 actual training
states and finished in 23.42 seconds while CUDA optimization advanced. Workers
used one Torch thread each and zero CUDA initialization. On the prior selected
checkpoint, training median residual is 0.0353517 and validation median 0.0385793.
Doubling residual has median correction-linearity error 3.38%/3.88% respectively;
sign reversal has 12.31%/14.01%. Eight repeated raw corrections reach validation
median 0.00429424; an operator-based minimum-residual step reaches 0.00373089.
Neither is a qualified solve or demonstrated simulation speedup.

Single 16-cubed CPU inference measures 3.26 ms at eight threads versus 541.07 ms
at 24 on this host. Profiling shows slowdown across small convolution, GroupNorm,
padding and fill operations, without establishing one universal cause. Persistent
recorded-state replay gives 1,455–1,468 states/s with 23 single-threaded workers,
versus 460–462 with one eight-threaded worker. Three eight-threaded workers show
little gain and one severe stall; passive waiting removes that stall but reaches
only 435–490 states/s. Startup is reported separately and can dominate small
batches. These are pressure-inference measurements, not three evolving native
simulations; existing deployment thread/witness contracts are unchanged.

Parallel native checking reproduces all scientific outputs from the historical
128-case serial validation exactly, excluding elapsed_s, solver_s and
verification_s. All native process groups drained and ownership was released.
The [next model experiments](next-steps.md) prioritize physically scaled residual
inputs, linearity and coarse/repeated correction before increasing corpus size.
No new reserved test was consumed and no new simulation corpus was generated.

The complete measured summary, configurations, Run IDs, artifact hashes, traces,
source closures and small test receipts are retained in [reports](reports).
Large artifacts remain in normal Tuldok custody and runs/water-conditioning-v1.
Source changes are committed separately for resource/diagnostic work, model
conditioning/normalization, and execution optimization. The task branch is
retained for review and continuation; the original Rheon and serving Tuldok
checkouts were not modified by this tranche, and no model weights/datasets are committed.
