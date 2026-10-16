# Pressure accuracy recovery

Completed. No replacement selected or deployed; eight reserved independent
scenes remain unused. [Plan](plan.md), [methods](reports/methods.json),
[execution ledger](execution-ledger.md).

## Saved-input convergence

Eighteen retained sparse states from consumed scenes17,60,69,73; both models see
the exact same RHS, base/history and physical settings. These are local mean-free
same-RHS pressure errors against the independently checked sealed-Neumann oracle,
not pressure differences between independently advanced trajectories.

| Checkpoint | Pressure target | Mean calls | Max calls | Worst observed pressure relative L2 |
|---|---:|---:|---:|---:|
| Harder-data control |1e-6|7.8333|11|5.8327e-7|
| Harder-data control |3e-7|8.8333|12|1.7124e-7|
| Harder-data control |1e-7|9.2778|13|7.8348e-8|
| Prior late |1e-6|9.2222|16|5.5122e-7|
| Prior late |3e-7|9.8889|18|1.7343e-7|
| Prior late |1e-7|11.0556|20|5.4709e-8|

The harder-data checkpoint retains a work advantage on these supplied inputs at
each matched target. Stricter stopping buys local accuracy; autonomous accuracy
and the full comparative work gates are separate pending measurements.

No optimizer or new simulation:36 fresh-process replays,0..32 calls each,45.978s
total command time. Peak individual RSS789880832B. Full curves, source/input/model
identities and per-band spatial signal/error energies are retained under
[frontier](reports/frontier/) and [summary](reports/frontier-summary.json).
Spatial cosine modes and per-state amplitude bins are not temporal frequency
bins. Zero/near-zero signal is reported separately; relative error alone does
not establish an input or arithmetic noise floor.

## Runtime qualification

New project assembly:3.216s smoke,3.266s weight 0.1,3.565s weight 0.25. All capture
the existing v9 trainer at3a6ec2d. CUDA smoke plus fresh CPU restore completed and
drained in592.978s managed wall time; the actual recipe times and CUDA allocation
are in [smoke receipt](reports/smoke-result.json). The managed wall includes
runtime attestation, input custody, approval and artifact retrieval; it is not
optimizer time. No runtime/portability changes were made.

## Consumed trajectory comparisons

All four checkpoint/policy combinations completed all 74 consumed scenes, with
original58/amended8/recent8 reported separately.
Existing physical limits remain unchanged. No final-informed tuning or relaxed
replacement gate is permitted.

### Stricter stopping, same harder-data checkpoint

All 148 AI trajectories and 74 native controls qualified in2185.8s concurrent correctness wall. No policy is eligible under the frozen combined accuracy/work gates.

| Population | Target | Mean calls | Max calls | Worst paired trajectory pressure relative L2 |
|---|---:|---:|---:|---:|
|original58|late-pressure|7.9373|12|3.1447947e-06|
|original58|tight-3e-7|8.5047|13|3.2255102e-06|
|original58|tight-1e-7|9.1844|14|3.9575718e-06|
|late-amended8|late-pressure|9.5619|12|1.3489549e-06|
|late-amended8|tight-3e-7|9.7378|12|1.0261382e-06|
|late-amended8|tight-1e-7|10.5420|13|1.3850956e-06|
|minimum8-failed8|late-pressure|9.0796|17|1.4151669e-06|
|minimum8-failed8|tight-3e-7|8.9900|12|9.5384598e-07|
|minimum8-failed8|tight-1e-7|9.6538|13|1.5042394e-06|

Target 3e-7 recovers pressure accuracy on the two newer populations, with extra work. Original58 still narrowly regresses against prior late pressure accuracy, and mean/maximum work gates also prevent replacement. Target 1e-7 is not monotonically better on paired autonomous trajectories. Same-input convergence alone would not reveal that. Full per-step data, sparse oracle checks and failed comparative gates are in [policy reports](reports/policy/).

## Matched CUDA fits

Both 5,000-update fits use the same original seed,192-pair bank, real/analytical/supplemental/direction sample streams, normalization and initial weights as the archived weight 0 control. Only first-correction pressure feedback differs. Both train/restore workflows completed and drained; real CPU fixtures passed. Neural architecture, inputs, inference code and float32 precision remain unchanged. Different weighted training-loss values are not directly comparable; qualification uses common pressure/work metrics.

| First-pressure weight | Optimizer seconds | Managed train+CPU restore seconds | Peak CUDA tensor allocation |
|---|---:|---:|---:|
|0.1|562.049|1256.730|770565632B|
|0.25|492.958|1080.380|770565632B|

Timing contexts differ:0.1 overlapped two CPU simulation workers;0.25 overlapped approximately94s of CPU diagnostics, with otherwise idle experiment CPU workers. Do not attribute the timing difference to the loss weight. The extra managed wall includes CPU restore and runtime/input/artifact custody. Both checkpoints are frozen before consumed trajectory evaluation.

### Gentler first-correction feedback

All 148 AI trajectories and 74 native controls qualified. Neither checkpoint meets all comparative gates; there is no promotion.

| Population | Model | Mean calls | Max calls | Worst paired trajectory pressure relative L2 |
|---|---|---:|---:|---:|
|original58|late-pressure|7.9373|12|3.1447947e-06|
|original58|gentle-010|7.4338|9|3.5956503e-06|
|original58|gentle-025|7.9586|11|3.8858956e-06|
|late-amended8|late-pressure|9.5619|12|1.3489549e-06|
|late-amended8|gentle-010|8.8772|12|1.07942e-06|
|late-amended8|gentle-025|9.0631|11|9.3300919e-07|
|minimum8-failed8|late-pressure|9.0796|17|1.4151669e-06|
|minimum8-failed8|gentle-010|7.6582|10|1.4333059e-06|
|minimum8-failed8|gentle-025|8.7434|14|1.6197044e-06|

Concurrent correctness wall:2139.076s. Weight 0.1 reduces original58 max12→9 and recent8 max17→10; mean calls also decrease in all three populations versus prior late. Amended8 worst pressure improves about20%. Original58 worst pressure regresses about14.3%, and recent8 about1.3%, so the complete gate rejects it. Weight 0.25 improves amended8 pressure further but has higher recent peak work and worse original pressure. These are comparative tradeoffs, not physical qualification failures. [Full trained-model reports](reports/training/).

## Current serial CPU latency

Prior late versus weight 0.1, same current host/runtime, CPU1/one thread, two reversed-order repetitions, first AI timing sample excluded. All 12 episodes qualified. These are paired whole Engine.advance AI timings; physical limits remain unchanged.

| Grid | Prior late median | Weight 0.1 median | Reduction |
|---|---:|---:|---:|
|17×13×9|0.036831s|0.034064s|7.51%|
|24×16×12|0.066232s|0.066383s|-0.23%|
|40×40×40|1.382127s|1.384489s|-0.17%|

Only the smallest case shows a clear measured improvement. The other two are effectively unchanged;40³ has only 14 steady samples per model and differences lie within repetition variability. No general speedup or resolution crossover claim. [Timing CSV](reports/timing-comparison.csv).

## Spatial amplitude diagnostics

On 18 identical supplied inputs, worst same-RHS pressure error at 8 calls is1.4153e-4 for prior late,8.3229e-6 for harder-data control,6.0483e-6 for weight 0.1, and4.1325e-5 for weight 0.25. This purposefully hard consumed set differs from the earlier15-state diagnostic; do not compare their maxima as if the inputs matched. [Full curves](reports/fixed-call-final-summary.json) and [per-state spectral CSV](reports/spectral-state-comparison.csv) retain absolute signal/error energy and relative error.

At 8 calls on9×11×33, weight 0.1 improves pure-low1Pa coefficient error from4.6344e-5Pa to2.9783e-5Pa, but high-only low-mode leakage rises from1.2682e-4Pa to3.5035e-4Pa. Weight 0.25 worsens both versus control. Gentler feedback is not a uniform spectral improvement. Pure-mode scaling and strong-mode masking remain distinct; this does not establish a universal noise floor. The temporal replay retains the prior 32-sample Hann method (lowest positive bin 1.875 Hz), not evidence about slower temporal cycles or a changed timestep. [Manufactured comparison](reports/manufactured-mode-comparison.json).

## Retention and decision

The predefined combined pressure/work gates reject all four candidates. Weight 0.1 is selected only for diagnostic timing. No second initialization seed was tried and the independent eight remain reserved; no final-informed refit, target sweep or replacement occurred. Model architecture, inference contract and deployed comparison remain unchanged.

Measured span:92.99min, from first timed source test through final timing command, including overlap/setup and documentation gaps; final sealing/commit time excluded. CPU correctness phases2185.8s and2139.1s; serial timing303.0s. Full GPU optimizers562.0s/493.0s; managed full workflows1256.7s/1080.4s; managed smoke593.0s. GPU tensor peak770565632B per full fit; maximum individual CPU diagnostic RSS870805504B. These are not summed system memory or isolated throughput measurements.

All 296 full AI trajectories qualified, 932 development oracle-bound checks passed, 462 native children reaped, 557 timed child command receipts complete/reaped, six managed Runs completed/drained in the original Tuldok catalog. No OOM or failed simulation/training episode. A CSV line-ending check was repaired in a follow-up commit; its parsed numbers were unchanged. [Sealed summary](reports/summary.json), [full small episode receipts](reports/episodes.json), [fixture inference proof](reports/fixture-inference-proof.json), and [ledger](execution-ledger.md). Source/config/scalar evidence is in Git; weights/raw tensors remain in ignored local artifacts and the existing catalog. No unrelated graph files were changed.
