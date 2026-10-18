# Completed C69 optimizer observation

C53 remains deployed. The published diagnostic replay reproduced C69's final
weight bytes exactly and all 32 CPU fitting residual values exactly (eight cases,
1/2/4/8 passes). This is a successful diagnostic pipeline, not an accuracy gain:
C69 still qualifies 6/8 at four passes and 7/8 at eight, with worst residuals
0.002026085796796365 and 5.727517748017874e-6 respectively.

Weights SHA-256: 593ff362fbd3c0b82573c9b18549cc942d085563a2fb0ea947fef855a6ce2e11.
No weights, trace arrays, datasets, or new holdout measurements are in Git.

Published Project dd27f770a5794504b2d7d46c54859b4b, training Run
c23d6eef1025497fa5eaead048c508b2, CPU restore Run
16dd74d4f6d24d0190070230e1903751. Both completed/drained. Standard artifact parity
and all five trace hashes match train/restore. The local controller reaped its
child. Existing owner/catalog, Pumas runtime and Tuldok publication/readback were
used; no private trainer or catalog bypass.

| Measurement | Seconds |
| --- | ---: |
| Final CPU qualification worker calculation | 11.209 |
| Instrumented CUDA trainer | 70.895 |
| Instrumented optimizer loop, including observations | 66.812 |
| Timed observation callbacks, included above | 4.009 |
| Managed training/restore/readback workflow | 275.136 |
| Outer managed controller | 280.016 |
| CPU optimizer/pressure analysis | 4.477 |
| Original C69 CUDA trainer, separate run | 68.804 |
| Original C69 optimizer loop, separate run | 64.605 |

The trainer difference is +2.091 s (about 3.0%), not a controlled overhead
estimate: separate runs vary. Callback time excludes final metadata/archive
closure. Optimizer work remains 6,400 forwards / 7,424,000 cells / 1,600 examples /
200 complete-cycle updates; 960 additional diagnostic forwards are separate.
Managed API calls totaled 190.636 s across admission, polling and readback and
are not all additional trainer idle time. Retained training artifacts total
31,088,149 bytes; 47 train/restore downloads, with 14 verified local-byte reuses.

CUDA peak allocation 20.061 MiB; trainer host peak 2,303.770 MiB; CPU analysis peak
775.734 MiB. GPU is RTX 5090 Laptop, Torch 2.14.0+cu132, NumPy 2.5.3; AMP/TF32 off,
deterministic algorithms on. CPU workers used one thread/core and never initialized
CUDA. Source/runtime/command hashes and actual worker bounds are retained.

Direct CUDA measurements on weak case 7 (16 × 7 × 11 at 16 cells/metre):

| Update | LR | Four-pass residual before | After | Full loss change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1e-4 | 1.5217229e-4 | 3.9383391e-4 | +1.976052 |
| 10 | 1e-4 | 1.8637601e-3 | 2.2332389e-3 | +0.371070 |
| 100 | 1e-4 | 2.4695764e-3 | 2.5650640e-3 | +0.078649 |
| 150 | 1e-5 | 2.0617954e-3 | 2.0742694e-3 | +0.012526 |
| 200 | 1e-8 | 2.0262236e-3 | 2.0258564e-3 | -0.000373 |

Only five updates were observed; these do not establish the fraction of harmful
updates across training. They do show that actual earlier updates hurt the weak
case, while the last selected update barely changes it. At update 1 the CPU
first-order estimate predicts improvement but the measured finite step worsens
loss; at 10/100/150 both indicate worsening. This motivates separating moment
transformation from finite-step magnitude before choosing a new policy.

CPU Adam reconstruction passed all five predeclared 2e-7 absolute checks; worst
parameter difference 7.45e-9, and relative delta L2 error at most 4.99e-6.
These use actual captured moments/gradients, not inferred historical states.
CPU reconstruction of the aggregate CUDA gradient failed all five 5e-6 relative
checks: 7.19e-5, 5.24e-4, 1.22e-3, 2.09e-3, 1.69e-1. The last gradient is especially
precision-limited. Component gradients agree exactly with the full CPU objective,
but that proves CPU algebraic consistency, not CUDA derivative identity. The
viewer marks this limitation. Do not promote its cosine/projection estimates into
a causal optimizer explanation or treat capped geometry as trained Adam.

The CPU reconstructed float32 moments suggest nonzero intended updates leave
about 0.065% of parameters unchanged at update 1 versus 33.39% at update 200.
This float64 formula is an approximate rounding diagnostic, not a change to
training precision or an exact decomposition of GPU rounding. Parameter-update
L2 falls from .008194 at update 100 to 3.800e-6 at 200. Tiny late updates and
quantization cannot by themselves explain earlier directly observed deterioration.

Failures are first-class records in issues.md and reports/. Two CUDA children
actually completed 200 updates each: attempt-03 was not published because its
trace violated the evidence shard bound; only attempt-04 is scientifically
analyzed. Another child failed before optimizer execution. Rejected preparation,
provider admission and request-key checks are retained separately. No unresolved
mutation was replayed. The interrupted study was closed after owner API custody
verification; its catalog state was not rewritten into success.

The first CPU analysis compared full historical row dictionaries (which include
extra grid labels) to measurement-only rows, falsely marking quality mismatch.
All 32 values were already identical. Corrected the comparator to compare the
explicit metric keys and independently verify geometry. Prior analysis/source/log
and its 4.400 s timing are retained under metadata-comparison-01; the final
4.477 s analysis is the accepted record. No training or historical data changed.

Offline browser verification passed all eight cases × three measurement controls,
retained scalar/tooltip/table checks, keyboard labels, narrow-screen overflow and
console checks. The owned browser was reaped; screenshot was visually inspected.
