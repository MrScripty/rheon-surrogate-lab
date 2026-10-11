# First experiment closed: frozen final comparison

Author: Puma. Completed 2026-10-11. This is the final, unretuned result of the
first fixed-operator experiment. The learned initial guess qualified with either
PCG preconditioner on all 16 reserved cases, but raw predictions failed one case.
The analytic spectral baseline qualified all cases without training, at much
smaller residual and comparable resident-model cost. Learned paths were slower
than zero-start PCG when Python/model initialization was paid each batch.
There is no demonstrated learned advantage over the analytic baseline, general
CFD accuracy claim, or complete-simulator acceleration result.

## Freeze and test access

Independent review admitted the sealed-only initial-pressure API, analytic
baseline, evaluator and development smoke before final access. Reviewed source
and clean binaries identify `44c470178a7411cea9b812fe485020af59094ff6`. The next
commit, `af6d2a10cd94b2a69b44bc70b470ea611ecc069c`, added only the freeze JSON,
BEFORE generation. It fixes 34 source files, two binaries, checkpoint, method
order, tolerances, 16 seeds and runtime. No numerical source, checkpoint,
protocol or tolerance changed after test access. Final evaluation performed
**zero optimizer steps** and no checkpoint selection, restart or retuning.

The tiny archived checkpoint is exactly the original 335-gain, 500-step model
from `cd6f61adceca5fca9fb479790b2931fa81782362`. The global cosine basis supplies
a strong prior specific to this operator; training learned positive modal gains.
The analytic baseline uses the same dense basis and the known sealed-box
finite-volume eigenvalues `sum_d 4V/(rho*h_d^2) * sin²(pi*k_d/(2*n_d))`.
It omits the constant mode, divides modal RHS by these eigenvalues, and fixes
pressure cell zero. It requires no training or PCG. This dense implementation
is not an optimized fast DCT or a claim of best possible analytic runtime.

The reserved oscillatory-mode seeds 9000–9015 were generated once after the
committed freeze: **16 final-test cases, 128 arrays, 233,728 raw bytes**, all
reference targets reverified. New seeds belong to an already-observed family;
this is not an unseen-family test. The reservation is now consumed and exposed.
The original reservation JSON remains the immutable declaration at freeze time;
[the outcome record](../../../../experiments/final/outcome.json) records closure.

## Identical numerical acceptance

All methods use the same 8×7×6 sealed box (336 cells), spacing
[0.25, 0.3, 0.4] m, density 1.25 kg/m³ and dt 0.03125 s. Inputs are deterministic
boundary-valid preprojection MAC velocities. RHS is rebuilt using public Rheon
APIs; pressure is Pa with `p[0]=0`. Qualification requires finite/shape/gauge/
compatibility checks, full original-RHS residual including the gauge row
`||b-Ap||₂ <= max(1e-12, 1e-10*||b||₂)`, predicted divergence at most `1e-9 s⁻¹`,
and actual corrected-f32-velocity divergence at most `1e-5 s⁻¹`.
Every path receives identical velocity correction and physical verification.
No tolerance was relaxed to accept predictions. Acceptance thresholds match;
actual residuals need not equal one another below those thresholds.

| Method | Qualified | Total PCG iterations (range) | Worst relative full residual | Mean reference-relative pressure RMSE |
| --- | ---: | ---: | ---: | ---: |
| Analytic spectral | 16/16 | — | 2.92574e-15 | 1.05129e-10 |
| Zero + Jacobi | 16/16 | 1210 (74–77) | 9.67794e-11 | 0.00000e+00 |
| Learned + Jacobi | 16/16 | 1 (0–1) | 9.73095e-11 | 2.37337e-10 |
| Zero + SGS | 16/16 | 472 (29–30) | 9.76740e-11 | 1.38388e-10 |
| Learned + SGS | 16/16 | 1 (0–1) | 9.73095e-11 | 2.37164e-10 |
| Raw learned (diagnostic) | 15/16 | — | 1.03700e-10 | 2.39286e-10 |

The raw failure is `oscillatory_modes-9005`: full residual is **1.036999 times**
the permitted linear threshold. Both warm starts correct this case in one
iteration; the other 15 require zero. No solver returned an error. The raw
model is not a qualified solver replacement.

Maximum actual f32 divergence is **1.6142924619e-7 s⁻¹ for every method**. The
analytic predicted-divergence maximum is 1.89663e-14 s⁻¹, while the largest of
all method maxima is 7.56106e-10 s⁻¹. Actual and predicted divergence differ
because velocity storage/correction rounds to f32; the rounded velocity must
be checked directly. Reference-relative pressure error is agreement with a
finite-tolerance PCG solution, not exact PDE error. Zero-Jacobi error is zero by
construction; the analytic pressure has a much smaller residual despite its
nonzero difference from that reference.

There are **96 quality projections plus 2,304 timed projections**: six methods,
one quality pass, and 12 batches in each of two timing regimes, with 16 cases
per batch. These are repeats of 16 cases, not 2,400 independent samples. Five
fully qualified methods contribute 2,000 accepted projections; raw learned
contributes 375/400, retaining its same failure on all 25 passes. Overall
acceptance is 2,375/2,400 (95/96 quality; 2,280/2,304 timed). Per-case pressure
hashes and physical outcomes remain equal across every timing repetition.

## End-to-end timing

Every number below is for a **16-case batch**, not one case. Times are descriptive
measurements from one Linux CPU environment (AMD EPYC 9V74, four-CPU cgroup quota;
worker OpenBLAS/OMP threads fixed to one), Python 3.12.14 and NumPy 2.3.5.
Twelve balanced Williams rows place every method in every position twice and
every ordered adjacent pair twice. Fresh/resident regime order alternates by
row. No results were used to change order, repetitions or implementation.

| Method | Fresh Python/model median [min–max], ms | Resident median [min–max], ms | Resident amortized total, ms |
| --- | ---: | ---: | ---: |
| Analytic spectral | 81.702 [78.191–93.059] | 5.197 [4.941–6.227] | 12.925 |
| Zero + Jacobi | 59.700 [57.572–65.648] | 27.975 [26.923–30.864] | 33.209 |
| Learned + Jacobi | 83.866 [80.211–105.158] | 5.477 [5.059–6.081] | 12.664 |
| Zero + SGS | 51.117 [48.903–55.617] | 19.723 [18.825–21.166] | 24.801 |
| Learned + SGS | 81.277 [78.876–101.030] | 5.458 [5.180–7.364] | 13.206 |
| Raw learned (diagnostic) | 82.834 [79.925–90.080] | 5.191 [4.809–6.155] | 12.345 |

Fresh-process timing encloses Python startup/imports, model/basis setup when
used, RHS reads/hashes, inference, prediction file writes/transfers, native
child startup, input hashing/RHS reassembly/allocation, PCG, correction,
qualification, JSON handling, cleanup and exit. Zero-PCG paths do not pay for
an unused NumPy/model setup. “Fresh” means a new Python process/model; OS and
CPU caches were uncontrolled, so this is not a cold-filesystem/cache benchmark.

Resident-model timing holds Python and the basis/checkpoint in memory. Each
batch still rereads inputs and starts a fresh native child; this is not a
persistent simulator integration. The steady median excludes one-time setup,
but the last column explicitly amortizes **startup + quality/warm-up + all 12
timed calls + teardown over 12 batches**. Idle time while other methods run is
not assigned to a method. Native/worker stage timers are nested diagnostics,
not additive substitutes for the enclosing wall measurement.

The entire comparison harness took **6.727683 s**, including orchestration and
journal handling; the gated comparison before receipt writing took **6.800770 s**,
including 0.048070 s initial gate/reference verification and post-run checks.
Generation plus its gate and reference verification took **0.080462 s**.
Offline training (historical 0.225573 s), compilation and environment provisioning
are outside operational batch timings, not hidden per-call speedups. No
amortization over a hypothetical larger workload is inferred.

Learned resident calls reduce PCG iterations and cost relative to these zero-PCG
paths, but fresh learned calls cost 81–84 ms versus 51–60 ms for zero-PCG.
Analytic resident median is 5.197 ms versus learned 5.458–5.477 ms; amortized
costs are similar (12.925 ms analytic, 12.664/13.206 ms learned). Small ordering
of those amortized values is not evidence of a robust learned advantage.
The analytic baseline is already accurate and inexpensive for this special box.

## Verification, retained evidence and boundaries

Before the freeze: 20 lab Rust tests, 14 Python tests, formatting and strict
Clippy passed. The isolated API patch passed all 309 no-default-feature Rheon
tests; a fresh independent review also reran 23 focused pressure/operator/API
contracts. Development smoke used only existing train/validation cases.
Independent source/protocol and frozen-artifact checks preceded generation;
the final receipt audit is recorded in [the review](review.md).

- [Full final comparison receipt](final-comparison.json): per-case quality,
  pressure hashes, 12 samples per method/regime, stage timing, min/max/median,
  amortized accounting, runtime, identities and all limits.
- [Final generation receipt](final-generation.json) and
  [committed freeze](../../../../experiments/final/freeze.json).
- Ignored local evidence: `data/final-9000/`,
  `data/final-9000-generation.json`, `runs/final-v1/receipt.json` and
  `runs/final-v1/journal.jsonl`. Raw data and repeated batch journal stay out of git.
- The 2,950-byte exact checkpoint and isolated API patch are tracked, so recovery
  does not require retraining. Recovery commands are in
  [the experiment directory](../../../../experiments/final/README.md).

No upstream publication occurred. The lab's production dependency is still
Rheon `34235d33c886aa05b5bdf95d920b863d1fb2ad6f`; the optional probe alone uses
local reviewed patch `61182a4c4b2ec49cbc19a952e6b1d0b86f590d21`. Original Rheon
main was not altered. The pinned API still lacks warm starts; upstream API
admission remains a separate owner decision, not part of this experiment.
The patch supports this sealed domain only, with finite/shape/gauge checks,
full initial `b-Ap`, original-b tolerance and reset Krylov scratch.

The early shallow CNN remains a negative result (0/16 qualified predictions);
it has not been relabeled successful by the later geometry-specific model.
No moving boundaries, free surfaces, changing grid, density, geometry, coefficient
field, trajectory rollout, larger scale or broader CFD generalization was tested.
This closes experiment one; no further model search or integration work is opened.

Publication remains paused at the exact requested destination
`https://github.com/MrScripty/rheon-surrogate-lab`. The authenticated integration
returned `GraphQL: Resource not accessible by integration (createRepository)`;
the exact-remote push returned `Repository not found`. No retry, alternate
repository/account or upstream publication was attempted during closure.
The public repository/remote commit has therefore **not** been verified.

## Exact evidence identifiers

- source/build commit: `44c470178a7411cea9b812fe485020af59094ff6`
- freeze commit: `af6d2a10cd94b2a69b44bc70b470ea611ecc069c`
- freeze SHA-256: `51e2e5882194f16f0e697d7fe0839b891bb12a0eef77c569d88211175c358a35`
- candidate weights SHA-256: `f97ffa3e083e5fbf8f7c8b69b79b2c6176af1ff3b01737664752bb66747d20b9`
- final manifest SHA-256: `620cb224547e93c67e2b54a8aa28250eabcd0f26430d63d5d16baae99a2d3192`
- comparison receipt SHA-256: `81e47c2d1e233eba042322d48f8394ef085c2d391c81598fa974f291212c7e00`
- generation receipt SHA-256: `81fc3ea382cec88f0e349bef5fcb485a29f4f52f22b8ae85fa44f9bb5708442c`
- ignored journal SHA-256: `ab152906dae7b813c7e55d7147da873c2778fd31a87e78960b5b8649a49b627a`
