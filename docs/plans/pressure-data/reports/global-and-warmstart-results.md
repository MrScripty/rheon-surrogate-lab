# Fixed-operator global candidate and corrected-solve result

Author: Puma. Date: 2026-10-11. Development evidence only; no new test predictions.

The single preregistered global candidate passes the initial-guess signal and its
predictions become fully qualified through the isolated PCG API. This establishes
a working fixed-operator warm-start path, not general CFD accuracy or an integrated
simulator acceleration result. The original shallow CNN's negative results stand.

## Source and experiment accounting

- Dataset generator: cc2820646601f3db51fade707bfac5a521653257; 18 cases, of which
  only 8 train/4 validation enter this candidate and corrected-solve comparison.
- Candidate implementation/run: cd6f61adceca5fca9fb479790b2931fa81782362. Exactly
  500 full-batch Adam updates, 335 positive gains, lr 0.05, final-only validation.
  No sweep, restart, early stopping, analytic inverse gain injection or test prediction.
- Probe implementation/run: 0ec9ee6166b0679e425891f5da3c060cfd730d71. Zero optimizer
  updates; original frozen prediction hashes replay exactly. Both release binaries
  were built from clean commits; hashes are recorded in the receipts.
- Numerical base: 34235d33c886aa05b5bdf95d920b863d1fb2ad6f throughout. Root lab git
  dependency and lock remain unchanged. Optional API patch local commit:
  61182a4c4b2ec49cbc19a952e6b1d0b86f590d21; base/tree/diff hash in patches/manifest.json.
- Final test reservation 9000–9015 remains ungenerated. Full dataset integrity
  verification still reads original exposed test references, as declared in ADR0002.
  No new model prediction or model metric on either test group was produced.

## Raw prediction result

| Measure | Training (8 cases) | Validation (4 cases) |
| --- | ---: | ---: |
| Mean relative pressure RMSE against reference | 5.832e-11 | 6.910e-11 |
| Mean relative full residual L2 | 3.601e-11 | 1.343e-10 |
| Maximum actual corrected f32 divergence (s^-1) | 1.909e-7 | 1.177e-7 |
| All reference gates qualified | 8/8 | 1/4 |
| Worst original-b linear gate ratio | 0.4791 | 1.7280 |

Three validation cases narrowly fail the original-b linear gate; predicted and
actual divergence pass for every development prediction. The result is far below
the predeclared diagonal baseline's validation mean relative residual 0.7581,
mean relative pressure error 0.9640 and maximum actual divergence 0.7247 s^-1.
The initial-guess signal passes without weakening any solver tolerance.

Training took 0.225573 s; one 12-case inference took 0.000203 s, excluding file writes
and process startup. These single-run candidate timings are not the later paired
wall-time comparison. Pressure error is against the qualified numerical reference,
not against a general PDE truth. Raw reference residuals are themselves finite.

The fixed tensor cosine basis provides global context and the sealed rectangular
operator's modal structure. Learning 335 gains therefore approximates this single
operator's inverse response. It is a much stronger geometry prior than the old
local CNN, with exact zero/gauge and linear scaling by construction, subject to floating-point
rounding. This does not demonstrate neural
generalization across boundaries, coefficients, shapes, resolutions or trajectories.
An analytic spectral solver is a relevant future baseline before an efficiency claim.

## Genuine corrected-solve comparison

The native probe calls the patched solve_rhs_with_initial_pressure for warm starts
and unchanged solve_rhs for zero starts. Both preconditioners use the same
original-b relative 1e-10 / absolute 1e-12 tolerance, full predicted-divergence 1e-9
and actual corrected f32-divergence 1e-5 gates. Full b-Ap includes the gauge row.
All 12 cases in all 20 repeats of all 4 paths qualify: **960 accepted solves**.
This is repeated measurement of 12 cases, not 960 independent samples.

| Path | Iterations per 12-case batch | Per-case range | Median batch wall time | Min–max |
| --- | ---: | ---: | ---: | ---: |
| Zero + Jacobi-PCG | 920 | 76–77 | 21.127 ms | 20.100–23.476 ms |
| Global prediction + Jacobi-PCG | 5 | 0–2 | 3.523 ms | 3.037–4.325 ms |
| Zero + SGS-PCG | 359 | 29–30 | 14.422 ms | 13.887–16.928 ms |
| Global prediction + SGS-PCG | 3 | 0–1 | 3.349 ms | 2.895–4.515 ms |

All eight training guesses need zero PCG iterations. The four validation cases
need five Jacobi iterations in total or three SGS iterations in total; all four
then pass every gate. Maximum relative residual after either warm correction is
8.399e-11, maximum predicted divergence 2.992e-10 s^-1, and maximum actual rounded
f32 divergence 1.909e-7 s^-1 over development cases. Maximum actual divergence on
validation is 1.177e-7 s^-1 for every path.

Observed batch wall time is lower for both warm paths in this local measurement.
The paired timer includes NumPy inference; fresh prediction-directory creation,
little-endian file transfer and cleanup; child startup/JSON return; input file
reads/hashes, RHS reconstruction, workspace allocation, PCG, velocity correction,
and independent residual/divergence qualification. It alternates zero/warm order
for 20 pairs per method. Mean warm-path inference is 0.263 ms (Jacobi sequence) /
0.296 ms (SGS sequence); writes 0.591/0.550 ms; cleanup 0.142/0.118 ms. Mean native
solve is 0.258/0.258 ms; correction 0.198/0.192 ms; qualification 0.141/0.143 ms.
Native input/setup and child-launch/output overhead are also included in wall time.
Means of stages and medians of totals are distinct statistics and need not add up.

This is a warm-cache 336-cell CPU/file-pipeline measurement in one environment,
reporting AMD EPYC 9V74 and a four-CPU cgroup quota; Python scripts default both
OpenBLAS and OMP to one thread (neither variable was externally set).
with preloaded model/basis/RHS in the Python parent. Preparation 0.052056 s separately
includes Git/hash checks, dataset reference re-verification, checkpoint/RHS/basis
loading and frozen prediction replay. Python parent startup/imports/argument parsing
are excluded. Training, reference-data generation and build costs are also excluded
from paired solve time. There is no GPU transfer, persistent simulator integration,
cold-cache test or deployment workload. No simulator acceleration is claimed.

## Checks, preservation and next milestone

- Lab: 19 Rust contracts, 9 Python tests, formatting and strict Clippy pass.
- Optional patched Rheon: 309 tests pass with default features disabled, including
  8 new warm-start contracts for both preconditioners; fmt/strict Clippy pass.
  Desktop/PNG features are outside this patch verification scope.
- Independent source review confirmed initial-guess validation, full residual,
  original-b tolerance, Krylov reset, old zero-start path and matched probe gates.
  Exact error-variant assertions and timing-scope wording were strengthened.
- The exported patch applies cleanly to the pinned reference (`git apply --check`).
  Original /workspace/Rheon and the read-only reference remain clean and unchanged.
- Publication remains blocked by GitHub createRepository permission and missing
  exact remote. No retries, alternate repository, upstream PR/push or merge occurred.

The next milestone is review/admission of the isolated API patch and a frozen
untouched-test protocol, including an analytic spectral baseline and integrated
cost accounting before broader claims. Keep test generation, API publication and
UI/Tuldok/Pantograph/Pumas integration as separate later actions.

[Candidate receipt](global-candidate.json) and [corrected-solve receipt](corrected-solve-probe.json)
retain commits, hashes, settings, per-case physics and wall-time samples. Bulk arrays,
operator export and weights remain ignored, reproducible from source commands.
