# ADR 0005: Freeze and close the first fixed-operator experiment

Author: Puma. Status: protocol fixed before reserved generation; implementation
and independent development review must finish before the freeze is committed.

The parent explicitly authorized the independent final comparison, including
materializing reserved oscillatory_modes seeds9000–9015 only after review/freeze.
This closes experiment1. No tuning, retraining, checkpoint selection, new solver
pin, upstream publication, GitHub creation retry or expansion to another domain.

## Candidate, data and physical gates

Preserve the exact335 learned log-gains from the sole500-step cd6f61a run. Archive
the unchanged2950-byte weights.npz under experiments/final/ as a justified tiny
frozen checkpoint; all other model/data bulk stays ignored. Record file hash,
original run receipt hash, source commit, numerical reference and patch tree.
No coefficient is replaced by an analytic inverse in the learned candidate.

Use a distinct rheon-pressure-final-v1 schema: exactly16 final_test cases from
the reserved family/seeds, with the existing strict physics/arrays and a freeze
SHA256. V1 development data semantics are preserved. Metadata-only reservation
tests must not generate reserved fields/targets. Final CLI materialization binds
to the frozen generator binary; the Python gate additionally checks a committed
freeze, clean sources, all source/checkpoint/binary hashes and reviewed patch tree.

All methods use the identical 8x7x6 sealed uniform grid, spacing[.25,.3,.4]m,
rho1.25, dt.03125. Original RHS remains unchanged. Full residual includes the gauge
row; p[0]=0. Linear gate max(1e-12,1e-10*||original b||2), predicted divergence1e-9,
actual corrected f32 velocity divergence1e-5. The same public correct_velocity
and divergence operators are called for every path. Record failures without
raising tolerances, dropping cases, changing the checkpoint or selecting a retry.

## Six fixed paths

1. Analytic cosine solve, no PCG. Orthonormal DCT-II basis, constant mode omitted;
   lambda(k)=sum_d 4V/(rho*h_d^2)*sin^2(pi*k_d/(2*n_d)). Divide nonconstant RHS
   coefficients by lambda and impose cell-zero gauge. Full original-b residual
   checks still apply; the RHS is not explicitly mean-shifted. No target fitting.
2. Original zero-start Jacobi-PCG.
3. Frozen learned prediction plus patched Jacobi-PCG.
4. Original zero-start SGS-PCG.
5. Frozen learned prediction plus patched SGS-PCG.
6. Raw frozen learned pressure, no PCG: diagnostic, not a matched-quality timing
   competitor if it fails any gate.

Use the same dense cosine basis for analytic and learned predictions to isolate
coefficient learning. This is not a fastest available FFT/DCT implementation or
a general analytic CFD solver. Check the analytic baseline against a hand chain
and pinned public operator action on geometry/development fixtures before freeze.
Report pressure error against the reference separately from numerical acceptance.
A solver flag alone never grants physical acceptance.

## Freeze and evaluation accounting

Commit reviewed evaluator/generator/protocol/test source first, build clean release
binaries, then write a freeze identifying that code commit, every relevant source
hash, the exact binaries and weights, all six paths/tolerances/order, runtime and
patch metadata. Commit the freeze before any reserved generation. Frozen binary
metadata identifies the preceding code commit; adding the freeze/report docs does
not change evaluator code. Verify all hashes before generation and before/after
comparison. Report-only changes may follow; no numerical/evaluation code changes
are permitted after final-test access. Any pre-freeze correctness finding must
be fixed, requalified and reviewed first.

After generation/re-verification, perform one untimed quality pass per path,
including reference pressure RMSE. Then run12 repetitions per path per regime,
16 cases per batch. These are repeated timings of16 cases, not independent samples.
There are96 quality projections plus2304 timed projections in total. Report
unqualified cases and solver errors; timings cannot rescue a failed quality gate.

Method order is a fixed balanced Williams design: base indices[0,1,5,2,4,3], all
six modular shifts, then each row reversed. Across12 rows each method occupies
each position twice and each ordered adjacent pair occurs twice. Alternate regime
order by row. No random timing-dependent reruns or discarded samples. Development
smoke exercises all methods and protocol paths only on existing train/validation.

## Cold and warm distinctions and total costs

Fresh-process regime: every batch starts a new Python worker. Parent wall time
starts before process launch and ends after exit/JSON parsing. Include imports,
model/basis or analytic-gain setup, RHS loading/hashing, inference, prediction file
write/cleanup, native child startup/input hashing/RHS build/allocation, solve,
correction, full residual/actual divergence qualification and output transfer.
Zero-PCG paths do not import NumPy or build an unused model.

Resident-model regime: one Python worker per method preloads only model/basis;
every request still reads/hashes RHS and starts a fresh native child. One quality
pass also warms numeric kernels. Report steady-call median/min/max, startup and
teardown, and **amortized active wall cost including startup, the quality/warm-up
pass, all12 calls and teardown divided by12**. Idle waits while other methods run
are scheduling overhead, reported globally rather than charged to one method.
This is a resident-model/file-pipeline measurement, not a persistent native solver.
The analytic baseline pays its own setup; learned paths pay checkpoint loading.

Both regimes use OS page caches as available. No cache flushing, privileged
system changes or claim of cold filesystem/CPU cache control. Cold means fresh
Python/model initialization only; warm means resident Python/model state.
One BLAS/OMP thread is forced for workers. All timed loops run serially. Report
source/dataset verification, builds, reference generation and training as offline
costs outside operational timings; measure final gate/generation orchestration
and total harness wall separately. No GPU transfer exists. Tiny fixed-grid CPU
results cannot establish simulator acceleration or general CFD accuracy.
