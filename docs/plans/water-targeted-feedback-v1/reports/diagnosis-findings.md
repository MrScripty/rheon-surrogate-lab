# Diagnosis result and selected recipe

The actual one-solve replay runs the existing correction owner on byte-identical
native fields. Control needs 11 calls; feedback needs 28 twice. Replays take
1.60–2.03 seconds. Capture includes 15 native warm-up steps plus one AI step,
completes/reaps the native child, and saves fields only in ignored local runs.
The 13-call diagnostic exits 0 for control, 1 for feedback (expected red signal).

The 66 bounded probes take 16.330 s, zero updates/CUDA. Feedback at eight fixed
calls leaves residual norm 0.0193690 times the initial correction residual.
Cyclic field/dimension permutations reduce that to 8.01737e-5 and 6.08290e-6.
The corresponding control is 1.67838e-4, 1.82035e-4 and 1.42064e-4. Feedback's
independent pure Z mode on 16×23×25 fails 32 calls at 2.83321e-6; control needs
13 calls. Shape/axis-dependent operator error is supported; the finite population
is not a global proof or a unique attribution to pooling versus learned filters.

Zeroing the base or scaling the residual by -2/0.25 barely changes fixed-call
curves (feedback eight-call residual remains 0.0193690). Reconstruction/precision
is rejected as an explanation of this large gap. Identical-input reproduction
also excludes earlier autonomous trajectory drift as its cause.

Selected reversible comparison: retain the original eight-grid primary stream
and its exact 625 exposures per grid. Add one weight-0.1 analytical sample each
update. Supplemental control reuses the primary bank with a private sampling
seed; targeted fit uses the three cyclic permutations of 16×23×25 plus three
of 10×7×25 at the corresponding scene spacing. Both retain intermediate loss,
5,000 updates, real draws/scales/seeds and two corrections. This matches extra
example count, not voxel work; all forward cell/operator work and time are
reported. CPU preparation of late labels and new CFD remains outside these fits.
