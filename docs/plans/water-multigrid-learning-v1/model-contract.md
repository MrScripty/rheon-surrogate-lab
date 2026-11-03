# Cohort development checkpoint contract

Development-qualified laboratory candidate; application deployment is a separate
integration. Parameters remain 370,016 float32 values; weights.npz is 1,485,452 bytes.
Weight SHA256: `0de683cccc3fb878c3310f098676d08bcda519defe3c13275b768fd4931bb4d6`.
Normalization SHA256: `f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1`.
Captured source commit: `c4e4a503d69e2bad080704e62c43a3fde2e5e3d6`.
Training Run 8a4f5c778e73408bb784375ae9a94689; fresh CPU restore
96a8c0a8014e4241a06fb25eb7d1fce4; development inference
d7c0504aa2e04a119bfcca6ec8f4a8fd. These use the existing Tuldok catalog.

## Input and reconstruction

The raw network accepts batch 1–4; qualified adaptive stopping requires batch 1.
One float32 neural tensor `[N,1,Z,Y,X]` contains
`q = ((rho/h) * residual) / 0.6197167573289654`. The scale's units are Pa.
The historical channel identifier is `dimensionless_laplacian_rhs`; the recorded
physical quantity before normalization is in Pa. There are no coordinate,
voxel-size, density, timestep or previous-pressure feature maps. Scalar geometry
and physics remain available to the physical reconstruction and stopping owner.

The supplied residual is `b - A(previous_pressure)`, with sealed uniform cubic
`A=(h/rho)L`. Each learned evaluation predicts a float32 raw correction. Convert
to float64, multiply by 22.254425460012698 Pa, subtract that correction's first
voxel value, then add the supplied float64 pressure estimate. Recompute residual
from the original RHS after each correction. Follow the captured authoritative
physical_residual.py and learned_iteration.py implementations for operation order.
The network/optimizer remain float32; existing physical/base arithmetic is float64.

Training uses two shared evaluations. Qualified adaptive inference checks after
each individual evaluation, with at most32 actual network evaluations and no
native pressure refinement. Stop at residual L2 no greater than
`max(1e-12,1e-6*RHS_L2)` and predicted divergence no greater than 1e-7. Afterwards,
apply the existing independent native physical profile; equation stopping alone
is not the complete qualification. Zero residual/already-qualified estimates
can bypass neural work. All repeated evaluations share the same weights.

## History, fallback and physical scope

The experiments supply actual previous pressure and 15 native warm-up steps.
Teacher-state inputs contain their real preceding snapshot. Inference labels
never enter the predictor. Budget exhaustion or failed physical checks remains
an explicit raw AI failure. Application fallback/cancellation is owned by the
Rheon comparison worker; a native fallback must retain separate provenance and
must not count as a successful learned solve.

The code enforces positive finite scalar cubic h, finite colocated physical
fields, cell-zero base gauge and a 65,536-cell tensor budget. One shared h defines
all three axes; extents are counts times h. The operator is the connected sealed
Neumann grid without geometry masks, so obstacle/free-surface/open-boundary
operators are unsupported by this representation.

The dataset and native fixtures enforce water density 1000 kg/m³, dynamic
viscosity 0.001 Pa·s, timestep 1/60 s, impermeable free-slip walls and internal forcing
without mass injection. Origin 0 is a fixture convention; the pressure network
has no world-coordinate input. Generic model helpers accepting positive rho/dt
are not qualification for other materials/settings. Tests cover real 5–16-axis
training/development grids, cubes 16/24/32/40, rectangular24x16x12 and 12x8x10,
and tested cubic spacings 1/8,1/16,1/24,1/32,1/40 m. Other admitted tensor shapes
are numerical input support, not demonstrated arbitrary-resolution accuracy.

Weights/data stay in managed artifacts and ignored local runs. The existing
application adapter must explicitly support this preprocessing, normalization,
iteration policy and physical/history contract before consuming this checkpoint.
