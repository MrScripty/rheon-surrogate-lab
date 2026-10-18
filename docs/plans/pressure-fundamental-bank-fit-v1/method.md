# Method

This is a capacity diagnostic, not a new physical simulation dataset. Each case
is a discrete Neumann fundamental pressure eigenmode on a sealed rectangular
water grid. Each voxel remains cubic. The eight existing grids vary cell counts
and scalar spacing (1/8 or 1/16 metre); each has three axis-aligned fundamentals.
The manufactured right-hand side uses the independent eigenvalue, then the
implemented full operator is checked against it before admission and CPU evaluation.

Pressure amplitude is the cell-weighted RMS of the original training corrections,
2.8535734640782584 Pa. Original training scalars and normalization are unchanged:
rho 1000 kg/m³, viscosity 0.001 Pa·s, dt 1/60 s, origin zero, closed free-slip
boundaries, no SGS. These are the captured physical scope, not a claim to support
new densities, boundaries or fluid types. There is no emitter or new fluid capture.

The model still consumes one dimensionless residual channel, [batch,1,Z,Y,X].
The float32 network produces a correction reconstructed in float64; the original
float64 residual operator and explicit shared recurrence remain unchanged.
One optimizer example unrolls four network passes; inference checks use 1/2/4/8.
Default deployment remains two passes. No architecture, parameter count, input
channel, normalization or precision changes are made.

The pure-fitting loss is the qualified C62 loss: canonical manufactured pressure,
gradient and equation feedback, final log equation feedback, and early equation
feedback weight 0.05. Teacher weight is zero. No real-frame, primary analytical,
supplemental, direction or current companion sample participates in optimization.
Inherited real-frame diagnostics are outside the optimizer work counter.

C64 has 24 cases and 4,800 updates. Conditional C65 has three axes on the first
13×15×16 grid, 600 updates. Conditional C66 has X fundamentals on eight grids,
1,600 updates. Each starts from the exact C53 inventory with fresh Adam and
clipping norm one. Every complete cycle visits every case once, privately shuffled
with PCG64 seed 2026102101. Exactly 200 cycles are used; per-case exposure and
learning-rate allocation match across arms. LR changes after complete cycles at
100/150/175/190 exposures. There are 20 full-bank learning measurements, every
ten cycles. Seeds are fixed, so these are bounded screens rather than seed-averaged
claims. Compared with historical C62, both training population and update budget
change; no single-factor causal claim against C62 is intended.

The existing Project assembly captures all sources and typed contracts. The
existing Tuldok catalog owns admission, CUDA execution, fresh CPU restore and
published artifact readback. No private catalog or unmanaged optimizer is used.
Final qualification is published only after all bank checks; the controller rejects
a generic base-only receipt before starting any child. Both Runs must complete and
drain, all published artifacts must match, and owned children must be reaped.

CPU evaluation loads one checkpoint per fresh process. The 128 consumed real
requests are timed serially across checkpoints, one thread on the same core,
one warm-up and three repetitions per case. Timing includes fixed-pass recurrence
and its final equation check. Fitted cases and the existing 240 manufactured
regression cases run afterward on three separate one-thread cores. Their concurrent
elapsed times are recorded, but not used as isolated inference speed comparisons.
The original untouched holdouts are not used. Host reserve is 12 GiB plus 2 GiB
per worker; fitting tensors are bounded at 8 MiB. CUDA wall cap is 1,800 seconds.

Every fitted case must satisfy residual <=1e-6 at four and eight passes and must
not regress against C53 at one or two. Only a failed C64 gate authorizes the two
subset fits. Passing does not promote a specialist; the real and manufactured
regression populations are separate guards. Record time, memory, numerical results,
source and weight identities, actual exposures and network/cell work. Retain failed
attempts without mutation replay. No model weights or full fields enter Git.
