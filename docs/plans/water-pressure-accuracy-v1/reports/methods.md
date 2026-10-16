# Method and evidence boundaries

Configuration: experiments/water-pressure-accuracy-v1/configs/study.json.
Two source-bound, previously managed CUDA checkpoints, restored on CPU, are
selected by reusable project/artifact references. Their existing receipts and
source-witness inventories establish checkpoint identity; no historical hash
or result was replaced. summary.json retains the exact identities used here.

Three consumed scenes are the prior pressure witnesses: 31×11×7 (96 steps),
24×16×12 (512 steps), 31×15×23 (128 steps). Preserve seeds, force properties,
cubic h=1/16 or 1/8 m, water rho=1000 kg/m³, mu=0.001 Pa·s, dt=1/60 s,
origin zero, filled connected sealed/free-slip boundaries, SGS off and 15 native
warm-up steps. No emitters, geometry masks, free surfaces, material variation,
GPU inference or application-adapter migration are introduced.

First qualify 15 native warm-up frames plus one AI frame. Capture base,
post-force residual, scalar h/rho/dt before the actual corrector. Replay both
checkpoints on identical byte-bound arrays, using the existing inference owner
with one actual network evaluation per proposal. Capture first/prior peak/last
inputs on each original-policy trajectory: 18 records, 15 unique arrays because
each scene's two first inputs share native warm-up. Replays have 0..32 call
frontiers; zero is a negative no-correction control, not a network invocation.

The independent cosine pseudoinverse uses the sealed uniform Neumann spectrum
lambda=2−2cos(pi*k/n), excludes the constant null mode and anchors cell000.
The existing independently enumerated-neighbor action qualifies its correction
at relative defect <1e-9; observed worst is 2.22164e-13. The oracle operates on
the exact same residual and base supplied to the model. It is an analysis target
only, never a submitted cleanup pressure or learned success. No captured native
teacher pressure is used as an exact late-residual label.

Pressure relative L2 uses the cell-zero gauge. Mean-free pressure subtracts the
mean from both error and reference. Gradient error uses adjacent cell differences
(uniform h cancels the ratio). Error modes use orthonormal cosine coefficients
of mean-free pressure error; low modes means indices 0..2 on every axis. These
are finite sampled observations, not arbitrary-grid/operator-norm guarantees.

Separately run complete paired native trajectories for two checkpoints and
three stopping policies. Original: relative residual1e-6. Diagnostic strict-7/8:
1e-7/1e-8, unchanged absolute1e-12 and predicted-divergence1e-7 limits, same 32-call
cap and native physical checks. Previous pressure remains the actual preceding
pressure on that trajectory. No reference overwrite, native cleanup or fallback
gets AI credit. Paired pressure metrics compare independently evolving native
reference/AI states; they are not same-RHS exact solve errors.

Neural inputs/weights/outputs remain float32. Existing float64 physical operator,
base and correction accumulation remain unchanged. Many small float32 learned
corrections accumulated through that existing path can reach a much smaller
equation error than one rounded correction. This is not evidence of float32
full-pressure storage to double precision, and it does not change the accepted
1e-6 target or introduce higher-precision neural training.

Use existing bounded native startup/timeout/termination/reaping and timed-command
owners. Initial 18 trajectories run at18-way concurrency, one CPU/thread each;
two long strict8 cases time out at120s and are reaped. A separate2-way fresh
recovery with unchanged inputs/limits completes them. Replay pool uses24 workers;
all24 affinity cores execute cases. Replayed timings are concurrent diagnostic
cost, not an isolated benchmark or speedup versus Rheon's native algorithm.
Initial fixed task-to-core assignment could reuse a busy core in the second
wave; the retained controller uses explicit core leases for future invocations.

After numerical execution, review tightened controller cancellation to propagate
SIGINT through the existing command owner's cancellation event, cancel queued
work and drain active children. Two real-child lifecycle tests pass. The exact
earlier executed controller is archived as executed-controller.py.txt, with
hash parity checked; diagnose.py and all numerical owners are unchanged. Do not
attribute the initial scientific timings to the final controller implementation.

Scientific failures and setup rejections are in reconciliations.json. Full
arrays, callbacks, frames, failed sources and traces remain ignored local files.
Retained small JSON/CSV includes source/sample identities, per-step measurements,
timing boundaries and terminal child outcomes. No model weights/datasets/auth
secrets are committed. There is no fresh-final consumption or model promotion.
