# Methods and reproduction

This continuation starts from the retained eight-grid shape-feedback checkpoint.
The sequence configuration names all packets, runtime, training recipes and
physical studies. There are still 16 real training simulations/32 states plus
1,024 manufactured examples on eight grids. Cubic spacing is scalar; physical
extents follow grid counts. Density1000 kg/m³, viscosity0.001 Pa·s, timestep1/60 s,
sealed free-slip walls and the existing model/native gates remain unchanged.

Two new initialization/sampling seed pairs test repeatability. The one-correction
5,000-update control keeps the original initial parameters and real sampling
order. The longer one-correction10,000-update fit doubles draws and moves the
learning-rate reductions to 6,000/9,000 updates. It approximately matches the
old two-correction forward work, not backward work, optimizer work or duration.
The network has 370,016 trainable float32 parameters. No model/objective change.

Scientific runtime:
`/media/jeremy/OrangeCream/Linux Software/repos/owned/ai-systems/Pumas-Library/torch-versions/v2.14.0/venv/bin/python`
(Torch2.14.0+cu132, NumPy2.5.3, Python3.14); GPU RTX5090 Laptop;
CPU Core Ultra9275HX,24 logical cores. The lab .venv runs authenticated Tuldok
HTTP orchestration. The existing catalog owns execution, approvals and custody.
No alternative catalog, dependency installation or secret persistence is used.

Source is committed before assembly. A clean detached temporary worktree avoids
touching unrelated untracked user files; its head remains on the retained branch.
For each trial, execute the existing conditioning pipeline assemble with the
sequence's packet/validation/reserved/runtime/Tuldok/config paths and a fresh
runs/water-shape-repeatability-v1/TRIAL/project output. Capture logs, exact argv,
zero-residual qualification, timings and copied-file parity are in capture.json
and per-trial assembly-timing.json; the small capture receipt is committed.

Execute the prepared sequence:

```bash
.venv/bin/python experiments/water-shape-repeatability-v1/run.py \
  --sequence experiments/water-shape-repeatability-v1/configs/sequence.json \
  --torch-python "$TORCH_PYTHON" \
  --managed-controller /tmp/run-shape-managed.py
```

The environment adapter delegates to water-conditioning-v1/training/workflow.py
using the already attached local owner session,600-second HTTP deadline and normal
preflight/approval/submission/Run/download/reopen. Its exact source/hash is retained
under ignored runs/, alongside the existing authentication adapter (no token values).
Managed Runs remain Tuldok-owned after any controller failure; reconcile before
any retry. Each selected stage must complete/drain and download identically twice.
CPU restore must preserve weights and initialize no CUDA context.

qualify.py delegates to existing multigrid evaluate.py on all 32 fitting states
and 128 familiar development states with adaptive cap 32; fixed 8 is a separate
policy. mode_probe.py supplies120 signed analytical cases, not additional CFD
simulations. Existing study.py/rollout.py own fresh physical trajectories,7 prior
grid cases and 3 lower-force512-step trajectories, with native controls first and
15 native pressure-history steps before AI. Each actual network call is counted;
no native cleanup qualifies an AI result. Raw managed output is checked separately
and remains distinct from adaptive corrected pressure. Raw fitting outputs cover
at most12 retained predictions, not all 32 fitting states.

CPU qualification uses cores1–23 and exclusive pool ownership, overlapping later
CUDA fitting. Native diagnostics use up to 23 single-thread workers; independent
trajectory cases use 8,7 or3 workers. The observer joins at sequence completion;
existing Tuldok and unrelated processes stay alive. Root activity is visible in
the registered project. Managed and numerical errors preserve original logs and
terminal outcomes; a scientifically unqualified trajectory is still a result.

Every delegated command retains UTC start/finish, monotonic elapsed seconds,
exact argv, PID, return code and reaping. capture.json records preparation;
execution.json records the managed/CPU sequence's elapsed makespan; each trial
has managed-timing.json and qualification-timing.json. Optimizer time comes from
training history; Run timestamps separately expose queue and execution intervals.
These intervals overlap: do not add CPU job times to claim elapsed pipeline time.
Concurrent timings establish throughput only. Isolated native-versus-AI inference
latency, its distribution and a resolution crossover remain unmeasured here.

Existing multigrid record.py seals each trial after complete managed/CPU custody.
The continuation summarize.py derives comparisons from those immutable records.
trajectory_summary.py retains pressure drift, actual terminal frames and hard
stops for fresh/prior/long/final populations. Final configurations were committed
before fitting. Select only from development evidence, freeze identities before
final execution, retain all failures and never reselect/retrain from final results.

Additional exploratory CPU work uses the same native controls/15-step history
on 65×9×7,7×65×9,95×7×5 and 127×5×5 domains, within existing native bounds.
It is outside the predeclared selection gates. The selected challenger is
evaluated on these reused development cases after selection freezes.
precision_probe.py reuses audit.spectral_correction only to diagnose ideal
F64/rounded-F32 corrections on actual learned inputs; original learned outputs
alone advance the simulation. Native physics/error-frame parity confirms the
probe does not refine or change the learned trajectory.

The proposed11-grid bank is rejected before optimization by the enforced
eight-grid limit. Its exact config, failure and timing are retained. A separate
three-grid auxiliary bank passes 384 operator-label checks with no optimizer or
CUDA initialization. CPU preparation also audits32/64-parent lossless views
from existing native captures, without reserved/final use. Neither prepared
input replaces current training data. Full preparation commands, successful
and rejected timings and bounded-failure dispositions are retained.
