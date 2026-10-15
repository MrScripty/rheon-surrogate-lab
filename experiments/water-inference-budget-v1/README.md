# CPU inference budget and scaling

See the canonical [plan](../../docs/plans/water-inference-budget-v1/plan.md).
Training reuses the two already committed single-pass configurations and the
existing Tuldok workflow. No checkpoint, dataset or private catalog is added here.

`evaluate.py` accepts explicit experiment root, trial, fitting/development packets
and native binaries. It requires a completed managed full sequence and retains
exact CPU command arrays, known-case fit, development diagnostics, independent
native evaluation and three raw trajectories. It performs no optimization.

The existing `learned_iteration.py --adaptive-profile configs/adaptive.json`
predicts from a fresh managed CPU restore. Maximum proposals are explicit;
recorded per-case work is distinguished from the maximum. Original RHS comes
from the supplied residual plus A(previous pressure), never a pressure label.
Residual and predicted divergence stop corrections; independent native checks
still decide physical qualification. Budget exhaustion is a retained failure.

`scaling.py` accepts captured project/restore paths, native binary, authored
study configuration, adaptive profile, output and CPU. It runs serially after
task training/audits finish. Three repetitions alternate fixed/adaptive ordering;
15 native steps supply history and the first AI step warms network execution.
Remaining seven steps supply each repetition's timing samples. Native controls
precede candidates. The two sweeps change either cells/metre at fixed physical
volume or slice counts at fixed spacing. The baseline 16-cubed scene is shared.
Force parameters/seed are matched across scenes; force support is defined by
the existing engine relative to physical domain extents. The report retains
physical failures as well as timings. No crossover is claimed for failed cases.
Existing native/AI residual profiles are 1e-8/1e-6 respectively, so this is a
comparison of supported tracks, not identical equation tolerances. Native
pressure-kernel time is not isolated; neural callback time and whole-track time
are recorded separately. CPU runtime overhead is part of the latter.
