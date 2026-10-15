# Prepared follow-ups; not executed

Use the qualified single-pass-trained checkpoint with explicit adaptive
iteration as the current development reference, retaining the earlier checkpoint
and policies as controls. Both completed fits miss the gates in one evaluation;
the new iterative policy passes development and trajectory checks. The higher-rate
combined fit failed numerically. Broader training must name its inference budget
and qualify fitting at that budget. No one-pass deployment candidate is qualified.

1. Isolate stopping granularity from learned weights: the latest comparison
   changes both. Compare each checkpoint with the same per-evaluation stopping
   rule before attributing the gain to training alone. Then profile actual CPU inference on the measured grid sizes. Separate convolution,
   physical residual updates and IPC from native whole-track work. Investigate
   the measured 24-cubed latency increase and compare suitable CPU thread counts
   and layouts with matched resources and all physical checks. The current
   benchmark deliberately uses one core per track; it does not establish the
   fastest possible multithreaded implementation. Retain accuracy versus latency,
   rather than attributing all cost to parameter count.
2. Test train-only analytical feedback across several grid sizes before adding
   many CFD trajectories. Four pooling stages turn the training 16-cubed grid
   into a 1-cubed bottleneck; 32-cubed becomes 2-cubed and 40-cubed 3-cubed.
   Distinct coarse spatial features are absent at the fitting shape. This is a
   reason to test multi-grid feedback, not proof of the cause of current errors.
   The existing analytical bank deliberately accepts one fitting frame; new
   multi-grid sampling, units, normalization and cache budgets need an explicit
   contract and independent eigenvalue/operator checks. Preserve the existing
   single-grid trial exactly as its control. See reports/pyramid-scaling.json.
3. Compare a small fixed network-evaluation budget, such as two, with the current
   four-evaluation training and the failed one-evaluation fits. Use a controlled
   stable optimization schedule; retain equal real/analytical exposure and total
   neural compute separately. First show better known-case fit and error reduction
   per evaluation. If that stalls, investigate capacity/global coarse correction
   with a separately reviewed architecture intervention. More parameters alone
   are not evidence of faster or more accurate pressure solves.
4. Expand to the prepared 16-simulation cohort after fitting is credible at the
   chosen inference budget. The existing cohort varies shape and force histories
   but uses the retained small-grid capture contract. Larger-grid CFD training
   needs a separately qualified dataset specification and bounded capture/view
   budgets. Several seeds, new force setups and an untouched final population
   must distinguish learning progress from development-set tuning.
5. Extend trajectories and deployment only under native-qualified settings.
   Current windows are short; higher resolution can require timestep/force
   changes to keep native controls valid. Preserve cubic spacing, physical gates,
   previous-pressure history and explicit inference counts. Model input support
   beyond 65,536 cells, obstacles, free surfaces and variable density remains
   outside the current contract. Coordinate the separate Rheon UI deployment
   rather than inferring it changed from laboratory diagnostic success.

The new sequence's method, unsuccessful fits, adaptive policy, repeated timings
and all managed Run custody are retained alongside historical results.
