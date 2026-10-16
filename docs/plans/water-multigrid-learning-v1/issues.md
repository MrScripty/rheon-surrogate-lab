# Issues and dispositions

- Resolved: checkpoint and stopping granularity were confounded. Matched stopping
  measures a6.22% gain from the previous weight change; the larger prior number
  included a policy change. Retained controls remain unchanged.
- Supported hypothesis, scoped: multi-grid analytical feedback improves large-grid
  correction work. One-evaluation training regresses small-grid mean work; two-
  evaluation/cohort results improve the tested development mean/tail. Voxel work,
  population and normalization changes remain explicit confounders.
- Retained negative results: raw one/two-evaluation outputs remain unqualified;
  fixed eight evaluations pass46/128 for the cohort versus11/128 previously.
  Adaptive policy is needed for full tested acceptance. No relaxed target.
- Retained tradeoff: pressure-field trajectory drift is mixed despite unchanged
  equation/physical passes. Cube maximum relative L2 worsens versus the reference;
  rectangular/compact maxima improve. Finite8.5333s windows do not prove stability
  at longer times. Force strength is explicitly reduced for these native controls.
- Retained performance limit: native remains faster in all measured paired tracks.
  Four Torch threads help this workload; channels-last microprobes show mixed
  results. Kernel/layout, GPU utilization and Tuldok preflight/custody optimization
  are separately owned future work, with source/physics verification preserved.
- Resolved operational setup: controller environments lacked dependencies until
  using the existing lab .venv; profile.py shadowed Python's stdlib; changed project
  snapshots needed a new ID. Failed startup/HTTP logs remain retained and are not
  counted as failed CUDA fits. A system-Python scheduling-test invocation lacked
  NumPy; rerun with the qualified scientific interpreter passed.
- Resolved measurement visibility: fixed-core scheduling now streams observations
  as each episode completes, with exclusive leases for candidate/native jobs.
  Independent scheduling/failure tests and real execution fixture pass.
- Deferred physical/capacity scope:65536 cells, sealed uniform water, no obstacles,
  free surface or variable-density operator. New final populations, longer/native-
  qualified conditions, larger capacity and Rheon deployment need their own
  scoped qualification. Prepared follow-ups are not additional running jobs.
