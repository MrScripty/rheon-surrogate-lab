# Ranked predictions before probes

The minimized real pressure solve reproduces the regression: identical sample
SHA, control 11 calls, feedback 28 calls on two replays. Removing 112 later
simulation steps preserves the symptom. Native warm-up remains 15 steps; one
AI solve remains. Replay command is diagnose.py replay with the captured sample,
checkpoint project/CPU artifacts and --maximum-calls 13. Feedback exits 1,
control exits 0, in about two seconds. Expected red results are diagnostic
success, not failed lifecycle or a claim that training has fixed the model.

1. Shape-dependent residual directions dominate. Prediction: fixed-call residuals
   retain coherent low/mixed cosine modes and independent manufactured modes on
   the hard shapes contract more slowly for feedback than control. Extra targeted
   operator examples could address this without an inference-policy change.
2. Axis/odd-grid alignment matters. Prediction: cyclically permuting all spatial
   fields and dimensions, preserving cubic h/gauge, changes fixed-call ratios
   materially; single-axis mode responses identify the differing axes.
3. Reconstruction/precision dominates. Prediction: using a zero base with the
   identical correction residual, or scaling that residual by -2 or 0.25, changes
   normalized fixed-call residual curves materially. Near-invariance would reject
   this explanation. Adaptive call counts with a changed RHS norm are not directly
   comparable; this probe uses fixed calls and original-residual normalization.

All probes are CPU diagnostics on consumed cases, zero optimizer updates, no
CUDA, no changes to deployment or final-selection policy. No global contraction
proof follows from a finite mode/axis population. The bounded stopping condition
is choosing targeted coverage versus a different training mechanism.
