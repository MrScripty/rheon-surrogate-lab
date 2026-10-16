# Prepared next experiments

These follow-ups are not running. Keep the previous two-correction checkpoint
as the qualified control. Retain the one-correction10,000-update checkpoint as
a useful development result, with its final failure attached.

1. Address the 16×23×25 convergence failure before claiming broader improvement.
   The current 16 final simulations are now consumed development cases; reserve
   new whole final simulations before another fit and freeze selection again.
   Rank development candidates using worst-shape convergence, autonomous work
   and pressure drift alongside physical gates. Familiar mean work alone chose
   a checkpoint that generalizes worse. Do not test alternate checkpoints on
   this same final population and call that untouched validation.
2. Run a two-correction10,000-update control using existing supported settings,
   scaled learning-rate schedule and unchanged data/scales/initialization. It
   tests whether more optimizer updates preserve two-correction robustness.
   Compare equal-update and equal-forward-work interpretations separately; keep
   the 600-second managed Run cap. Repeat any qualified improvement across seeds.
3. Test explicit intermediate/residual contraction feedback on training-only
   examples, including residuals after the model's own corrections. The failing
   final case contracts monotonically by a median factor0.7574 per call, which
   is too slow for 32 calls. Target these slow error modes and mixed odd-domain
   geometries, rather than enlarging the inference cap or relaxing the target.
   Test axis permutations as physically valid cubic-grid augmentation, preserving
   operator/label/gauge consistency. Treat any objective/sampling change as its
   own controlled comparison.
4. Prepare broader operator coverage around mixed odd rectangles and elongated
   shapes while protecting existing cube/odd regressions. The 3-grid auxiliary
   bank is already qualified;32/64-real-simulation views are already audited.
   Current banks enforce at most8 grids. Combining extra grids needs a reviewed
   trainer/configuration-contract change with resident-budget and legacy-parity
   evidence; retain the rejected11-grid proposal. Never silently replace the
   successful historical bank or confound count, normalization and objective.
5. CPU-only investigation can separately test a named residual-minimizing step
   policy while CUDA trains. Record its extra operator/reduction work, actual
   network calls and physical gates. It is a distinct inference policy; preserve
   the current pure learned-iteration control, same thresholds and 32-call cap.
   Qualify it on fresh final cases after policy selection freezes.
6. Profile Tuldok preparation/approval/submission/download costs and CPU tensor/
   convolution work separately from model learning. Retain full pipeline,
   optimizer and qualification elapsed times. Use isolated matched-resource
   inference measurements before claiming speed or a resolution crossover.
   Continue independent CPU preparation alongside exclusive CUDA training.
7. Larger/longer native contracts and application deployment remain separate.
   Preserve cubic spacing,15-step history, float32 neural values, native physical
   limits, explicit fallback and visible comparison. No inference benchmark or
   arbitrary-resolution guarantee follows from this finite experiment.
