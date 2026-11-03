# Prepared follow-ups

These steps are prepared, not running. The longer two-correction fit passes
all 34 development trajectories and 8 new final simulations but regresses worst
work/pressure error; the prior control remains preferred. The next accuracy
experiment should isolate mixed-odd coverage or intermediate feedback rather
than add updates alone. Keep all physical gates and failed attempts intact.

1. Review the longer two-correction control on the consumed 16, especially
   16x23x25, alongside the familiar/fixed-eight and fresh final results. If it
   improves robust qualification/work without pressure regression, repeat the
   same recipe across independent seed pairs before replacing the laboratory
   control. If the gap persists, change feedback or coverage in a separate
   controlled comparison rather than extending the inference cap.
2. Add an explicitly named intermediate residual-contraction objective as a
   separate trainer change. Current two-pass training already uses the first
   prediction's residual in its second pass and backpropagates through both;
   loss is measured on the final pressure. The new experiment should add
   feedback at intermediate corrections and train-only examples of later
   residual distributions, preserving the existing canonical-loss control.
   Record actual forward/backward work and elapsed time, including any extra
   operator/reduction work. Keep float32 neural values and current gates.
3. Introduce a versioned analytical-bank capacity contract before combining the
   old eight grids with the three qualified mixed-odd grids. Combined projected
   tensors are 642,528,256 bytes (612.76 MiB), below 768 MiB; 11 grids exceed the
   currently enforced 8. Preserve legacy tensor/order/evidence and RNG parity,
   exact historical configs and the rejected 11-grid proposal. Uniform sampling
   across 11 grids changes each old grid's exposure; disclose/control that
   sampling change and compare matched work. Do not silently replace old grids.
4. The prepared 32/64-real-parent cohorts provide more distinct shapes, but all
   have at most 16 cells per axis and neither gets closer to 16×23×25 by ordered
   count L1 than the original 16-parent cohort. A native CFD coverage expansion
   should deliberately include larger mixed odd rectangles with new force seeds,
   centers, radii, swirl and fluctuations, preserving cubic h and sealed water.
   Use multiple post-warm-up frames per whole simulation. Keep final simulations
   untouched and separate by whole scene. Count expansion also changes fitted
   normalization/amplitude under the current trainer: control or explicitly
   separate that effect. See reports/prepared-cohort-coverage.json.
5. Axis permutations can augment analytical/native training fields only with
   all related fields, labels and dimensions transformed consistently. Cubic
   scalar spacing is invariant; cell-zero gauge is preserved by permutations.
   Reflections require regauging and are a separate implementation decision.
6. A residual-minimizing step-size policy remains a separate CPU inference
   experiment, with the same 32-call budget and physical gates. Record its extra
   operator/reduction cost and actual network evaluations. It must not be called
   a checkpoint improvement or credited with teacher/native pressure cleanup.
7. Profile the managed preparation/submission/download intervals and CPU
   convolution separately from learning. Benchmark isolated matched-resource
   inference before speed or resolution-crossover claims. More GPU utilization
   or simultaneous fits require a separate throughput experiment; this control
   intentionally uses one optimizer. CPU preparation can overlap future fits on
   disjoint cores, with bounded independent workers rather than excessive
   threading inside each small convolution.

Reserve new whole final simulations before any subsequent fit and freeze
selection before testing them. This run's final population becomes consumed
once evaluated. Deployment into Rheon and broader physical/boundary contracts
remain separate work.
