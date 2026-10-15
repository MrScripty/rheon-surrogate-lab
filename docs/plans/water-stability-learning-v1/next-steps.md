# Prepared follow-ups; not executed

Make the inference budget explicit before selecting a deployment candidate.
The successful policy currently costs 32 network evaluations per pressure solve,
all on CPU; these corrections do not disappear outside training. Four-evaluation
predictions still fail the physical gates. No one-evaluation result is qualified.

1. Test one network evaluation with the successful analytical training feedback.
   Reusable configurations are stability-modes-single-pass.json and
   stability-combined-single-pass.json in the existing configs directory. They
   change only correction.steps from four to one in their corresponding controls,
   preserving 5,000 updates, seed, real-frame sampling, objective, modes and scales.
   Compare equal real/analytical exposure and report the lower compute separately.
   Neither configuration has been captured or trained. Verify fresh CPU inference,
   known-frame fit, all development physical checks and actual native trajectories.
   A failed one-frame fit should trigger capacity/optimization investigation before
   a much larger data run. No one-pass success is promised.
2. As a separately labelled iterative alternative, investigate adaptive stopping
   and/or fewer proposals using the existing 1e-6 equation and physical gates.
   Stored-state equation checks first pass after 2–7 proposals, typically three;
   this is a compute opportunity, not proof of physical acceptance or a speedup.
   Measure independent native gates and full CPU cost. Do not call adaptive or
   repeated inference a one-pass predictor or hide native refinement in AI success.
3. The 16-simulation/32-state varied cohort remains prepared in the previous
   tranche. Broader training is a follow-up after the chosen inference budget
   and fitting/stability criteria are credible. Current auxiliary bank generation
   deliberately accepts one fitting frame; multiple grids require an explicit
   train-only sampling/normalization contract and bounded cache. Then compare
   several seeds, larger populations and reserve a new untouched final test.
4. Extend temporal/generalization evidence under native-qualified settings.
   The present 4-second windows are useful but short. The cube's fixed timestep
   reaches the native Courant limit at step242. Longer tests need a separately
   qualified timestep/forcing configuration, preserving the physical limits.
   Cover new domain sizes, force histories and cells-per-metre values with cubic
   spacing. The current contract excludes obstacles, free surfaces and variable
   density; do not infer support from the current results.
5. Qualify an explicit one-channel deployment contract before UI integration.
   Preserve supplied previous-pressure history/warm-up, scalar cubic spacing,
   physical residual preprocessing, selected output scale, neural dtype and
   counted inference policy. Coordinate Rheon integration separately from model
   experiments; its existing seven-channel deployment has not been replaced.

Historical captures, failed fits, native control failure, CPU timings and all
managed artifacts remain retained. Future experiments get distinct IDs and
source captures; do not rewrite a successful multi-pass result as single-pass.
