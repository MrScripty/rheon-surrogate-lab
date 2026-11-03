# Next proposal: learn a spatial coarse pressure correction

Stop scalar current-feedback weight sweeps. Neither smaller weight met the
unchanged population gates, and isolated amplitude scaling does not explain the
remaining low-frequency error. The next controlled change should test whether a
learned, spatially varying coarse correction improves long-range pressure response
at fewer network calls. No further training has started.

1. Establish a control using the retained current-0.1 initialization and training
   recipe. Audit the existing global-average branch: in exact arithmetic its
   constant output is removed by pressure gauge subtraction. Existing local
   encoder/decoder paths still communicate spatially. Removing the constant path
   can change float32 rounding, so compare actual outputs and qualify that change
   separately rather than assuming bitwise equivalence.
2. Prototype one small, nonconstant coarse spatial mixing path. Keep the existing
   nonlinear encoder/decoder behavior (including SiLU and GroupNorm) fixed; test
   a linear, bias-free coarse path as the controlled addition. Preserve arbitrary
   supported grid dimensions, cubic spacing, fixed scalar physics metadata and
   CPU float32 inference. Decide the
   exact architecture after measuring shape handling, parameter/file size and
   memory. Do not add a native or spectral inverse solver at inference.
3. Before expensive fits, use already consumed analytic fields and training RHS
   pairs to measure amplitude linearity, low-mode gain, high-to-low leakage,
   pressure gradients and equation error at 1/4/8/12 calls. Include rectangular
   grids, pure/mixed amplitudes and all diagnostic tails. Measure standalone CPU
   inference separately from the existing paired comparison branch, whose timer
   includes an extra native diagnostic solve. Run bounded CPU preparation while
   the GPU is occupied, with sequential broad pools and retained reserve checks.
4. Freeze a matched control and one architectural candidate before CUDA capture.
   Keep the native bank, seed, initialization rule, update/forward-cell budget,
   other loss weights, physical arithmetic, stopping and evaluation rules fixed.
   Record CUDA optimizer/trainer/Study clocks separately, memory and artifact
   hashes. Fresh CPU checkpoint restoration and first-frame qualification precede
   all broad trajectories.
5. Evaluate all 74 development cases and 233 independent checks per candidate.
   Apply every original population pressure, mean-call and maximum-call gate.
   Retain physical failures, tails and cost even if average accuracy improves.
   Only after one package meets every gate should it be frozen for the eight
   unused whole-scene holdouts. No automatic promotion or threshold relaxation.

This is a hypothesis about spatial communication, not a promised improvement.
If fixed-call diagnostics show no benefit or CPU cost dominates, reject the
prototype before another large sweep and retain the negative result.

A separate Tuldok performance investigation can profile the repeated parent
validation implicated by the retained one-byte artifact probes. Establish a
function-level baseline first. Any request-local reuse must preserve current
authority/revocation checks, artifact hashes, lineage/cycle checks and bounded
memory. Keep that application work on its own branch; no cross-request cache or
weakened custody is part of this model proposal.
