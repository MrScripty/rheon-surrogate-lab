# Proposed controlled late-residual training comparison

Prepared, not implemented or running. This study performs zero optimizer
updates. Keep retained checkpoint 17b423b7… and the existing 1e-6 acceptance
target. CUDA owns future training; fresh CPU restore and native paired checks
own inference qualification. Do not promote a stricter diagnostic policy as a
checkpoint improvement.

The deciding observation is smooth remaining pressure error: the targeted
checkpoint removes most equation residual rapidly, but its remaining error is
concentrated in long-axis low modes. Extra calls remove it on the same input.
An appropriate next fit should learn those residual directions at useful
amplitude, instead of merely running more initial-state updates.

1. Prepare a frozen train-only early/late pair bank from the current targeted
   checkpoint's eight-call predictions. Use only the original 16 training
   parents/32 states and already admitted analytical training banks. Do not
   label the consumed pressure witnesses as training examples. Retain the
   original bank tensors/order and whole-parent splits. The existing 32 late
   labels are from the retained control; they cannot silently stand in for
   matching targeted-checkpoint labels or larger-grid directions.
2. Independently solve both early and late compatible residuals with the qualified
   sealed cosine oracle, checking its operator/gauge and the removed null mode.
   Match early/late shape, h, rho, sample counts and private sampling order.
   Include both real and analytical train-only directions so larger grids are
   represented without using development trajectories. Keep exact source
   checkpoint, physical units and per-case label provenance.
3. The bias-free linear network is homogeneous. Test that property at the
   selected amplitudes before rescaling each extra correction/residual pair to
   the original train-correction RMS. Record every gain and reject nonfinite or
   degenerate labels explicitly. Keep original normalization. This is a
   deliberate weighting of directions, not new fluid physics or a normalization
   refit. It prevents tiny late labels being lost under global loss scales;
   do not assume the present relative-loss floors suit these labels unmodified.
4. Implement a versioned, closed extra-bank selection in the existing trainer
   owner. Compare two matched 5,000-update CUDA fits from the same original
   initialization: extra early-direction control versus extra late-direction
   treatment, same weight 0.1 and one extra draw per update. Preserve the
   original real/primary/supplemental exposures, learning-rate schedule, seeds,
   two training corrections and intermediate loss. Both extra streams use the
   same amplitude rule and loss. This isolates the extra residual-direction
   effect. The unchanged targeted checkpoint remains an additional reference.
   Current v7 only admits analytical supplemental feedback; this new pair-bank
   path is not yet supported and needs parity/gradient/lineage evidence first.
5. Qualify one example/gradient before a managed full fit. One CUDA optimizer at
   a time, existing catalog/custody and fresh CPU restore, original per-Run
   resource bounds. Independent CPU work can prepare labels and qualification;
   choose concurrency from measured workload and retain timeout outcomes.
6. Compare fixed-call accuracy at 4/8/12 calls as well as the unchanged adaptive
   policy and 32-call cap. Measure exact pressure/gradient error on common inputs,
   low-mode contraction, all fitting/familiar cases and all 58 now-consumed
   standard development simulations. The 127×5×5 failure stays attached as
   exploratory evidence. Do not mix a stopping-policy change into this fit test.
7. Reserve a new whole-scene final population and commit its protocol before
   fitting; freeze candidate selection before a single final evaluation. The
   unused independent-seed confirmation population from the previous study
   remains reserved for its original conditional protocol. Require a complete
   accuracy/work/physical improvement before independent-seed confirmation or
   definitive promotion. No final-informed refitting or favorable-seed search.

The 16 larger CPU spectral-teacher water parents remain a separate data-expansion
experiment. Their 784 pressure frames are visible in Tuldok, but they still need
an explicit versioned packet/view contract, matched post-warm-up sampling and a
train-only scale audit before a 16-versus-32-parent fit. Do not combine that data
effect with the late-direction comparison.
