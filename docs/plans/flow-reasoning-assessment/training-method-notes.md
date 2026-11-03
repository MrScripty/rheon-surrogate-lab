# Training-method verification and pressure-solver adaptation

Checked 2026-10-18 against the included v3 PDF and the authors' live sources.
This note proposes an experiment; it does not implement or launch training.

## What Fixed-Point Forcing actually changes

For the paper's categorical denoiser `D_theta(x, t, condition, carry)`:

1. Draw `t`; form `x_t = (1-t)*noise + t*target`.
2. Draw `t_start ~ Uniform(0,t)`; form `x_start` with the same noise and target.
3. Run self-conditioned integration from `t_start` to `t`; detach its final clean prediction.
4. Supervise `D_theta(x_t, t, condition, detached_prediction)` against the original target with endpoint cross-entropy.

The rollout changes the carry distribution, **not the supervised interpolant**.
The rollout evolves flow state, time and carry together; it is not simply
held-time fixed-point iteration. Table 3 reports rollout probability/depth
`0.5/16`, not a randomly sampled recurrence depth. Stage B starts from selected
Stage-A self-conditioning weights with a fresh optimizer and EMA.
The paper leaves the integration helper abstract and does not fully specify
its time grid or the interaction of the two reported conditioning probabilities.
[Included paper, §§2–3, Fig. 4, Appendix A.2/Table 3](flow-reasoning-models-2606.29150v3.pdf)
([versioned source](https://arxiv.org/pdf/2606.29150v3)).

The authors' repository still contains README/assets and announces an upcoming
implementation; there is no released training code to resolve those details.
[Authors' repository, checked 2026-10-18](https://github.com/helblazer811/Flow-Reasoning-Models).

## What transfers to our pressure solver

The transferable hypothesis is **learning a useful next correction from the
current model's own detached intermediate states**. Our existing inference
already computes

```text
r_k = r_0 - A(p_k - p_0)
p_(k+1) = step_theta(p_k, r_k, physical_metadata)
```

Here `step_theta` includes the captured feature preparation, normalization and
pressure reconstruction, not just the neural network call.
[Inference source](../../../experiments/pressure-pass-sweep-v1/inference.py).

An adaptation would choose a correction depth `k`, produce `p_k` under the
current weights without recording gradients, recompute `r_k` with the same
physical operator, and differentiate one more correction. Score its resulting
**total pressure** against the original physical problem. This exposes the
network to its own remaining error without backpropagating through its history.
It changes the input-state distribution and gradient horizon, whereas literal
FPF changes a separate carry channel. Call this detached rollout training
inspired by FPF; a faithful FRM reproduction would also require its flow state,
time input, separate carry and canonical denoising objective. That broader
architecture change would answer a different experimental question.

This is an interpretation of the paper applied to the local recurrence, not a
CFD claim established by the authors. The existing assessment makes the same
distinction. [Assessment](research.md).

## Details that can invalidate the comparison

- **Detach the history, keep the new correction differentiable.** Generate the
  prefix under `no_grad`, detach the resulting pressure and reconstructed inputs,
  then leave gradient recording enabled through the final network call and
  physical operator. Detaching the final residual would remove its training
  signal. This is a proposed gradient boundary; the existing objective instead
  differentiates through its shared two/four-pass chain.
  [Current objective](../../../experiments/pressure-four-pass-distillation-v1/fine_tuning.py).
- **Keep the original problem bound to the loss.** The model's local sample may
  use `base=p_k` and `native_input_residual=r_k`; the loss must retain `p_0`,
  `r_0`, the original target and physical metadata. Passing the modified sample
  into all existing loss code can silently change labels or denominators.
  Existing equation feedback divides by `||r_0||²`; changing this to `||r_k||²`
  heavily reweights late states and defines a different objective. Preserve the
  original denominator for the first controlled comparison.
  [Loss and normalization](../../../experiments/pressure-four-pass-distillation-v1/fine_tuning.py).
- **Match inference arithmetic.** Recompute the physical residual after every
  pressure update using the deployed boundary, gauge, precision, reconstruction
  and normalization conventions. Preserve the original residual scale used by
  the acceptance gate; a locally normalized residual alone is insufficient to
  reconstruct it. Match module mode as well: `no_grad` does not disable dropout
  or running-statistic updates. These are implementation requirements inferred
  from the captured inference contract.
  [Inference source](../../../experiments/pressure-pass-sweep-v1/inference.py).
- **Keep early and late states represented.** Define the sampled depths and
  mixing weights before training, including `k=0`; regenerate prefixes from
  current weights. A frozen C53 rollout bank tests exposure to C53 states, a
  different hypothesis. Sampled depth is our design choice, not a paper setting.
  Current frozen-teacher outputs are auxiliary targets rather than current-model
  state generation. [Existing teacher](../../../experiments/pressure-four-pass-distillation-v1/fine_tuning.py).
- **Separate state exposure from gradient truncation.** Replacing unrolled
  training changes both. To attribute an effect to later-state exposure, compare
  matched-depth differentiable prefixes, matched-depth detached prefixes, and
  variable-depth detached prefixes, or explicitly report that the effects remain
  confounded. Count actual prefix forwards alongside optimizer updates; matching
  updates alone does not match training compute.

## What success would mean

For the schematic map `p -> p + B_theta(r)`, a true solution requires
`B_theta(0)=0` (up to pressure gauge). That condition is insufficient: a nonzero
residual with `B_theta(r)=0` is a stalled, incorrect fixed point. Locally, for a
linear pressure operator, error propagation is approximately
`e_(k+1) = [I - J_B(0) A] e_k`. Contraction on the relevant pressure subspace,
particularly its slow modes, is the useful property; repeated predictions that
agree do not establish it. These are deductions from the recurrence, not
convergence guarantees supplied by FPF.

Measure the physical residual after each pass and the residual gate at four
passes, retaining first-pass and eight-pass regressions as checks. Also inspect
low-mode pressure errors: smaller equation residuals can conceal large pressure
errors along small-eigenvalue modes. Keep pressure gauge handling consistent.
Training on later states is a plausible way to improve these corrections; it
does not establish that exposure mismatch caused the existing low-mode tail.
[Existing objective and gauge treatment](../../../experiments/pressure-four-pass-distillation-v1/fine_tuning.py),
[retained C53 diagnosis](../pressure-four-pass-distillation-v1/C53-diagnosis.json).
