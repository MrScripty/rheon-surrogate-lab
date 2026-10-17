# Proposed next comparison: current-policy late correction states

Prepared only; no new admission or fit. Weight0.03 is a development lead, not a
promoted model. Preserve all existing population/physical gates and the unused
eight whole scenes. The goal is lower worst-pressure error without losing its
call improvement.

Existing training already includes a late-direction bank from a frozen older
reference, independently solved and rescaled to train-correction RMS. Repeating
that experiment is not the proposal. The new question is whether states reached
by the current0.03 policy at later calls, at their actual small amplitudes,
expose a mismatch that the existing two-correction objective and scaled bank
miss. The selected scene73trace motivates investigation; it is development
evidence, never a new training parent or label source.

First prepare a bounded train-only diagnostic. Freeze current0.03 checkpoint;
use only the original16training parents/32states and already admitted analytic
patterns. Record prefixes2/4/8/12, compatible residual, exact remaining correction,
physical spacing and amplitude/contraction statistics. Verify the pressure
operator identity and independent oracle. Process one grid/state at a time with
bounded caches; do not materialize a state-by-mode tensor. Count generation
network calls, CPU/GPU time and peak memory. Keep held-out development and unused
eight out of sampling/normalization. Compare weak/strong mode errors on common
RHS states before attributing any effect to noise or frequency.

Only if those diagnostics support a distinct learnable target, freeze a matched
CUDA pair: same initialization,5000updates, seeds, base samples, existing losses,
weight0.03, architecture and supplemental draw/cell budget. Compare extra early
versus current-policy late-state supervision, retaining native amplitude and an
explicit finite numerical guard. Declare the supplemental weighting/normalization
before results, and show that both early and late exposure are retained. Avoid
long backpropagation through12passes: frozen prefix states can be generated
outside the gradient graph. An extra early arm controls extra training compute;
report actual forward/cell counts and elapsed time rather than equating update
count with equal work. Do not reuse compare_training's identical-input assertion
for intentionally different banks; bind the allowed difference explicitly.

Qualify the altered target/loss and one-frame path before full fitting; CUDA
training and fresh CPU restore remain required. Then all74consumed development
scenes, same-RHS oracles, each population's pressure/mean/max-call gates, fixed-call
manufactured tests and repeated serial timing. Freeze an eligible package before
any unused whole-scene validation. Reject honest tradeoffs; do not relax gates or
add native refinement to obtain a passing model. No new simulation campaign is
needed merely to generate algebraic within-step residual prefixes.
