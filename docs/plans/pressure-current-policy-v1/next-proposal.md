# Proposed next experiment

Status: Proposed, not admitted. No further training has started. Preserve all
current results, rejected candidates, the failed publication attempt, and the
unchanged selection gates. The eight unused scenes remain unused.

The late-state arm improved median local pressure error and overall average
calls, but lost accuracy against the prior accurate model on original58 and
minimum8-failed8. Its hardest call count rose to 15 versus the retained 0.03
model's 11. The early-state arm made the opposite tradeoff. This supports testing
whether the new current-policy stream has too much weight before changing the
architecture. Improvement is a hypothesis, not an assured next result.

1. Freeze two smaller weights for **current_feedback**: 0.01 and 0.03, both selecting
   the same native late bank02. Keep the original lowband weight 0.03 and every
   other stream, initialization, seed, sampling sequence, optimizer schedule,
   5,000 updates and physical contract unchanged. This is a different parameter
   from the previous lowband-weight ablation. Do not regenerate the bank or add
   data in response to development outcomes.
2. Capture and verify both configurations through the existing reusable Project
   assembler. Each arm should execute the same 50,000 network forwards and cell
   budget. Use CUDA training, then exact fresh CPU restore and first-frame
   qualification. Retain the current 0.1 late arm as measured context; the older
   no-current-stream0.03 candidate has a different compute budget and must be
   labelled accordingly.
3. During GPU fitting, use bounded CPU work on existing training states to compare
   frequency transfer at fixed call counts, retaining absolute spectral energies
   and pressure, gradient and equation errors. Include high-to-low leakage and
   per-case tails. Keep this preparation separate from admission, loss choices
   and the unused scenes. Limit parallel CPU pools by available memory and keep
   the host reserve; measure full pipeline time because CPU/GPU overlap can also
   affect the GPU's wall time on this laptop.
4. Qualify each candidate on all 74 development trajectories with 233 independent
   checks, then apply the existing pressure/mean/max-call gates independently to
   every population against all three historical comparators. Record clocks,
   memory, call distributions, manufactured results and serial timing. Freeze
   one package before touching the unused eight scenes only if every gate passes.
   No automatic promotion, threshold relaxation or native refinement.

If smaller weights keep the same pressure/work tradeoff, stop scalar-weight
sweeps and propose a spatial global correction path. The CPU probe found that
this model's global-average shortcut contributes a constant removed by pressure
gauge in exact arithmetic; the observed remaining changes are consistent with
float32 rounding. The spatial encoder/decoder still communicates information.
A later architecture experiment should preserve nonconstant coarse pressure
structure, with learned outputs and CPU cost measured explicitly. It must not
be presented as a proven fix or substitute an oracle/native solver at inference.
A shortcut-removal control would need fresh qualification because float32 output
is not bit-identical even when its exact-arithmetic contribution is constant.

A separate performance proposal may examine Tuldok artifact readback: successful
Study wall time substantially exceeds optimizer time. First profile and reproduce
that overhead; keep any custody/verification optimization separate from model
changes and preserve exact source/input/output validation. Do not weaken those
checks to gain throughput.
