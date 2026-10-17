# Next: qualify overflow-only clipping in a capped managed stability screen

The diagnostic identifies an outer float32 norm-reduction overflow at replay
update 1398. All gradient components and individual parameter norms were finite;
their true total 3.044506589e19 is representable. A prospective float32-scaled
repair passes arithmetic qualification, but is not installed or trained.

First add the narrow overflow-only repair at the original one_update clipping
owner. Keep the original finite Torch path byte-for-byte in behavior, so every
ordinary update retains its original clipping norm and gradient bits. Preserve
max_norm 1, epsilon 1e-6 and rejection of actual nonfinite gradients or an
unrepresentable final norm. Do not relax numerical or physical gates. The
experimental function currently supports only present, colocated float32 L2
training gradients; keep that supported contract explicit. Review exception
meaning and integrated qualification before capture/admission.

Verify the integrated seam on consumed actual fields, not just scalar norms:
same weights, gradients, Adam and CPU/CUDA RNG when ordinary norms are finite;
original guard errors unchanged for bad components; corrected finite clipping
at the minimized actual overflow. Record repair counts and exact norms, retaining
the original exception if the rejected operation is not the qualified norm
overflow. Do not broaden this into general Torch API compatibility.

Then admit one new 1500-update managed CUDA screen with the exact original 5000
configuration and all full-length streams. Only clipping arithmetic changes;
no gain, coarse learning-rate, loss, sampling or model change at the same time.
The retained 13 loss checkpoints should match exactly through 1300. The first
repair should occur at 1398 if the unchanged prefix is reproduced. Require 1500
successful optimizer/CUDA finite-state checks, and record each repair, remaining
loss behavior, elapsed time and memory. A labelled cap is not completed training
and produces no eligible final pressure model. Preserve journalled one-attempt
admission/closure, source/custody binding and original Tuldok/Rheon observation.

If actual nonfinite components appear or recurrent corrections remain extreme,
retain the failing sample/stage before proposing one controlled gain/rate or
feedback-denominator intervention. The failing direction stream had large
finite loss, so arithmetic recovery alone does not guarantee learning quality.
Do not infer a coarse-matrix-only problem: local CNN gradients dominate the
observed norm. Do not silently suppress small-amplitude or low-frequency error.

Only after finite integrated screen evidence should the next matched full CUDA
control/candidate run be considered. Restore on CPU, qualify the first frame,
then all 74 development trajectories / 233 independent checks and unchanged
population accuracy/call gates. Keep raw-forward and paired-worker timings
separate. Freeze one eligible model before any untouched-scene access; no
automatic promotion. Record failed attempts and all command/Run/loop clocks.

No repaired managed screen or additional fit has started in this experiment.
