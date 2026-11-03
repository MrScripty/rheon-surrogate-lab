# Generative velocity representation inside the full-state network

The completed matched loss comparison finds that increment-weighted MSE helps
learning: cold training velocity median falls from 0.8418 to 0.3664, and warm
training tracer median from 0.00190 to 0.000677. It still fails development and
240-step accuracy, with worse equation error. No temporal recipe qualifies.

Test one controlled representation change on the same K2 endpoint model.
Interpret its three velocity-output channels as learned edge potentials. A
fixed signed linear curl readout maps them to staggered face increments, with
tangential edge-potential wall masks. Add these increments to the current
velocity. This defines the neural output representation; it does not project a
separately predicted velocity or invoke a numerical pressure solve. Existing
nonzero input divergence is not corrected. Float32 physical interchange has
roundoff.

The curl readout is a fixed mathematical layer inside candidate inference,
just as geometry-dependent bases and physical reconstruction are fixed layers.
It is explicitly part of complete block cost. Pressure and tracer outputs,
all intermediate states and the original error criteria remain required.
The endpoint and intermediate decoder both use the same new representation.

The independent readout checks cover a four-face circulation impulse, exact
normal wall conditions, four-grid double-precision divergence cancellation and
small-grid matrix ranks spanning the closed divergence-free face subspace.
Maximum tested double divergence ratio is 1.29e-16. These are representation
checks, not prediction accuracy. Actual two-block CPU qualification also passes
all four grids and cold/warm starts, with identical initial tensors and initial
physical outputs to the direct-increment control. CUDA qualification must pass
before the managed fit is admitted.

Use the same 163,068 parameters, 10,000 updates, sample stream, optimizer schedule,
training-only scales, increment-MSE loss and two-block curriculum. The direct
increment-MSE fit is the comparison control. Only velocity representation and
its fixed readout arithmetic change; FLOPs are not matched. The managed Project
and sources are frozen, and completed fitting evidence admits this trial.

Evaluate every pressure, velocity and tracer state and the pressure equation
against native development references, including 12/48/240-step own-history.
All four worst errors must remain at or below 0.004. The two-state complete
block budget remains below 4 ms. Constraint satisfaction alone cannot qualify
a model. No new CPU latency, production result or reserved evaluation is claimed
from the prototype; all original objectives remain active.

[Frozen registration](temporal-curl-registration-01.json) · [Admission evidence](temporal-curl-admission-01.json) · [Structural qualification](../../../runs/full-ai-next-priorities-v1/curl-readout-representation-qualification-01.json)

## Completed outcome

The managed10,000-update fit, exact CPU restore, development evaluation and
comparison audit finish. Although the representation greatly reduces divergence,
worst small-grid single-block velocity error rises from0.8224 to1.2467. Every
full-state accuracy criterion still fails, including240-step own-history.
Constraint satisfaction did not resolve prediction learning. No recipe qualifies
for replication or reserved evaluation; no new CPU latency is claimed.
See [all measured outcomes](temporal-curl-results.md).
