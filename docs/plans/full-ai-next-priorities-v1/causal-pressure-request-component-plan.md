# Causal pressure-request component before endpoint and feature reuse models

The learned pressure component now has adequate offline accuracy on the consumed
temporal scenes. Its remaining missing input is a pressure request predicted
from current physical state and known forcing. This component study tests that
learning problem before integrating complete velocity/tracer outputs. It does
not replace the requested K=2,3,4 full-state comparisons or their 240-step tests.

The prepared view retains all 16 original training scenes and all 129 states,
with lossless float32 velocities and float32 q targets. Its 213,993,244-byte
archive fits the existing managed input budget. Maximum q target conversion
relative error is 3.74e-8. Future q is supervision only; current pressure, future
pressure and future reference velocities are excluded from component inputs.

The new local network uses learned linear filters, learned products and
absolute-value products to represent local velocity/forcing interactions.
Geometry and h, dt, rho and viscosity supply unit conditioning. All spatial
coefficients are trained; there is no explicit divergence assembly or pressure
solve in its forward path. The parameter counts for K=1,2,3,4 are
3,936 / 7,488 / 11,952 / 17,328.

Its offline training objective combines relative request error and relative
pressure error computed from a full-modal analytical teacher. That inverse is
confined to the training objective. Candidate inference imports the neural model
and learned pressure component, never the analytical objective. Native q labels
remain offline. CPU and CUDA qualification pass all four K values and all four
grid groups: full field transport hashes, finite complete gradients, repeated
bitwise deterministic gradients and independent teacher/stencil reconstruction.
No component fit has run yet.

Start with one bounded K1 fit of 10,000 Adam updates, using learning rate 3e-3,
then 1e-3 at update 6,001 and 3e-4 at update 9,001, gradient norm cap 1, initialization
seed 2026101994 and sampling seed 2026101995. Draw all 16 training parents uniformly.
Use one quarter startup windows at step zero, one quarter uniformly from steps
one through eight, and one half from steps nine through 120. This is a separately
registered new fitting strategy, not a claim of matched sampling with the failed
generic temporal model. It gives cold forcing-driven pressure explicit exposure.
Retain checkpoints at 0, 2,000 and 10,000; use the final update for the bounded
screen. Capture managed Project inputs and qualify the actual trainer before fit,
then use an independent exact CPU restore.

Evaluate every development start and horizon for request error and the pressure
produced by the learned component from that predicted request. Include startup
and 32³ separately. Future true RHS may enter metrics only. Report component
latency separately, outside the full-state temporal frontier. A component pass
still requires full-state endpoint and retained-feature architectures, all
intermediate outputs, combined block timing and own-history continuation.

After the K1 diagnosis, include the requested K=2,3,4 in the new architecture
comparisons. Batch/shared pressure features can be reused across a block; all
encoding, pressure work, velocity/tracer heads, intermediate decoding and cache
updates remain inside the 4/6/8 ms combined budgets. Preserve failures and keep
the eight reserved temporal scenes unopened until a full-state finalist qualifies.

[Combined budgets and architecture options](multi-step-generation-options.md) ·
[Data registration](causal-pressure-request-data-registration.json)
