# Execution ledger

2026-10-15: admitted continuation on a new branch from 9512221. The retained
recurrent model improves saved-state median error but amplifies finest modes and
hard-stops independent simulations after 4–6 AI steps. Implement mode feedback
first, with output scaling isolated as the next intervention. Historical records
and deployed comparison adapter remain unchanged. No new training started yet.

Implementation qualification: 18 focused Python checks passed (7 conditioning,
6 retained residual contracts, 5 analytical/output-scale checks). Analytical
labels match the independently specified rectangular Neumann eigenvalues;
zero correction incurs loss 1 and a reversed solution incurs loss 13, proving
non-amplification alone cannot satisfy the objective. Signed sample generation
is deterministic and leaves Torch initialization randomness untouched.

Force-path inspection: existing native paired engine applies advection, the
scene force, viscosity, RHS assembly and previous-pressure residual before
requesting inference separately on each track. The one-channel input contains
the force contribution to the pressure equation. Force parameters are not neural
features; the model replaces pressure projection, not whole simulation dynamics.
No force-path change is needed. Source: runs/water-learning-sequence-v1/Rheon/
src/bin/desktop/ai_comparison/engine.rs, prepare/force/advance.
