# Learned pressure representation controls

Two predeclared execution controls reduce the worst 32-cubed reference-equation difference from about 1.06e-4 to 6.67e-5 over 240 own-history steps. The 1e-5 criterion still fails. No new fitting, native pressure solver, analytical eigenvalue or pressure inverse enters candidate inference.

## Complete own-history results

The fixed explicit cosine transform and every raw managed-restored parameter remain unchanged. The baseline multiplies modal inputs by exp(log_gain). One control divides by the reciprocal of that learned gain; the other divides by exp(-log_gain). Gauge mode handling remains explicit. Derived constants come only from existing learned parameters, and their double inventories are recorded.

Each control scores all pressure, velocity and tracer fields at all 240 steps on the existing eight development scenes: 1,920 states per policy, including 480 at 32 cubed. This is K1 development diagnosis, with no new independent scene consumption, timing qualification, K2–4 result or promotion claim.

| Policy, 32 cubed | Worst reference equation | Worst pressure | Worst velocity | Worst tracer |
| --- | ---: | ---: | ---: | ---: |
| Learned gain multiplication | 0.000105727 | 4.72522e-07 | 1.43301e-07 | 5.43239e-08 |
| reciprocal-division | 6.66728e-05 | 4.72522e-07 | 1.26684e-07 | 5.43239e-08 |
| negative-log-division | 6.66728e-05 | 4.72522e-07 | 1.26684e-07 | 5.43239e-08 |

Both division controls preserve zero velocity difference on the smaller supported grids, with worst equation difference about 1.18e-13. At 32 cubed, 311 of 480 states satisfy all 1e-5 criteria; the first failing state is step 42 in the density-16 development scene. The worst equation difference is about 37% lower than the multiplication control, but remains about 6.7 times the limit. This metric compares pressure with the independently evolved reference RHS and therefore includes accumulated trajectory differences; it is not an isolated pressure solve residual.

The two controls have matching reported carrier errors and almost identical pressure/equation errors. Their benefit is evidence of numerical sensitivity in this learned operator, not proof that either is independently qualified or fastest. Every raw learned parameter inventory matches the multiplication baseline.

## Training label encoding diagnosis

A separate offline test uses all 144 original pressure/RHS pairs from the 16 training scenes. It compares the original projected pressure labels, explicit cosine projection, and inversion of only the small fixed cosine-coordinate basis for label encoding. The latter is a data-coordinate round-trip control: it constructs no physical pressure inverse, gain ratio or inverse-operator target and fits no model.

| Geometry, z/y/x | Native teacher worst equation | Original label round trip | Explicit projection round trip | Fixed-basis inverse label round trip |
| --- | ---: | ---: | ---: | ---: |
| g16_16_16 | 2.40087e-14 | 1.3536e-13 | 1.35302e-13 | 6.91252e-14 |
| g11_11_9 | 6.3228e-15 | 4.83701e-14 | 4.90176e-14 | 2.97911e-14 |
| g8_6_15 | 2.40027e-15 | 1.11712e-14 | 1.11292e-14 | 4.86995e-15 |
| g32_32_32 | 1.15356e-13 | 7.5352e-13 | 7.54904e-13 | 2.48259e-13 |

At 32 cubed, the coordinate control reduces the pressure round-trip equation difference from 7.54e-13 to 2.48e-13. This supports testing decoder-compatible supervision in a separately registered managed CUDA fit. It does not prove that label encoding is the sole cause of rollout drift or that retraining will improve it. Merely changing projection to explicit einsum did not improve this round-trip control.

## Disposition and next experiment

Preserve both execution controls as development evidence. Before the next pressure fit, prepare decoder-compatible labels from the same training-only physical data, qualify the actual double objective on CPU/CUDA without updates, and register a bounded managed CUDA refinement plus exact CPU restore. Retain the old labels and models as paired controls. A direct learned-divisor parameterization is a separate possible experiment if encoding alone is insufficient; no analytical divisor table may be inserted.

Single-core execution specialization and the existing parallel implementations remain separately available. The complete generator must still satisfy the latency and all-field own-history criteria, then independent confirmation, for K1–4. The 16 reserved precision-confirmation scenes remain unconsumed. Densities 50/100 and the 600-step drift stage remain gated.

[Machine-readable evidence](learned-pressure-representation-summary-08.json) · [Single-core execution improvement](frozen-point-execution-results-07.md) · [CPU resource policy](cpu-resource-policy-04.md)
