# Frozen pressure and representation diagnosis 01

Completed offline diagnosis on 32 consumed development states: eight scenes, native states 1, 64, 121 and 240. No fitting, numerical inverse, fresh holdout or causal prediction claim. The correct native RHS is a teacher input in this diagnostic.

| Regime | Offline policy | Worst equation relative L2 | Worst mean-free pressure relative L2 |
| --- | --- | --- | --- |
| small | native_reference_control | 2.0350286e-14 | 0 |
| small | native_reference_through_current_codec | 4.078394e-06 | 6.2390692e-08 |
| small | frozen_learned_pressure_on_native_rhs | 0.00041536501 | 0.00010232081 |
| small | frozen_learned_pressure_and_current_codec | 0.00041528498 | 0.00010232169 |
| 32cubed | native_reference_control | 1.583209e-13 | 0 |
| 32cubed | native_reference_through_current_codec | 2.3213327e-05 | 6.9356729e-08 |
| 32cubed | frozen_learned_pressure_on_native_rhs | 0.00060033924 | 0.00053009171 |
| 32cubed | frozen_learned_pressure_and_current_codec | 0.00060060355 | 0.00053009372 |

The native pressure control agrees with the independent forward edge-flux operator to at most 1.59e-13 at 32 cubed, qualifying units and reference precision. No numerical solve is run in this check.

With the correct native RHS, the frozen learned pressure component reaches 6.0034e-4 worst equation residual and 5.3009e-4 worst mean-free pressure error at 32 cubed. It therefore does not independently meet the 1e-5 component objective. These are measured component errors on the declared diagnostic population, not a universal lower bound for every possible learned input or architecture.

Even the exact native pressure, normalized, cast and reconstructed through the existing float32 pressure codec, reaches 2.3214e-5 worst equation residual at 32 cubed. Its pressure field error is only about 6.94e-8: small field error does not guarantee the residual target. The codec alone fails the new equation threshold on this population.

Next architectural contracts must include a separate precision-preserving pressure output and qualify derivative/residual error after actual physical reconstruction. Float32 velocity/tracer need their own tests; this result does not require all tensors to become float64. Replace or retrain the frozen pressure approximation and avoid the current expensive repeated CNN path. Do not claim that more temporal-head updates alone remove either observed limitation.

Raw evidence: [pressure-floor-01.json](../../../runs/full-ai-tight-targets-v1/pressure-floor-01.json). Protocol: [pressure-floor-protocol-01.json](pressure-floor-protocol-01.json).

Source SHA-256: `7cf16dd728a965b6c46437f9b0d9ba4feed27261504dbd55b29792aa2958e905`.
