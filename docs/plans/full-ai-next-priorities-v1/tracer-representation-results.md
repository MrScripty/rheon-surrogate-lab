# Native tracer representation controls

A convex displacement-based readout reproduces all 256 native primitive outputs exactly at float32 precision when supplied native displacement labels. The 64 captured next-tracer states also match the native primitive bitwise. The probes cover all 16 training scenes, four starting times, and captured, zero, constant and random tracer fields.

| Tracer pattern | Native displacement worst relative error | Offline first-order median | Offline first-order worst |
| --- | ---: | ---: | ---: |
| captured | 0 | 1.754578e-07 | 1.931064e-05 |
| zero | 0 | 0 | 0 |
| constant | 0 | 0 | 0 |
| random | 0 | 9.754837e-08 | 7.105585e-06 |

Maximum observed CFL is 0.0251227. The qualified local support contains at most eight active donors. Zero and constant tracer fields are preserved. This supports a small learned-displacement representation in place of 27 unconstrained softmax weights.

These are offline numerical and representation controls. The displacements are supplied by the native sampler; no learned predictor, optimizer update, complete-state forecast, rollout pass or CPU inference timing is established. A candidate must predict its displacements from causal model inputs, qualify gradients and state behavior, and pass every full-state horizon and own240 check. Native backtracing or reference displacements cannot be inserted into candidate inference.

The active matched provisional-supervision continuations keep the existing tracer architecture. Assess their actual outcomes before selecting the next representation comparison.

[Evidence and source hashes](tracer-representation-results.json) · [Frozen protocol](tracer-representation-protocol-01.json) · [Full-state target](plan.md)
