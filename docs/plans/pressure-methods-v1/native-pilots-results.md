# Native domain, timestep and spacing pilots

Nine training-only pilot trajectories completed with all native steps qualified.
Retained snapshots passed independent RHS, residual, divergence, flux, Courant
and viscosity checks. No model has been trained on these new captures yet.

| Grid | Cells/metre | Snapshots | Native steps | Payload MiB | Capture seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| [32, 32, 32] | 8 | 2 | 64 | 3.80 | 12.452 |
| [32, 32, 32] | 16 | 2 | 64 | 3.80 | 12.102 |
| [64, 64, 64] | 8 | 2 | 64 | 30.19 | 172.094 |
| [64, 64, 64] | 16 | 2 | 64 | 30.19 | 174.198 |

| Time s | Field | dt vs dt/2 | dt/2 vs dt/4 |
| ---: | --- | ---: | ---: |
| 0.266667 | velocity | 0.0103964 | 0.00519941 |
| 0.266667 | tracer | 0.000834721 | 0.000410902 |
| 0.266667 | pressure | 0.0580645 | 0.028552 |
| 1.066667 | velocity | 0.0032544 | 0.00162233 |
| 1.066667 | tracer | 0.00169259 | 0.000842359 |
| 1.066667 | pressure | 0.0319047 | 0.0148624 |

Temporal differences use the finest field norm for both pairs. Smaller pair
distance supports further temporal study; it does not establish exact-solution
accuracy or that a fine pressure field labels the original coarse request.

[Evidence hashes and scalar results](native-pilots-results.json).
