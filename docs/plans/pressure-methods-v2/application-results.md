# Single-track application comparison

Three serial repetitions per policy on CPU 23, one thread, 64 steps including 15 native warmup steps. Every candidate frame uses one fluid track; there are no reference or same-request oracle solves. Fifteen Rust diagnostic tests and Clippy passed, including exact native pressure/velocity/tracer parity and rejection without advancing history. Original selected Rheon sources were not modified.

| Grid | Policy | Completed repeats | Accepted steps | Median frame ms | Median pressure ms |
| --- | --- | ---: | --- | ---: | ---: |
| 16³ | before (6 passes) | 3/3 | [64, 64, 64] | 49.315 | 43.287 |
| 16³ | coarse (5 passes) | 3/3 | [64, 64, 64] | 45.951 | 39.930 |
| 16³ | coarse (6 passes) | 3/3 | [64, 64, 64] | 52.858 | 46.768 |
| 16³ | native | 3/3 | [64, 64, 64] | 14.562 | 8.643 |
| 16³ | native-loose | 3/3 | [64, 64, 64] | 13.637 | 7.688 |
| 16³ | neural (7 passes) | 3/3 | [64, 64, 64] | 56.130 | 49.961 |
| 16³ | neural (8 passes) | 3/3 | [64, 64, 64] | 62.485 | 56.416 |
| 16³ | spectral | 3/3 | [64, 64, 64] | 10.160 | 4.192 |
| 32³ | before (6 passes) | 0/3 | [20, 20, 20] | 359.940 | 307.970 |
| 32³ | coarse (5 passes) | 3/3 | [64, 64, 64] | 311.032 | 260.576 |
| 32³ | coarse (6 passes) | 3/3 | [64, 64, 64] | 362.762 | 313.530 |
| 32³ | native | 3/3 | [64, 64, 64] | 192.886 | 145.338 |
| 32³ | native-loose | 3/3 | [64, 64, 64] | 180.509 | 132.396 |
| 32³ | neural (7 passes) | 0/3 | [15, 15, 15] | — | — |
| 32³ | neural (8 passes) | 3/3 | [64, 64, 64] | 440.078 | 392.744 |
| 32³ | spectral | 3/3 | [64, 64, 64] | 80.581 | 32.636 |

Frame time includes advection, forcing, viscosity, pressure request/reply conversion and IPC, projection and host physical gates. Simulation wall time additionally includes native startup/setup and output IO. The JSON retains cold model loading, setup, total worker time, per-frame measurements and peak memory. No model loading or setup cost is amortized into the frame medians.

The spectral policy is fastest on these two sealed uniform scenes. The trained coarse hybrid improves over the retained neural policy but does not beat native PCG with this CPU/Python integration. These results motivate numerical baselines and integration work alongside training. They do not establish performance for an embedded implementation, GPU inference, arbitrary geometry or production grid sizes.

Rejected-policy timings cover accepted prefixes only; they are ineligible as successful speed comparisons. Native-loose uses a 1e-6 solver tolerance for all steps, while the other native prefixes use 1e-8. All candidate admission gates retain the existing relative 1e-6 residual and physical checks.

[Full measurements and hashes](application-results.json)
