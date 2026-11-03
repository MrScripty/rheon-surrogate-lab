# Displacement error separation and CPU profile

The learned viscosity head is the main remaining fluid-error source in these offline controls. The normal model reproduces its previous fixed-window metrics bitwise, and all candidate coefficients remain unchanged. Controls use 16 training scenes and four starts; the two autonomous replays use 128 steps on training scenes. They do not establish the 240-step goal or a new candidate.

| Control | Warm worst P | Equation | Velocity | Tracer |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.006457352 | 0.0109775 | 0.0001389866 | 7.481297e-06 |
| zero_viscosity | 0.0003525958 | 0.0006586259 | 2.824856e-05 | 7.6926e-06 |
| offline_native_displacement | 0.006467435 | 0.01097104 | 0.0001386925 | 5.328488e-07 |
| offline_native_viscosity | 0.0003526276 | 0.0006592426 | 2.824846e-05 | 7.650691e-06 |
| offline_both | 0.0003526277 | 0.0006206933 | 2.824845e-05 | 9.885477e-08 |

| Training replay | Control | Worst equation through 128 | Worst velocity | Worst tracer |
| --- | --- | ---: | ---: | ---: |
| temporal-g1-cpm8-r0 | normal | 0.003533345 | 0.0009398915 | 0.0001707457 |
| temporal-g1-cpm8-r0 | offline_native_displacement | 0.003540732 | 0.0009403531 | 0.0001601958 |
| temporal-g1-cpm8-r0 | offline_native_viscosity | 0.0001492839 | 2.829021e-05 | 3.113313e-05 |
| temporal-g1-cpm8-r0 | offline_both | 5.375461e-05 | 2.829021e-05 | 4.770696e-07 |
| temporal-g1-cpm16-r0 | normal | 0.01165581 | 0.003993998 | 0.0009954874 |
| temporal-g1-cpm16-r0 | offline_native_displacement | 0.0116266 | 0.003994307 | 0.001001249 |
| temporal-g1-cpm16-r0 | offline_native_viscosity | 0.0005656477 | 3.464949e-05 | 0.0001095511 |
| temporal-g1-cpm16-r0 | offline_both | 4.715711e-05 | 2.639506e-05 | 4.386876e-07 |

The native viscosity primitive and exact dyadic label path pass 56 cases. The next neural fit can learn viscosity coefficients from those labels without introducing a numerical viscosity routine into candidate inference. The first helper build failed on a crate-private axis constant before any numeric query; the preserved second version uses the identical public axis enumeration.

The separate CPU profile leaves predictions and weights unchanged. Inclusive small-grid block medians include:

| Stage | Inclusive block ms |
| --- | ---: |
| fluid | 0.4682 |
| local_update | 0.1338 |
| displacement | 0.7516 |
| displacement.context | 0.0689 |
| displacement.curvature | 0.0529 |
| viscous_head | 0.1871 |
| flux | 0.1977 |
| pressure_core | 0.7412 |
| momentum | 0.1805 |
| network | 4.7419 |
| field_readout | 1.5872 |
| physical_preparation_and_reconstruction | 0.9418 |

Inclusive stages overlap and are not additive. Instrumented timings are not frontier measurements. The fixed field readout is the largest measured local stage and supports testing a fused implementation with unchanged learned weights. Actual full-block benchmark remains 6.35 ms, with accuracy failures retained.

[Raw controls, profile and hashes](displacement-component-results.json) · [Full-state accuracy](temporal-displacement-results.md) · [Actual CPU costs](temporal-displacement-cost-results.md)
