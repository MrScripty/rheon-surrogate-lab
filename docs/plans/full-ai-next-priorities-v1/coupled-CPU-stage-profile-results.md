# Coupled CPU stage profile

This diagnostic uses the restored auxiliary checkpoint, one CPU thread on core23, eight consumed development scenes and starting steps0/63. Inclusive module measurements use hooks and are not the serial benchmark latency.

| Module | Small-grid inclusive block ms | 32³ inclusive block ms |
| --- | ---: | ---: |
| network | 4.2137 | 245.9038 |
| tracer_readout | 1.0797 | 36.9127 |
| pressure_core | 0.7765 | 134.7782 |
| fluid | 0.4524 | 22.3216 |
| left | 0.2468 | 7.5050 |
| right | 0.2187 | 7.7615 |
| flux | 0.1915 | 9.9921 |
| momentum | 0.1786 | 2.9389 |
| viscous_head | 0.1715 | 3.4880 |
| local_update | 0.1520 | 3.1322 |
| transport_context | 0.0695 | 1.7554 |
| transport_head | 0.0261 | 0.5990 |

Input encoding, prescribed forcing and output reconstruction outside the network take a median 0.9276 ms on the selected small-grid profile cases. The tracer decoder is the largest individually measured small-grid module. These medians are inclusive diagnostics and do not sum into the benchmark median.

The preliminary denormal screen shows no speed benefit and is not adopted. Selected-output classifications remain unchanged, but no full-window/own240 execution-variant qualification is claimed. The original weights and execution policy remain unchanged.

[Profile evidence](coupled-CPU-stage-profile-results.json) · [Five-repeat measured CPU costs](temporal-provisional-cost-results.md) · [Matched accuracy results](temporal-provisional-results.md)
