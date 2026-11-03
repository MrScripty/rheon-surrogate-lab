# Coupled-model transport diagnosis

The train-only controls identify provisional transport as the immediate fluid limitation. Replacing pressure alone leaves warm worst velocity error at 0.002496007, compared with 0.002439474 normally. Replacing the provisional velocity instead reduces it to 2.824841e-05. These reference-field substitutions are offline controls and are excluded from candidate frontiers.

| Warm training control | Worst pressure | Worst equation | Worst velocity | Worst tracer |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.08195473 | 0.2428143 | 0.002439474 | 0.00175786 |
| saved_initialization | 0.4000973 | 0.8292476 | 0.00631044 | 0.06258844 |
| zero_transport | 0.4010321 | 0.8278319 | 0.006305711 | 0.001768801 |
| zero_viscosity | 0.0820674 | 0.2434057 | 0.002454479 | 0.001757958 |
| zero_both | 0.4000973 | 0.8292476 | 0.00631044 | 0.001768893 |
| offline_native_pressure | 8.123638e-08 | 1.808595e-05 | 0.002496007 | 0.001757154 |
| offline_native_provisional | 0.0003526307 | 0.0006315574 | 2.824841e-05 | 0.00175967 |

The learned transport improves on the saved initialization and zero-transport controls. Its warm provisional error is nevertheless about 33% of the native transport increment on median and up to 106% in this probe. Direct provisional-velocity supervision is the next controlled objective comparison, alongside the full-state loss. Tracer changes little under either fluid substitution and needs its own transport/generalization diagnosis.

The offline reconstruction passes 24 independent native fixtures and 128 captured targets. Corrected velocities round-trip exactly; maximum RHS disagreement is 3.430088e-06. That qualification uses original float64 pressure. Labels reconstructed from the actual normalized float32 training targets still require a separate check.

All 896 diagnostic outputs are retained. The unchanged model reproduces its previous fitting errors bitwise and its coefficients remain unchanged. No optimizer update, native pressure solve, native timestep, latency claim or reserved capture occurs. Every physical output, 240-step accuracy gate and combined 4/6/8 ms budget remains unchanged.

[Machine-readable summaries and evidence](temporal-coupled-transport-results.json) · [Diagnostic protocol](temporal-coupled-transport-diagnostic-protocol-01.json) · [Actual full-state result](temporal-coupled-results.md)
