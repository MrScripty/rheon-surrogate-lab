# Frozen component diagnosis 01

Status: completed, zero optimizer updates; all K=1/2/3/4 physical outputs remained bitwise unchanged with instrumentation. Every neural tensor inventory remained unchanged; CUDA was not initialized. Only already-consumed development scenes were used.

These are inclusive instrumented median block costs, not benchmark or acceptance scores. Nested rows overlap and must not be summed.

| K | Complete block | Learned pressure | Fine pressure CNN (inside pressure) | Shared encoder | Displacement | Field readout | Input/output preparation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 105.620 ms | 64.255 ms | 61.884 ms | 21.473 ms | 8.039 ms | 2.347 ms | 1.721 ms |
| 2 | 194.296 ms | 131.646 ms | 126.785 ms | 21.287 ms | 17.270 ms | 4.758 ms | 2.865 ms |
| 3 | 278.214 ms | 203.826 ms | 196.357 ms | 10.063 ms | 26.700 ms | 7.424 ms | 4.097 ms |
| 4 | 366.044 ms | 262.084 ms | 252.523 ms | 21.388 ms | 34.259 ms | 9.435 ms | 5.152 ms |

At 32 cubed, the larger-grid pressure route runs two fine CNN stages for each generated state, rather than the small-domain compiled learned matrix. It dominates the measured cost. One encoder per block is already shared; adding encoder sharing alone would repeat the existing implementation.

Removing only the pressure cost cannot establish the new target: the current field readout alone takes about 2.35–2.47 ms per generated state by these instrumented medians, and input/output preparation alone is also costly. These findings justify replacing repeated state-generation work and qualifying a new data path before expensive learning. They do not prove a lower bound for a different implementation.

Next, isolate the frozen pressure approximation and current codec with native RHS/pressure controls. These use teacher inputs offline and cannot count as causal full-AI predictions. Then preregister bounded learned spectral and shared-output cost experiments; no target has passed.

Raw evidence: [frozen-component-profile-01.json](../../../runs/full-ai-tight-targets-v1/frozen-component-profile-01.json). Protocol: [frozen-component-profile-protocol-01.json](frozen-component-profile-protocol-01.json).

Source SHA-256: `ecb28e1bb442a2eda5f6428edba5b2318f935e96219817f8d33c957aee98f671`.
