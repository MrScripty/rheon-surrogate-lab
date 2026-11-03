All 12 advisory measurements completed with the frozen quality gates rechecked from raw frames and callbacks. Both repetitions used CPU1 and one thread; old/new order reversed in the second repetition. No performance acceptance or speedup is claimed.

| Repetition | Scene | Old first model ms | New first model ms | Old steady median ms | New steady median ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 36.941 | 35.556 | 39.067 | 37.749 |
| 0 | 16 | 88.237 | 75.949 | 69.479 | 66.366 |
| 0 | 13 | 1542.292 | 1484.068 | 1420.987 | 1420.584 |
| 1 | 0 | 35.558 | 39.011 | 36.545 | 35.820 |
| 1 | 16 | 80.029 | 86.821 | 73.285 | 72.049 |
| 1 | 13 | 1556.474 | 1451.267 | 1549.827 | 1460.874 |

Scene0 is the original96-step workload; scene16 is24×16×12 over512 steps; scene13 is40³ over23 steps. First inference is step16, after15 native steps. Each steady sample count is80,496 or7 respectively. Full-step native/AI clocks, min/max/spread, startup/hash/restore scopes, memory, commands and immutable receipt hashes are in [the timing report](q7-advisory-timing-v1.json).

The OS cache was warm and uncontrolled, and no globally idle machine is claimed. Startup profiling contributes overhead; its scopes can overlap. Current startup includes full installation verification, while the old local owner startup does not. CUDA remained uninitialized. The scientific gate passed separately before these timings; no model, policy or checkpoint selection occurred.
