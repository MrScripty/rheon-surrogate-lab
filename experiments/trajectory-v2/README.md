# Measured 3D trajectory experiment

This checkpoint was trained through Pumas Library's managed PyTorch
2.14.0+cu132 on the RTX 5090 Laptop GPU. It fits 32,767 supervised cosine-mode
pressure responses on 96 actual Rheon frames from 12 training trajectories.
It replaces PCG entirely inside the declared 32³ fixed-box simulation.

`model.pt` and `freeze.json` preserve the exact model and pre-test conditions.
`result.json` records the five-repetition/four-seed first complete-rollout
comparison: median 22.01 s Jacobi, 19.08 s SGS, 5.18 s learned for 120 frames
(two simulated seconds), including transfers and all simulation stages with a
resident model. Native CPU kernels are single-threaded in that first comparison.
The separate parallel comparison does not alter these historical receipts.

`parallel-freeze.json`, `parallel-result.json` and `parallel-verification.json`
preserve validation-selected thread counts and the complete parallel comparison.
It measured 16.35 s Jacobi (4 threads), 9.67 s SGS (8 threads) and 1.76 s learned
(8 native threads plus GPU inference): 9.30× / 5.50× speedups. All 4,320 timed
frames qualified; independent full-field replay over four test seeds and both
CPU implementations found maximum velocity relative L2 error 9.00e-8.

`verification.json` independently recomputes full-field velocity/tracer errors,
checks artifact hashes, every frame's gates, timing arithmetic and memory limits.
It verifies 7,200 benchmark frames. Worst independently recomputed velocity
relative L2 error is 8.89e-8; tracer relative L2 error is 5.89e-8. Both describe
agreement with Rheon discretized flow, not general physical truth.

`viewer-check.json` and `viewport.png` record browser behavior and rendering.
The interactive viewer lives in `viewer/`; bulk fields remain in ignored runs.
Run the documented training/evaluation procedure in a new output directory to
regenerate fields. A replay is not an untouched test for future model selection.

The source hashes identify the exact numerical/runtime code; the freeze need
not be rewritten when this receipt is copied for preservation. Binary hashes
are local-build identities, not portable build guarantees. Replication on a
different machine requires its own source/runtime/binary receipt.

This is a sealed, constant-density box with prescribed forcing and a passive
tracer. It has no free liquid surface, viscosity, obstacles or moving boundaries.
An analytic spectral inverse also exists for this operator. No advantage over
that analytic method or applicability to untrained physics is claimed.

The final failure-path audit found a buffered-pipe cleanup race after correctly
rejected pressure. The current driver catches that close-time `BrokenPipeError`
and always reaps the child/closes pipes. The exact original timed driver is kept
in `frozen-source/trajectory_experiment.py`, matching its recorded source hash.
No checkpoint or numerical kernel changed, and no test-based model selection or
timing replacement occurred. New runs freeze the updated driver themselves.

`fresh-process.json` additionally measures three runs of the current driver from
a fresh Pumas-managed Python process, including imports, CUDA/model startup,
120 complete frames and Python teardown, without explicit warmup. Median wall
time was 3.20 s; Python peak RSS was 942 MB and CUDA reserved peak was 37.7 MB.
This three-sample, seed-900 startup check is separate from the balanced resident
benchmark and does not replace its speedup statistics.
