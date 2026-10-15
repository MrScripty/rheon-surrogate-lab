# CUDA training and CPU inference qualification

CUDA training support is qualified through the user's existing Tuldok catalog.
The unchanged seven-channel model completed 20 GPU optimizer updates, exported
portable float32 weights, and restored in a separate managed CPU Run with zero
updates. All twelve CPU predictions repeated exactly. This establishes device
portability; **the learned pressure proposals still fail all twelve accuracy
checks**, including the current float32 target. Longer scientific runs have not
started.

Measured evidence is in [acceptance.json](acceptance.json),
[native diagnostics](reports/native-evaluation.json) and
[runtime events](reports/runtime-events.json) and
[final UI/reopen verification](reports/live-final.json). Local complete source, material,
Run, artifact and re-download evidence remains under
`runs/cuda-pressure-v1-workspace-2`, with historical failure evidence preserved
under `runs/cuda-pressure-v1-workspace-1`.

| Property | Observed result |
| --- | --- |
| Existing catalog | a7da4a8de53541a2a0e2aad08005bb71 |
| Immutable execution project | 0ae8c995d2ec4f0194c6828dfb8aad35 |
| CUDA training Run | 097f84f0493f45d1a52c1180c1c1032b; completed and drained |
| CPU restore Run | 284b8c39020e41a5a00468181b7d12bf; completed and drained |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU |
| Runtime | Python 3.14.7, Torch 2.14.0+cu132, CUDA 13.2, NumPy 2.5.3 |
| Actual training | 20 updates; CUDA parameters, batches, finite gradients and Adam moment buffers |
| Precision | Float32 network; AMP and TF32 disabled; existing physical reconstruction/loss accumulation retained |
| GPU memory | Peak allocated 10,272,256 bytes; reserved 12,582,912 bytes |
| Model training function elapsed | 5.234 s; excludes managed startup/runtime verification |
| Validation loss | 2.6075728388 → 0.3414271193 |
| Checkpoints | Initialization, update 10, update 20; per-tensor float32 byte inventories |
| CPU transfer/restore | Every weight bit preserved; zero restore updates; CUDA not initialized in CPU process |
| Prediction parity | Twelve CPU cases exact; eight shared artifacts byte identical |
| CUDA/CPU forward difference | Maximum absolute reconstructed-pressure difference 9.7152e-9 Pa; cross-device bit identity not required |
| Publication | Ten artifacts per Run; completed/drained Runs reopened and every whole artifact downloaded again with exact byte/hash verification |
| Physical inputs | Six existing cubic grids; h=0.125 or 0.0625 m; no voxel-size input maps |
| Learned pressure success | 0/12 under original native gates and 0/12 under current float32/physical profile |
| Native terminal solves | 12/12 qualified after conventional fallback; no learned acceleration claim |

The same train-only normalization digest is retained:
`2ce6c3ab6722269b97b6ae61e978d219c0ee8d3d56c471cba0b3d85188a64da1`.
The final weights digest is
`064a19f3f678be31aaa49f93278531a2054c3500bf2fa9a3997bcaa59a3c38c9`.

The first managed attempt, bf6d12d3e90e4e728baa20c5ad82aa8f, stopped before
process launch because the dispatch supervisor compared the legacy host closure.
It performed zero optimizer updates. A focused admission-to-launch regression
covers the fix; that interrupted attempt was retained rather than retried in place.

Source separation: lab model/trainer code was frozen from 59397ff (implementation
641f499); generic Tuldok runtime code was committed separately as b881ebd and the
dispatch correction as d852e21. A later registration-only bound check stops inventory
hashing as soon as its aggregate asset budget is exceeded; it does not alter the
qualified child execution path (final Tuldok head d3514c6). The current CUDA Run,
CPU restore and historical CPU Run artifacts also reopened and downloaded
successfully after the final server restart. Runtime assets were measured/verified before and
after both actual Runs. Host OS/driver/hardware remain explicit external
trusted-local dependencies, not a hermetic captured CUDA installation.

Fourteen model CPU/CUDA checks and 51 runtime/legacy recipe/frozen input/custody
checks passed, with a later focused asset-budget regression. CUDA average-pool
backward lacks a deterministic implementation in this Torch release; explicit
disjoint backward reductions preserve the native average/nearest forward model.
The old CPU forward path, physical operator arithmetic and seeded parameter
inventory remain covered by focused numerical checks. All 38 initial parameter
tensors also match the historical seven-channel CPU Run byte for byte.

The normal Tuldok project now links these results and current sources. Actual
native learned-pressure previews were published to the already-open Rheon egui
live-project viewer. Live preview is observational; retained Run artifacts and
independent native diagnostics remain the acceptance evidence.

Next scientific work remains D0 amplitude probes and R1 one-case fitting. The
reusable trainer already supports up to 10,000 updates, bounded sparse checkpoints,
explicit learning-rate changes and independent sampling seeds. Explicit single-frame
selection and scheduled fitted-frame target inspection are still needed before R1;
normalization, loss, architecture and generalization changes remain separate.
