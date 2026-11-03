# CUDA training with CPU deployment inference

Status: Accepted. Serial owner: root. The user requested CUDA training support;
CPU-first refers to inference. The completed scope is portability qualification,
not pressure accuracy or execution of the longer scientific training proposal.

Canonical evidence: [results](results.md), [acceptance](acceptance.json),
[ledger](ledger.md). All requested implementation and acceptance work is complete.
No active next slice. Separate future work is described in the
[next scientific sequence](../pressure-learning-next-runs/proposal.md).

Branches: lab feat/cuda-training-cpu-inference from d04116f; clean Tuldok worktree
feat/managed-torch-cuda from 9eeb653. Keep both local; no push or merge. Retain the
Tuldok worktree under runs/live-integration/Tuldok because it serves the user's
open application and existing catalog. Retain runs/cuda-pressure-v1-workspace-1
and-2, the measured runtime manifest, and qualified native/capture assets for
review/replay. Do not clean the original dirty Tuldok or conflicted Rheon checkout.

Implemented owners and write set: Tuldok owns managed_torch_python_v1 registration,
measured bootstrap, explicit CUDA/CPU mode, generic recipe integration, dispatch,
authorization and normal Run lifecycle. Lab experiments/cuda-pressure-v1 owns
versioned device portability, bounded reusable training/checkpoints, managed
project assembly/execution and native acceptance. This plan/results and current
README/project/proposal pointers expose the completed qualification. Rheon
numerical source and every historical experiment namespace remain unchanged.

Design: reference explicitly selected Pumas Python/package assets and verify their
full content/topology before and after execution. Capture the approved manifest
and bootstrap; declare host OS, GPU driver/kernel and hardware as external
trusted-local dependencies with observed versions/mappings. This is not a
hermetic captured CUDA closure. Registration is in the existing catalog, not a
second database or scheduler. The generic custodian remains sole process owner.
Legacy captured CPU profiles retain their original material/cold-read semantics.

Model fidelity: seven input channels, cubic scalar h, arbitrary positive per-axis
counts within 65,536 cells, 371,729 float32 parameters, the same train-only scales,
initialization, forward architecture and loss. CUDA uses equivalent deterministic
backward reductions where native Torch operations lack them. CPU retains native
operations. AMP/TF32 are disabled. Float32 checkpoint tensor inventories survive
CPU transfer; same-version CPU predictions repeat exactly. Cross-device forward
roundoff is measured separately. No normalization/architecture/loss ablation.

Acceptance achieved: optimizer-free fixture/grid preflight; actual CUDA parameters,
batches, gradients and Adam moments; 20 finite GPU updates; portable checkpoint
export; separate managed zero-update CPU restore with CUDA uninitialized; 12 exact
CPU predictions; both Runs completed/drained, reopened and every artifact downloaded
again. Native diagnostics independently checked all 12 predictions using unchanged
physical gates and the current 1e-6 float32 output profile. Learned proposals pass
0/12; conventional fallback passes 12/12. This is successful device portability,
without an accuracy or acceleration claim. Full numbers and source identities are
in the linked acceptance report.

Verification: 14 real-host model CPU/CUDA checks; 51 runtime/legacy material/frozen
input/custody checks, followed by 25 focused checks including a new early aggregate
asset-budget regression. Initial 38 parameter tensors match the historical CPU Run.
The first managed attempt stopped before process creation because dispatch selected
the legacy host closure; it remains preserved. The corrected fresh attempt passed.
The final inventory-only resource fix does not change the qualified child path.

No blockers remain in this scope. D0/R1 population/probe/inspection changes and any
scientific comparisons require their own implementation slice. Adopted Coding
Standards 188beda implementation, contracts, verification and commit workflows
were used; source was committed before approved training material was frozen.
