# CUDA training with CPU deployment inference

Status: Active. User explicitly requested CUDA training support after correcting
CPU-first to mean inference. Root agent owns serial integration. Private branches
feat/cuda-training-cpu-inference (lab, base d04116f) and feat/managed-torch-cuda
(Tuldok clean worktree, base 9eeb653) remain local for review. No push or merge.

Objective: train the unchanged seven-channel float32 model on CUDA under normal
Tuldok custody; export portable checkpoints; restore and infer on CPU; perform
actual native physical diagnostics with the accepted float32 profile. Longer
scientific experiments and normalization/architecture changes remain outside this
portability slice. Preserve every historical source, profile, checkpoint and result.

Write sets: Tuldok adds an explicit managed_torch_python_v1 runtime owner,
bootstrap, generic external-recipe integration, tests and documentation. Lab adds
experiments/cuda-pressure-v1 with versioned device portability, bounded reusable
training/checkpoint policies, project/qualification helpers and focused tests;
this plan/evidence and current user-facing pointers. Rheon numerical code is not
changed. Tuldok worktree and existing catalog are used, never dirty originals.

Runtime design: the new profile references explicitly selected installed Pumas
Python/package assets, recording and verifying their full content inventory before
and after execution. It does not claim a captured/hermetic CUDA closure. Host OS,
GPU driver/kernel and GPU device remain declared host dependencies, with observed
versions and mappings recorded. The installation must remain immutable for the
attempt. The manifest is owned by the existing catalog and its exact bytes are
captured in approved material. CUDA and CPU modes are explicit in the recipe;
training cannot silently fall back. Historical captured CPU snapshots keep their
existing semantics and checks. Existing generic child custody remains the sole
process owner; no detached local training or wrapper CPU Run hiding GPU work.

Composed-design review: applicable. Canonical responsibilities are separated:
Tuldok owns authorization, runtime identity, material capture and Run lifecycle;
the lab owns device-portable model/training and numerical acceptance; Rheon owns
simulation/operator/physics. The runtime bootstrap uses the existing generic
trusted-local execution seam, adding no second scheduler or catalog. Source and
runtime manifests bind the cross-process interface, while typed Run artifacts
carry checkpoint custody into CPU restore. Native physics remains independently
measured. Physical data and model architecture are reused unchanged.

Acceptance: real CUDA parameters, batches, gradients and Adam moment buffers;
finite updates with GPU memory/device evidence; float32 checkpoint bytes preserved
through CPU transfer; fresh deterministic CPU predictions repeat exactly from
managed restore. Report CUDA/CPU differences rather than requiring cross-device
bit identity. Verify all existing physical gates and current 1e-6 model-output
profile honestly. Qualification success is portability, not pressure accuracy.
No optimizer during preflight. Same Torch runtime release in both device modes.

Current phase: runtime/model portability implementation. Exactly one next slice:
qualify the new generic Tuldok runtime and device-capable lab model, then commit
source before freezing managed qualification recipes. Follow with a small real
CUDA training Run and CPU restore, source/weight checks and native diagnostics.
Blockers: none observed. Replan if installed assets cannot be measured reliably,
CUDA lacks a required operation, or runtime/Run ownership cannot be preserved.
Use adopted Coding Standards 188beda implementation, contracts, verification,
planning and commit workflows. Detailed evidence will live beside this plan.

Implementation evidence (2026-10-14): installed runtime registration succeeded in
the existing catalog a7da4a8de53541a2a0e2aad08005bb71. Manifest identity
74bbe92e160f14266b850cfebad01655192f3cb272083f8a87c1ae224c9e1e4e measures
25,765 files / 5,744,383,179 bytes. Runtime manifest and descriptor are retained in
runs/cuda-pressure-runtime-1 (local measured assets, not checked-in packages).
Fourteen model checks pass on the real CUDA host, including arbitrary grids,
finite gradients, physical operator agreement and no fallback. Fifty-one Tuldok
checks pass for runtime/legacy recipe/frozen input/process custody contracts.
CUDA average-pool backward lacks deterministic support; preserve its forward and
supply equivalent disjoint backward reductions, with differential CPU tests.
Nearest interpolation and replicated padding receive the same deterministic CUDA
portability treatment. CPU forward/native operator/model weights remain unchanged.

Current phase: Implemented, verifying managed execution. Exactly one next slice:
commit runtime and model portability sources, freeze a 20-update qualification,
then run approved CUDA training and fresh CPU restore in the existing catalog.
No longer R1/normalization/cohort experiment has started. The retained Tuldok
worktree serves the user's open UI and must remain until explicit replacement.
