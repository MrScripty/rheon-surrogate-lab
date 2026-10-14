# Conditional full training on a local GPU

This path is separately documented and **not authorized to execute by the design
PR**. No GPU hardware, CUDA runtime, full data or full training result is claimed.
Tuldok's captured CPU profile explicitly excludes CUDA/JIT/compiler/plugin execution.
A direct local GPU training receipt may be opened as external evidence in a Tuldok
project; it is not a managed GPU Run. Actual managed full-GPU training would require
its own generic runtime/execution-owner admission, beyond this lab plan.

## Preregistered full preset and cost cap

Counts32³; 48 train/12 validation whole trajectories of128frames; new full namespace
seeds and the same geometry/forcing definitions as data-and-model.md. Three unseen
suites each reserve8 trajectories of600frames. Smoke final test histories are excluded
from all full training/selection; resolution transfer is not tested or claimed.
The encoder/decoder and parameter inventory stay the same. One local GPU with
>=12GiB VRAM and host RAM>=16GiB is a proposed requirement to validate, not an
observed capability. Raw reference payload/train shards live in an exclusive ignored
local experiment directory; no oversized Tuldok captured inputs or implicit paths.

Train float32, no mixed precision/TF32, batch2, fixed seed20261014/sample-order
seed20261015, exactly10,000 Adam updates, same lr/betas/epsilon/loss and gradient
cap as smoke; final-only weight selection. Log train every100updates and validation
at0/10000 only. No architecture/lr schedule/sweep/restart/early stopping. Fixed
candidate can fail without another attempt. A new model after failure gets new
review, experiment ID and reservation.

Proposed budgets: <=4GPU-wall hours training, peak VRAM<=10GiB, CPU reference
generation/verification<=2hours, raw/work artifacts<=32GiB, full final evaluation
<=16CPU-wall hours on the admitted local host, with serial methods. Provision/build time is reported separately. Preflight a tiny
train-only batch and measure memory/throughput before whole dataset/training;
if projected cost or qualified native generation cannot meet those bounds,
stop and replan before expensive work. The complete raw reservation alone is23,423,877,120bytes (~21.8GiB):
6,517,948,416train +1,629,487,104validation +15,276,441,600test. Admission must
measure container/manifest overhead, checkpoints, witnesses, journals and transient
copies against the32GiB cap and actual free space; the cap is not a filesystem
reservation. Keep successful raw source arrays; do not delete historical inputs to
make room. The full evaluation is705,600method-frames, including three resident
and three cold repetitions. The prior cloud32³ receipt measured30.475s per120frames
Jacobi and24.234s SGS under a different flow/thread allocation; it warns that the
old4hour proposal was too optimistic and does not qualify this workload. Use
train-only measurements of every method to admit the16hour cap before freeze;
otherwise defer the full preset. No provider spend, multi-GPU/distributed
training or cloud job is implied. GPU model/runtime availability is pending.

## Reproduction procedure to implement

1. Resolve exact CPython/Torch-CUDA/NumPy versions, supported driver, hardware,
   licenses and hashes in an isolated environment; no ambient-version fallback.
   Set deterministic algorithms, record CUDA/cuDNN/driver/BLAS/thread flags and
   disable TF32/benchmark-based kernel selection. Verify real forward/backward
   and memory on a train-only fixture. If deterministic qualification fails,
   stop; do not silently relax it.
2. Build clean locked native generator/verifier from the reviewed pin/patch,
   create only full train/validation data, independently verify all frames and
   persist group/source/payload hashes and train-only preprocessing.
3. Run the fixed train entrypoint explicitly on local GPU, bounded by monitored
   wall/time/memory. Record synchronized device-step and enclosing wall time,
   data loading/host↔device transfers, actual optimizer updates and every failure.
   Store finite trained arrays, exact config/normalization and receipt exclusively.
4. Fresh local GPU process restores final weights with zero updates and bitwise
   reproduces witness predictions on that exact runtime. Export numeric arrays
   for CPU without refitting/replacing any parameter. Fresh qualified CPU runtime
   loads identical arrays; cross-device prediction parity is a tolerance claim:
   relativeL2<=1e-5, max absolute normalized correction error<=1e-4. Report both.
   If this fails, CPU restore/evaluation is unqualified, not a different checkpoint.
5. Record any optimizer-state checkpoint only for audit; v1 forbids continuation.
   Commit independent source/evidence review and full freeze before full-test
   generation. Smoke freeze/weights cannot stand in for full provenance.
6. Evaluate all autonomous full tests on CPU with original gates and baseline
   comparisons. Optional **local GPU inference** is a separately measured path:
   synchronize GPU, include preprocessing, host↔device copies, native CPU PCG/
   correction/fallback and setup in total cost. Do not compare GPU kernels to CPU
   whole solves or omit fallback. No CPU gain implies a GPU gain or vice versa.

Tuldok can inspect compact source/evidence references and run a bounded CPU
restoration/evaluation of the exported checkpoint if its actual source/input/
runtime/output bounds are met. Larger full evaluation stays an explicit local
experiment, with honest external-evidence scope, unless a qualified generic
profile suffices without changes. No new Tuldok GPU executor is part of this plan.

## Full acceptance mapping

E1/E2/E5/E7/E8/E9 retain their numerical/source/split/freeze/rollout/cost meanings
at the full counts and durations. Their real full execution is required; smoke
results are not inherited as full physics/performance evidence. Full-specific
execution criteria replace the smoke-only wording explicitly:

| Gate | Full criterion | Scope |
| --- | --- | --- |
| Full E3 | Exactly10,000 actual local-GPU Adam updates, fixed configuration, train-only preprocessing, complete source/data/runtime/weight/cost receipt | Explicit local GPU process, **external evidence**, no managedGPU Run |
| Full E4 | Fresh zero-update same-GPU bitwise witness restore, identical-array CPU export and the stated CPU/GPU numerical parity bounds | Qualified same-device process plus fresh capturedCPU restore of the actual full trained checkpoint |
| Full E6 | Actual Tuldok completed/drained boundedCPU restore and dev-evaluate Runs with full32³ checkpoint, verified downloads/reopen and desktop/narrow UI; fullGPU train/final-local-eval receipts inspected as external evidence | ManagedCPU workflow is observed; UI/evidence cannot certify managedGPU training or full managed rollout |

Full E6's bounded dev-evaluate uses a declared training/validation witness subset,
not a cropped final trajectory. Whole final rollouts remain the local full E8/E9
procedure. Keep the full source/config/weights/normalization identities bound across
external and capturedCPU paths; if a bound fails the corresponding criterion stays
pending or blocked. A future managedGPU runtime would require a separately reviewed
replacement of Full E3/E6, not relabelling this local result.

Acceptance also requires GPU provisioning/memory, same-device restore,
cross-device tolerance, frozen test access and measured total cost. If the local
GPU or a necessary qualified dependency is unavailable, status is blocked for
full training; the completed CPU smoke remains independently useful evidence.
