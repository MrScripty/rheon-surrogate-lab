# Water pressure conditioning experiments

Status: Accepted bounded diagnostic/execution tranche; model accuracy unqualified.

Completed operation: implement and execute Jeremy's approved diagnosis sequence.
Owner: root, serial source integration. Branch experiment/water-conditioning-v1
from 4402db4915ecd1f5267d217317c011595ee7ab3e. Preserve historical source, results,
captures and unrelated dirty Rheon checkouts. Retain this private branch for
review and continued experiment consumers; retire after accepted integration.
Do not push or merge as part of this sequence.

Write set: experiments/water-conditioning-v1, this plan/results/reports, README
and root Tuldok project references. No existing experiment implementation or
physical acceptance profile is edited. Reuse the measured water trainer,
selected-frame views, managed CUDA/CPU recipes and native physical oracle.
The ordinary Tuldok catalog owns training and restore Runs, not a private catalog.

First slice: reproduce the single-frame failure, check float32 reference output,
captured residual consistency and loss derivatives, then run three controlled
10,000-update schedule variants. Same initial seed, frame, seven-channel model,
loss, batch 1, CUDA float32 and CPU deployment inference as the historical fit.
Each managed training Run retains the existing 600-second elapsed bound.
Constants 1e-4 and 1e-3, plus delayed decay at 5,000/8,000, are compared to the
existing decay at 1,000/6,000. Intermediate checkpoints diagnose fitting curves;
they are not retrospectively selected deployment candidates.

CPU work overlaps GPU optimization: restored-model amplitude tests with both
signs, raw repeated corrections and minimum-residual scalar step control on the
existing development validation set. These are algebraic diagnostics, not new
physical rollouts or independent test data. All correction tests keep cell size
cubic, preserve pressure gauge and use the same full original equation.

Resources: host has 24 available cores, 64 GiB RAM and RTX 5090 Laptop GPU.
Reserve CPU 0 from the diagnostic pool for GPU host work and interactive apps;
use up to 23 process workers, each pinned to one different remaining core with
one Torch/BLAS thread and no CUDA initialization. Benchmark actual restored CPU
inference at multiple thread counts before choosing inner versus outer
parallelism. All cores remain available across the composed CPU/GPU workload;
do not claim that 24 threads per tiny inference is necessarily faster. Retain
worker affinity, timings, task completion and observed concurrency. Process-pool
owner drains workers on completion/error; no detached scientific jobs. Keep
additional artifacts below 768 MiB and at least 1 GiB filesystem reserve.

Acceptance: source checks and independently computed stencil regression pass;
the old fitting command fails its 1e-6 equation gate; known reference controls
pass that gate; schedule variants have completed/drained managed Runs and fresh
CPU restore; final one-case and validation native gates reported for every Run.
No loosened physical gate, hidden native refinement or higher precision model.
The diagnostic gate is not a substitute for full native physical qualification.
Keep source/configuration commits before source capture. All results, including
failures, are preserved. CPU diagnostics and managed progress are linked from
the existing Tuldok project. Rheon continues showing actual native playback;
optimizer updates do not advance recorded simulation time.

Next slices depend on evidence: test amplitude-preserving normalization if the
amplitude checks fail; strengthen coarse corrections or train repeated correction
if fitting still stalls after implementation/optimization checks pass. Broaden
training and temporal coverage only after the known-case gap is understood.
Use a new untouched reserved population for eventual final qualification.
Keep model changes in separate commits from resource/diagnostic changes.

Standards: Coding Standards MCP snapshot
snapshot:v1:c266ea73-ebb6-4955-baf2-bb0b592dd813; implementation, verification,
commit, planning/discovery, concurrency, performance, contracts and code design.
Diagnosing-bugs skill applies; existing one-frame evaluation is the red-capable
loop. Ranked hypotheses: early learning-rate decay; loss/pressure reconstruction
mismatch; normalization obscuring amplitude; insufficient coarse error correction.

Initial measured loop: cpu.py fit-check on the existing one-frame CPU restore
returns exit 2 with relative residual 0.0017782435122407469. Rounded-reference
float32 correction gives 9.733520186283732e-8, captured-residual relative mismatch
1.1685694665632907e-15, and loss directional derivative absolute mismatch
9.284525772859649e-20. This case is representable within the existing target.


Accepted diagnosis slice: independent rectangular-grid stencil test passes; native
one-frame reference and rounded-correction controls both pass the float32 physical
profile with all physical gates preserved. The new 23-worker CPU audit completed
128 development validation states and all 1,024 actual training states. All
training-audit worker cores 1–23 were used; all workers stayed CPU-only and
single-threaded. Training audit overlapped CUDA optimizer progress. Every ideal
float32 reference correction passed the equation gate (1,152 states); full native
physical qualification is measured separately for the one-frame controls.

Current evidence justifies the next model slice while schedule Runs continue:
constant density enters the historical network as 1,000 and timestep as 1/60,
while other fields use train-derived scales. Compare fixed physical feature
units (1,000 kg/m3 and 1/60 s) against the historical preprocessing with the same
network, initialization, optimizer and loss. A separately declared amplitude
variant divides residual by per-sample RMS and restores that amplitude after the
network. This guarantees positive scaling up to arithmetic error; negative
scaling remains a measured learning property. On the one training frame the
amplitude factor is effectively one, so do not waste a duplicate full fit there.
A two-update managed amplitude smoke qualifies its CUDA/CPU reconstruction; the
full one-frame scalar-units run isolates the scalar-feature change. New versioned
configuration, source witness and architecture profile distinguish these models
from historical seven-channel checkpoints; do not deploy them through the old
comparison adapter without a matching deployment-contract update.

The training namespace copies the prior measured owner as a versioned scientific
snapshot. New policy extends configuration, model.py owns the conditioning,
project/train/test_inference select it explicitly, and trace adds post-update
physical residual, rate and clipping. Unmodified data/scenes preserve historical
view interpretation; original source hashes are not replaced. Exactly one next
slice: qualify the conditioned model and source capture, then execute its managed
smoke and scalar-units comparison after the already queued schedule controls.


User steering: diagnose why 24 CPU threads are slower than eight, benchmark three
CPU workloads at once, and improve GPU throughput before considering concurrent
training. This replaces the next-slice authority with a measured execution slice.
Current Tuldok runs.py owns a single serial supervisor; two submissions queue,
so parallel managed GPU jobs require a separate scheduler/lifecycle change. Do
not modify the serving application or bypass managed custody for convenience.

CPU profiler identifies time in many small convolution, GroupNorm, padding and
fill operations at 24 threads. The host uses an Intel Core Ultra 9 275HX with
24 cores. No blanket single-cause claim is established. Compare persistent
1x8, 3x8, 6x4, 12x2 and 23x1 core partitions on the same 1,024 recorded states,
retaining startup and steady-state separately. This is pressure inference, not
full time-evolving simulations. The default 3x8 layout has a severe second-pass
stall; a passive wait-policy control removes the stall but does not outperform
23x1. Do not promote an unstable layout or silently change deployed witnesses.

The accepted GPU performance slice adds version-2 execution configuration:
resident immutable training fields, combined finite reductions, an explicitly
stacked same-shape batch path, and a ten-update managed CUDA/CPU profiler window
after warm-up. Existing version-1 configuration retains its serial/per-tensor
execution defaults. Finite checks still cover every parameter, gradient and Adam
moment every update. Combined checks remove repeated host scalar round trips;
resident inputs remove repeated copies. Compare 500 updates for original and
optimized batch-1 execution with the same initialization and amplitude-model
conditioning; verify update-2 parity against the completed managed smoke and
full update-500 tensor inventory parity between the two paths. Batch-4 is a
throughput diagnostic with four times the sample exposure, not an accuracy
comparison. Reject heterogeneous stacked shapes; do not pad physical grids.
Batched float32 reductions can differ from serial arithmetic, so it has its own
fresh CPU restore and no exact state-parity claim. Mathematical loss-gradient
comparison uses a norm-relative tolerance and an absolute floor for near-zero
bias gradients, separate from unchanged physical acceptance limits.

Exactly one next slice: qualify and commit the execution changes, run managed
baseline/resident/batch-4 profiles, and retain CPU/native and CUDA metrics with
fresh restores. Scalar-units training continues from its earlier immutable
capture while this work proceeds. Later fitting/architecture decisions depend
on these actual native gates; no additional dataset expansion is admitted by a
performance benchmark result alone.


GPU execution acceptance measured: baseline and resident batch-1 controls have
identical checkpoint inventories at updates 0, 2 and 500. Measured training
interval rates are 31.1926 and 42.0984 updates/s; profile CPU/CUDA scalar extractions fall
from 2,520 to 1,030 per ten updates, tensor copy/conversion calls from 100 to 30. Batch-4 gives
41.3233 updates/s, 165.293 samples/s, on repeated fixture data. These intervals
include optimizer/cache setup and the bounded profiler window; they are not
steady-state kernel-only or full Tuldok workflow times. CPU restores and
aggregate device-utilization observations remain separately retained evidence.

The agreed normalization alternative is now a bounded model slice: compare the
scalar-units one-frame fit with parameter-free GroupNorm replaced by Identity.
Preserve every seeded convolution/head parameter, scalar units, Adam schedule,
frame and 10,000-update budget. Use the numerically equivalent resident-input
execution. Version-3 configuration declares normalization explicitly; the new
architecture profile identifies the decoder contract. Same tensor shapes do not
make it an old deployment model. CPU fixtures prove all initial parameter bytes
match, GroupNorm is absent and rectangular output is finite. Gate a full fitting
run on qualified execution controls; retain failures and do not infer held-out
accuracy from memorizing one frame. No additional corpus is generated.
Exactly one next scientific slice: complete the managed no-GroupNorm fitting and
fresh CPU/native checks, then use fitting evidence to select the next architecture
experiment. Existing GPU performance captures are immutable and unaffected by
this separately committed model change.


Resource replan: measured retained artifacts reached 488 MiB after the added
CPU/GPU throughput controls. Increase the task-owned retention bound from 500 to
768 MiB for the approved normalization comparison and final receipts; 1.9 GiB
currently remains free, preserving the unchanged 1 GiB reserve. This is a local
experiment retention bound, not an increase to any Tuldok recipe/runtime cap.
Do not delete scientific artifacts or overwrite old captures to fit the bound.


Further measured execution slice: the resident path still performs 103 scalar
reads per update, including validation of immutable conditioning/physics fields.
Version-4 execution can prepare those fields and loss denominators once using the
canonical contracts. It retains network input, raw correction, reconstructed
pressure, loss, parameter, gradient and Adam-state finite checks during updates.
The cached operator keeps the canonical arithmetic order and coefficients; CPU
regression proves identical prediction, loss and all parameter gradients. Require
identical CUDA inventories at 0/2/500 before acceptance. This is an execution
change; model/loss semantics, float32 parameters and physical gates are unchanged.
The opt-in currently supports resident, fully active, serial training samples;
ordinary deployment and exact-zero handling retain the checked public path.
No-GroupNorm continues from its earlier immutable capture. Exactly one next
execution slice: qualify the prepared-fields managed CUDA/CPU run, then retain
all results and next-model recommendations. Device utilization observations are
aggregate samples including open apps and setup; do not substitute them for
measured training throughput or full-workflow timings.


Metric clarification: legacy profile artifacts call aten::_to_copy counts
host_device_copies; that count includes device dtype conversions. Scalar extraction
counts include CPU and CUDA scalars. Retain the original artifacts, interpret them
with those exact scopes, and use cudaStreamSynchronize for direct wait counts
(1,960 baseline versus 400 resident per ten updates). Future profile schema v2
names copy/conversion and synchronization counts explicitly. No optimization or
accuracy claim depends on counting all scalar events as GPU readbacks.


Completion: 25 normal managed stages are completed/drained, 52,002 CUDA updates
and fresh CPU restores/inference retained. Every new accuracy candidate still
fails its fitted/validation native gates. The prepared execution path preserves
all baseline parameter bits at 0/2/500 and reaches 52.181 versus 31.193 updates/s.
See results.md and reports/summary.json for full measurements and limits. CPU
next-model work is prepared in next-steps.md. No additional dataset expansion or
unqualified deployment follows from these results. Branch remains retained under
the declared review/continuation contract; no worktree was created by this tranche.
The next scientific slice is the physically conditioned residual/linear correction
experiment in next-steps.md, with unchanged gates and a fresh deployment contract.
