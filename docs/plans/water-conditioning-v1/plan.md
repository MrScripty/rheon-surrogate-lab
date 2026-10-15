# Water pressure conditioning experiments

Active operation: implement and execute Jeremy's approved diagnosis sequence.
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
additional artifacts below 500 MiB and at least 1 GiB filesystem reserve.

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
