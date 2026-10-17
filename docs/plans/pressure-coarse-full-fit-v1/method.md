# Full coarse fit and CPU pressure qualification

This experiment completes the already qualified clipping repair's original
5,000-update fit. It does not introduce another model or loss change.
Execution records, weights, datasets, private catalog records and original logs
remain under workspace-root `runs/pressure-coarse-full-fit-v1`; only source,
small receipts and analysis are retained in Git.

## Controlled training comparison

The source freeze is `8c102a913cdccca7cb9783a0f413bc31a36ae254` on
`experiment/coarse-full-fit-20261017`. The new full-fit entry observes the original
loop and allows its normal final evidence publication. Its two-update CUDA
qualification establishes exact predecessor weights, gradients, Adam state and
CPU/CUDA RNG. It rejects premature or incomplete final checks.

The immutable Project capture preserves the original 5,000-update configuration,
all five complete sampling arrays, original seeded initialization, inputs,
normalization, precision, gains, schedule and feedback losses. Float32 Adam and
model state are unchanged. The qualified overflow-only clipping repair preserves
the native clipping path when its float32 norm is finite. Repairs and successful
CUDA finite-state checks are recorded separately from sampled loss observations.

The retained full-fit control is reused, not rerun. `control-comparison-03.json`
verifies its actual CUDA/CPU artifacts, training inputs/configuration/schedule,
allowed model/source differences and the package-to-74-case campaign chain.
Control and candidate are different architectures: the candidate adds the coarse
pressure module. The previous candidate's accepted 1,500-update clipping screen
is the exact prefix comparator, with all 15 logged losses checked through update
1,500. These are sampled losses, not claims about every intermediate loss or a
physics accuracy test.

One journalled fit and one fresh CPU restore use Tuldok's original catalog,
Project, Study and Run owners. Unknown dispatch is never replayed. The Study is
closed after training. CPU restoration must use zero optimizer updates, leave
CUDA uninitialized and reproduce final weights/checkpoint, normalization,
predictions and history byte for byte. The read-only custody collector joins
actual admitted materials, staged source hashes, process specifications and
public artifact inventories to the retained downloaded bytes. Both managed Runs
must terminate with exit zero and kernel-confirmed ECHILD/spawn closure.

## CPU accuracy and timing

The CPU source freeze is `2f99494`; collection freeze is `87195fb`.
Only the orchestration and custody helper changed. The retained control's
scientific benchmark, rollout, oracle and common-RHS producer sources are
unchanged, preserving comparison identities.

1. Package the exact restored final candidate with the existing adapter.
2. Run scene 0 for 16 steps: 15 native warm-up steps, then one actual AI request
   with previous-pressure history. Check exact input closure and independent
   same-RHS pressure/operator oracles. Scientific rejection is retained and blocks
   admission of the broad development campaign; infrastructure failure stops.
3. Run fixed-input frequency/amplitude transfer diagnostics and the same 144
   training RHS/label pairs used by the retained control. These run no optimizer
   updates or advancing simulations. Report spatial frequency bands and absolute
   energies separately from ratios; they do not establish a causal noise floor.
4. Only after first-frame acceptance, run the 74 consumed development windows
   and 233 scheduled independent pressure checks with unchanged thresholds,
   warm-up/history, fallback, call and population gates.
5. Only after all 74 cases complete without an infrastructure failure and the
   pool drains, run two raw forward timing repetitions. Admit two paired serial
   repetitions only if all four fixed timing scenes qualified. Use fixed single
   cores. Paired diagnostic branch
   times include IPC/logging and an additional native reference solve. Raw network
   timing excludes those costs. Neither is mislabeled standalone solver speed.

Broad CPU work uses 16 one-thread workers on cores 5–20, 2 GiB per worker and
8 GiB available reserve, with broad pools sequential. A local Linux subreaper
owns every phase's descendants, including separate-session workers. Cancellation
is deferred during drainage until kernel ECHILD; no later phase may start while
previous descendants remain. The retained four-case process qualification
reproduces leader-only orphaning and verifies cooperative/forced cleanup for the
real benchmark and common-RHS pool plus interruption during normal drainage.

No unused holdouts are consumed and no candidate is promoted. Accuracy and work
must be compared per population to the retained control and prior accepted
models; passing individual physical checks does not automatically pass those
population gates.

## Clocks and observation

Record optimizer-loop wall time from training history, trainer time, managed Run
start-to-drainage, complete command/API/download time, each CPU phase wall time,
concurrent campaign wall time, and separately scoped serial timings. Startup,
queue/admission, evidence readback and probes are not called pure inference time.
Resource samples are sparse machine-wide observations, not process memory peaks;
trainer CUDA and timing-process RSS peaks have their own measurement scopes.

The original Tuldok and Rheon applications remain open. Bounded observation calls
publish actual managed Run/CPU progress. A recorded control preview is explicitly
labeled until a hash-bound passed candidate first-frame/case is available. There
is no separate catalog or detached monitoring service. Final records include
controller/worker cessation and unchanged original application identities.

## Actual stopping decision

Classification-only continuation was frozen at `da7df62`, with successful prefix
reuse after eleven fail-closed negatives and exact first-frame numerical replay.
The fresh campaign then encountered the independent conjunctive stopping gate on
scene 7. Its native branch completed, but callbacks at steps 64/96 exhausted32 calls
without passing the conservative pressure-error certificate. The read-only replay
confirms independent operator and oracle coverage validity. Actual measured
pressure error remained below the target at those captures; the certificate did
not. This is distinct from the explicit native pressure rejections on scenes 1/5/6.

The raw scene 7 receipt remains labeled infrastructure by the unchanged episode
classifier. Its diagnostic interpretation is recorded separately; it was not
rewritten into a passed or native-rejection receipt. Stop at the failed independent
gate. Retain the failed/incomplete74 campaign, identify cancelled/unstarted cases,
and do not admit serial or raw timing afterward. No complete-population comparison
or pass rate follows from these partial results.
