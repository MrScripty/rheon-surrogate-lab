# Balanced shape aggregation

Status: Active — CUDA screen and CPU restore. Canonical operation: Jeremy resumed model
improvement after the completed Tuldok performance repair, admitting the bounded
proposal in pressure-loss-components-v1/next-steps.md. Keep C53 deployed.

Objective: test whether one shared update from all eight shape gradients improves
four-pass fitting. C69 uses the exact C66 cases, C53 initialization, normalization,
model, four-pass loss, early feedback, seeds and exposure-based learning rates.
Do not treat gradient geometry as proof of training causality.

One complete cycle backpropagates each of eight cases with loss divided by eight,
releasing each graph immediately. Zero gradients once, globally clip once, Adam
step once. Retain 200 exposures/case: 1,600 examples, 6,400 network forwards and
7,424,000 forward cells, excluding diagnostics. This is 200 optimizer updates
versus C66's 1,600; changing Adam frequency/moments is part of the experiment.
No equal-update claim or multiplying the example budget. Rates by cycle:
1–100 1e-4, 101–150 1e-5, 151–175 1e-6, 176–190 1e-7, 191–200 1e-8.
Keep C66's seeded case order and diagnostics every ten exposures/case. Counters
for examples, exposures, optimizer updates and network work remain distinct.

Before admission: verify identical captured sources except the explicit bank
aggregation adapter, config and trainer wiring. Compare original bank/order,
train-only amplitude, every sample tensor and normalization. Check each label
against the pressure operator (<1e-10). Independently collect captured per-example
full gradients at frozen C53, average in float64, and compare the actual float32
accumulation (relative L2 <=5e-6). This tolerance is numerical, not a changed
physical target. Verify one global clipping operation and one Adam step, no
intermediate parameter change, final weights against the explicit-gradient Adam
oracle, finite values and unchanged default inference. A literal scalar oracle
must distinguish averaging from summing and per-case clipping. No fabricated Adam
history; both this candidate and C66 start fresh optimizer state at C53.

Then exactly one visible Tuldok CUDA Run using the existing Project/catalog/Pumas
owners, one-shot journaled admission, 1,800-second trainer cap, fresh CPU restore
and published artifact parity. Reuse the qualified scheduling repair, not a new
execution/catalog implementation. No raw private Run outputs or admission retry.
Report all-case four/eight-pass fitting (<=1e-6), one/two-pass no regression
against C53, the same 128 consumed real requests and 240 manufactured cases.
Record accuracy, example/update/pass counts, elapsed training/workflow/CPU times,
peak host/CUDA memory, source identities, failures and process cessation.

Stop at a failed fitting gate; do not automatically lengthen training or run the
24-case test. A passing screen supports a separate 24-case/retention proposal,
not promotion. Preserve C67/C68 history and untouched holdouts.

Resources: qualification one CPU/Torch/BLAS thread, no CUDA initialization, 12 GiB
host reserve plus 2 GiB worker headroom. Fitting tensors <=8 MiB; stored gradient
workspace <=16 MiB; graphs released per example. CUDA via the existing one-GPU
slot. Restored inference timings serial on one CPU; then up to three independent
one-thread fitting/diagnosis workers on distinct CPUs, same 12+6 GiB headroom,
existing cancellation/TERM/KILL/reap owners. Do not stop user applications.

Write set: experiments/pressure-balanced-shapes-v1/ and matching docs directory;
only bounded model reservation/report-link integration in four_pass_study.py and
its owned cohort-progress.html region if needed. Runtime data under ignored runs/
pressure-balanced-shapes-v1/. Preserve unrelated dirty chart/source files.
Composed-design scope: no new product architecture or physical/model contract;
the aggregation adapter owns optimizer-cycle semantics and delegates bank/loss/
sample generation to the captured unchanged owner. Existing Project, execution,
inference and report owners retain their boundaries. Replan before changing loss,
model, capture, runtime, physical thresholds or gating semantics.

Branch experiment/pressure-balanced-shapes-20261018, base
cbb1c5e728d61f9822417edeb701826a5e14f63f, private review proposal. Jeremy owns
integration into the preceding lab experiment line; no automatic push/merge/
synchronization/rewrite. Retain unchanged for review; retire after integration or
explicit disposition. No worktree created. Standards snapshot:
snapshot:v1:ffcfc3f8-b52a-434e-b57f-d259a0355711. Ledger: execution-ledger.md;
findings: issues.md; receipts: reports/. Scientific outcome is distinct from
correct execution/recording. CPU qualification passed: literal averaging/clipping oracle, exact case/order identity,
relative mean-gradient error 8.62e-8 and Adam parameter max error 3.73e-9.
Next slice: one journaled CUDA Run, fresh CPU restore and fitting/regression gate.
