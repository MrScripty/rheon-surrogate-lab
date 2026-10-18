# C69 Adam observation replay

Authorized continuation of the completed aggregate-gradient diagnosis. Keep C53
in deployment; this is one diagnostic replay, not a new training policy or model
promotion. Reuse C69's eight train-only fundamental cases, C53 initialization,
float32 CUDA optimizer, seeds/order, complete-cycle mean, global clipping at 1,
four-pass objective (teacher 0, early .05), 200 updates / 1,600 examples and LR
schedule. Relative pressure-equation target remains 1e-6; cubic spacing unchanged.

Before admission, compare instrumented and retained uninstrumented C69 code for
10 disposable CPU updates: require bitwise parameters, gradients, Adam moments,
steps, RNG, losses, order and exposures. Independently reconstruct observed Adam
steps with maximum absolute parameter error <=2e-7. Check source/normalization
custody, unchanged objective/configuration and recipe declarations. No retained
training before these checks pass. Preparation uses the existing Tuldok app
Python environment (schema validators); numerical workers use Pumas' runtime.

Capture updates 1, 10, 100, 150, 200 only. Retain six float32 vectors per update:
parameters before/after, Adam first/second moments before, raw mean gradient and
clipped gradient. Record parameter layout, actual step counters/group settings,
and per-case losses and 1/2/4/8-pass residuals before/after. Uninitialized first
moments are explicitly marked and represented by Adam's literal zero initial
state. Actual delta is computed from the two observed parameter arrays in
float64. No invented historical moments or unbounded vector history.

Vectors total 44,401,920 uncompressed bytes plus bounded headers/metadata.
Final transport is five compressed NPZ artifacts, one per selected update,
each capped at 8 MiB; complete managed output cap 64 MiB. Stream one vector at
a time. This respects both the 40 MiB restore-input declaration bound and the
8 MiB evidence-decoder shard bound; the evidence aggregate cap is explicitly
64 MiB. Measured byte-only preflight compressed each artifact to 4.1–5.6 MB.
No raw vectors are retained in GPU history. Five observations add 960 diagnostic
forwards, excluded from the unchanged 6,400 optimizer forwards. Record observation
wall time separately; copies synchronize CUDA and compression costs host CPU, so
timing is diagnostic overhead, not a clean throughput benchmark. Keep the
1,800-second managed wall cap per phase.

One existing-catalog Tuldok CUDA Run, then fresh CPU checkpoint restore. Retain
Run custody/terminal receipts and exact artifact hashes, including trace. Compare
final weights to original C69 before attributing dynamics to its old trajectory.
A mismatch is a reproducibility result, not permission to rewrite history.

CPU analysis uses one worker/core, <=2 GiB host budget and 12 GiB host reserve,
120-second bounded qualification and bounded analysis process with TERM/KILL/reap.
Stream each selected pre-state; reconstruct CPU Adam within 2e-7; compare actual
parameter delta with each case's full/final/early gradients and negative capped
mean. CPU/CUDA gradients are tested against 5e-6 relative L2; failures restrict
interpretation and stay in the report. Compare predicted first-order loss change
with measured loss change and pressure residual, and update magnitude across
rate decay. Distinguish observational evidence from causal proof.

Retain source/configuration hashes, commands, elapsed times, memory, failures,
results and next proposal as small Git records; arrays remain ignored locally.
No longer training, new loss, per-case capping, 24-case expansion, holdout use,
weights/data commit, Tuldok restart or automatic promotion in this study.

Branch context: private `experiment/pressure-adam-replay-20261018`, admitted base
7082d9e on `experiment/pressure-aggregate-gradients-20261018`. The lab user owns
integration. Retain this experiment branch as the reviewable source/result
proposal until whole-branch acceptance or an explicit retirement decision;
synchronize by new commits only, with no automatic rewrite/push/merge. No new
worktree was created. Concurrent chart/THIRD_PARTY edits are outside this slice.
