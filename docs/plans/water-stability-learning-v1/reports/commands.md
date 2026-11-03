# Methods and reproduction

Use the qualified Pumas Torch v2.14.0 interpreter selected by the runtime manifest
`runs/cuda-pressure-runtime-1/manifest.json`. Each captured project pins source,
configuration, input bytes and runtime. No model weights or datasets belong in
Git. Existing Tuldok catalog: a7da4a8de53541a2a0e2aad08005bb71.

The recurrent baseline is retained in the previous tranche. Each full candidate
uses its same seed, one real training frame, 5,000 real-frame exposures, batch one,
Adam settings and schedule, four shared residual corrections, unchanged canonical
real-frame objective and identical development population. Auxiliary runs add
5,000 sampled analytical exposures (four additional network passes per update).
They sample 32 patterns at four signed amplitudes, 128 cache entries. First eight
patterns include smooth axis modes and the finest checkerboard; remaining
patterns alternate arbitrary modes and three-mode mixtures. Exact labels use
Neumann cosine eigenvalues, not a learned or numerical teacher.

The additional loss is weight 0.1 times: 0.25 relative pressure MSE + 0.25
relative pressure-gradient MSE + 0.5 relative equation MSE + squared positive
part of (relative equation MSE minus one). Each analytical sample has zero base;
its physics and RMS amplitude derive only from the fitting frame. This augments
operator coverage, not the count of independent fluid simulations. Loss totals
with/without the auxiliary term are not comparable accuracy metrics.

`stability-modes.json` isolates auxiliary training. `stability-scale.json` changes
only output reconstruction to train correction RMS, with zero auxiliary weight.
`stability-combined.json` combines them; `stability-smoke.json` checks the latter
for two updates. Old pressure/input/loss scales remain unchanged. The new output
scale changes initialization in physical units, not the seeded parameter values.
All candidates are selected at the fixed terminal update.

Assembly, with TRIAL selecting a committed configuration:

```sh
"$TORCH_PYTHON" experiments/water-conditioning-v1/training/pipeline.py assemble \
  --packet runs/water-learning-sequence-v1/views/fit-one \
  --validation runs/water-learning-sequence-v1/views/validation \
  --reserved runs/water-learning-sequence-v1/views/reserved-test \
  --runtime-manifest runs/cuda-pressure-runtime-1/manifest.json \
  --tuldok runs/live-integration/Tuldok \
  --output "runs/water-stability-learning-v1/$TRIAL/project" \
  --config "experiments/water-conditioning-v1/configs/$TRIAL.json"
```

Execute captured train/restore/validation recipes with the existing
`training/workflow.execute` and authenticated loopback client. Session wrappers
`/tmp/run-stability-managed.py --trial TRIAL --phase smoke|train|post|full` use
that owner and preserve preflight, approvals, normal Run IDs, completed/drained
custody, whole downloads, reopening and redownload parity. Credentials stay in
memory. The wrappers select the new local output root; they do not implement an
optimizer or a separate catalog. Reserved-test inference is not selected.

CPU evaluation reuses the existing retained owners with the new project and
fresh CPU restore paths. Run `audit.py` on fit-one (--training-only) and the
128-state development view; `modes.py` on the same fitting packet, testing the
24 retained modes over three grids; `native_parallel.py` on fitted and development
predictions. Run `rollout.py` on each retained trajectory-{cube,long,compact}.json
with the qualified native binary, preserving 15 warm-up steps and hard stops.
Session wrappers `/tmp/evaluate-stability-trial.py` and
`/tmp/run-stability-trajectories.py` retain their exact command arrays/logs.
Audits use available distinct cores with one Torch thread per worker. The native
build/source hashes are checked against retained build and three native/native
controls in native-controls.json; unchanged native code needs no rebuild.

Retain each finished experiment using the existing immutable record owner:

```sh
python3 experiments/water-residual-learning-v1/record.py --trial "$TRIAL" \
  --root runs/water-stability-learning-v1 \
  --reports docs/plans/water-stability-learning-v1/reports
```

The per-trial report includes config/source/checkpoint lineage, Run custody,
optimizer timestamps, training trace, analytical bank/order identity when used,
fresh CPU audits, physical evaluations, mode probes and actual trajectory
receipts. Re-recording different results under an existing ID is rejected.
The existing seven-channel UI deployment remains unchanged; these laboratory
CPU comparisons do not imply deployment qualification or simulation speedup.

Repeated CPU policy: run experiments/water-stability-learning-v1/learned_iteration.py
with the same --project, --artifacts, --packet, fresh --output, --population
train|validation and --proposals 8. It creates a separately typed local CPU
diagnostic from a managed restore, never a fabricated managed inference Run.
Native_parallel accepts that explicit diagnostic only with matching restore,
weights, normalization and population identity. Run the native evaluator on its
predictions with unchanged gates. For trajectories add --proposals 8 to the
existing rollout.py command; default remains one proposal. Eight calls execute
32 f32 neural passes, recomputing residuals using the existing f64 physical
operator/base accumulation. There is no native pressure refinement. Record
both the failed single-proposal and successful repeated-policy results.

Extended scenarios are committed under experiments/water-stability-learning-v1/
configs. Run native controls first, then learned comparisons. Rectangular and
compact use 256 steps; cube uses 240 after the native 256-step attempt fails
Courant at step242. Keep the failed control. The session script
/tmp/run-stability-long.py ran independent scenes on distinct CPU cores, and
/tmp/run-stability-cube-240.py retained the bounded cube comparison. Exact scene,
binary, process cleanup, proposal count and result are retained in each receipt.
The local CPU helper source hashes and qualification evidence are in
iteration-policy-evidence.json; six negative/feedback/process checks distinguish
source identity, proposal bounds, residual feedback and process reaping.
