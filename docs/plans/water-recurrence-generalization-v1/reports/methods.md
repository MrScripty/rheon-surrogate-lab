# Methods and reproduction

The Pumas scientific interpreter is
`/media/jeremy/OrangeCream/Linux Software/repos/owned/ai-systems/Pumas-Library/torch-versions/v2.14.0/venv/bin/python`.
It supplies Torch 2.14.0+cu132, NumPy 2.5.3 and Python 3.14. CUDA training uses the
RTX5090 Laptop; CPU qualification uses the Core Ultra9275HX. The lab `.venv`
executes the authenticated Tuldok HTTP controller, not the optimizer. Host
execution is required because the sandbox cannot see the GPU or service process.

`configs/sequence.json` names training, validation, reserved, runtime and Tuldok
paths. Assemble each trial using the existing trainer:

```bash
"$TORCH_PYTHON" experiments/water-conditioning-v1/training/pipeline.py assemble \
  --packet runs/water-residual-learning-v1/views/cohort-16 \
  --validation runs/water-learning-sequence-v1/views/validation \
  --reserved runs/water-learning-sequence-v1/views/reserved-test \
  --runtime-manifest runs/cuda-pressure-runtime-1/manifest.json \
  --tuldok runs/live-integration/Tuldok \
  --config experiments/water-recurrence-generalization-v1/configs/cohort-two-seed2.json \
  --output runs/water-recurrence-generalization-v1/cohort-two-seed2/project
```

The working tree must be committed/clean before assembly. Captures retain exact
source/input/runtime witnesses, zero-residual fixture qualification and recipes.
All four initial captures use source ebad78c. Later reporting code does not
modify these captured model/trainer sources.

The existing local authenticated session helper invokes conditioning
`workflow.execute(client, project, output, observer, stage_names=...)` through
normal Tuldok graphs, concrete preflight/grants and managed Runs:

```bash
.venv/bin/python /tmp/run-budget-managed.py \
  --root runs/water-recurrence-generalization-v1 \
  --trial cohort-two-seed2 --phase full
```

The helper reads the already attached local owner session in memory and logs
out at exit. It creates no new token file/catalog. Its temporary location is a
controller detail; captured recipes and `training/workflow.py` own scientific
execution/custody. Train, CPU restore and validation complete/drain, artifacts
are downloaded, reopened and downloaded again for exact parity. No reserved
managed inference stage is selected. Preserve failed attempts; do not blindly
retry a submission with an unknown outcome.

The CPU follow-up controller uses these existing entry points for each trial:

- `water-multigrid-learning-v1/evaluate.py` on the cohort training packet and
 128-state validation packet, default adaptive 32-cap policy checked after each
 individual network evaluation.
- The same evaluation with `--network-budget 8 --fixed-budget` on validation.
 This uses exactly eight shared network evaluations rather than adaptive stopping.
- `water-multigrid-learning-v1/mode_probe.py` on the retained one-frame packet:
 60 independent mixtures at two signed amplitudes,120 operator probes. These
 are not120 CFD simulations.
- `water-multigrid-learning-v1/study.py --config STUDY --output FRESH --workers 8`
 for fresh physical development and `--workers 3` for the 512-step study. Each
 study has explicit model/project/restore paths, native controls first,15 native
 warm-up steps and independent AI trajectories with no native pressure cleanup.

Exact argv arrays/logs remain under ignored run roots. A single observer combines
per-trial activity into `runs/live-project/activity.json` for the existing UI.
Numerical diagnostic pools use up to 23 distinct single-thread workers. First
reference long-window qualification uses cores21–23 while CUDA trains; all its
children drain before the first trial's CPU follow-up starts. These are
correctness/throughput observations, not isolated latency benchmarks.

`water-multigrid-learning-v1/record.py` reconstructs real sampling from the
declared PCG seed and verifies the retained closure before reporting real/analytical
exposures and voxel-evaluation work. It also checks completed/drained custody,
repeated-download parity and CPU restore identity. The new `summarize.py` derives
the recurrence comparison from those sealed records and fixed-eight native reports.
Equal5,000-update counts are not equal compute at different recurrence depths.

The final 16 whole simulations are declared in two `final-holdout` configs before
training. Candidate selection must be recorded/frozen from development evidence
before running either. Retain failures and pressure drift versus native as well
as equation gates; a1e-6 equation target is not a1e-6 pressure-field error bound.

Raw-output native checks additionally read managed restore and validation
+predictions. The trainer intentionally retains at most 12 fitting predictions;
+that raw subset is distinct from the adaptive evaluator's complete 32 training
+states. Do not report a12-case raw check as an all-training-state result.
+
Final recording uses summarize.py followed by finalize.py with --root
+runs/water-recurrence-generalization-v1 --reports
+docs/plans/water-recurrence-generalization-v1/reports. trajectory_summary.py
+exports the fresh, long, prior-grid and two final populations separately; final
+batches have different scenes and are combined only by the final index.
+final-source-verification.json independently checks the index against3,616
+native-produced AI frames and callback counts. Inputs/configuration/checkpoint
+identities and source-report hashes are retained in summary.json.
+
+The original one-pass full workflow fails during CPU approval at its180-second
+HTTP timeout, before submitting that CPU Run. Recovery calls normal
+workflow.execute with stages restore/validation and previous.train bound to the
+existing completed CUDA Run, using a600-second HTTP deadline. managed-recovered
+is a derived index of those real Runs, not another execution path or a synthetic
+Run. It preserves managed-full/failure.json and managed-post-recovery receipts,
+reopens all 3 Runs and downloads all artifacts again. record.py --managed
+selects this explicit index. The normal record remains byte-identical.
+
+Late CPU qualification, raw native checks and grid/final studies share an
+exclusive advisory file lease so their23-worker pools do not overlap. Fewer
+independent jobs use fewer cores. No latency claim is drawn from these studies.
+The five CUDA optimizer intervals do not overlap. Both activity observers stop
+before final completed publication; the Tuldok service stays running.
+