# Commands and environment

Run from the laboratory repository. Set TORCH_PYTHON to the qualified Pumas
v2.14.0 interpreter identified by runs/cuda-pressure-runtime-1/manifest.json.
Runtime manifest/input hashes and exact package versions are captured in each
project source witness and managed receipt. Hardware: Intel Core Ultra 9 275HX
(24 available cores), NVIDIA RTX 5090 Laptop GPU. CPU audits own one thread/core
per worker and leave one core for the controller. CUDA training is deterministic
float32 with AMP/TF32 disabled; physical stencils/reconstruction use f64.

Focused commands:

```sh
"$TORCH_PYTHON" -m unittest discover -s experiments/water-conditioning-v1/training -p test_conditioning.py
"$TORCH_PYTHON" -m unittest discover -s experiments/water-conditioning-v1/training -p test_residual.py
"$TORCH_PYTHON" -m unittest discover -s experiments/water-residual-learning-v1 -p test_audit.py
```

Assembly (TRIAL selects a committed reusable configuration):

```sh
"$TORCH_PYTHON" experiments/water-conditioning-v1/training/pipeline.py assemble \
  --packet runs/water-learning-sequence-v1/views/fit-one \
  --validation runs/water-learning-sequence-v1/views/validation \
  --reserved runs/water-learning-sequence-v1/views/reserved-test \
  --runtime-manifest runs/cuda-pressure-runtime-1/manifest.json \
  --tuldok runs/live-integration/Tuldok \
  --output "runs/water-residual-learning-v1/$TRIAL/project" \
  --config "experiments/water-conditioning-v1/configs/$TRIAL.json"
```

The reserved input is registered to prove population disjointness; its inference
stage is not executed. Execute through existing `training/workflow.execute` with
an authenticated loopback client for the existing catalog, selecting train,
restore and validation. The local session wrapper is /tmp/run-residual-managed.py
(`.venv/bin/python ... --trial TRIAL --phase smoke|full`); credentials stay in
memory. This wrapper does not implement a trainer or substitute catalog. Managed
preflight, approval, submission and terminal/reopened/download receipts are
retained under each trial's managed-* directories. Run IDs are in committed
per-trial reports. The Tuldok lifecycle owns cancellation and process reaping.

CPU diagnostics (RESTORE names the completed managed CPU artifact directory):

```sh
"$TORCH_PYTHON" experiments/water-residual-learning-v1/audit.py \
  --project "runs/water-residual-learning-v1/$TRIAL/project" \
  --artifacts "$RESTORE" --packet runs/water-learning-sequence-v1/views/validation \
  --output "runs/water-residual-learning-v1/$TRIAL/validation-audit"
"$TORCH_PYTHON" experiments/water-conditioning-v1/native_parallel.py \
  --packet runs/water-learning-sequence-v1/views/validation --inference "$INFERENCE" \
  --binary runs/sealed-water-v1-3/evaluation-build/rheon-sealed-water-evaluation \
  --output "runs/water-residual-learning-v1/$TRIAL/validation-evaluation"
python3 experiments/water-residual-learning-v1/record.py --trial "$TRIAL"
```

For the fitted frame use fit-one and --training-only with the fresh CPU restore.
Keep analytic diagnostics separate from native physical qualification. Reopen
completed Runs and verify artifact hashes/whole downloads through workflow.py.
The root observer is `.venv/bin/python experiments/water-conditioning-v1/observe.py
--root runs/water-residual-learning-v1 --seconds 3600`; it never owns training.
