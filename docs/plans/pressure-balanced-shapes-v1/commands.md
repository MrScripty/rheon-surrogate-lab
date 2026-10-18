# Commands and retained environment

Run from the lab root. Application controller:
`runs/dependency-migration-v1/app-env/bin/python3.12`. Numerical workers use the
Pumas-qualified manifest `runs/dependency-migration-v1/runtime/current-manifest-v2.json`:
CPython 3.14.7, Torch 2.14.0+cu132, NumPy 2.5.3, RTX 5090 Laptop GPU. CPU launcher
hides CUDA, sets one Torch/BLAS thread and binds distinct cores. Exact commands,
runtime identities, durations and cessation are retained in reports/.

Preparation and disposable CPU qualification:

```bash
runs/dependency-migration-v1/app-env/bin/python3.12 -B experiments/pressure-balanced-shapes-v1/prepare.py \
  --config experiments/pressure-balanced-shapes-v1/config.json \
  --output runs/pressure-balanced-shapes-v1/balanced-shapes-project-01
runs/dependency-migration-v1/app-env/bin/python3.12 -B experiments/pressure-balanced-shapes-v1/qualify_worker.py \
  --config runs/pressure-balanced-shapes-v1/balanced-shapes-experiment.json \
  --project runs/pressure-balanced-shapes-v1/balanced-shapes-project-01 \
  --output docs/plans/pressure-balanced-shapes-v1/reports/C69-qualification-01.json \
  --execution docs/plans/pressure-balanced-shapes-v1/reports/qualification-worker-01.json
```

One-shot managed CUDA training and CPU restore; **do not replay** this completed
study. Inspect its existing intent, mutation journal and terminal receipts.
Private session contents are excluded from source, logs and reports.

```bash
runs/dependency-migration-v1/app-env/bin/python3.12 -B experiments/pressure-balanced-shapes-v1/run.py \
  --config runs/pressure-balanced-shapes-v1/balanced-shapes-experiment.json \
  --assembly runs/pressure-balanced-shapes-v1/balanced-shapes-project-01/assembly.json \
  --qualification docs/plans/pressure-balanced-shapes-v1/reports/C69-qualification-01.json \
  --session-file runs/tuldok-performance-v1/private/owner-session.txt \
  --output runs/pressure-balanced-shapes-v1/balanced-shapes-controller-01
python3 -B experiments/pressure-fundamental-bank-fit-v1/evaluate.py \
  --config runs/pressure-balanced-shapes-v1/balanced-shapes-experiment.json \
  --projects runs/pressure-balanced-shapes-v1 \
  --controller runs/pressure-balanced-shapes-v1/balanced-shapes-controller-01/execution.json \
  --output runs/pressure-balanced-shapes-v1/balanced-shapes-cpu-01
python3 -B experiments/pressure-balanced-shapes-v1/report.py \
  --config runs/pressure-balanced-shapes-v1/balanced-shapes-experiment.json \
  --projects runs/pressure-balanced-shapes-v1 \
  --cpu runs/pressure-balanced-shapes-v1/balanced-shapes-cpu-01 \
  --controller runs/pressure-balanced-shapes-v1/balanced-shapes-controller-01/execution.json \
  --qualification docs/plans/pressure-balanced-shapes-v1/reports/C69-qualification-01.json \
  --output docs/plans/pressure-balanced-shapes-v1/comparison
```

Scalar/source verification and offline report:

```bash
python3 -B experiments/pressure-balanced-shapes-v1/verify.py \
  --directory docs/plans/pressure-balanced-shapes-v1/comparison \
  --output docs/plans/pressure-balanced-shapes-v1/reports/verification-01.json
python3 -B experiments/pressure-balanced-shapes-v1/render.py \
  --directory docs/plans/pressure-balanced-shapes-v1
node experiments/pressure-balanced-shapes-v1/browser.cjs \
  /home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell \
  runs/pressure-balanced-shapes-v1/browser-01
```

The browser owns a temporary profile and is reaped; it does not attach to or close
the user's browser. No model weights, simulation arrays or screenshots are added
to Git. The cohort page receives only the new report link and model ID reservation;
pre-existing global chart/model-import edits remain outside this experiment.
