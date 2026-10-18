# Commands and environment

Run from the repository root. Runtime files stay under ignored `runs/`; the
private owner-session file is never copied to the report. The existing Tuldok
catalog and Project execution path are used. Do not replay managed admission
commands for an existing study. The journal is authoritative if interrupted.

The Pumas runtime manifest is
`runs/dependency-migration-v1/runtime/current-manifest-v2.json`: CPython 3.14.7,
PyTorch 2.14.0+cu132, NumPy 2.5.3, RTX 5090 Laptop GPU. The app controller uses
`runs/dependency-migration-v1/app-env/bin/python3.12`. Managed receipts retain
actual runtime/device details and source identities; the gradient worker receipt
retains the exact Python executable and worker commands. CUDA is hidden for CPU
workers, each with one Torch/BLAS thread and distinct affinity. Only one managed
CUDA training/restore workflow is admitted at a time.

Initial diagnosis and C67 preparation/qualification:

```bash
python3 -B experiments/pressure-shape-decay-v1/diagnose.py \
  --config experiments/pressure-shape-decay-v1/config.json \
  --output runs/pressure-shape-decay-v1/gradient-workers-02 \
  --results docs/plans/pressure-shape-decay-v1/reports
python3 -B experiments/pressure-shape-decay-v1/prepare.py \
  --config experiments/pressure-shape-decay-v1/config.json \
  --arm shape-later-decay \
  --output runs/pressure-shape-decay-v1/shape-later-decay-project-01
python3 -B experiments/pressure-shape-decay-v1/qualify.py \
  --config runs/pressure-shape-decay-v1/shape-later-decay-experiment.json \
  --project runs/pressure-shape-decay-v1/shape-later-decay-project-01 \
  --output docs/plans/pressure-shape-decay-v1/reports/C67-qualification.json
```

One-shot managed execution, followed only after completion by fresh CPU checks:

```bash
runs/dependency-migration-v1/app-env/bin/python3.12 -B \
  experiments/pressure-shape-decay-v1/run.py \
  --config runs/pressure-shape-decay-v1/shape-later-decay-experiment.json \
  --assembly runs/pressure-shape-decay-v1/shape-later-decay-project-01/assembly.json \
  --qualification docs/plans/pressure-shape-decay-v1/reports/C67-qualification.json \
  --session-file runs/pressure-four-pass-distillation-v1/private/owner-session.txt \
  --output runs/pressure-shape-decay-v1/shape-later-decay-controller-01
python3 -B experiments/pressure-fundamental-bank-fit-v1/evaluate.py \
  --config runs/pressure-shape-decay-v1/shape-later-decay-experiment.json \
  --projects runs/pressure-shape-decay-v1 \
  --controller runs/pressure-shape-decay-v1/shape-later-decay-controller-01/execution.json \
  --output runs/pressure-shape-decay-v1/shape-later-decay-cpu-01
python3 -B experiments/pressure-shape-decay-v1/report.py \
  --config runs/pressure-shape-decay-v1/shape-later-decay-experiment.json \
  --projects runs/pressure-shape-decay-v1 \
  --cpu runs/pressure-shape-decay-v1/shape-later-decay-cpu-01 \
  --controller runs/pressure-shape-decay-v1/shape-later-decay-controller-01/execution.json \
  --qualification docs/plans/pressure-shape-decay-v1/reports/C67-qualification.json \
  --output docs/plans/pressure-shape-decay-v1/reports/shape-later-decay
```

Conditional C68 preparation requires `--arm joint-later-decay`, its fresh output
directory, and `--decision` pointing to the completed passing C67 decision.
It is prohibited if that full gate fails. The plan states the remaining matching
and restore requirements; a failed screen is not permission for a longer fit.

Report generation and focused verification:

```bash
python3 -B experiments/pressure-shape-decay-v1/collect.py \
  --directory docs/plans/pressure-shape-decay-v1
python3 -B experiments/pressure-low-mode-tail-v1/view.py \
  --directory docs/plans/pressure-shape-decay-v1
python3 -B experiments/pressure-shape-decay-v1/verify.py \
  --directory docs/plans/pressure-shape-decay-v1 \
  --output docs/plans/pressure-shape-decay-v1/reports/verification.json
node experiments/pressure-shape-decay-v1/browser.cjs \
  /home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell \
  runs/pressure-shape-decay-v1/browser-final-02
```

The browser command owns a temporary headless browser and verifies/reaps it;
it does not attach to or close the user's browser. Screenshots stay in `runs/`.
The final retained browser receipt names the exact rendered HTML hash.
