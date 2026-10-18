# Commands and environment

Run from the repository root. Numerical workers use the Pumas-qualified
`runs/dependency-migration-v1/runtime/current-manifest-v2.json` (CPython 3.14.7,
Torch 2.14.0+cu132, NumPy 2.5.3). The launcher reads that exact interpreter and
site-packages selection, hides CUDA, sets OMP/OpenBLAS/MKL/Rayon to one thread,
and binds workers to three distinct logical CPUs. Exact command, runtime-manifest
hash, CPU, durations and reaping are in reports/execution.json.

Literal oracle and actual-pool cancellation checks:

```bash
python3 -B experiments/pressure-aggregate-gradients-v1/check.py \
  --output docs/plans/pressure-aggregate-gradients-v1/reports/math-check.json
python3 -B experiments/pressure-aggregate-gradients-v1/check-cancel.py \
  --config experiments/pressure-aggregate-gradients-v1/config.json \
  --scratch runs/pressure-aggregate-gradients-v1/cancellation-02 \
  --output docs/plans/pressure-aggregate-gradients-v1/reports/cancellation-final.json
```

The actual captured C69 derivative-guard command is retained in joint-worker.json.
Its worker uses the same manifest-derived `common.environment(runtime)` as the
pool, a 120-second deadline and process-group TERM/KILL/reap. The worker command
is `check-joint.py --config .../config.json --scratch .../joint-guard-01 --output
.../reports/joint-guard.json`. It first proves the valid joint derivative, then
detaches only final feedback without changing values. The exact expected
diagnostic is retained. No optimizer runs in this fixture.

Completed numerical diagnosis and reporting:

```bash
python3 -B experiments/pressure-aggregate-gradients-v1/diagnose.py \
  --config experiments/pressure-aggregate-gradients-v1/config.json \
  --output runs/pressure-aggregate-gradients-v1/workers-01 \
  --results docs/plans/pressure-aggregate-gradients-v1/reports
python3 -B experiments/pressure-aggregate-gradients-v1/report.py \
  --config experiments/pressure-aggregate-gradients-v1/config.json \
  --execution runs/pressure-aggregate-gradients-v1/workers-01/execution.json \
  --directory docs/plans/pressure-aggregate-gradients-v1
python3 -B experiments/pressure-aggregate-gradients-v1/verify.py \
  --directory docs/plans/pressure-aggregate-gradients-v1 \
  --output docs/plans/pressure-aggregate-gradients-v1/reports/verification.json
python3 -B experiments/pressure-aggregate-gradients-v1/render.py \
  --directory docs/plans/pressure-aggregate-gradients-v1
node experiments/pressure-aggregate-gradients-v1/browser.cjs \
  /home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell \
  runs/pressure-aggregate-gradients-v1/browser-01
```

Use fresh worker/receipt destinations for any replay; retained completed outputs
are evidence. Raw float32 vectors stay under ignored runs/, with payload about
56.46 MiB per checkpoint. Their identities are in each worker receipt; Git keeps
only small scalar geometry, provenance and test outcomes. The offline browser
uses a temporary profile and reaps its own process; the user's browser/apps remain
open. No Tuldok admission, CUDA training or simulation capture occurs here.
