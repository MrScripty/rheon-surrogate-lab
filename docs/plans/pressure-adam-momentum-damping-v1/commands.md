# Commands and environment

Run from the repository root, with a fresh output/result directory on repeat:

```bash
python3 -B experiments/pressure-adam-momentum-damping-v1/run.py \
  --config experiments/pressure-adam-momentum-damping-v1/config.json \
  --output runs/pressure-adam-momentum-damping-v1/workers-02 \
  --results runs/pressure-adam-momentum-damping-v1/final-measurements
python3 -B experiments/pressure-adam-momentum-damping-v1/report.py \
  --config experiments/pressure-adam-momentum-damping-v1/config.json \
  --execution runs/pressure-adam-momentum-damping-v1/workers-02/execution.json \
  --directory docs/plans/pressure-adam-momentum-damping-v1
python3 -B experiments/pressure-adam-momentum-damping-v1/lifecycle.py \
  --output runs/pressure-adam-momentum-damping-v1/cancellation-01
node experiments/pressure-adam-momentum-damping-v1/browser.cjs \
  /home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell \
  runs/pressure-adam-momentum-damping-v1/browser-03
python3 -B experiments/pressure-adam-momentum-damping-v1/verify.py \
  --directory docs/plans/pressure-adam-momentum-damping-v1 \
  --output docs/plans/pressure-adam-momentum-damping-v1/reports/verification.json
```

Before final report generation, copy final update/qualification records to
`docs/plans/pressure-adam-momentum-damping-v1/reports`, and the controller's
execution.json there too. Copy cancellation/browser receipts to that reports
directory. The retained repeatability receipt compares initial/final candidate
hashes and scalar metrics after the workspace accounting review. Existing
destinations are retained evidence; the runner rejects overwriting them.

Numerical child commands, PIDs, CPU affinity, returncodes, RSS, elapsed seconds,
runtime and config SHA256 are in reports/execution.json. Runtime is selected
from the existing Pumas manifest in config: Python 3.14.7, Torch 2.14.0+cu132,
NumPy 2.5.3. `common.environment` hides CUDA, disables bytecode and fixes
OMP/OPENBLAS/MKL/RAYON threads to 1. Parent uses the system Python standard
library only. Network parameters and disposable Adam operations are float32;
captured physical loss/residual calculations retain their float64 owner.

Historical step-screen Adam endpoint code is imported unchanged, so its prior
measurements and source witnesses remain valid. Trace custody and model/data
sources are checked against the published Tuldok replay receipts; no separate
catalog, Project, private model runtime or training Run is created. Existing
Tuldok-linked activity shows execution/completion. No login or external message
is sent. No weights or datasets are part of the commit.

Standards: existing resolved MCP snapshot
`snapshot:v1:ffcfc3f8-b52a-434e-b57f-d259a0355711`; commit workflow reread from
the MCP for staged-scope, verification and conventional-message guidance.
Scalar recurrence is an independent numerical oracle. Historical control
identity proves replay consistency, not new external physics conformance.
