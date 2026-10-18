# Commands, runtime and reproducibility

From the repository root:

```bash
python3 -B experiments/pressure-adam-step-screen-v1/run.py \
  --config experiments/pressure-adam-step-screen-v1/config.json \
  --output runs/pressure-adam-step-screen-v1/workers-03 \
  --results docs/plans/pressure-adam-step-screen-v1/reports
python3 -B experiments/pressure-adam-step-screen-v1/report.py \
  --config experiments/pressure-adam-step-screen-v1/config.json \
  --execution runs/pressure-adam-step-screen-v1/workers-03/execution.json \
  --directory docs/plans/pressure-adam-step-screen-v1
node experiments/pressure-adam-step-screen-v1/browser.cjs \
  /home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell \
  runs/pressure-adam-step-screen-v1/browser-04
```

The numerical children use Pumas' Python/site-packages selected by
`runs/dependency-migration-v1/runtime/current-manifest-v2.json`: CPython 3.14.7,
Torch 2.14.0+cu132, NumPy 2.5.3. `common.environment` hides CUDA, selects CPU,
disables bytecode and sets OMP/OPENBLAS/MKL/RAYON threads to 1. Workers pin
separate CPUs. All actual numerical commands/PIDs/returncodes, config/runtime
hashes and cessation are retained in reports/execution.json. The parent needs no
Torch installation. Existing Tuldok-linked activity publishes the task; no new
catalog, Project, model, training Run, external message or login was created.

Literal oracles run as the first owned child using the same managed numerical
runtime; qualifier receipt identifies the independent scalar recurrence, literal
interpolation, initialized-state control and exact negative diagnostics. The
cancellation fixtures' actual commands and every owned process are in
reports/cancellation.json: SIGTERM during qualifier startup (cancel-02), then
SIGINT after three snapshot workers start (cancel-03). Both use fresh ignored
destinations and signal only their own controller.

These completed destinations are evidence, not rerun paths. Use a fresh process
root/result directory for a replay. Five published trace archives and all model
sources are checked against their existing managed custody/identity records.
No historical hashes were replaced, raw trace/weights altered, or vectors
written into this study's Git records. Policy/fraction aliases refer to hashes
of ephemeral float32 parameter vectors.

Branch experiment/pressure-adam-step-screen-20261018 starts at
be223233ba941914dc43d2c3316e66201bfb6788 on the prior replay branch. The lab user
owns integration and retirement. No worktree or merge/rebase/push was performed;
concurrent chart and THIRD_PARTY edits remain outside this commit slice.

Final record verification:

```bash
python3 -B experiments/pressure-adam-step-screen-v1/verify.py \
  --directory docs/plans/pressure-adam-step-screen-v1 \
  --output docs/plans/pressure-adam-step-screen-v1/reports/verification.json
```
