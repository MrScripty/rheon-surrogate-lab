# Commands and environment

Run from the repository root. Schema preparation/controller commands use the
existing app environment. Numerical children use the Python/site-packages in
`runs/dependency-migration-v1/runtime/current-manifest-v2.json`, provisioned by
Pumas (Torch 2.14.0+cu132, NumPy 2.5.3, CPython 3.14.7). The retained launcher
receipts contain the exact interpreter, arguments, source/runtime hashes and
process cessation. Never print the existing owner session.

Completed successful Project preparation and qualification:

```bash
runs/dependency-migration-v1/app-env/bin/python3.12 -B \
  experiments/pressure-adam-replay-v1/prepare.py \
  --config experiments/pressure-adam-replay-v1/config.json \
  --output runs/pressure-adam-replay-v1/attempt-04/adam-replay-project-01
runs/dependency-migration-v1/app-env/bin/python3.12 -B \
  experiments/pressure-adam-replay-v1/qualify_worker.py \
  --config runs/pressure-adam-replay-v1/attempt-04/adam-replay-experiment.json \
  --project runs/pressure-adam-replay-v1/attempt-04/adam-replay-project-01 \
  --output runs/pressure-adam-replay-v1/attempt-04/qualification.json \
  --execution runs/pressure-adam-replay-v1/attempt-04/qualification-execution.json
```

Managed execution (existing owner API, catalog, current-lab alias, visible Project):

```bash
runs/dependency-migration-v1/app-env/bin/python3.12 -B \
  experiments/pressure-adam-replay-v1/run.py \
  --config runs/pressure-adam-replay-v1/attempt-04/adam-replay-experiment.json \
  --assembly runs/pressure-adam-replay-v1/attempt-04/adam-replay-project-01/assembly.json \
  --qualification runs/pressure-adam-replay-v1/attempt-04/qualification.json \
  --session-file runs/tuldok-performance-v1/private/owner-session.txt \
  --output runs/pressure-adam-replay-v1/attempt-04/adam-replay-controller-01
```

CPU analysis uses the same Pumas runtime with CUDA hidden, one thread/core,
12 GiB host reserve plus a 2 GiB worker budget, 120-second deadline and owned
process-group TERM/KILL/reap. The managed controller also owns/reaps its child;
remote cessation is established separately by Tuldok terminal drained receipts.

```bash
runs/dependency-migration-v1/app-env/bin/python3.12 -B \
  experiments/pressure-adam-replay-v1/analyze_worker.py \
  --config runs/pressure-adam-replay-v1/attempt-04/adam-replay-experiment.json \
  --project runs/pressure-adam-replay-v1/attempt-04/adam-replay-project-01 \
  --study runs/pressure-adam-replay-v1/attempt-04/adam-replay-study-01 \
  --output runs/pressure-adam-replay-v1/attempt-04/analysis.json \
  --execution runs/pressure-adam-replay-v1/attempt-04/analysis-execution.json
python3 -B experiments/pressure-adam-replay-v1/report.py \
  --base runs/pressure-adam-replay-v1/attempt-04 \
  --output docs/plans/pressure-adam-replay-v1
node experiments/pressure-adam-replay-v1/browser.cjs \
  /home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell \
  runs/pressure-adam-replay-v1/browser-01
```

These destinations are retained evidence, not rerun targets. Every new admission
needs fresh paths and request keys; never move failed evidence back into an old
request-key path. This study records failures below and does not retry ambiguous
mutations. Sources/custody come from ordinary managed readback. Private Run inspections were read-only stderr/custody diagnosis and byte-format
compression qualification of copied but unpublished trace payloads. No numerical
results or checkpoint restore were recovered through that path. Scientific
analysis uses only fresh completed managed publication/readback.

Final retained-record verification:

```bash
python3 -B experiments/pressure-adam-replay-v1/verify.py \
  --directory docs/plans/pressure-adam-replay-v1 \
  --output docs/plans/pressure-adam-replay-v1/reports/verification.json
```

`reports/resource-policy-regression.json` records the exact historical resource
rejection and corrected acceptance. `reports/evidence-shard-regression.json`
records the actual decoder's bounded-file rejection and corrected shard reads;
`reports/transport-fixture.json` verifies lossless decompression of all 35 members.
These format fixtures did not import unpublished scientific outcomes.
