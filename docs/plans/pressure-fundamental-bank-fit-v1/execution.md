# Execution commands

Working directory is the rheon-surrogate-lab repository. Python `-B` avoids
bytecode writes to captured sources. Project preparation uses the existing app
environment; qualification and CPU inference use the Pumas managed Torch runtime
and `common.environment(runtime)` from the four-pass workflow owner. That sets
CPU device, hides CUDA and limits native CPU thread pools to one. Never substitute
an ambient Torch installation.

```sh
runs/dependency-migration-v1/app-env/bin/python3.12 -B \
  experiments/pressure-fundamental-bank-fit-v1/prepare.py \
  --config experiments/pressure-fundamental-bank-fit-v1/config.json \
  --arm combined-fundamentals \
  --output runs/pressure-fundamental-bank-fit-v1/combined-fundamentals-project-01
```

Run `qualify.py --project <project> --output <fresh-composite-receipt>` using the
manifest's `python` and CPU environment. A base receipt alone cannot admit a run.
The bounded bank checks add independent operator, gradient, sampling, initialization
and memory evidence without CPU optimizer updates.

```sh
runs/dependency-migration-v1/app-env/bin/python3.12 -B \
  experiments/pressure-fundamental-bank-fit-v1/run.py \
  --config runs/pressure-fundamental-bank-fit-v1/combined-fundamentals-experiment.json \
  --assembly runs/pressure-fundamental-bank-fit-v1/combined-fundamentals-project-01/assembly.json \
  --qualification docs/plans/pressure-fundamental-bank-fit-v1/combined-qualification-01.json \
  --session-file runs/pressure-four-pass-distillation-v1/private/owner-session.txt \
  --output runs/pressure-fundamental-bank-fit-v1/combined-fundamentals-controller-01
```

The session file is private, not committed. Existing local Tuldok catalog
`a7da4a8de53541a2a0e2aad08005bb71`, Project alias `current-lab`; endpoint
`http://127.0.0.1:8091`. The controller records the exact child command and source
hash and publishes visible project activity. Tuldok owns bounded remote Runs;
terminating the controller is not proof that remote work stopped. Do not replay an
unresolved admission. Read the durable journal and known Run instead.

Run `monitor.py` from the four-pass workflow with `--completion <execution.json>`
and a fresh GPU JSON destination. It samples whole-GPU load, including desktop
activity; trainer receipts separately record allocated GPU memory.

After managed completion, run new `evaluate.py --config <frozen-arm-config>
--projects runs/pressure-fundamental-bank-fit-v1 --controller <execution.json>
--output <fresh-arm-cpu-directory>` using system Python. It launches the verified
Pumas CPU runtime, times C53 and candidate serially, then runs three independent
fitting/manufactured workers with separate cores and a memory reserve.

Run new `report.py --config <frozen-arm-config> --projects <run-root> --cpu <cpu-dir>
--controller <execution.json> --output <fresh-arm-docs-dir>`. Its completed
`capacity-decision.json` is required for any conditional subset preparation:
`prepare.py --arm axis-isolation|shape-isolation --decision <combined-decision>`.
Each has fresh qualification, controller, study, CPU and report directories.

After the required screens finish, run `collect.py --directory <study-docs>`,
`verify.py --directory <study-docs> --output <verification.json>`, and the existing
`pressure-low-mode-tail-v1/view.py --directory <study-docs>`. Keep all per-arm
immutable results alongside the aggregate. Browser receipts verify scalar bindings,
controls, report discovery and owned browser reaping.
