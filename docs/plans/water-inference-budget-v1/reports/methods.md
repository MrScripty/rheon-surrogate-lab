# Reproduction and measurement

Use the Pumas Torch interpreter selected by the existing measured runtime
`runs/cuda-pressure-runtime-1/manifest.json` (Torch 2.14.0+cu132, NumPy 2.5.3).
Managed stages use the user's existing Tuldok catalog and `training/workflow.py`.
No experiment optimizer runs outside Tuldok. Reserved-test recipes are captured
for disjointness but never selected for execution.

Both single-pass configurations were captured at source 8c6ddc7, using the
existing `training/pipeline.py assemble` with fitting packet
`runs/water-learning-sequence-v1/views/fit-one`, development packet
`runs/water-learning-sequence-v1/views/validation`, and existing reserved packet.
Outputs are under `runs/water-inference-budget-v1/TRIAL/project`. Configurations
are the committed `stability-{modes,combined}-single-pass.json` files. Each keeps
5,000 real-frame exposures and 5,000 auxiliary analytical exposures, seed, sample
order, train-only scales, optimizer schedule and physical gates matched to its
four-pass control. Four versus one shared network evaluations per example is
the intervention. Final checkpoints are predetermined terminal updates.

Session authentication wrapper `/tmp/run-budget-managed.py --root
runs/water-inference-budget-v1 --trial TRIAL --phase full` invokes the existing
workflow for CUDA train, fresh CPU restore, and development inference. It reopens
normal Runs, downloads all artifacts twice with complete hash/byte parity, and
logs out. Credentials remain in memory. Wrapper commands/logs are retained
locally; all scientific sources and settings are captured/committed.

CPU qualification uses the reusable `experiments/water-inference-budget-v1/
evaluate.py` with explicit `--root`, `--trial`, `--packet`, `--validation`,
`--native-binary` and `--rollout-binary`. This retains every command, audit,
native evaluation and three independent trajectory results. It does not relabel
direct CPU diagnostics as managed Runs.

Adaptive reference inference uses `experiments/water-stability-learning-v1/
learned_iteration.py` with the retained stability-modes captured project and
fresh CPU restore, `--population validation --proposals 8 --adaptive-profile
experiments/water-inference-budget-v1/configs/adaptive.json`. The output is
`runs/water-inference-budget-v1/reference-adaptive-validation`. The existing
`native_parallel.py` consumes its explicit diagnostic receipt and independently
checks the unchanged float32 physical profile. Source ad89d11 owns the optional
stopping rule and diagnostics. No pressure label is used for choosing corrections.
Actual calls and traces are retained per case and per native callback.

`/tmp/budget-adaptive-trajectories.py` runs three 64-step scenarios on distinct
CPU cores 1,2,3; `/tmp/budget-adaptive-extended.py` uses the previously qualified
240-step cube and 256-step rectangular/compact scenes. Exact argument arrays,
per-step native physics and callback work live under the new run root. Their
timings overlap GPU/preflight work and are not isolated speedup measurements.

Scaling uses `scaling.py --config configs/scaling.json --adaptive-profile
configs/adaptive.json --project CAPTURE --artifacts CPU_RESTORE --binary
NATIVE_ROLLOUT --output NEW_OUTPUT --cpu CPU`, with config paths relative to the
new experiment directory. Execute after all managed jobs and task CPU workers
finish. Whole-track timings include preparation, projection, JSON/IPC and
Python callbacks; native pressure-kernel timings are unavailable. Three repeated
episodes alternate fixed/adaptive policy order. Each has 15 native history steps,
one excluded neural warm-up step and seven measured AI steps. Both policies are
checked physically; timing alone never establishes qualified acceleration.
Existing native equation tolerance is 1e-8 versus AI 1e-6; all other physical
limits stay unchanged. The 16-cubed scene is shared by both scaling sweeps.

Data/checkpoints, failed native controls and full process logs stay outside Git.
Small receipts/results in this directory retain source/input/checkpoint hashes,
normal Run IDs, lifecycle outcomes and numerical failures for future review.
