# Methods and reproduction

Training uses the existing Tuldok catalog and qualified Pumas Torch 2.14.0+cu132,
NumPy 2.5.3, Python 3.14 runtime. The lab .venv executes the authenticated HTTP
controller; it does not execute an optimizer. No new catalog or token file is
created. Controller authentication stays in memory and logs out on exit.

Source must be committed and the working tree clean before assembling a capture:

```bash
"$TORCH_PYTHON" experiments/water-conditioning-v1/training/pipeline.py assemble \
  --packet runs/water-learning-sequence-v1/views/fit-one \
  --validation runs/water-learning-sequence-v1/views/validation \
  --reserved runs/water-learning-sequence-v1/views/reserved-test \
  --runtime-manifest runs/cuda-pressure-runtime-1/manifest.json \
  --tuldok runs/live-integration/Tuldok \
  --config experiments/water-conditioning-v1/configs/multigrid-modes-single-pass.json \
  --output runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project
```

The normal authenticated controller calls conditioning training/workflow.execute
with that captured project and a fresh managed-full directory, selecting train,
restore and validation. It uses exact graph preflight/approval, completed/drained
Run inspection, whole artifact downloads, reopened reads and identical repeated
downloads. No reserved_test stage runs. The smoke selects train and restore only.
Local invocation uses the retained session helper `/tmp/run-budget-managed.py`
with `--root runs/water-multigrid-learning-v1 --trial NAME --phase full` (or
`--phase smoke`), under `.venv/bin/python`. Helper/authentication paths are local
controller details, not reusable experiment configuration or checkpoint identity.
The captured source-witness and managed receipts own reproducible material.

Bank qualification: run qualify_bank.py using the new configuration, original
fit-one packet, and the retained previous single-pass capture's analytical.py and
original configuration. This checks all 640 bank entries and exact legacy parity.

CPU study: run study.py with configs/existing-checkpoints.json and a fresh output.
Seven physical scenes cover 16³, 24³, 32³, 40³ with either a fixed 1 m cube or
fixed 16 cells/metre (the common 16³ baseline occurs once). Each has 15 native
history steps, one excluded neural timing warm-up and seven measured steps.
Use separate fixed cores for numerical throughput, then one core and alternating
three repetitions for timing. These distinct conditions are labelled in reports.
All native controls must qualify before comparing learned tracks. No native
cleanup contributes to raw AI success. Pressure labels never enter inference.

Development evaluation: evaluate.py accepts explicit project, managed CPU restore,
packet, population, native binary and fresh output. It retains exact inference
and native commands, 128 independent native physical results and actual neural
counts. Every diagnostic pool uses one thread/core per worker, up to 23 workers;
case count and available independent work bound effective CPU occupancy.

Profile: cpu_profile.py uses explicit checkpoint, fitting amplitude, analytical mode
mixture and cubic grid. One warm-up, three unprofiled repetitions and one Torch
CPU profile separate network and physical operator work. Explicit core affinity
and 1/4/8 thread counts characterize this mechanism only; whole-track IPC/native
work remains measured by trajectory studies. No performance claim from concurrent
CPU/GPU execution or from this microprofile alone.

All experiment outputs use fresh directories under ignored runs/. Small receipts,
methods and decisions are committed here. Checkpoints/data stay in the existing
Tuldok managed artifacts and local ignored captures. The preceding tranche remains
unchanged. Fixed32, original adaptive and refined adaptive are distinct policies.

The optional rollout `--threads 1|4|8` applies the same explicit thread budget
to native and AI tracks. The study gives each worker a disjoint core group and
accepts `--timing-study` only for serial repeated observations; caller isolation
is still required. The four-thread positive fixture uses one AI step after
native history and records actual affinity. Default policy remains one thread.

Longer-window configuration long-window.json doubles the previous largest
bounded time window to 512 steps (8.5333 s), with quarter-strength internal
forcing. This is a separate set of native-qualified conditions, not an extension
of the earlier full-force claim. Report early/intermediate/final pressure-field
relative L2 difference alongside every step's unchanged physical gates.

Prepared-cohort audit reads the existing lossless cohort-16 view, inspects only
training parents, verifies 16 distinct physical-setup identities and 32 real
snapshots, and records cubic spacing/force/history metadata. It neither trains
nor infers on the untouched final population. Broad-cohort feedback needs an
explicit amplitude-reference contract because the current controlled bank takes
one fitting frame; do not silently use the first parent as the global statistic.

Final timing design: final-timings.json names each restored checkpoint explicitly,
uses three alternating repetitions on fixed-volume16/24/32/40-cubed and a
rectangular24×16×12 scene, and retains native controls. Run only admitted/completed
checkpoints; a prepared cohort entry must be removed if its fit is not admitted
or completed. Two serial studies use `--threads 1` and `--threads 4`, one worker,
and `--timing-study`, after GPU training and task CPU qualifications drain. All
tracks get the same core/thread allocation within each study. Existing distinct
native 1e-8/AI1e-6 residual profiles remain stated; do not present them as a
matched-tolerance pressure-kernel benchmark. Exclude first AI timing step and
retain seven samples/repetition and all physical outcomes.

Equal-budget comparison: evaluate.py accepts `--network-budget 8 --fixed-budget`
with the same managed CPU restores and 128-state validation packet. It omits
adaptive early stopping and retains a distinct fixed-budget result schema. Both
predictions use identical case/population identities, are checked by the same
native oracle/profile, and contribute no native refinement to AI success.
The default remains32-cap adaptive inference. Retained raw/fixed/adaptive cases
are separate evidence; no budget failure is erased by another policy's success.

summarize.py derives the 13-stage/15020-update inventory and comparison CSV from
sealed trial/smoke records and completed timing reports. trajectory_summary.py
checks common scenes/native binary and original receipt hashes, then retains
60 pressure-drift samples from 12 paired trajectories. No hard-coded pass totals
replace the retained per-episode observations. Final source/witness/checkpoint
and model-interface details are in model-contract.md.
