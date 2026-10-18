# Method and reproduction

The configuration and preparation/qualification receipts in this directory bind
the actual experiment. `results.json` will retain scalar measurements and source,
input and checkpoint identities. Full training artifacts stay in ignored `runs/`
and the existing Tuldok catalog. Historical checkpoints and results are preserved.

## Scientific comparison

C60 and C61 start from the exact C53 weights, share fresh Adam state, 1,500 updates,
sampling/model seeds, learning rates, training data, train-only scales and all
analytical banks. Their sole objective difference is final low-mode feedback
weight 0 versus 0.1. Both retain the frozen C53 eight-pass teacher at weight 0.1
and earlier-pass equation feedback at total weight 0.05. C59 is a historical
reference, with different sampling seeds and supplemental bank; its difference
from these arms cannot isolate the bank's causal effect.
Their total loss scalars have different definitions and are not an accuracy
ranking. Qualification uses the same independent physical error measurements.

The primary broad bank is unchanged. The supplemental bank has 32 pressure
patterns × four signed amplitudes × eight grids = 1,024 examples. Equal grid
exposure gives each grid 187 or 188 of the 1,500 draws. Exact resident tensors use
38,035,456 bytes. Shapes informed by consumed validation failures use entirely
new manufactured labels and seeds. No recorded validation target enters training.
The existing generator combines discrete Neumann cosine modes, deriving RHS from
their analytical eigenvalues rather than the implementation under test.

The diagnostic separately tests six pure spatial modes and five amplitude scales
on each grid: 240 cases. Labels satisfy the captured operator to relative defect
below 1e-10. Amplitude is scalar pressure RMS relative to all original train
corrections. The range 1e-4 to 100 spans a millionfold; -1 tests sign reversal.
This is not a temporal frequency or timestep sweep. Identical mode indices on
different physical extents do not imply identical wavelengths in metres.

The final low-mode pressure objective projects mean-free pressure error into the
orthonormal cosine block with each axis index <=3, excluding DC. It normalizes by
the larger of low-mode target energy and 0.01 of full mean-free correction energy,
then applies `0.1 * log1p(normalized_squared_error / 1e-12)` in C61. Independent
pure-mode, constant-gauge, excluded-high-mode and finite-gradient checks passed.
Both arms keep canonical physical and final logarithmic equation objectives.
The fine-tuning wrapper owns the complete effective loss; it calls the baseline
physical criterion rather than baseline `training_loss`. Inherited configuration
labels such as baseline band-feedback weight 0.03 describe the captured C53 source,
not another added loss term in these fine-tunes. The explicit `fine-tuning.json`
and captured wrapper source define the terms actually optimized.

Real snapshot qualification divides equation error by full pressure RHS L2.
Training logarithmic feedback divides by initial correction-request residual L2.
Manufactured zero-base requests have the same initial residual and RHS. Do not
interchange these denominators, pressure error and trajectory drift.

## Execution

The existing visible catalog is `a7da4a8de53541a2a0e2aad08005bb71`, project root
alias `current-lab`, at `http://127.0.0.1:8091`. Managed Pumas Torch performs CUDA
training; a separate fresh managed Run restores each checkpoint on CPU without
optimizer updates. Source/closure identities and all artifact parity checks remain
required. No private trainer, additional catalog, model promotion or relaxed
physical acceptance is used.

The diagnostic execution receipt records complete commands and CPU affinity.
Preparation uses the app environment and `experiments/pressure-low-mode-tail-v1/prepare.py`.
Qualification uses the managed Torch interpreter with CUDA hidden and one thread
per worker. `registration.json` identifies the two qualified visible Projects.

The owned managed controller command is:

```sh
python3 -B experiments/pressure-low-mode-tail-v1/run.py \
  --config experiments/pressure-low-mode-tail-v1/config.json \
  --session-file runs/pressure-four-pass-distillation-v1/private/owner-session.txt \
  --directory runs/pressure-low-mode-tail-v1
```

This command is **not replayable** against an existing execution. Read durable
mutation journals/current owner API state before recovering an interrupted call.
It never automatically resubmits unknown mutations. Credentials are not retained
in Git. Tuldok owns bounded CUDA/CPU admission; two controllers do not imply two
simultaneous GPU trainers.

After both managed controllers and artifact readbacks finish:

```sh
python3 -B experiments/pressure-low-mode-tail-v1/evaluate.py \
  --config experiments/pressure-low-mode-tail-v1/config.json \
  --projects runs/pressure-low-mode-tail-v1 \
  --output runs/pressure-low-mode-tail-v1/cpu-01
```

Before rendering the final report, run `exposure.py` with the managed Torch
interpreter and CUDA hidden, using the C60 project/study and output
`docs/plans/pressure-low-mode-tail-v1/bank-exposure.json`. The exact interpreter,
arguments, elapsed time and reaping result are in `exposure-execution.json`.
This scalar-only postfit analysis uses the captured generator and retained history,
verifies the actual sampling-order hash, and counts spectral exposure. Then:

```sh
python3 -B experiments/pressure-low-mode-tail-v1/report.py \
  --config experiments/pressure-low-mode-tail-v1/config.json \
  --projects runs/pressure-low-mode-tail-v1 \
  --cpu runs/pressure-low-mode-tail-v1/cpu-01 \
  --output docs/plans/pressure-low-mode-tail-v1
python3 -B experiments/pressure-low-mode-tail-v1/view.py \
  --directory docs/plans/pressure-low-mode-tail-v1
```

Snapshot timing is serial, one CPU core/thread, one checkpoint per process,
after managed work drains. Three measurements per request follow one warm-up;
recurrence and final equation check are timed, data assembly is excluded.
Manufactured postfit diagnostics run afterward on two distinct cores. Keep a
12 GiB host reserve plus 2 GiB per CPU worker; teacher cache is capped at 256 MiB.
Record trainer time, training-loop/bank time and complete workflow time separately.
GPU observations include the desktop; per-run peak allocation comes from trainer
receipts. No inference GPU support or deployment default is changed.

## Readback measurement

The separate workflow change measures typed owner operation calls. Direct owner
session negotiation and the raw restore-approval route are included in whole
workflow elapsed time, but are not individually timed by this profile. The collector
reuses bytes only after a current authorized terminal Run declares the same whole
SHA256 and length as an artifact already downloaded in this controller. Source
bytes are checked again, destinations are created exclusively, and published
length/hash are checked afterward. Current owner authorization, Run custody,
typed contracts and restore-specific receipts are not skipped. No private Run
storage is accessed. A focused corruption regression proves changed local bytes
are refused. This optimization cannot eliminate admission or lineage validation
overhead; the profile records actual costs without assuming their internal cause.
