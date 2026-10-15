# Rheon Surrogate Lab

Current data work trains on [varied sealed domains full of water](experiments/sealed-water-v1/README.md), with cubic cells, viscous native references and localized fluctuating forces. The configuration reserves 256 training, 64 validation and 64 final-test simulations. Results will distinguish execution qualification from pressure accuracy.

Current model work uses [CUDA training with CPU deployment inference](experiments/cuda-pressure-v1/README.md)
for the unchanged seven-channel variable-shape pressure network. A managed
20-update CUDA run and fresh CPU restore passed [portability qualification](docs/plans/cuda-training-cpu-inference/results.md);
all twelve learned proposals still fail the pressure accuracy target.
Voxel size is one scalar in physical metadata, with no voxel-size network channels.
Current learned-output acceptance uses a [float32-calibrated residual target of
1e-6](experiments/float32-pressure-target-v1/README.md), with existing physical
limits retained. Historical experiments keep their original targets and results.
The [proposed next training sequence](docs/plans/pressure-learning-next-runs/proposal.md)
requires CUDA training with CPU deployment inference, then one-case fitting and
controlled normalization comparisons.
Explore its `[batch, channel, Z, Y, X]` inputs in the interactive
[pressure tensor page](viewer/tensor.html), with [recorded-data instructions](viewer/TENSOR-README.md).
Variable-domain capture uses [cubic physical cells](experiments/cubic-pressure-v1/README.md):
16 cells per metre, with per-axis counts derived from domain size. Historical
experiments below retain their original grids and results.

Author: Puma.

Live work is visible through the repo's [Tuldok project](tuldok-project.json) and
[Rheon egui observation workflow](experiments/live-project-v1/README.md).

Fresh local GroupNorm CPU experiments use the separate
[reusable CPU pipeline](experiments/local-pressure-v1/README.md), with configurable
development reservations, fixture-first qualification and managed train/restore.
Its [acceptance plan](docs/plans/reusable-local-cpu-pipeline/plan.md) records current
verification status. Historical pressure experiments and results remain frozen.

The project now includes a **trained pressure model inside a time-evolving 3D
Rheon simulation**, with GPU training through Pumas Library's managed PyTorch
2.14.0, complete-rollout timing, numeric accuracy and a synchronized 3D timeline.
The first pressure-only experiment below remains frozen and historical.

## Timed 3D simulation

**CPU-only control:** the unchanged trained model also runs entirely on CPU.
Giving every method eight native CPU threads and the model eight PyTorch CPU
threads produced median 120-frame times of **15.21 s Jacobi, 9.39 s SGS and
1.86 s learned**: **8.20× / 5.06× speedups** without GPU execution. Three balanced
repetitions across four seeds qualified all 4,320 timed frames; independent
full-field replay found worst velocity relative L2 error **9.00e-8** against both
CPU references. CUDA was hidden and all model tensors were checked to reside on
CPU. [CPU experiment receipts](experiments/trajectory-cpu-v1/README.md) preserve
this hardware-matched control. The original GPU-model comparisons below remain
separately labelled.

Experiment two learns 32,767 global pressure responses from real Rheon frames.
During inference it replaces the iterative pressure solve completely; Rheon
performs advection, forcing, gradient correction, tracer transport and gates.
The 32³ uniform sealed unit box uses prescribed three-component forcing and a
passive tracer. It has no free liquid surface, viscosity or obstacles.

Across four independent 120-frame test trajectories, maximum full-grid velocity
relative L2 error is **8.89e-8** and tracer relative L2 error is **5.89e-8**. All
7,200 benchmark frames passed physics gates, and every AI frame met the native
reference's stricter observed relative residual tolerance. No teacher states
enter the AI rollout. Untrained coefficients fail the divergence gate.

| Path | Median complete 120-frame wall time | Speedup |
| --- | ---: | ---: |
| Rheon Jacobi CPU | 22.01 s | reference |
| Rheon SGS CPU | 19.08 s | reference |
| Learned pressure + Rheon | 5.18 s | 4.25× vs Jacobi; 3.69× vs SGS |

These first timings use single-thread native kernels, a resident CUDA model and
five balanced repetitions across four seeds. Each includes native startup,
every simulation stage, model/CPU/GPU transfer, qualification, reply parsing and
native teardown. Imports/model initialization, capture and rendering are separate.
The parallel CPU execution comparison is retained as a separate experiment.
This is time to calculate two simulated seconds, not a real-time claim.

With Rheon's parallel kernels enabled, validation selected 4 threads for Jacobi
and 8 for SGS and the AI path's remaining native stages. A separate three-repeat,
four-seed comparison measured **16.35 s Jacobi, 9.67 s SGS and 1.76 s learned**:
**9.30× and 5.50× faster**. These results are retained with their own execution
freeze and receipts alongside the first experiment.
Independent full-field replay at those thread counts found maximum velocity
relative L2 error **9.00e-8** across both CPU references and four test seeds.

CUDA training peaked at 264 MB reserved; Python training peaked at 1.50 GB RSS.
Inference/evaluation peaked at 37.7 MB CUDA reserved and 2.00 GB Python RSS,
comfortably within the 20/8 GB budgets. CUDA driver/context allocations are
outside PyTorch allocator statistics. Float64 weights already occupy only 266 KB;
no four-bit quantization is needed. A 500M parameter model is unnecessary for
this fixed-operator contract. The analytic spectral alternative remains a valid
nonlearned solution; this model does not claim to beat it.

Keep this checkout alongside Rheon at the recorded commit. Select v2.14.0 in
Pumas Library, then run from the lab root using a new output directory:

```sh
export CARGO_TARGET_DIR="$PWD/target"
cargo build --release --locked --manifest-path probes/trajectory/Cargo.toml
python3 scripts/pumas_trajectory.py train runs/my-fluid --device cuda
python3 scripts/pumas_trajectory.py evaluate runs/my-fluid
python3 scripts/package_trajectory_viewer.py runs/my-fluid/evaluation
python3 -m http.server 8098 --bind 127.0.0.1
```

Open `http://127.0.0.1:8098/viewer/?run=my-fluid`. The view replays two independent
accepted trajectories, with orbit/pan/zoom, play/pause, frame scrubbing, simulated
time, actual frame calculation times and divergence. It uses recorded 3D fields.
Compact opacity exports are for display; accuracy uses full saved numeric fields.

For the default CPU-only view, first generate its comparison and replay:

```sh
python3 scripts/pumas_trajectory.py cpu-compare runs/my-fluid runs/my-fluid/cpu-only-8threads --threads 8
# Use the Pumas interpreter path printed by the launcher:
PATH_TO_PUMAS_PYTHON scripts/trajectory_cpu_viewer.py runs/my-fluid runs/my-fluid/cpu-only-8threads
```

The Comparison selector switches between CPU inference and the earlier GPU
recording. `?compute=gpu` selects the GPU recording directly. An analytic
spectral inverse also exists for this fixed box; superiority over that algorithm
has not been established.

The [model, freeze, receipts and screenshot](experiments/trajectory-v2/README.md)
preserve this run. The [book chapter](docs/book-physics-chapter.md) explains units,
MAC shapes, global context, supervised fitting, leakage, independent rollouts,
timing/memory scopes, Pumas execution, quantization limits and reproduction.
It is also prepared for the requested training book's current manuscript and web
edition, with its own executable companion.

## Tuldok pressure project

The [lab-owned project](tuldok/README.md) packages fixed CPU training and restored-checkpoint comparison in Tuldok's generic project/recipe formats. It preserves the experiments above, pins the trajectory dependency for clean builds, retains full numeric replay fields, and emits separate resident-rollout and fresh-worker receipts. Managed Tuldok execution and new cloud measurements have their own acceptance evidence; the historical eightfold result is not a promised replication.

## Historical pressure-only experiment

A small reproducible experiment: can a neural network predict the pressure used
by a numerical projection step? Rheon supplies the reference solve. This is a
learning and measurement lab, not a complete learned fluid simulator.

The first milestone is a real Rust data generator and verifier. It pins Rheon to
`34235d33c886aa05b5bdf95d920b863d1fb2ad6f` and uses its public pressure APIs.
It never reconstructs preprojection input from already-projected simulation
states. The generator uses the dependency directly, with no vendored solver.
An optional isolated API patch for the later warm-start experiment is kept separate.

**Experiment one is closed.** Independent review and a committed freeze preceded
the final 16 new-seed cases. Both learned warm-start paths qualified all 16;
raw learned pressure qualified 15. An analytic spectral baseline qualified all
16 without training and with much smaller residual. There was no post-test tuning.

| Path | Qualified | Fresh Python/model median | Resident-model median | Resident total amortized |
| --- | ---: | ---: | ---: | ---: |
| Analytic spectral | 16/16 | 81.70 ms | 5.20 ms | 12.92 ms |
| Zero + Jacobi | 16/16 | 59.70 ms | 27.98 ms | 33.21 ms |
| Learned + Jacobi | 16/16 | 83.87 ms | 5.48 ms | 12.66 ms |
| Zero + SGS | 16/16 | 51.12 ms | 19.72 ms | 24.80 ms |
| Learned + SGS | 16/16 | 81.28 ms | 5.46 ms | 13.21 ms |
| Raw learned (diagnostic) | 15/16 | 82.83 ms | 5.19 ms | 12.34 ms |

Times are per **16-case batch**, with 12 balanced repetitions per regime.
Fresh timing includes Python/import/model initialization; resident amortization
includes startup, quality/warm-up, all calls and teardown. Native processes are
fresh for every batch; OS caches are uncontrolled. Learned warm starts were slower
than zero-PCG in fresh processes and showed no established advantage over the
analytic baseline. These are one fixed box/operator and new seeds of an observed
family. The [closing report](docs/plans/pressure-data/reports/first-experiment-closure.md)
contains all results, timing scope, negative findings, evidence hashes and limits.
[Frozen experiment instructions](experiments/final/README.md) preserve the exact
checkpoint without retraining. Source and receipts are now public in
[MrScripty/rheon-surrogate-lab](https://github.com/MrScripty/rheon-surrogate-lab).

## Run the reference experiment

Prerequisites: Git, Rust 1.92.0 with Cargo (the toolchain file selects it), and
internet access for the small first dependency fetch. No GPU or paid service is
needed. Work from this repository's root, with a clean committed checkout.

```sh
cargo test --locked
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo run --release --locked -- generate data/smoke 4
cargo run --release --locked -- verify data/smoke
```

`4` means four seeds per distribution family: 8 training, 4 validation, 4 test
cases, plus 2 check-only cases = 18 total. Existing output directories are
refused. Use a new path for another run. Generator errors return nonzero.
Failures before final manifest publication
leave an incomplete manifest; such data is rejected by the verifier. A complete
manifest must still pass all re-verification checks before use.
Generation requires a clean source commit and a matching freshly built binary
so manifests identify the source used. Tests use temporary fixtures.

Each field occupies an 8×7×6 grid, with spacing [0.25, 0.3, 0.4] m, density
1.25 kg/m³ and dt 0.03125 s. The boundary normal velocities are zero. Training
families are white noise and polynomial mixtures; validation uses localized
blobs and test uses oscillatory modes. Seeds and whole families are disjoint.
All three velocity components vary in 3D. These isolated artificial fields are
not simulated trajectories. Zero and manufactured-gradient cases are checks,
never the training distribution.

The verifier checks finite values, shapes, boundary conditions, zero-cell gauge,
RHS compatibility, the full residual `b − Ap` including the gauge row, and actual
divergence after f32 velocity correction. It rebuilds RHS from saved MAC faces,
recomputes diagnostics and correction, checks hashes, and replays the seeded
input and pinned solver. The true linear tolerance is
`max(1e-12, 1e-10 * ||b||₂)`, predicted divergence limit is `1e-9 s⁻¹`, and actual
rounded-velocity divergence limit is separately `1e-5 s⁻¹`.

## Optional CPU CNN smoke

Python 3.12 and NumPy 2.3.5 are needed only for this path. Install the pinned
small dependency in a virtual environment if it is not already available:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s scripts -p 'test_*.py' -v
python scripts/train_cpu.py data/smoke runs/tiny --tiny-overfit --steps 400
python scripts/train_cpu.py data/smoke runs/heldout --steps 200
```

Build the Rust release binary first using the commands above. Scripts refuse
uncommitted source or an existing output directory. They verify the dataset
through Rust before reading it. The 1,760-parameter model uses a 3×3×3 convolution
with 16 tanh channels and a 1×1×1 pressure output, with RHS plus x/y/z coordinate
channels. It subtracts the predicted gauge-cell value. Full-batch Adam trains
supervised pressure MSE on CPU in float64. This is a transparent small baseline,
not a claim that a local receptive field captures the global pressure solve.

Normalization records all training IDs and is fitted only on training data.
The tiny mode updates weights using just the first training case; remaining
training cases are unoptimized in that run. Validation/test fields never enter
optimizer updates or fitted statistics. Development runs now omit test predictions
by default; `--include-test` is an explicit final-evaluation option. No checkpoint selection or hyperparameter
search uses test results. Runs save ignored weights, prediction arrays and a JSON
receipt with source/dataset/weight hashes, losses, normalization, raw physics
metrics and limited timing. `evaluate` also works on a directory of per-case
little-endian f64 pressure predictions:

```sh
target/release/rheon-surrogate-lab evaluate data/smoke runs/heldout/predictions
```

## Diagnose qualification without training

Existing checkpoints can be compared with five fixed nonlearned baselines using
only training and validation cases:

```sh
cargo build --release --locked
python scripts/diagnose_cpu.py data/smoke runs/diagnostic runs/tiny runs/heldout
```

This performs zero optimizer steps. It reports zero pressure, training-mean
pressure, one diagonal inverse step, and pinned Rheon Jacobi/SGS solves, with
all residual/divergence gates. It also checks checkpoint replay, zero-input and
linearity behavior, centered pressure error, and a deterministic nonlocality
witness. It reads no test predictions. Full dataset integrity verification is
still required. [ADR 0002](docs/decisions/0002-development-evaluation.md) reserved
16 new test seeds, subsequently generated only after the committed final freeze.
That final set is now exposed; old test results below remain historical.
The reservation did not claim a new independent physics family.

The first diagnostic run used **0 new optimizer steps** and took 0.339 s total.
Both reference methods qualified all 12 development cases. On validation, mean
relative residual was 1.000 for zero pressure, 0.758 for one diagonal inverse
step, and 8.623 for the existing eight-case CNN. The CNN has a measured nonlocal
blind spot and predicts 4.245 Pa RMS pressure for zero RHS. [The diagnosis](docs/plans/pressure-data/reports/qualification-diagnosis.md)
separates observed defects, scale imbalance and limits of those explanations.
Final pre-freeze lab checks: **20 Rust tests and 14 Python tests**, formatting and
Clippy pass.

## One bounded global-context candidate

The preregistered follow-on in [ADR 0003](docs/decisions/0003-global-candidate.md)
learns 335 positive gains in fixed global 3D cosine modes. It enforces zero response,
linearity and gauge by construction. This geometry-specific spectral prior is
stronger than a generic neural architecture. It uses no analytic inverse gains.

```sh
cargo build --release --locked
python scripts/global_candidate.py data/smoke runs/global-v1
```

The command runs exactly 500 full-batch Adam steps (lr 0.05), balancing each
training case through relative pressure and full-residual squared losses. It
performs one final validation evaluation and no test prediction. Checkpoints,
predictions and the pinned operator export used for the loss stay ignored.
A useful raw initial guess need not meet PCG tolerances: corrected solve quality
and total inference/transfer/solve/correction overhead are the eventual criterion.
The one recorded run at `cd6f61adceca5fca9fb479790b2931fa81782362` took 0.226 s
for training. Validation mean relative pressure error was 6.91e-11, mean relative
full residual 1.34e-10, and maximum corrected f32 divergence 1.18e-7 s⁻¹. Raw
predictions qualified 8/8 training and 1/4 validation cases: three validation
linear residuals narrowly exceeded the strict gate (worst ratio 1.728). This
passes the preregistered initial-guess signal. See the [recorded receipt](docs/plans/pressure-data/reports/global-candidate.json).

The global basis is tailored to this uniform sealed box and effectively learns
its modal inverse response. This is a fixed-operator result, not evidence for
unseen geometries or a general neural simulator. This development run left the
reserved seeds unused; their later frozen comparison is reported above.

## Isolated local warm-start experiment

The lab dependency remains pinned. A separate Cargo probe links a sibling Rheon
checkout with a reviewable initial-pressure patch; no upstream publication occurs.
[Patch instructions](patches/README.md) prepare that checkout without modifying an
existing Rheon main branch. Then, from this repository root:

```sh
export CARGO_TARGET_DIR="$PWD/target"
cargo build --release --locked
cargo build --release --locked --manifest-path probes/warmstart/Cargo.toml
python scripts/probe_warmstart.py data/smoke runs/global-v1 runs/warmstart-v1
```

The probe reuses frozen weights and performs zero optimizer steps. It compares
20 paired 12-case development batches for each preconditioner, including inference,
file transfer, child process startup, PCG, f32 correction and qualification.
[ADR 0004](docs/decisions/0004-corrected-solve-probe.md) fixes timing scope and gates.
The native probe expects a dataset already fully verified by its Python driver.
The recorded run qualified all 960 corrected solves (12 cases × 20 repeats ×
4 paths). Median 12-case wall times were 21.13 ms zero / 3.52 ms warm for Jacobi,
and 14.42 ms zero / 3.35 ms warm for SGS, with all declared file/process overhead.
Warm guesses needed 0–2 Jacobi or 0–1 SGS iterations per case. These are limited
local development measurements, not simulator acceleration evidence. See the
[full result and timing limits](docs/plans/pressure-data/reports/global-and-warmstart-results.md).

## Data format

`manifest.json` declares schema, completion, source commits, seeds/families/splits,
trajectory identity, grid axes/spacing/origin/units, boundary/operator/gauge,
solver settings, iterations, recomputed diagnostics and every array's SHA-256.
Eight raw little-endian arrays per case store preprojection u/v/w (f32), RHS and
pressure (f64), and corrected u/v/w (f32). Shapes and MAC offsets are explicit.
X is fastest: `i + nx*(j + ny*k)`; reshape a cell array as `[z,y,x]` in Python.
The cell pressure offset is [0.5,0.5,0.5]; each MAC component has offset zero
along its own axis. RHS/residual units are m³/s²; pressure is Pa.

Schema v1 supports exactly this fixed experiment. Changed physical contracts or
families need a deliberate schema/generator version. Hashes detect corruption;
they do not authenticate who produced a manifest. Byte determinism is tested on
the same platform/toolchain, not promised across different libm implementations.
Bulk data and checkpoints belong in ignored `data/` and `runs/` directories.
The final schema `rheon-pressure-final-v1` binds its exact 16-seed plan to the
committed freeze hash. One tiny 2,950-byte final checkpoint is intentionally
tracked for recovery; full generation/comparison receipts are also retained.

## Historical CNN result and limits

The generator checkpoint passes 16 Rust contract tests, formatting and Clippy.
A clean-commit CLI run generated and reverified all 18 cases (144 arrays, 262,944
bytes). Maximum corrected f32 divergence was 1.91e-7 s⁻¹; maximum predicted
divergence was 6.21e-10 s⁻¹. The CPU model has four passing unit tests. Both CPU
runs completed at source commit `e45a4ceefc3ecc89531a2dcf859935c8459f652a`:

| Run | Fitted cases / steps | Final / initial training MSE | Test mean relative pressure RMSE | Test max corrected divergence | Qualified predictions |
| --- | --- | --- | --- | --- | --- |
| One-case fitting check | 1 / 400 | 0.0001001 | 1.430 | 4.861 s⁻¹ | 0 / 16 |
| Held-out smoke | 8 / 200 | 0.03651 | 7.664 | 17.458 s⁻¹ | 0 / 16 |

The model learns the one-case fitting task but fails reference qualification and
held-out generalization. Even the fitted case does not meet the solver's strict
physical gates. These are useful pipeline checks and a negative model result.
No learning-rate or architecture search was performed on the test set. Training
wall times were under one second in this environment; this is not a benchmark
or a solver speed comparison. See [generator receipt](docs/plans/pressure-data/reports/generator-smoke.json),
[one-case receipt](docs/plans/pressure-data/reports/tiny-cpu-smoke.json),
[held-out receipt](docs/plans/pressure-data/reports/heldout-cpu-smoke.json),
[the completed local plan](docs/plans/pressure-data/plan.md) and
[numerical/source review](docs/plans/pressure-data/reports/review.md).

The pressure-only experiment made no acceleration, general CFD accuracy or learned-simulation claim. Its grid, geometry, boundary conditions, density and time step are fixed.
The family split is a small distribution-shift check, not a broad generalization
benchmark. An initial-guess API remains absent in the pinned dependency. The
isolated local patch adds one sealed-domain entry point; it is not an upstream
release. [ADR 0001](docs/decisions/0001-pinned-reference.md) records integration
requirements, including matched-residual and actual-divergence timing for both
Jacobi and SGS. UI slices and Tuldok/Pantograph/Pumas adapters are
later milestones, not dependencies of this repository.

## Publication and licensing

Public source: [MrScripty/rheon-surrogate-lab](https://github.com/MrScripty/rheon-surrogate-lab).
The owner created this exact repository after the earlier integration creation
denial. Its initial license commit is preserved alongside the complete experiment
history, including closure commit `f80a7e871852cebfce9d22459e9a75ac511a7176`.
Publication used a normal push with no history overwrite or alternate destination.
The original closure receipts retain their historical publication-blocked state;
[the publication receipt](docs/plans/pressure-data/reports/publication.json)
records the later successful release of the source. No GitHub Actions workflows
were configured at initial publication; the recorded validation was run locally.

The owner supplied the [Apache-2.0 project license](LICENSE) when creating the
repository. Dependencies retain their own licenses. See
[dependency provenance](THIRD_PARTY.md). The optional Rheon API patch remains a
separate proposal; this publication changes no upstream Rheon code or dependency.
