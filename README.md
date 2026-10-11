# Rheon Surrogate Lab

Author: Puma.

A small reproducible experiment: can a neural network predict the pressure used
by a numerical projection step? Rheon supplies the reference solve. This is a
learning and measurement lab, not a complete learned fluid simulator.

The first milestone is a real Rust data generator and verifier. It pins Rheon to
`34235d33c886aa05b5bdf95d920b863d1fb2ad6f` and uses its public pressure APIs.
It never reconstructs preprojection input from already-projected simulation
states. It does not copy or fork Rheon's solver.

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
still required. [ADR 0002](docs/decisions/0002-development-evaluation.md) reserves
16 new test seeds that have not been generated; old test results below remain
historical. The reservation does not claim a new independent physics family.

The first diagnostic run used **0 new optimizer steps** and took 0.339 s total.
Both reference methods qualified all 12 development cases. On validation, mean
relative residual was 1.000 for zero pressure, 0.758 for one diagonal inverse
step, and 8.623 for the existing eight-case CNN. The CNN has a measured nonlocal
blind spot and predicts 4.245 Pa RMS pressure for zero RHS. [The diagnosis](docs/plans/pressure-data/reports/qualification-diagnosis.md)
separates observed defects, scale imbalance and limits of those explanations.
Current checks: **18 Rust tests and 6 Python tests**, formatting and Clippy pass.

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
Conditional local upstream patch work leaves this repository's dependency pin
unchanged and is not an upstream publication.

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

## Status and limits

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
[the active plan](docs/plans/pressure-data/plan.md) and
[numerical/source review](docs/plans/pressure-data/reports/review.md).

No acceleration, general CFD accuracy, warm start or learned-simulation claim is
made. The grid, geometry, boundary conditions, density and time step are fixed.
The family split is a small distribution-shift check, not a broad generalization
benchmark. An initial-guess API is absent in pinned Rheon; `solve_loaded` zeros
pressure. [ADR 0001](docs/decisions/0001-pinned-reference.md) records the exact
future integration need, including matched-residual and actual-divergence timing
for Jacobi and SGS baselines. UI slices and Tuldok/Pantograph/Pumas adapters are
later milestones, not dependencies of this repository.

## Publication and licensing

Requested public destination: `https://github.com/MrScripty/rheon-surrogate-lab`.
Creation is currently blocked: the authenticated integration returned
`GraphQL: Resource not accessible by integration (createRepository)`. A push to
the exact remote returned `Repository not found`. No alternate account or
repository was used. Local source is committed; push requires that exact public
repository to exist and the integration to have write access. Once available,
run `git push -u origin main`; then verify the remote commit and public visibility.

No project license has been selected; this repository does not invent one.
Dependencies retain their own licenses. See [dependency provenance](THIRD_PARTY.md).
