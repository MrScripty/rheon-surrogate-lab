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
refused. Use a new path for another run. Generator errors return nonzero and
leave an explicitly incomplete manifest; such data is rejected by the verifier.
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

The initial generator checkpoint passes 16 Rust contract tests, formatting and
Clippy. The clean-commit CLI smoke receipt and CPU model work follow next; see
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
`GraphQL: Resource not accessible by integration (createRepository)`. No alternate
account or repository was used. Local source is committed; push requires that
exact repository to exist and the integration to have write access.

No project license has been selected; this repository does not invent one.
Dependencies retain their own licenses. See [dependency provenance](THIRD_PARTY.md).
