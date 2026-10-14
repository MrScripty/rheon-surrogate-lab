# Pinned sealed native dependency

This boundary owns the lab's initial-pressure API qualification at Rheon
`2209012bc5f902693d0f695ec1c2d854126af317`. It retains the upstream Apache-2.0
[license](LICENSE). The patch modifies `src/pressure.rs` and adds
`tests/initial_pressure_contract.rs`; each carries a modification notice.
The reviewed older patch is identified in [manifest.json](manifest.json), and
the complete exported source closure is [source-files.json](source-files.json).
No upstream checkout is patched. No remote fork is published.

## API contract

```rust
PressureWorkspace::solve_rhs_with_initial_pressure(
    &mut self,
    operator: &PressureOperator<'_>,
    rhs: &[f64],
    initial_pressure: &[f64],
    dt: f64,
    settings: PressureSettings,
    cancelled: impl FnMut() -> bool,
) -> Result<PressureReport, PressureError>
```

The operator must be sealed (`has_gauge()`), both arrays must have the exact
cell count and finite f64 values, and `initial_pressure[0]` must equal zero.
The guess is copied without a shift. Unsupported free surfaces and nonzero
gauges produce explicit typed errors. Existing RHS compatibility, settings,
allocation and cancellation errors retain their meaning.

The initial residual is full `b - A p`, including cell zero. Relative tolerance
uses the original `b`, and both the residual and predicted divergence gates
remain necessary. A successful report still requires independent actual f32
velocity-divergence qualification after correction. The warm path resets
Krylov buffers for both Jacobi and SGS. Pressure scratch following any failure
is unaccepted; the workspace may be reused. Cancellation occurs at the same
pre-solve and iteration boundaries as the original solver. The existing
`solve_rhs` and `solve_velocity` entrypoints retain their zero-start behavior.

## Preparation and consumption

From the lab root, supply an explicitly selected clean Git repository root:

```sh
python experiments/global-pressure-v1/prepare_dependency.py \
  --source /tmp/global-pressure-rheon-reference \
  --destination "$PWD/runs/global-pressure-v1-dependency/rheon"
python experiments/global-pressure-v1/prepare_dependency.py \
  --verify "$PWD/runs/global-pressure-v1-dependency/rheon"
python experiments/global-pressure-v1/test_dependency.py -v
```

Preparation verifies clean exact HEAD/tree, patch and inventory hashes, then
exports committed Git blobs into staging, applies the patch and checks every
file and executable bit. Ambient Git overrides are removed and source-index
refresh is disabled. Ignored source files never enter the archive. Output
must stay below the lab's task-owned `runs/global-pressure-v1-dependency/`;
existing paths, traversal and symlinks fail. Publication exclusively creates
its final directory and publishes `source-receipt.json` last. If publication
fails, only the newly claimed destination is cleaned. Concurrent foreign
destinations remain untouched. Source files are read-only. Build targets must
live outside this source directory.

The receipt is deterministic JSON schema version 1: `base_commit`, `base_tree`,
`patch_sha256`, `prepared_tree_sha256`, `source_manifest_sha256`,
`source_files_sha256` and `license_sha256`. Tree identity hashes canonical JSON
of sorted relative file paths, Git modes and SHA-256 contents; only the receipt
itself is excluded. Identity JSON rejects duplicate keys and nonfinite numbers;
schema version must be integer 1. Receipts require exact fields, JSON types and
values. `--verify` rejects missing, changed or additional files, mode changes,
symlinks and receipt drift. Consumers verify before compilation
and execution, and separately prove Cargo actually selected this exact source.

The probe declares Git URL `https://github.com/MrScripty/Rheon.git` at the exact
base revision, disables unused native default features, and supplies an
explicit Cargo patch override to this prepared path. The probe owns its lock,
resolver-source check and runtime receipt. A sibling checkout or an environment
path alone is not proof of the dependency consumed.

Focused native evidence is the patched `initial_pressure_contract` test target;
it tests zero equivalence for both existing entrypoints, exact/poor/shifted/
nonfinite/wrong-length guesses, original-b scaling, the full gauge-row gate,
cancellation, numerical and iteration failures, reuse and actual corrected
f32 divergence for both PCGs. The [slice plan](../../../docs/plans/global-pressure-native-capture-v1/plan.md)
owns acceptance and measured run evidence.
