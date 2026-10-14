# Native warm start and development capture

Only the new global-pressure namespaces participate. Old probes, receipts,
reservations and negative results remain unchanged. No training or final arrays.
The dependency helper/patch is described in [dependency/README.md](dependency/README.md).
The caller/physics/payload contract is [capture-contract.md](capture-contract.md).

Use the isolated Rust 1.92.0 toolchain and a prepared, qualified native source.
All build/capture destination names must be fresh; a partial directory cannot be
reused. Direct Cargo compilation is unsupported for qualified capture.

```sh
export CARGO_HOME=/workspace/task-cargo
export RUSTUP_HOME=/workspace/task-rust
python3 experiments/global-pressure-v1/capture.py build \
  --cargo /workspace/task-cargo/bin/cargo \
  --prepared-source /workspace/rheon-surrogate-lab/runs/global-pressure-v1-dependency/rheon \
  --output /workspace/rheon-surrogate-lab/runs/global-pressure-v1-build/reviewed-build
python3 experiments/global-pressure-v1/capture.py generate \
  --preset smoke --split train --trajectory smoke-train-100000 --fixture-frames 1 \
  --binary /workspace/rheon-surrogate-lab/runs/global-pressure-v1-build/reviewed-build/rheon-global-pressure-capture \
  --prepared-source /workspace/rheon-surrogate-lab/runs/global-pressure-v1-dependency/rheon \
  --output /workspace/rheon-surrogate-lab/runs/global-pressure-v1-capture/train-fixture
python3 experiments/global-pressure-v1/verify_capture.py \
  /workspace/rheon-surrogate-lab/runs/global-pressure-v1-capture/train-fixture
python3 experiments/global-pressure-v1/test_capture.py
```

After the tiny fixture qualifies, complete local development generation uses
`--split development` and omits `--trajectory`/`--fixture-frames`. Separate
`--split train` and `--split validation` calls are available for bounded managed
shards. `fixture_complete` is diagnostic and ineligible as a complete dataset.
Full/test requests always fail before output creation. A scientific failure stops
for parent review; no case/tolerance/config tuning is authorized.

Each qualified build directory retains actual Cargo metadata, its hashed binding,
immutable copied binary and successful build receipt. Each accepted capture
retains raw payloads, per-trajectory stderr, candidate, verification report and
final manifest. Independent verification checks the current source closure;
scientific/source changes require a fresh qualified build and capture. Do not
reinterpret old capture receipts as validation of later source.

The focused tests and schema/source/timeout fixtures use standard
Python only. Literal dyadic fixtures and analytic cosine action test finite-volume
semantics. Independent corruption tests start from a valid fixture and assert the
specific failing boundary. Real native source/build/fresh-process evidence remains
separate and is recorded in the slice1 plan reports by the integration owner.
