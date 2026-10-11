# Frozen first experiment

Author: Puma. Candidate checkpoint is the exact335-parameter,500-step global-v1
model from cd6f61adceca5fca9fb479790b2931fa81782362. The2950-byte weights.npz is a
small intentional exception to ignored checkpoints so recovery needs no training.
Do not modify it or refit anything after reading final results.

Before final access: independently review patch/source, run all development checks,
commit source and review record, and build clean release binaries. Freeze from the
lab root, then commit the resulting JSON **before** materializing the final set:

```sh
cargo build --release --locked
CARGO_TARGET_DIR="$PWD/target" cargo build --release --locked --manifest-path probes/warmstart/Cargo.toml
python3 scripts/final_evaluate.py freeze
git add experiments/final/freeze.json
git commit -m 'experiment: freeze final comparison before reserved generation'
python3 scripts/final_evaluate.py generate
python3 scripts/final_evaluate.py run
```

Do not rebuild between freeze and final execution: hashes bind to those exact
binaries, which identify the preceding source commit. Source, patch, checkpoint,
runtime and committed-freeze checks run before generation and comparison. Final
output directories are never overwritten; partial failures remain inspectable.

Artifact hashes protect this execution's provenance; binary bytes are not promised
portable across build paths/platforms. To reproduce on another environment, use
the reviewed numerical source and identical checkpoint, build there and create a
separately named/committed replication freeze before regenerating the declared
seeds. Such a replication is not a new untouched test and must not be used for
model selection. The original freeze/result remain historical evidence.

ADR0005 defines every method, tolerance, seed, order, repetition and timing scope.
Fresh-process costs include imports and model/basis setup. Resident-model costs
include all call overhead plus reported amortized initialization/warm-up/teardown;
the native child is still freshly launched per batch. OS caches are uncontrolled.
Raw predictions that fail physics are retained as negative diagnostic results.

The reservation file under docs/ is an immutable declaration at freeze time;
final materialization/completion is recorded separately with dataset/freeze hashes.
Only same-family new-seed generalization is tested. No upstream publication,
root dependency change, GitHub retry or further optimization is part of closure.
