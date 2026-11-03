# Detached auxiliary pressure correction experiment

Run the three declared objectives from C53, C44 and C37 using the existing
managed training catalog. U retains the original four-pass fitting objective.
A adds an attached auxiliary correction. D uses the same auxiliary scalar with
detached history. The shared numerical template comes from C53; each exact
initialization retains its own weights and weight inventory.

The [experiment plan](../../docs/plans/flow-reasoning-assessment/training-experiment.md)
defines the objective, populations and gates. The
[execution ledger](../../docs/plans/flow-reasoning-assessment/execution-ledger.md)
records the authorized launch and pre-admission recovery.

`campaign.py prepare` captures fresh Projects using the app environment and
qualifies each with the managed Torch interpreter on CPU. It writes an immutable
`runs/pressure-detached-correction-v1/campaign.json` inventory and copies scalar
qualification reports to `docs/plans/flow-reasoning-assessment/execution/`.
This preparation has already completed; do not replay it over existing captures.

`launch.py --session-file PATH` executes that frozen inventory through the
existing managed controller and then runs CPU evaluation. Every mutation uses
the existing durable journal. Existing execution/custody prevents an automatic
restart. The explicit `--resume-before-admission` option is limited to the
recorded first authentication failure with no studies or mutation journals; it
does not recover an uncertain admission or training failure. That continuation
has already been used.

The active launch command uses the app interpreter:

```sh
runs/dependency-migration-v1/app-env/bin/python3.12 -B \
  experiments/pressure-detached-correction-v1/launch.py \
  --session-file runs/tuldok-performance-v1/private/owner-session.txt \
  --resume-before-admission
```

This is an execution record, not a replay command. Inspect the campaign state,
per-arm controller logs and mutation journals before any recovery. Full artifacts
stay in ignored `runs/`. Tuldok retains CUDA/CPU Run ownership; terminating a local
controller does not establish remote cessation.

`evaluate.py` requires all nine completed training/restore receipts. It evaluates
one CPU worker at a time, with CUDA hidden, one thread, explicit CPU affinity,
a 12 GiB host reserve plus 2 GiB worker allowance, bounded child lifetimes and
reaping. It freezes all original and candidate checkpoint identities, reports
capacity and regression gates separately, and writes final scalar JSON and
Markdown reports. It does not start simulations, fit models or promote a candidate.

The completed run needed a CPU-only evaluation recovery: the original C44 input
view mixes training and consumed validation parents, so its old calibration path
was rejected. `fitted.py freeze` now freezes one common bank from the same C53
train-only calibration used by all nine fits. `fitted.py evaluate` applies every
checkpoint to those exact tensors without opening that checkpoint's historical
training dataset. The common bank reproduces all original C53 fitting values
exactly. `evaluate.py --resume` preserves the stopped attempt, verifies retained
result identities and continues only missing CPU measurements, without any remote
admission. This recovery has completed; do not replay it over finished results.

The [outcome](../../docs/plans/flow-reasoning-assessment/outcome.md) retains C53:
no candidate passed the full fitting or independent-scene gate. Four reporting
and worker-lifecycle tests run with `python3 -B -m unittest discover -s
experiments/pressure-detached-correction-v1 -p 'test_*.py' -v`.
