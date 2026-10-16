# Reproduction and retained methods

Captured training source commit: b2a7e78bef82bbd1bd394d90d00a382ed288162a.
Configuration: experiments/water-two-pass-long-v1/configs/sequence.json and its
one trial. Capture receipt retains the exact clean-worktree assembly command,
UTC boundaries, elapsed duration and byte parity. Worktree lifecycle is recorded
separately; the commit remains reachable from the experiment branch.

Controller command, from the lab root:

```sh
.venv/bin/python experiments/water-two-pass-long-v1/run.py \
  --sequence experiments/water-two-pass-long-v1/configs/sequence.json \
  --reports docs/plans/water-two-pass-long-v1/reports \
  --torch-python '/media/jeremy/OrangeCream/Linux Software/repos/owned/ai-systems/Pumas-Library/torch-versions/v2.14.0/venv/bin/python' \
  --managed-controller /tmp/run-two-pass-long-managed.py
```

The authenticated adapter is copied exactly to the ignored run root. It uses
the existing Tuldok local-owner session and normal workflow.execute; credentials
stay in memory. Training, restoration and validation use three catalog-owned
Runs. Runtime qualification, source witnesses and double artifact downloads are
retained by those owners. No unmanaged training substitutes for those Runs.
The original adapter and source bytes remain available in the local run root.

The existing shape runner performs managed training and standard CPU fitting,
familiar validation, fixed-eight, signed-mode, fresh/prior-grid/512-step and raw
native checks. The new runner adds the consumed 16 development regressions,
exploratory elongated 4, frozen selection and eight fresh final simulations.
Historical comparisons reuse exact scenes and native binary hashes. Case and
checkpoint identities, full pressure-error milestones, native verdicts and
actual network call counts remain in the reports and ignored native receipts.

Hardware/runtime: RTX 5090 Laptop GPU, Core Ultra 9 275HX, 24 logical CPUs,
Pumas Torch 2.14.0+cu132/NumPy 2.5.3/Python 3.14.7; local controller Python 3.12.
CUDA weights, outputs, gradients and Adam moments use float32; no AMP/TF32.
Physical operator/base accumulation remains float64. Qualification uses fresh
CPU restores and single-thread worker processes, with bounded independent pools;
standard state qualification can use 22 workers under its reserved core affinity,
trajectory studies use 8 workers. These are throughput observations under shared
machine use. They do not establish isolated native-versus-AI inference latency.

Training starts fresh from the original seed, with 16 real parents/32 states and
the existing 1,024-example eight-grid analytical bank. Each update draws one real
state and one analytical example, each undergoing two shared network corrections.
10,000 updates therefore request 40,000 forward network evaluations, compared
with 20,000 for both the two-pass 5,000 control and the one-pass 10,000 prior result.
The counts omit backward work and do not imply equal runtime. CPU inference
checks each individual correction, capped 32, after 15 native warm-up steps.

Each scientific delegated command has a retained argv, UTC start/finish,
monotonic elapsed duration, child PID, terminal outcome and reaping status.
The historical training interval includes cache/bank preparation, checkpoint
publication and update checks inside its retained timer; it is not pure GPU
kernel time. Optimizer interval, complete managed workflow, CPU qualification, regressions, final
qualification and preparation are reported separately. Concurrent duration sums
are not elapsed makespan. Failed model qualification remains a scientific result;
execution/custody failures require explicit reconciliation and no blind replay.

The initial postprocessor used a mismatched historical reference and stopped.
See postprocessing-recovery.json for the retained failure, corrected identity
bindings and explicit development-only recovery. Recovery invokes the controller
command above with --resume-development; it cannot launch managed training.
Original controller status and wrong-reference drift remain retained.

Final indexing (after the recovery controller exits):

```sh
.venv/bin/python experiments/water-two-pass-long-v1/finalize.py \
  --sequence experiments/water-two-pass-long-v1/configs/sequence.json \
  --reports docs/plans/water-two-pass-long-v1/reports \
  --completion-status runs/water-two-pass-long-v1/development-recovery-status.json
```

The existing one-channel physical/deployment restrictions are documented in
../water-shape-repeatability-v1/model-contract.md. Rheon's deployed adapter,
solver architecture, pressure gates and boundary conditions are unchanged.
