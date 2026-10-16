# Retained methods and reproduction

Training source capture: 6b9ef674b25a738f87bb24e4f04b4d9f85e8c549. Each project
is assembled from that clean committed worktree, copied byte-for-byte to the lab
run root, and executed by the existing Tuldok workflow. Exact assembly argv,
UTC boundaries, elapsed duration, source commit and copy parity are in capture.json.
The task-created worktree is removed with its head protected by the branch.

Controller command, from the lab root:

```sh
.venv/bin/python experiments/water-coverage-feedback-v1/run.py \
  --sequence experiments/water-coverage-feedback-v1/configs/sequence.json \
  --reports docs/plans/water-coverage-feedback-v1/reports \
  --torch-python '/media/jeremy/OrangeCream/Linux Software/repos/owned/ai-systems/Pumas-Library/torch-versions/v2.14.0/venv/bin/python' \
  --managed-controller /tmp/run-coverage-feedback-managed.py
```

The adapter is retained exactly in runs/water-coverage-feedback-v1, uses the
existing local-owner session and catalog, and stores no credential values. Each
fit has three managed Runs: CUDA training, fresh CPU restore, validation CPU
inference. Normal preflight, exact approval, submission, terminal custody,
reopening and duplicate artifact downloads remain the managed owner's contract.
No private catalog or unmanaged optimizer substitutes for it.

Both new fits use 5,000 updates, batch 1, two shared corrections, original seeds,
Adam and learning-rate reductions 3,000/4,500, 16 real training parents/32 states,
train-only normalization/amplitude, 1,024 historical analytical examples and
unchanged 370,016-parameter float32 network. No AMP/TF32. Physical assembly and
pressure accumulation remain float64, as before. The coverage fit adds 384 mixed
odd-grid examples (three cyclic axis permutations of 16×23×25 at 16 cells/metre).
Original analytical tensors remain identical. Balanced 11-grid sampling reduces
old-grid exposure from 625 to 454/455 draws; this is disclosed rather than called
an equal-exposure comparison. Real sampling order and scales remain fixed.

The feedback-only fit keeps the eight-grid samples/order. The named objective
adds 0.1 times mean squared intermediate residual / squared initial correction
residual to the final canonical loss on real samples. It also adds that term to
the final analytical loss, retaining the analytical sample's outer 0.1 weight.
With two training corrections, only the first correction contributes intermediate
feedback. Exact-zero residual contributes zero. Evaluation's final-pressure loss
remains canonical; inference collects no training intermediates. No residuals
from beyond two corrections are included in either fit.

Training-only future preparation, run with the qualified Pumas Python:

```sh
/path/to/qualified/pumas/python experiments/water-coverage-feedback-v1/prepare_late.py \
  --sequence experiments/water-coverage-feedback-v1/configs/sequence.json \
  --reports docs/plans/water-coverage-feedback-v1/reports
```

This delegates eight fixed learned corrections on 32 training snapshots from16
parents to the existing CPU owner, with a bounded pool on independent cores.
Actual used cores, command, 8.401s duration, remaining correction scale and
teacher-label defect are retained. It performs zero updates and initializes no
CUDA context. Median late/initial residual is 2.69826e-5; the maximum captured
teacher defect is 1.6597% of the late residual. This is preparation evidence,
not new CFD simulations, a current-fit input, or a finalized late-label contract.

Execution environment: RTX 5090 Laptop GPU; Core Ultra 9 275HX / 24 logical CPUs;
Pumas Torch 2.14.0+cu132, NumPy 2.5.3, Python 3.14.7. Local HTTP controller uses
Python 3.12. Standard CPU state/probe qualification may use 22 single-thread
workers under reserved affinity; autonomous studies use 8 workers. Actual core
use is retained per receipt. Concurrent CPU work and managed preparation mean
these elapsed times measure throughput, not isolated inference latency.

The scientific selection includes 42 previously consumed development trajectories,
fitting/familiar qualification, worst-case work and pressure drift. Elongated 4
are exploratory. Eight new whole final simulations are committed before fitting;
selection and checkpoint identity freeze before their execution. All cases have
15 native warm-up steps and cubic h; the equation/physical gates and 32-call cap
are unchanged. Source/data identity and native controls precede model judgment.

All scientific delegated commands retain argv, UTC start/finish, monotonic
elapsed duration, PID, return code and reaping. Training interval includes bank/
cache preparation, checks and checkpoint publication inside the historical timer;
it is not pure GPU kernel time. Report optimizer, full managed workflow, CPU
qualification, postprocessing and capture-to-final spans separately. Sums of
concurrent durations are not makespan. Training-loss reduction is not native
qualification; native fallback/refinement is not credited as AI success.

Final indexing, after the controller exits:

```sh
.venv/bin/python experiments/water-coverage-feedback-v1/finalize.py \
  --sequence experiments/water-coverage-feedback-v1/configs/sequence.json \
  --reports docs/plans/water-coverage-feedback-v1/reports
```

The model's one-channel input, gauge reconstruction, history and deployment
restrictions remain documented in ../model-contract.md.
Source weights and raw simulation arrays remain local and are not committed.
