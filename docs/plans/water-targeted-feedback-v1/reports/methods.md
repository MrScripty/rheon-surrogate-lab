# Method and reproduction

The diagnosis, v7 sampler/configuration and training sources are captured at
commit a1c5d6f (full identity in capture.json). Source assembly runs from a clean
temporary worktree, copies exact project bytes to the lab and preserves that
source head on the experiment branch before removing the worktree. Capture
argv, source witness, duration and copy parity are retained. Training code is
not edited during the fits.

Controller command from the lab root:

```sh
.venv/bin/python experiments/water-targeted-feedback-v1/run.py \
  --sequence experiments/water-targeted-feedback-v1/configs/sequence.json \
  --reports docs/plans/water-targeted-feedback-v1/reports \
  --torch-python '/media/jeremy/OrangeCream/Linux Software/repos/owned/ai-systems/Pumas-Library/torch-versions/v2.14.0/venv/bin/python' \
  --managed-controller /tmp/run-targeted-feedback-managed.py
```

The execution adapter's exact code/hash is retained locally. It uses the existing
Tuldok catalog and normal preflight/approval/submission, terminal custody,
reopening and duplicate artifact downloads. Each fit has managed CUDA training,
fresh CPU restore and CPU validation. No private catalog or unmanaged optimizer.
Environment: RTX 5090 Laptop GPU, Core Ultra 9 275HX / 24 logical CPUs, Pumas
Torch 2.14.0+cu132 / Python 3.14.7 / NumPy 2.5.3. HTTP/catalog control uses the
existing lab Python 3.12 environment. Neural parameters/activations are float32;
physical accumulation and operator arithmetic retain their existing float64.
No AMP/TF32 or GPU inference. Runtime facts also appear in the sealed receipts.

Both fits use 16 real training parents/32 states, 5,000 updates, batch one,
original initialization/real-sampling seeds, Adam, the original learning-rate
schedule, two shared training corrections and intermediate residual feedback.
The original eight analytical grids retain exact tensors/order/evidence and
625 primary exposures per grid. Train-only normalization/amplitude is unchanged.

Each update adds one supplemental analytical example at weight 0.1. The control
reuses primary tensors with a private seed and 625 extra exposures per grid;
it allocates no duplicate bank. The targeted recipe uses three cyclic axis
permutations of 16×23×25 and three of 10×7×25, all at 16 cells/metre. Its 5,000
extra draws are balanced across those six grids. Both supplemental streams use
seed 2026101607. This matches extra example count, not cell work or FLOPs.
Primary and supplemental gradients accumulate before the unchanged gradient
clipping/Adam update. Intermediate loss applies to both streams, including their
outer analytical weights. Actual forward cell counts and durations are retained.

The one-solve native replay and 66 CPU probes are documented in hypotheses.md,
diagnosis-findings.md and probes.json. They support targeted shape/axis examples,
not a unique pooling/filter cause or a global convergence proof. The red-capable
replay calls the actual correction code with the same captured fields: feedback
needs 28 calls twice, control 11. Failed diagnostic bounds do not mean execution
failure, and completing a controlled experiment does not mean that bound is fixed.

CPU preparation is independent of both fits and final data. prepare_cpu.py first
qualifies late correction targets for 32 training states using an independent
cosine pseudoinverse and independently enumerated neighbor operator. It measures
and removes the negligible incompatible constant component, preserves the cell
zero gauge and reports float32 output-rounding error against the original RHS.
No precision upgrade or relaxed inference gate follows. The saved predictions
from eight learned corrections are source/checkpoint bound; no redundant training.

The old full-field native reservation restricts axes to at most 16. Larger future
parents therefore use the existing arbitrary-grid native paired bridge, with an
explicit CPU spectral pressure teacher. A 15-step native warm-up plus one teacher
solve qualifies the fixture first. Sixteen varied sealed-water parents then run
64 steps each; each subsequent pressure reply solves the native predictor's
actual force-dependent RHS. Independent operator checks and native paired
reference/physical gates qualify the outputs. All native children are bounded,
terminated/reaped through the existing owner. Source/binary/scene identities,
actual affinity, duration and every native frame are retained. There are zero
neural calls or optimizer updates in this preparation.

These are real native water dynamics with CPU-teacher labels, rather than
manufactured cosine-mode simulations or exported native iterative pressure.
Each prepared parent retains 49 post-warm-up pressure/base/residual arrays, force
configuration and native physics telemetry. No velocity/tracer export is claimed.
export_preparation.py creates standard scientific_field_sequence_v1 pressure/base
timelines, validated by Tuldok's existing parser. The existing collection publisher
imports draft records into the same catalog and reopens exact content hashes.
No review/release is implied. Future packet/view integration and fitting on these
parents remain separate; they cannot silently enter the current fit inputs.

Preparation commands and their actual environments are retained in timings and
preparation-methods.json. The first worker initialization failed before any label
or native simulation; its source/log/timing remains local. After an import-path
fix, a fresh output/command identity completed. A publication attempt in Pumas
Python lacked psycopg and failed before authentication/catalog mutation; the
existing controller Python then completed publication. reconciliations.json binds
both failed timings to exact successful recoveries. No training replay resulted.
An unsupported project registration route returned 404 before mutation; the
canonical existing registration route then succeeded.

The 50 development trajectories include all previously consumed cases. Four
elongated shapes remain exploratory. Fit/familiar native qualification, worst
calls, pressure drift and mean work govern frozen selection. Eight new whole
final simulations are committed before fitting and run after selection freezes.
A single-seed improvement is provisional pending independent-seed replication.
Native warm-up, scalar cubic spacing, 32-call cap and physics/equation gates stay
fixed; no native solver cleanup counts as neural success. The application
adapter is unchanged. CPU work overlaps managed execution; these durations are
throughput observations, not isolated solver-speed measurements.

Final indexing after owned scientific/preparation jobs exit:

```sh
.venv/bin/python experiments/water-targeted-feedback-v1/finalize.py \
  --sequence experiments/water-targeted-feedback-v1/configs/sequence.json \
  --reports docs/plans/water-targeted-feedback-v1/reports
```

The audit retains actual failed-command states and requires exact recovery
receipts, completed managed custody, reaped native children, matched final
callbacks/frames and source hashes. No weights/raw simulation arrays enter Git.
