# Inspected code and retained negative evidence

Inspected 2026-10-14 in a clean clone of the verified public lab at
`20dd90f6c28a7e3be98e95bc7d99e64e48a7da9c`. GitHub metadata confirms lab PR1
merged at that commit and Tuldok PR58 at
`f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6`. These are source/evidence inspections,
not reruns of numerical work. The existing workspace Tuldok checkout is older
`d856a984535a430909761ed13ce42eadd55b0710`; use the merged GitHub contract,
not an assumption that its local CLI already exists.

## CNN: actual implementation and split

`scripts/train_cpu.py` encodes normalized RHS and x/y/z cell coordinates as
`N,C,Z,Y,X`; zero padding and 3³ windows yield 108 local features. The model is
`conv3d(4,16,3,pad=1) → tanh → 1x1 output`, implemented in NumPy, with 1,760
float64 parameters. Subtracting the first raw prediction fixes `p[0]=0`.
Apart from that shared gauge offset, pressure differences depend on radius-one
patches; this offset supplies no general distant-RHS pressure response.
The loss is normalized pressure MSE, with the gauge derivative included. Adam
uses lr 0.01, beta=(0.9,0.999), epsilon=1e-8, seed 2026 and full-batch updates.

`tuldok/cnn_project.py` packages verified native RHS/pressure pairs of shape
8×7×6: 8 train (white_noise, polynomial_mix), 4 validation (localized_blob),
4 test (oscillatory_modes); 2 native check cases are excluded. `tuldok/cnn.py`
checks unique case IDs/seeds and allowed families. Each purported trajectory
ID is just its standalone field ID, not a multi-frame simulation. Train avoids
loading test array members. `fit_normalization` selects only train cases and
records their IDs; no normalization leakage was observed in that code.

The packaged split therefore holds out families/seeds, but has **one geometry,
one density/dt and no time history**. The test family is now historically exposed.
It is unsuitable as a new trajectory/geometry/regime generalization test.

The restored-weight validator fixes `kernel:(108,16)`, `bias:(16,)`,
`output:(16,)`, dtype `<f8` and finite values. Fresh restore binds dataset,
splits, weights, normalization, prediction and source hashes; compares original
train/validation predictions exactly and performs zero optimizer updates.
This is valid restore evidence, not solver qualification.

## Observed results (from committed receipts)

`docs/plans/tuldok-pressure-project/reports/results.json` records 200 actual Adam
updates; final/initial train loss 0.03651220558, exact restored predictions,
and native qualification **0/16**. Mean/max relative pressure L2 are:

| Partition | Cases | Mean | Maximum |
| --- | ---: | ---: | ---: |
| Train | 8 | 0.5304807264 | 1.1640031103 |
| Validation | 4 | 4.3841482080 | 9.0245037067 |
| Exposed historical test | 4 | 7.6643041930 | 16.2091135621 |

Tuldok PR58's measured evidence records max corrected-f32 divergence
17.457689086596172 s^-1 and the same 0/16 failure. The source's native evaluator
rebuilds original RHS, applies the full operator including the gauge row, corrects
MAC velocities through Rheon and measures actual f32 divergence. A falling
training loss did not satisfy those gates.

`docs/plans/pressure-data/reports/qualification-diagnosis.md` already diagnosed
local context, scale balance and zero/linearity behavior. Limited data, strong
family shift and the local pressure-difference representation are plausible
contributors. Architecture causality is an inference; no new ablation was run.
Adding more Adam steps alone is not justified by this evidence.

## Earlier global model and analytic competitor

`docs/decisions/0003-global-candidate.md` and `scripts/global_candidate.py` define
335 positive learned cosine-mode gains, 500 Adam updates and balanced
pressure/residual loss on the same 336-cell fixed operator. The conditional
initial-guess signal passed. `patches/manifest.json` retains a sealed-only patch
based on Rheon `34235d3`, SHA256
`0edee18922a6bc52bb30d7a4aad23894f115d59d08b64d6cbf8670d98fdbb804`.
Its solve starts from full `b-Ap`, uses original-b tolerances and resets scratch.
This patch is not proof of API support at the newer trajectory reference.

`docs/plans/pressure-data/reports/first-experiment-closure.md` reports all 16
reserved cases qualified after learned+PCG, raw learned 15/16, analytic cosine
16/16. Fresh learned batches cost more than zero-PCG; resident analytic cost was
comparable. There was no robust learned advantage over analytic. Reservation
9000–9015 was consumed. Preserve that conclusion and the historical publication
blocker prose as dated evidence; current verified public metadata is separate.

`scripts/trajectory_model.py` is a different full-spectrum model on one sealed
32³ unit box: orthonormal cosine transforms, 32³−1 fitted gains and a zero constant
mode. `fit` uses supervised closed-form least squares and reports **zero optimizer
steps**. That useful modal reproduction cannot establish neural Adam training.
The reproduced cloud CPU rollouts record 6.8940109923× Jacobi and
5.4822303575× SGS, 4,320 qualified timed frames, max relative residual
9.9875884787e-9 and actual divergence 1.4305114746e-6 s^-1. Those numbers belong
to that model, allocation and stated timing scope, not to this proposal or CNN.

## Native rollout semantics to preserve

`probes/trajectory/src/main.rs` uses public MAC-grid velocity/tracer advection,
seeded forcing, original RHS assembly, full operator, pressure correction and
actual divergence. State evolves independently for each method; no teacher
states feed a rollout. It is a sealed uniform unit box, passive tracer and
prescribed forcing, not validated engineering aerodynamics or submerged-body flow.

Native settings are relative residual 1e-8, absolute 1e-12, predicted divergence
1e-7, max 3,000 iterations; the shared learned-path check is looser (relative
1e-5) and actual f32 divergence limit 1e-5. This proposal requires **the stricter
native residual and predicted-divergence gates for every accepted path**, with
actual f32 divergence checked independently. Do not copy the looser learned gate.
Tracer integral is measured, but source injection, clamping and semi-Lagrangian
transport mean it is not an exact mass-conservation proof. New accounting must
separate source change, transport drift and model/reference difference.

## Instructions and standards provenance

Checked `/AGENTS.md`, `/workspace/AGENTS.md`, the lab tree and `/workspace/.agents`.
No applicable AGENTS.md or .agents/skills was present; the mounted .agents directory
was empty. No absence is represented as having read a nonexistent file.
The existing pressure plan references MrScripty/Coding-Standards
`188beda1fa477d21d576c233dd8c7f4c4c267d23`; standards consulted for this design
are recorded in [the ledger](execution-ledger.md), after verifying that exact revision. The repository's established
plan format (status/phase/acceptance, scoped slices, owners, composed-design review,
evidence and blockers) is preserved. No code/API compliance is inferred from docs.
