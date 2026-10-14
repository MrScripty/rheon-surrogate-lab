# Proposed activation-normalized global correction smoke

Status: **proposal only**. This diagnosis authorizes no implementation or training
of this candidate. Parent review must accept a new experiment freeze before a
scientific Run. The failed original checkpoint, source and configuration remain
immutable under their original identities.

## Question and one intervention

Can per-case feature normalization prevent the measured decoder amplification/
negative-tail shutdown, allowing a global pressure correction to retain input
sensitivity after the same modest200-update training? This is a representation/
conditioning hypothesis. It promises neither an optimizer cure nor a useful solver.

Create a new model/experiment ID; retain the10channels, multiscale widths, pooled
path, replicate convolutions, interpolation, linear head and float64 gauge/operator.
Insert `GroupNorm(num_groups=4, num_channels=C, eps=1e-5, affine=False)` immediately
after **every convolution in a ConvBlock and before its SiLU**, in encoders,
bottleneck and decoders. Leave the head unchanged. All widths divide by4; each
bottleneck group contains12channels at1³, avoiding single-value normalization.
Non-affine normalization adds no learned parameters and cannot learn a shifted
normalization bias or gain. Convolution/head shapes and parameter count stay
372,377. Initialize every convolution/head with byte-identical original seeded
values, checked by explicit mapping; do not resume failed weights/Adam state.

GroupNorm computes within each sample's channel group/spatial field and uses input
statistics in train and evaluation, rather than dataset or batch running statistics.
This is an additional deterministic feature transformation; train-only persisted
physical RMS normalization still belongs to the original training histories. See
[PyTorch2.8 GroupNorm](https://docs.pytorch.org/docs/2.8/generated/torch.nn.GroupNorm.html)
and[Wu/He](https://arxiv.org/abs/1803.08494). Their vision results supply no expected
pressure-solver gain. Per-group normalized variance is controlled in exact
arithmetic, but f32 accuracy, spatial feature collapse and optimization must still
be measured. GroupNorm may suppress useful physical amplitude; measure forcing-amplitude
proportionality and derivative sensitivity on development arithmetic. The residual,
pressure and physical conditioning channels and labels remain unchanged.

## Alternatives considered

| Practical choice | Decision from current evidence |
| --- | --- |
| Higher precision or bias-free head alone | Defer: same-f32-feature and full-f64 final probes remain constant; this does not address the observed upstream zeros |
| Change loss weighting, learning rate or number of steps | Defer: no intermediate states identify optimizer cause; component-gradient samples alone do not justify several simultaneous changes |
| One fixed non-affine feature-normalization intervention | Proposed: directly tests observed huge internal amplitudes while keeping architecture capacity, loss, data and optimizer policy fixed |

Do not add an architecture/lr/normalization sweep, multiple seeds, a schedule,
restarts, early stopping or final-only checkpoint selection based on validation.
The original final model and zero correction are retained comparators; no second
baseline training is required. No inherited modal/cloud speedup belongs here.

## Pre-training qualification and freeze

Before any new Run, independently qualify the new model ID, exact initialization
mapping, source/runtime metadata and372,377-parameter inventory,16³/32³ shapes,
finite/group statistics, train/eval consistency, gauge/head invariance, full native
stencil, composite loss derivatives and global dependence. Check that normalization
of one row is unchanged when batch neighbours or validation rows change, and that
the original exactly-zero-residual bypass still returns previous pressure unchanged. Preserve
original288train/96validation whole-history split, data/native anchors and train-only
physical RMS bytes. Include small near-constant and large-amplitude arithmetic
fixtures to test finite GN behavior; do not use final arrays.

Freeze source/config/recipe/runtime/input/output identities and scientific policy
in a new lab namespace and exact generic Tuldok project. Reuse original captured
CPU profile only after actual runtime compatibility/preflight qualifies GroupNorm.
Use existing CLI/UI/grant/run-artifact restore contracts; add no Tuldok/Rheon
model-specific executor or upstream dependency. Independently clear exact material
before an owner grant, if a future parent authorizes that action.

## One modest CPU experiment, if later authorized

Exactly200 Adam updates, batch2, initialization20261014/order20261015,
lr0.001/betas0.9,0.999/epsilon1e-8/weight decay0, clip1; same original composite
pressure-plus-full-residual loss and physical normalization. Validation is
observational at0/200. CPU one thread, no GPU/JIT; <=600s existing child wall,
<=4GiB observed process high-water and32MiB total output/8MiBcheckpoint. Actual
preflight capture/input/output limits remain the original256/320MiB policy.
This is a proposed new scientific configuration, not a modification or rerun of
PR4's frozen model. Stop if qualification or bounds fail; no extra training attempt.

Retain full original sampled row IDs and an immutable checkpoint at each20-update
log (numeric states only, no resume). Eleven checkpoints including initialization
would total about16.5MB at the existing1.50MBfile size, leaving bounded room for
final weights/predictions/receipts under32MiB. Declare every flat checkpoint and
measure actual sizes before admission; adjust the recipe output inventory, not
hide excess files. Persist per-log train-batch layer magnitude/zero fraction,
correction spatial variation and pressure/residual gradient components before/after
the update. Logged intermediate checkpoints are observational: final200 remains
the only scientific candidate and no checkpoint search is authorized. If output
bounds cannot hold, replan before grant, never omit newly required evidence silently.

## Acceptance and interpretation

| Gate | Required evidence |
| --- | --- |
| Workflow | Exact200 updates; actual completed/drained typed artifacts; zero-update fresh restore bitwise final weights/splits/scales/predictions; zero-skip custody contracts |
| Conditioning question | On all384 existing frames, report last-decoder pre-SiLU ranges, within-channel feature variation, all-zero/nonconstant output counts and input/global gradients; distinguish prevention of the observed shutdown from residual solver usefulness |
| Numerical question | Keep all32 development teacher cases and unchanged original-b residual/divergence/wall/flux/Courant gates, warm<=32 then one fullp_base reset<=3000; retain every rejection/failure and full-budget temporal/zero/analytic comparisons |
| Cost | Actual inference, verification, failed warm/reset work, runtime/setup, train/checkpoint/restore and total enclosing cost; no speedup from iteration counts or selective cases |
| Decision | If all384 again collapse or native/cost results remain poor, retain the failure and stop; if feature transmission improves, report that separately from physics acceptance |

Nonzero output alone is not a solver gate. Do not expose reserved final geometry,
regime or whole-trajectory suites until a separately reviewed transition decides
that this development candidate warrants autonomous evaluation. Future rollout
stability/conservation and full local-GPU training retain the original separate
review gates; this proposal supplies no GPU/full training authorization.
