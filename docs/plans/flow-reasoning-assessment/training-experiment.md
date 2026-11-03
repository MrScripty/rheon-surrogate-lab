# Flow reasoning inspired pressure training experiment

2026-10-18. Status: the nine-fit screen and CPU evaluation are complete.
No candidate passed the full fitting gate; C53 remains the retained reference.
[Outcome and interpretation](outcome.md), [all nine results](execution/results.md),
[execution state](execution/execution.json),
[qualified Project inventory](execution/prepared.json),
[execution ledger](execution-ledger.md). The initialization manifest remains the
immutable pretraining plan; actual Run progress is recorded separately.

Test **detached supervision on the current network's intermediate pressure states**
while retaining the existing four-pass objective. Start with the eight manufactured
shape-fitting cases, then require a separate real-frame retention experiment before
considering deployment. Repeat the complete three-arm screen from **C53, C44 and
C37**, for nine fits total. C53 remains the retained deployment reference.

The useful question is whether a correction learns more effectively when its input
pressure is treated as fixed, rather than differentiating through the preceding
corrections. Our model already generates its own intermediate training states.
This comparison tests how those states receive gradients; recurrence itself is
already present.
[Existing training objective](../../../experiments/pressure-four-pass-distillation-v1/fine_tuning.py),
[existing recurrence](../../../experiments/pressure-pass-sweep-v1/inference.py).

## Evidence from completed experiments

| Experiment | Recorded result | Implication for this proposal |
| --- | --- | --- |
| C53 four versus eight corrections | Four passes meet the equation target on 7/128 consumed requests; eight meet it on 128/128. Median measured inference components are 12.20 and 24.22 ms. | Four-pass quality remains a useful target. These timings exclude simulator and IPC costs. |
| C57 through C59 four-pass fine-tuning | No qualified reduction to four passes; typical improvements did not protect the tail. | Preserve worst-case and early-pass checks. |
| C60/C61 low-mode feedback | Both fail the complete four-pass qualification. | Additional low-mode weighting alone has not solved the problem. |
| C62 single-mode capacity | Passes its fitted-case capacity gate. | The network can fit that particular difficult mode. |
| C63 frequency curriculum | Fails four-pass and retention gates. | Increasing fundamental-mode exposure was insufficient in this screen. |
| C64/C65/C66 fundamental fitting | C64 fits 15/24 four-pass cases; C65 fits all three axes on one grid; C66 fits 5/8 shapes. | Joint fitting across shapes remains a useful bounded test. |
| C67 delayed decay | Fits 6/8 shapes, but worsens the worst four-pass residual versus C66. | A higher success count alone is insufficient. |
| C69 balanced aggregation | Fits 6/8 at four passes, 7/8 at eight; fails every declared capacity check. | Aggregation did not remove the shared fitting difficulty. |
| Adam reset and first-moment damping screens | No proposal satisfies the declared retention guards. | Another optimizer change is not an established shortcut. |

Sources: [four-pass decision](../pressure-four-pass-distillation-v1/decision.json),
[low-mode decision](../pressure-low-mode-tail-v1/decision.json),
[single-mode decision](../pressure-single-mode-fit-v1/capacity-decision.json),
[curriculum decision](../pressure-single-mode-fit-v1/curriculum-decision.json),
[fundamental results](../pressure-fundamental-bank-fit-v1/results.json),
[decay results](../pressure-shape-decay-v1/results.json),
[aggregation results](../pressure-balanced-shapes-v1/comparison/results.json),
[optimizer follow-up](../pressure-adam-momentum-damping-v1/next-steps.md).

The earlier current-policy experiments already trained on a **frozen** bank of
early or eight-correction C53 residuals and did not qualify a replacement. Online
states from the weights currently being optimized are a distinct comparison.
[Frozen bank construction](../../../runs/dependency-migration-v1/lab/experiments/pressure-current-policy-v1/freeze_bank.py),
[current-policy decision](../../../runs/dependency-migration-v1/lab/docs/plans/pressure-current-policy-v1/reports/qualification-decision.json).

The original [assessment](research.md) recommends finishing C62 first. That step
and its conditional curriculum are now complete. The subsequent CUDA gradient
diagnostic has a [recorded plan](../pressure-cuda-gradient-conflicts-v1/plan.md);
no result report was present in that directory at this inspection. Its eventual
findings should inform interpretation without being treated as completed evidence.

## Relationship to Fixed Point Forcing

In the paper, Fixed-Point Forcing generates a detached conditioning carry through
a model rollout, while preserving the target-derived flow input for the supervised
update. Our pressure network has a residual input and no separate denoising carry.
Sampling a pressure correction depth is therefore a CFD adaptation. The paper
does not establish CFD convergence or a wall-clock speedup for this model.
[Method verification](training-method-notes.md),
[paper sections 2 and 3](https://arxiv.org/pdf/2606.29150v3).

Retain the pressure representation, architecture, operator, precision and physical
acceptance gate. A categorical denoiser or a new carry channel would add a separate
architecture experiment.

## First experiment and matched controls

Use C66's eight X-fundamental training cases, with fresh Adam in every arm.
Run the same experiment independently from each checkpoint below. These are
three different initializations, not three random-seed repetitions or a sequence
where one checkpoint's fitted result initializes the next.

| Starting checkpoint | Historical label | Historical corrections per request | New fits |
| --- | --- | --- | --- |
| C53 | current-late | 2 | C53 U, C53 A, C53 D |
| C44 | extra-late | 2 | C44 U, C44 A, C44 D |
| C37 | shape-one-longer | 1 | C37 U, C37 A, C37 D |

The [initialization manifest](initializations.json) binds each exact weights file,
source/configuration identities and the nine proposed fit keys. The three weight
inventories have the same 19 tensor names, shapes and dtypes, and their saved
normalization files are byte-identical. Their historical one-step numerical
sources must still pass prediction parity under the shared experiment wrapper;
matching tensor inventories alone is not numerical qualification.
[Captured checkpoint material](../../../runs/pressure-pass-sweep-v1/cpu-01/material.json),
[historical recurrence verification](../pressure-pass-sweep-v1/verification.json).

Retain C66's 1,600 single-example updates, 200 exposures per shape, sample order,
learning-rate schedule, clipping, normalization and
float32 network with float64 physical arithmetic. Teacher weight remains zero.
Use four training corrections in every fit, including those initialized from
C37. Its historical one-pass setting describes its training provenance, not this
screen's pass budget. The shared numerical owner, objective, data and schedule
must be identical across starting checkpoints; only initialization weights change.
Historical direction/current/supplemental feedback configuration must not silently
enable additional losses in any arm. These are fitting diagnostics, not new fluid
simulations. Execute the groups in order C53, C44, C37 under the same admission
limits, retaining all three groups even if one fails the accuracy gate. A numerical
qualification or resource failure stops the affected fit for diagnosis.
[Configuration](../../../experiments/pressure-fundamental-bank-fit-v1/config.json),
[fitting stream](../../../experiments/pressure-fundamental-bank-fit-v1/fit_bank.py).

| Arm | Objective | Auxiliary history |
| --- | --- | --- |
| U | Existing four-pass fitting loss | None |
| A | U plus `0.05 * Q` at a sampled correction depth | Gradients traverse the sampled prefix |
| D | Same scalar objective as A | Detach the sampled prefix before the auxiliary correction |

The proposed auxiliary weight `0.05` is a fixed screening choice, not a value
established by the paper. Keep the existing early-feedback weight `0.05` in all
nine fits. Use the final checkpoint for the decision; intermediate measurements
describe learning, rather than selecting an advantageous checkpoint afterward.

For each shape, precompute a privately seeded, balanced order of depths
`k in {0,1,2,3}`: 50 exposures at each depth. A and D use the same depth sequence
across all three starting checkpoints.
Generate it independently of the existing sample RNG so the original case order
does not change. State generation always uses the current pre-update weights.

Let `s0` retain the original problem, with base `p0`, initial residual `r0`,
reference pressure and physical metadata. The ordinary U forward already computes
`p1` through `p4`. Reuse `pk` from that forward, with `p0` selected when `k=0`:

```text
carry = pk                  # A
carry = pk.detach()         # D
rk = r0 - A_operator(carry - p0)
local = s0 with base=carry, native_input_residual=rk
p_next = existing_one_correction(network, local)

Q = analytical_loss(p_next, original_s0)
    + log1p((||r0 - A_operator(p_next - p0)||² / ||r0||²) / 1e-12)
loss = original_four_pass_loss + 0.05 * Q
```

Use the existing analytical pressure, gradient, equation and amplification
criterion in `Q`; retain its original weights and reference. The local sample is
only for predicting the correction. In particular, never replace the loss's
original `r0` denominator with `rk`, which would give late states a different
relative weight. The existing helpers' denominator handling remains authoritative.
The manufactured fitting cases have nonzero RHS; separately qualify zero-residual
behavior before applying this design to broader data.
[Analytical criterion](../../../runs/pressure-fundamental-bank-fit-v1/shape-isolation-project-01/analytical.py),
[equation feedback](../../../experiments/pressure-four-pass-distillation-v1/fine_tuning.py).

A versus D isolates the auxiliary gradient boundary: at identical weights, sample
and depth they have the same pressure and scalar loss, with different derivatives
through the prefix. U versus either augmented arm also changes supervision and
work, so that comparison alone cannot attribute a benefit to detachment. This
screen does not independently isolate online versus frozen state generation.

All sampled states already occur in U's four-pass chain. A adds intermediate
pressure supervision and changes equation-loss weighting; D changes that added
term's gradient horizon. At `k=0`, A and D have identical gradients. At `k=3`,
the auxiliary prediction repeats the terminal prediction at fixed weights, so
A adds terminal weight and D adds a local derivative there. Report these depths
explicitly. Testing states beyond this chain would be a separate exposure study.

Reusing the existing prefix avoids a second rollout. A and D each perform five
network forwards per update, versus U's four: 8,000 versus 6,400 forwards across
the fit. A and D match forward cells as well. Record actual backward work, trainer
time, memory and complete managed workflow time separately; equal forward counts
do not establish equal wall time. The U graph remains present, so this proposal
does not promise the memory savings of wholly detached training.

Each initialization requires 4,800 optimizer updates and 22,400 network forwards
across U/A/D. The full nine-fit screen requires 14,400 updates, 14,400 original
case presentations, 9,600 auxiliary losses and 67,200 network forwards, excluding
qualification and diagnostic work. That is three times the original screen's
training work; wall time must be measured rather than inferred from these counts.

## Qualification before fitting

Keep the change in a new experiment wrapper and preserve captured historical
sources. First verify these numerical properties without optimizer updates:

1. At fixed weights and depth, A and D match U's corresponding next pressure and
   match each other's loss within a declared CPU/CUDA numerical tolerance.
2. The D auxiliary loss has no gradient path to the prefix pressure, while its
   final correction and physical residual retain finite, nonzero parameter
   gradients. Inspect the auxiliary branch separately because U still supplies
   prefix gradients. At `k=0`, A and D parameter gradients should agree.
3. Original targets, residual denominators, normalization and all physical metadata
   stay bound to `s0`; gauges and recomputed residuals match the inference owner.
4. Depth draws leave case order unchanged; actual counters confirm exposures,
   forwards and cells. Reusing the prefix does not alter parameters or module
   state before the optimizer step.

For C53, replay U as the numerical control against retained C66 quality and weights
under the qualified runtime, disclosing any platform difference. C44 and C37 each
need a newly trained U control; C66 is not their matched control. First measure
each untouched starting checkpoint at 1/2/4/8 passes on the same fitting and
regression populations. Verify shared-wrapper prediction parity against each
checkpoint's captured numerical owner before any optimizer steps. C37's older
helper lacks the intermediate-pressure collector, so qualify that interface
adaptation explicitly rather than replacing its initialization or historical files.
Use the existing
managed training and fresh CPU restore path, artifact parity checks and resource
bounds. The [existing method](../pressure-single-mode-fit-v1/method.md) describes
that path; do not create a separate trainer or catalog for this comparison.

## Decisions after the screen

The capacity gate remains: **all eight** fitted cases must have restored CPU
four-pass and eight-pass relative residuals at most `1e-6`, with no per-case
one-pass or two-pass regression against that group's untouched starting checkpoint.
Report paired D-versus-A, D-versus-U and D-versus-starting-checkpoint results for
each group, including the worst case, each pass's contraction, and low-mode
pressure error. Also report every candidate against C53 under the unchanged
deployment guards. Preserving C37's own early-pass quality alone does not make a
candidate eligible to replace C53. A lower mean loss does not substitute for a gate.

Keep the 128 consumed real requests and 240 manufactured regression cases as
separate populations, using the existing early-pass, four-pass worst-case and
eight-pass guards. A specialist fitting success permits consideration of a
broader retention study; it does not qualify deployment. Existing real-data
regressions must be carried into that study explicitly rather than hidden by the
specialist capacity result.
[Existing guards](../pressure-low-mode-tail-v1/decision.json),
[shape capacity checks](../pressure-fundamental-bank-fit-v1/shape-isolation/capacity-decision.json).

Evaluate these decisions separately for each starting checkpoint. If D improves
the typical case but still misses the capacity gate, retain C53
and analyze which depths and shapes help or regress before proposing a longer
fit. If A and D behave similarly, the evidence favors extra supervision over a
specific detachment benefit. The three starting checkpoints have different training
histories, so differences between them do not isolate a particular historical
training mechanism. Report all three groups rather than selecting one favorable
result. A single seed per initialization is a screen; these three initializations
do not replace confirmation with additional predeclared random seeds.

If D passes the fitting gate and compares favorably with A, the next proposal
should use the existing broad training mixture with matched A/D objectives,
retaining frozen-teacher and early-feedback settings identically across arms.
Start again from the successful group's original checkpoint, rather than its
specialist fitted weights. Any teacher must be frozen and identical within that
group, with its identity declared before fitting. Use training-only states and
labels; do not incorporate
consumed validation targets into training. Require four-pass success on all 128
requests, the retained eight-pass/pressure-error guards and the existing
manufactured threshold before independent whole-scene qualification.

Only a qualified candidate should advance to fresh scenes and paired timing
against C53 and the native solver. Previously evaluated scenes remain consumed
regressions. Keep the physical residual gate for stopping: agreement between
successive predictions can also mean the model has stalled. Any eventual speed
claim must include actual network calls, equation checks and simulator/IPC costs.
