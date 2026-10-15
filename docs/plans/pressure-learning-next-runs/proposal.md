# Next pressure-model training runs

Status: proposed scientific sequence; no D0/R1 or later experiment submitted. This is
not an executable recipe or an admitted implementation plan. Machine-readable
budgets and factors are in [proposal.json](proposal.json). Existing historical
results and current float32 target remain unchanged.

User correction: **training uses CUDA; CPU-first applies to inference**. Every
optimizer run below requires CUDA, with no automatic CPU training fallback.
CPU restore/inference is a separate deployment acceptance step. The previous
CPU-training wording was an interpretation error and is superseded here.

The first question is whether the current model can accurately fit a known
frame. Selected training snapshots currently have relative residuals 0.22–0.62,
versus the calibrated 1e-6 target. The prior run used only 200 updates.

| Stage | Question and controlled change | Budget and next decision |
| --- | --- | --- |
| D0 | Measure amplitude/sign response of the existing model at residual multipliers 0, 0.25, 0.5, 1, 2, 4 and -1. Use training-parent snapshots only. | No optimizer updates. These are algebraic probes, explicitly labelled rather than presented as new simulations. |
| R1 | Fit one captured frame with the current seven-channel GroupNorm network: variable-train-510100, frame 0, 16³ cells, h=0.0625 m. | One initialization, batch one, up to 10,000 CUDA updates / 10 minutes elapsed. Judge deployed CPU predictions against the full current target profile. |
| R2 | If fitting or amplitude fails, compare GroupNorm removal and explicit residual-amplitude scaling, each independently against R1. | Up to two separate runs, identical case, initialization and optimizer schedule. Do not combine both changes. Define amplitude scaling before implementation; it must preserve units, gauge and exact zero bypass. |
| R3 | Train the unchanged baseline and the selected variant longer on all existing 96 training frames. Keep the 48 validation frames out of optimization and normalization. | Up to 10,000 updates per run, three paired initialization seeds per variant; identical sample order. This distinguishes additional training from the normalization change. |
| R4 | If needed, train two successive shared-weight pressure corrections, recomputing residual between them. Compare with the matched one-step R3 control. | Three paired seeds. Same data, architecture and loss; report model-call count and elapsed cost as well as accuracy. Do not treat a later conventional solve as model success. |
| R5 | Train the selected method on a larger independent cohort, then evaluate a genuinely reserved test set. | Proposed 24 train / 8 validation / 8 test parents across 8–32 cells/metre, with unseen shapes/resolutions reserved. Freeze assignments and new native seeds before capture; qualify one-frame fixtures first. |

Use float32 throughout the learned model and cubic cells. Optimization runs on
CUDA; deployment inference is checked on one CPU thread. The
shared proposed optimizer schedule is Adam at 1e-3, reducing to 1e-4 at update
2,000 and 1e-5 at 6,000. Each run has a 600-second wall cap; reaching the cap
without a pass is budget-limited evidence, not proof that fitting is impossible.
Save only the declared checkpoints plus any terminal state, bounded by 32 MiB.
R1/R2 may stop at the first scheduled inspection that passes all fitted-frame
gates. R3–R5 use the terminal reached state as the candidate, without selecting
a better intermediate checkpoint. Apply this consistently to paired runs.

At every inspection point record equation residual, worst predicted and actual
divergence, pressure/gradient error, amplitude behavior, model inference time and
any native correction/fallback cost separately. Success requires the current
float32 profile: residual <= max(1e-12,1e-6*||b||₂), with all physical limits
preserved. Mean training loss alone is insufficient. For replicated comparisons,
report each seed and median/worst performance, rather than choosing a lucky seed.

Preparation begins with managed CUDA runtime/device qualification and portable
checkpoint export into a separate CPU restore/inference Run in the existing
Tuldok catalog. Confirm model parameters, training inputs, gradients and Adam
moment buffers actually use CUDA; record GPU/driver/Torch/CUDA versions and
peak allocated/reserved VRAM. CPU restore must preserve checkpoint tensor bytes
and repeat CPU inference. CUDA-versus-CPU predictions may differ numerically;
record differences and apply unchanged deployment accuracy gates rather than
claiming bitwise cross-device equality. Runtime-version differences require
explicit qualification before controlled scientific comparisons.

The current Tuldok captured runtime profile intentionally excludes CUDA and the
current seven-channel trainer asserts CPU tensors. A new versioned portability
path is needed; do not remove these constraints from frozen historical runs or
claim that CUDA integration already works. The installed Pumas runtime and host
GPU can be probed read-only before any training; that smoke check does not
qualify managed lab-model training or fabricate a Tuldok Run.
The [installed-runtime smoke](reports/cuda-kernel-smoke.json) passed on the
RTX 5090 Laptop GPU with Pumas Torch 2.14.0+cu132 / CUDA 13.2. No optimizer
updates were performed. Managed CUDA training and CPU checkpoint portability subsequently passed a
[separate 20-update qualification](../cuda-training-cpu-inference/results.md).
The new versioned trainer supports 1–10,000 updates, sparse bounded checkpoints,
explicit learning-rate changes and independent sampling/initialization seeds.
It preserves the architecture and all older CPU sources/results. Both actual
managed Runs completed and drained; twelve CPU predictions repeated exactly.

Before D0/R1, implement the labelled algebraic probes, explicit fitted-frame
selection and scheduled native target inspection. The current qualification uses
all training parents; it is not a one-case fitting experiment. Define each R2
variant independently and add a genuine reserved test split before R5. Keep
physical thresholds fixed and project artifact limits before admission.

If one-case fitting remains unsuccessful after the controlled comparisons,
inspect gradients, learning-rate behavior and coarse-grid communication before
expanding the cohort. Further architecture changes require their own comparison;
none are silently included in this sequence. The first execution slice would be
trainer preparation plus D0 and R1, not the entire conditional queue.

Run CUDA training and CPU restore/inference through the user's existing Tuldok
catalog with their qualified runtime profiles. Publish actual
captured cases, model predictions and progress to the existing Rheon/Tuldok
interfaces. Preserve historical runs, keep model and portability work separate,
and use fresh output locations. This proposal authorizes no jobs by itself.
