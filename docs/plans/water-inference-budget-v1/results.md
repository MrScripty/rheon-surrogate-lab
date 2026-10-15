# Inference budget and scaling results

In progress; isolated grid-scaling timing and the lower-rate fit are pending.
The retained checkpoint now passes the same development physical gates with
adaptive CPU stopping. Neural weights remain float32; physical operator/base
arithmetic and every acceptance threshold are unchanged.

## Model sizes

| Model | Trainable parameters | Actual checkpoint bytes |
| --- | ---: | ---: |
| Current one-channel learned corrector | 370,016 | 1,485,452 |
| Original successful 32-cubed modal model | 32,767 | 272,149 |
| Earlier ten-channel neural model | 372,377 | 1,500,186 |

Current versus modal: 11.29 times the parameters and 5.46 times the file size.
The modal checkpoint uses float64 gains and a fixed cosine-basis buffer for
one fixed grid. Current parameters are float32 and shared between corrections.
Repeating inference does not duplicate the weights. See [inventory](reports/model-sizes.json).

## Adaptive inference

The existing stability-modes checkpoint passes all 128 development physical
checks using 8–28 network evaluations, average 14.96875 instead of fixed 32:
53.22% less network work. Median equation residual 2.33592e-7; worst 9.95478e-7.
None pass the stricter original native 1e-8 profile at this stopping point.
Native refinement methods in evaluator reports are separate benchmarks and do
not contribute to the raw adaptive candidate's acceptance.

All 854 AI steps qualify across the same three short and three extended
trajectories used for the previous fixed-32 comparison. The extended cube,
rectangular and compact scenes average 13.24, 12 and 16.27 network evaluations;
final pressure-field relative L2 differences from native are 9.77e-8, 2.37e-7
and 8.10e-7. Windows remain about four seconds. Timings overlap managed work;
no elapsed-time speedup is claimed from these runs. See [evidence](reports/adaptive-evidence.json).

## Single-pass training

Mode-feedback single-pass fitting completed 5,000 actual CUDA updates with a
fresh managed CPU restore and development inference. Its known-frame equation
residual is .00737933; development median .0164437, worst .106447, physical
qualification 0/128. All three short trajectories finish, with 0/147 AI steps
qualified. Initial parameter bits, real-frame order, analytical bank/order and
normalization match the four-pass control. Equal updates/exposure use fewer
network evaluations; optimization runs about 45.7 updates/s.

Independent analytical mixtures reduce error on all 72 first proposals, but
worst residual after eight one-network corrections is 6.36510e-5. This model
has not reached the required one-pass accuracy. The detailed
[trial record](reports/stability-modes-single-pass.json),
[control parity](reports/single-pass-control-parity.json) and
[mode addendum](reports/single-pass-mode-addendum.json) preserve the negative result.

Combined feedback/output-scale single-pass fitting failed with non-finite
gradient norm after at least 2,400 updates; last progress loss was 77,382.87.
The Run drained and published no model artifacts. Its exact terminal update is
unavailable. The [failure](reports/single-pass-combined-failure.json) is retained;
a separately captured comparison lowers the entire learning-rate schedule tenfold.
No failed checkpoint is substituted for a completed model.

## Scaling and disposition

Pending: serial repeated CPU comparisons at 16,24,32,40 cells per axis, separating
fixed volume from fixed spacing, with native controls and all numerical failures
retained. See [method](reports/methods.md) and the [current plan](plan.md).
Broader cohort training and architecture changes depend on fitting/scaling
outcomes. No fresh final test was consumed and no Rheon UI deployment changed.
