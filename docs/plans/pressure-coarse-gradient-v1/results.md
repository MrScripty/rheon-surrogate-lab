# Coarse gradient diagnostic: results

The diagnostic reproduced the finite-gradient-norm stop at update **1398**, with
**1397 completed updates**. Every gradient component and every individual
parameter norm was finite. The outer float32 norm aggregation overflowed to
infinity; the diagnostic float64 reference total is **3.044506589355366e19**.
All 13 shared logged losses exactly match the earlier failed candidate. This
establishes the immediate numerical failure in the faithful replay, not eventual
model accuracy or stability after a repair.

## What was executed

One managed CUDA diagnostic in the original Tuldok catalog:

- Run 5770116825f647ddabd2ba7e5413d885.
- Study 0f5e68cc15d34b0d9cd6a47b37fdfe52.
- Project bb3bf812d71f4bd5aa08881bc3564710.
- Captured source freeze 0edd151; separate cap 1500, original configuration 5000.

All 27 original numerical files and configuration bytes are unchanged. Nine
Project inputs, normalization, seeded 374112-parameter initialization, all five
full 5000-entry sampling arrays and their 1500-entry prefixes verify against
retained evidence. Keeping original array lengths matters: shortening updates
would change RNG positions in balanced feedback banks. The captured observer
does not change original losses, gradients, clipping, Adam, schedule or dtype.

Network parameters/gradients/Adam remain float32 CUDA; physical arithmetic remains
float64. Pumas owns Python 3.14.7/Torch 2.14.0+cu132/NumPy 2.5.3 on the RTX 5090 Laptop
GPU. There is no CPU optimizer, new simulation, CPU inference campaign, new
boundary/fluid setting, final candidate checkpoint, model promotion or holdout
access. No additional managed Run or replacement fit was started.

The original guard exception propagated. Tuldok records failed/nonzero_exit,
exit 1, drained/ECHILD and closed spawn; the single Study is closed and no CPU
restore was admitted. Zero artifacts were published. Diagnostic-controller
exit 0 means terminal evidence and closure were collected, not successful model
training. The earlier historical Run's exact failed update remains unobserved;
1398 is the exact replay update. Its13 retained loss checkpoints all agree.

## Failure anatomy

The largest two individual float32 parameter norms are:

| Parameter | Observed L2 gradient norm |
|---|---:|
|encoders.0.layers.0.weight|1.7409796856577262e19|
|decoders.3.layers.3.weight|1.2629991112216412e19|

Each is finite, but squaring and summing them in float32 exceeds float32's
approximately 3.40e38 maximum. Their actual combined norm is only 2.150854951e19,
which is representable. The full gradient norm 3.044506589e19 is also
representable. This is intermediate reduction overflow, not a requirement for
wider network weights or a relaxed physical/accuracy target.

The local CNN gradient reference norm is 3.044506050e19, versus 1.812322686e16 for
the added coarse matrix. The immediate overflow is therefore not dominated by
that matrix's gradient alone. All 10 observed forward outputs and all five loss
values were finite. The late-direction feedback stream on 24³ had loss
1.244640359e18; real-frame loss was 0.0492177924. These identify the problematic
update's context; sparse samples do not establish a steady loss-growth trend or
prove that repairing clipping will produce a useful pressure solver.

[Analysis](reports/analysis.json), [gradient table](reports/gradients.csv) and
[per-correction contribution table](reports/forward-contributions.csv) retain
all parameter norms, actual sample IDs, learning rate, finite checks and raw
network coarse/local/combined output and normalized-residual input norms.
Those raw-output norms are not gauge-corrected pressure errors. Two correction
rows per stream are observations of the existing recurrence, not new inference
or solver passes added by this experiment.

## Reproduction and prospective repair

A 0.433833 s CPU/CUDA reproduction uses the two actual observed parameter norms as
scalar gradients at the original Torch clipping seam. Both reproduce the guard
failure without changing gradients; deleting either element removes it. This
is the same outer norm operation, not the original full gradient tensors. See
[minimal reproduction](reports/minimal-reproduction-01.json).

The separate norm_repair.py prototype keeps original Torch clipping exactly
when its ordinary norm is finite. For rejected nonfinite reductions with finite
float32 components, it tests scaled float32 arithmetic: divide gradients by their
largest absolute component, measure the scaled norm, then rescale the scalar
norm. Actual nonfinite components and unrepresentable final norms still reject
before any gradient change. Original unit clipping and epsilon remain unchanged.

Its 0.750577 s CPU/CUDA qualification covers five exact ordinary-path cases,
three repaired cases (actual pair, all observed parameter norms, known 3–4 vector)
and three negative cases per device. Returned norms and unit directions agree
with independent Python math.hypot expectations. Every tensor in the repair
remains float32. NaN/Inf and finite components with an unrepresentable final norm
still reject without changing gradients. This prototype is **not installed in
the trainer**, and no repaired model update or stability replay is claimed.
See[arithmetic qualification](reports/norm-repair-qualification-01.json).

## Clocks and resources

| Measurement | Seconds | Scope |
|---|---:|---|
| Qualified observation seam02 |3.065495|Two original vs two observed CUDA seam updates; not fitting sequence|
| Capture construction |1.304029|Sources, declarations and Project decoding|
| Sampling/capture verification |3.423433|CPU construction, zero optimizer/forward/simulation calls|
| Managed Run |221.935134|Run start through drainage, including bootstrap/verification|
| Instrumented loop |165.504368|Initial-tensor event through clip failure; includes observer/checks/periodic diagnostics|
| Diagnostic command |383.922574|Admission, managed Run, terminal readback and Study closure|
| Custody collection command |9.238619|Public SDK and exact source/log readback|
| Analysis command |0.028221|Hash checks and scalar/table analysis|
| Minimal reproduction |0.433833|Internal CPU/CUDA arithmetic probe; zero model updates|
| Repair qualification |0.750577|Internal CPU/CUDA arithmetic checks; zero model updates|

Optimizer-only replay time is unavailable. Overlapping clocks must not be summed
as elapsed wall time, and the instrumented replay is not a throughput benchmark.
Sparse host samples are observations, not peak model memory or utilization.
Exact commands/environment/source identities and observed resource samples are
retained with the experiment. The reused runtime is the same as the original
candidate; its driver/OS dependencies retain the existing trusted-host scope.

## Evidence and disposition

The collector binds the Study intent and acknowledged Project/admission to the
actual public/internal Run, exact recipe and Project material, execution spec
hash and custody identity, all 32 actual staged source files and actual entry,
cap/reference argv. The analyzer requires that chain to the independently
verified capture and qualified observer before interpreting the failure.
See[custody summary](reports/diagnostic-custody-summary.json),
[custody receipt](reports/diagnostic-custody-receipt.json),
[stdout](reports/diagnostic-stdout.log) and [stderr](reports/diagnostic-stderr.log).
Internal execution/environment records, inputs and initial weights remain
outside Git. Historical results and root user assets remain unchanged.

Next: integrate the qualified overflow-only repair at the training clipping
owner, verify original finite-prefix behavior, then run one preregistered capped
CUDA stability screen. Only a successful screen can justify another matched
full fit and 74-case CPU qualification. All physical and population gates remain
unchanged; the eight unused scenes remain untouched. The separate
[next proposal](next-proposal.md) owns those prospective steps.
