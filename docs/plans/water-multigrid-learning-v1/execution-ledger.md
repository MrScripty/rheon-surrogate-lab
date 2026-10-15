# Execution ledger

2026-10-15: inspected clean HEAD 5bf5c2d and accepted previous tranche. Admitted
the recorded follow-ups on experiment/water-multigrid-learning-v1. Historical
results remain immutable. Two changes are controlled separately: per-evaluation
CPU stopping for both existing checkpoints, and multi-grid analytical training.
The broader cohort and architecture changes remain conditional on measured fit.

Implemented optional per-network-evaluation stopping in the existing corrector,
trajectory adapter and native receipt consumer. Captured recurrence owner is
reused without changing model configuration. Seven focused tests pass, including
an actual four-evaluation recurrence versus four separate evaluations: identical
pressure bits, four observed forwards in each, original metadata unchanged.
Existing two-cell RHS/divergence and budget checks remain passing. This is an
inference-policy measurement change, separate from training-population changes.

Added closed schema-v4 feedback with explicit grid counts, one cubic spacing per
grid, balanced private-RNG sampling and an exact 768 MiB cache projection.
Legacy analytical bank dispatch retains tensors/order/evidence. Three new
contract/eigenvalue tests, five historical analytical tests and six residual
contract tests pass. Scope explicitly includes the new test_multigrid.py and
training configuration files. The 10-update smoke covers every grid twice.

The complete authored bank qualifies on CPU: 640 entries across five grids,
worst analytical-eigenvalue versus implemented-operator relative mismatch
6.90286e-14. The original captured bank source independently reproduces every
legacy tensor, sample order and evidence byte; model RNG stays unchanged.
Resident tensor budget and the exact per-grid exposures are retained in
reports/bank-qualification.json. No optimizer or CUDA context in this check.

Controller environment correction: the Pumas interpreter lacks psycopg; the
Tuldok interpreter lacks NumPy for the observer. Both failed before Run
submission. The repository .venv has the established controller dependencies;
managed training still selects the qualified Pumas CUDA runtime. Both startup
logs remain under runs/. No extra model attempts are counted for these errors.

Added reusable configured CPU study driver. Every checkpoint episode uses a
fresh Python process, exact captured sources and a distinct output/log. Fixed
worker/core partitions prevent contention between candidate jobs. Concurrent
checks are numerical/throughput evidence; repeated timings must run separately.

Matched fine-stopping CPU study completed all seven native controls and fourteen
candidate episodes: 112/112 AI steps qualify, every native/controller child is
reaped. Latest weights need 22.875 evaluations on 40-cubed versus 29.75 for older
weights, under the same per-evaluation stopping. These overlapping timings carry
no latency claim. Older checkpoint development recheck passes 128/128 with mean
13.4375 evaluations, versus its prior block-of-four mean 14.96875. Retain fresh
latest-checkpoint development verification separately before final attribution.
The two existing native process failure/deadline tests also pass.

Fresh latest-checkpoint development replay again qualifies 128/128. With stopping
fixed per evaluation, the earlier/current means are 13.4375/12.6015625, a 6.22%
reduction attributable to changed training/checkpoint on this population. The
previous 15.81% comparison also included stopping granularity and is not a
training-only claim. Larger-grid qualification is short and development-only.

Tuldok enforces immutable manifest snapshots: reusing the previous project ID
for new content returned HTTP409. Preserved that snapshot and registered the new
multi-grid project a5e968a9ceb740c3b775ed14103bc519 in the same catalog. This is a
project snapshot, not a separate dataset catalog. Local CPU profile startup also
caught a stdlib module-name collision; renamed profile.py to cpu_profile.py and
retained its failed startup log. Neither issue is a model numerical result.

Smoke completed/drained training Run 6327bdba13f3465ab82474241f23d52f and fresh
CPU restore d488ff6ca3a84764ad9b33f9909d62fc. Ten CUDA updates, two analytical
exposures per grid, 488,651,776 resident tensor bytes, peak CUDA allocation
531,972,096 bytes; all weight/prediction bits restore exactly and artifacts
reopen/redownload identically. Full 5,000-update one-pass multi-grid sequence
was then submitted through the same workflow (source capture fb0eff6).

CPU profiles cover 16/24/32/40-cubed with 1/4/8 Torch threads, identical mode
mixtures and stopping. All sampled results satisfy the equation gate. Profiled
convolution and tensor-fill cost dominate; four threads help this mechanism,
but eight regress relative to four at 32-cubed. These microprofiles do not
prove whole-CFD latency or fastest production settings. Added optional 1/4/8
threads to the existing rollout owner and disjoint core groups to the study
controller for a matched native/AI follow-up. Defaults remain one thread.
Two process ownership tests still pass. Scientific source and deployment stay
separate; this extension is execution policy, not a model change.

Long-window native controls and current checkpoint complete 512 steps on all
three small-grid scenes, with 1,491/1,491 learned steps qualified after warm-up.
Forcing is explicitly quarter strength to keep the native controls physically
valid at the longer fixed timestep; this does not extend the original full-force
claim. Readiness of the broader cohort remains conditional on new fit results.

The optional four-thread trajectory fixture qualifies native and learned tracks
with one AI step after 15 native steps, explicit CPUs1–4 and observed four-thread
budgets. All children reaped. Retained separate-seed 120-mixture CPU baseline
uses up to all23 independent worker cores while CUDA fitting proceeds; low modes
and arbitrary-frequency mixtures are distinguished rather than averaged away.

Full one-evaluation fit completed 5,000 CUDA updates at 44.3782 updates/s.
The retained post-update fitted equation ratio is .0082280, still above 1e-6.
This satisfies the recorded condition for the prepared two-evaluation training
trial: compare whether recurrence during training improves error reduction.
Admit its unchanged 5,000-update config, same seeds, real/analytical sample
exposure and learning rates; it uses twice the neural evaluations/voxel work.
The first trial's full CPU restore/validation qualification remains pending;
no learned acceptance is inferred from this training diagnostic. Its optimizer
has ended before the second job is submitted through Tuldok's serial owner.

Prepared the conditional cohort feedback contract separately from the running
captured trials. Schema v5 names train-correction-cell-rms: all cached real
training cell-states determine analytical amplitude, with common rho/dt checked.
It does not select an arbitrary first parent or use validation statistics.
Historical v3/v4 one-frame paths remain unchanged. Cohort preprocessing still
fits only real training data; that population change is explicit, not a
single-factor learning claim. Added independent weighted-RMS, authority and
physics-mismatch tests. Admission of a cohort Run remains conditional on the
current trials' actual iterative fit/development/grid evidence; no Run starts
merely because these configurations exist.

One-pass multi-grid development completes 128/128 physical passes, averaging
13.46875 evaluations (worst23), versus current reference12.6015625 (worst26).
This is a mean-work regression on small saved states despite a better tail;
retain it rather than presenting every new training run as an improvement.
Larger grids and two-evaluation weights still need their full comparison.
Two-evaluation optimizer completed5,000 updates at23.5147 updates/s; fitted
post-update ratio .00309170 is diagnostic, not final CPU physical acceptance.

Prepared cohort bank independently qualifies all640 entries, worst relative
operator mismatch6.44257e-14, global real-train correction RMS2.853573464078259Pa.
Legacy bank remains bit-identical and model RNG unchanged. No cohort Run has
been admitted yet. Corrected the mode-probe description to60 distinct mixtures
at two signed amplitudes (120 cases); generated inputs are unchanged.
