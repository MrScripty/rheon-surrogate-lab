# Execution ledger

2026-10-15: admitted continuation on a new branch from 9512221. The retained
recurrent model improves saved-state median error but amplifies finest modes and
hard-stops independent simulations after 4–6 AI steps. Implement mode feedback
first, with output scaling isolated as the next intervention. Historical records
and deployed comparison adapter remain unchanged. No new training started yet.

Implementation qualification: 18 focused Python checks passed (7 conditioning,
6 retained residual contracts, 5 analytical/output-scale checks). Analytical
labels match the independently specified rectangular Neumann eigenvalues;
zero correction incurs loss 1 and a reversed solution incurs loss 13, proving
non-amplification alone cannot satisfy the objective. Signed sample generation
is deterministic and leaves Torch initialization randomness untouched.

Force-path inspection: existing native paired engine applies advection, the
scene force, viscosity, RHS assembly and previous-pressure residual before
requesting inference separately on each track. The one-channel input contains
the force contribution to the pressure equation. Force parameters are not neural
features; the model replaces pressure projection, not whole simulation dynamics.
No force-path change is needed. Source: runs/water-learning-sequence-v1/Rheon/
src/bin/desktop/ai_comparison/engine.rs, prepare/force/advance.

Committed source 042d839 and captured all four configurations before execution.
Tuldok project 9a5cbb1f6b9c4a49953645bc109094e1 exposes the active plan,
results and ledger. Two CUDA smoke updates completed in normal Run
97b2b4faccc14336a44af87c27bfe8d4; actual float32 parameters, gradients and Adam
moments verified, with AMP/TF32 off. Fresh CPU restore is still pending runtime
qualification. Full-trial runtime preparation overlaps this restore; execution
is owned and serialized by Tuldok. Input/preprocessing parity with the retained
recurrent control is established in reports/comparison-inputs.json.

Unchanged native build source and binary hashes verified against the retained
qualified build. Reuse all three native/native 64-step controls: each has 49
post-warm-up qualified solves and no hard stop (reports/native-controls.json).
Analytical sample generation and loss details are in reports/commands.md.

Execution scheduling clarification: full-mode admission overlapped the fresh
smoke CPU-restore queue. The real two-update CUDA smoke and its in-process CPU
transfer plus 18 CPU source checks had passed; fresh managed CPU restore remains
mandatory before accepting results. This overlap changes no scientific control
or deployment gate. Tuldok owns all actual Run scheduling. Added a CPU-only
72-case mixed-mode development probe while the GPU fits: independent seed,
signed amplitudes, three grids, analytical labels, ideal f64/f32 controls,
eight proposals, immutable captured CPU model loader and one core per worker.
This is sampled development evidence, not a global convergence guarantee.

CPU baseline mixture probe completed while CUDA was training: 14/72 first
proposals reduce equation error, worst first gain 7.1994, worst after eight
wrapper proposals 7.5678e10. Exact analytical f64 solutions passed all 72 operator
checks; no CUDA context. This strengthens the sampled amplification diagnosis.
Full rows and identities retained in reports/baseline-mixed-modes.json.

Fresh smoke CPU restore completed/drained/reopened with whole-artifact parity in
Run f63f31bda2904b65a3da282f875f8ef1. Smoke record retained. Its first recording
exposed a relative --reports path bug in the new generic recorder: the record
was written correctly, but the final display path failed. Normalize the selected
output root to absolute, then re-record the same immutable report successfully
with identical bytes (reports/record-cli-check.json). No training retry needed.
Full mode-feedback fit completed 5,000 updates at 12.8053 updates/s including
bank/setup: final training-frame residual 0.000436322 versus retained 0.000349860.
Canonical development loss improves from 0.00171135 to 0.0000134667; independent
CPU physical checks remain pending. All initial parameter bits and real-frame
sample order match the retained recurrent control (reports/control-parity.json).

Scale-only fit completed 5,000 updates: fitted residual 0.000322258 (7.9% lower
than retained recurrent), canonical development loss 0.000794065 versus baseline
0.00171135. Mode feedback improves the same development objective, while scaling
improves fitted residual; admit the already committed combined configuration as
a bounded factorial interaction test, fixed terminal 5,000 updates. All three
configurations were captured at source 042d839 before these outcomes. The new
combined run adds no CFD data or new parameter seed. Native qualification and
fresh CPU acceptance remain separate; cohort expansion is still conditional.

Mode-feedback CPU evidence: first-proposal development median 0.000747636
versus matched baseline 0.0143348 (94.8% lower); worst 0.0328209 versus 0.221805.
All 24 single modes and 72 independently seeded mixtures reduce error; worst
first gains 0.03563 and 0.02265. All three full 64-step simulations complete,
with 49 AI steps each and no hard stop, versus 4–6 before. Raw four-pass pressure
still fails physical qualification on every simulated step and the fitted frame.
Repeated recorded-state corrections now converge: all 128 equation checks pass
by seven proposals; eight-proposal worst relative residual 5.8231e-8. Replan the
next CPU inference slice around that evidence: explicitly budget eight proposals
(32 neural passes), no native refinement, same captured checkpoint and f64
physical accumulation, verify all native physical gates on fitted/development
states and full trajectories. It is a distinct inference-policy experiment,
not a relabelled successful single-proposal model. Two independent two-cell
feedback/budget tests and two existing real process-cleanup tests pass.

Eight-proposal native evaluation passes the fitted frame and all 128/128
float32-profile development cases (121/128 also pass the stricter original
native 1e-8 profile). No native pressure cleanup contributes to this candidate.
All three 64-step trajectories pass all 49 AI steps. Longer 256-step controls
and learned trajectories pass all 241 AI steps for rectangular and compact
scenes. The cube native control fails at step242 on Courant 0.502272 > 0.5;
its pressure/divergence checks pass, and every earlier 241 native frame qualifies.
This is a native fixed-timestep/forcing horizon limit, not a learned-model fail.
Retain the failed control and bound the cube comparison at 240 steps with a fresh
native control. No timestep, force, density, cell spacing or gate is changed.

All three full CUDA fits and their fresh managed CPU restores/validation Runs
completed and drained, reopened/redownloaded with whole-artifact parity: 11
stages and 15,002 optimizer updates including smoke. Combined four-evaluation
model median 0.000741674, worst 0.00612879, physical passes 0/128; three actual
64-step trajectories complete with no hard stop but no qualified AI pressure.
The mode-only 32-evaluation policy remains the separately qualified inference
result. Its six short/extended trajectories qualify all 854 AI steps. The cube
240-step replacement control/AI run passes all 225 AI steps. Native 256-step cube
failure remains retained. No threshold changed; no untouched test was consumed.

User questions clarified that 32 network evaluations occur at CPU inference,
not only during training. The same network executes 32 times before advancing
one simulation step. These passes do not disappear in deployment. Keep that
result as an iterative solver diagnostic; it does not establish one-pass
pressure accuracy or a faster solver. Prepared two one-evaluation configurations
and validated their policy contracts; neither was captured/trained. Subsequent
one-pass training, adaptive inference, broader data and deployment are prepared
follow-ups, not completed results. Source 54365af retains the qualified repeated
CPU policy. Final derived summary indexes four immutable trial records and all
11 normal Tuldok stages, excluding retained baseline updates from new totals.
