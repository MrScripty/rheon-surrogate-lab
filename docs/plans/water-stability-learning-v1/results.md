# Stability-aware training results

The training change substantially improves accuracy and removes the measured
short-horizon instability. A separately tested **iterative inference policy**
passes the development physical checks. It evaluates the same network **32 times
per simulation step on CPU**. This is not training-only feedback, a one-pass
pressure model, or a qualified production deployment.

## Controlled training comparison

Each full fit uses 5,000 CUDA updates, the same initial parameter bits and real
training-frame sampling, one actual native frame, float32 parameters/gradients/
Adam moments, no AMP/TF32, four recurrent network evaluations per proposal.
Mode-feedback fits add 5,000 analytically labelled operator examples, sampled
from 32 patterns at four signed amplitudes. They are not additional independent
fluid simulations. All 128 evaluation states remain development evidence.

| Training intervention | Fitted equation residual | Development median | Development worst | Physical passes, four network evaluations | 64-step trajectories |
| --- | ---: | ---: | ---: | ---: | --- |
| Retained recurrent control | 3.49860e-4 | 1.43348e-2 | 2.21805e-1 | 0/128 | Hard stops after 4–6 AI steps |
| Analytical mode feedback | 4.36320e-4 | 7.47636e-4 | 3.28209e-2 | 0/128 | All three complete; 0/147 AI steps qualify |
| Correction-RMS output scale | 3.22263e-4 | 1.35599e-2 | 1.46414e-1 | 0/128 | Hard stops after 12–14 AI steps |
| Feedback plus output scale | 3.56120e-4 | 7.41674e-4 | 6.12879e-3 | 0/128 | All three complete; 0/147 AI steps qualify |

The combined fit lowers the matched control's median by 94.8% and worst error by
97.2%; its fitted-frame error is 1.8% worse. Mode feedback is the important
stability intervention. All 24 single modes and 72 independently seeded mixtures
reduce error for both feedback models. Output scaling alone reduces first-proposal
error on all probes but still amplifies some mixtures under repetition and fails
actual trajectories. A first-proposal reduction does not establish convergence.

## Repeated inference: separate policy and compute cost

The mode-feedback checkpoint is retained in Tuldok Run
`96011191e0514ec8a61bafa7abdd4254`, weight SHA256
`7ca7a02e2454d6b79ff6f68fe27118991d9fbbeff1e768aae943ee0b0e2d3179`.
Fresh managed CPU restore and validation Runs are linked in its immutable report.
No weights are committed to Git.

Eight proposals each contain four network evaluations: **8 × 4 = 32 evaluations
of the same network for one simulation step**. All inference runs on CPU; weights
stay fixed. Residuals are recomputed between corrections. The existing float64
physical operator and pressure-base accumulation remain unchanged; neural
parameters and outputs stay float32. No native pressure refinement contributes.

This policy passes the fitted frame and **128/128 development cases** under the
unchanged float32 profile. Median equation residual is 1.44317e-12; worst is
5.82311e-8. Also 121/128 pass the stricter original native 1e-8 profile. The native
evaluator's separately reported algorithmic benchmark refinements do not enter
the learned candidate's predictions or its float32 qualification.

| Independent simulation | Total steps | Native warm-up | Qualified AI steps | Hard stop |
| --- | ---: | ---: | ---: | --- |
| Cube, rectangular, compact; retained short runs | 64 each | 15 each | 49/49 each | None |
| Cube; extended | 240 | 15 | 225/225 | None |
| Rectangular; extended | 256 | 15 | 241/241 | None |
| Compact; extended | 256 | 15 | 241/241 | None |

All **854 tested AI steps** qualify. Extended windows cover 4.0–4.267 seconds total,
3.75–4.017 seconds after warm-up. Final pressure-field relative L2 differences
from native are 1.15e-7, 2.60e-7 and 2.94e-7 for the extended cube, rectangular
and compact cases. These are finite development windows, not a global stability
or arbitrary-resolution accuracy guarantee.

The cube's attempted 256-step **native control** fails at step242 on Courant
0.502272 > 0.5, with pressure/divergence still within their limits. The failed
control and all earlier 241 qualified frames are retained. A fresh native control
qualifies the 240-step cube window; no force, timestep or threshold was changed.

Accuracy costs inference time. Observed extended-run mean AI track times are
0.201 s (cube), 0.205 s (rectangular), 0.0884 s (compact), versus native track means
0.0149, 0.0190, 0.00345 s. These include physical preparation, projection and the
IPC/model callback, under concurrent development workloads; they are not isolated
kernel benchmarks. They show no speed advantage. Reduce inference work before
considering practical deployment.

## Completion and retained evidence

Three full fits plus the two-update smoke completed **15,002 actual CUDA updates**
in **11 completed/drained managed stages**, with fresh CPU restores, zero CPU
optimization/CUDA initialization, reopened Runs and whole-artifact redownload
parity. Training interval rates were 12.805 (feedback), 27.151 (scale), 11.684
(combined) updates/s; auxiliary fits perform twice the network work per update.
Runtime/preflight/queue time is separate. Independent CPU audits used 23 worker
cores with one Torch thread each; distinct-core extended simulations ran in
parallel. All native children were reaped. Retained local artifacts are about
259 MiB of the 2 GiB budget; datasets/checkpoints remain outside Git.

Eighteen focused model/analytical checks, two two-cell feedback/budget checks,
two real process-cleanup checks, two real-source negative receipt checks, and
immutable report CLI regression passed. New scientific results did not change
physical gates or consume an untouched final test. No Rheon/Tuldok source or
deployed UI model changed.

See [methods](reports/commands.md), [comparison CSV](reports/comparison.csv),
[derived summary](reports/summary.json), [iteration evidence](reports/iteration-policy-evidence.json),
and the immutable [feedback](reports/stability-modes.json),
[scale](reports/stability-scale.json), [combined](reports/stability-combined.json)
and [smoke](reports/stability-smoke.json) reports. Their source/capture/normalization/
sample/checkpoint hashes, Run/catalog IDs, actual traces and lifecycle receipts
preserve both successful and unsuccessful outcomes.

The user's inference questions exposed an important distinction: the passing
32-evaluation policy is iterative at deployment, not extra training corrections
that disappear afterwards. **One-network-evaluation accuracy is not established.**
Prepared single-pass configurations have not been trained. The proposed next
tests and deferred broader cohort are in [next steps](next-steps.md).
