# CPU Adam state and step-size counterfactuals

Authorized next step from the C69 optimizer replay. Keep C53 deployed, float32
network parameters and the 1e-6 physical target unchanged. No CUDA training,
new simulation data, holdouts, deployed model or historical-result edits.

Use only the five published pre-states (updates 1/10/100/150/200), moments and
clipped CUDA gradients from the completed replay. Verify source/trace/custody
hashes against its managed receipts. Compare observed endpoint, preserved-state
CPU Adam endpoint and reset-moments/reset-step CPU Adam endpoint at fractions
0, 1/8, 1/4, 1/2, 1. Reset uses the same saved gradient and LR, not a recomputed
CPU gradient. These are independent one-step counterfactuals, not a new training
trajectory. Fractions interpolate rounded parameter endpoints; they are not an
exact LR change or retrained optimizer history.

Require all preserved endpoints to reproduce the observed GPU step within the
previous 2e-7 maximum absolute parameter tolerance. At update 1 the fresh/reset
and preserved endpoints must be bitwise equal. Qualify endpoint interpolation
with a literal vector and Adam with an independent scalar recurrence. Evaluate
all eight cases' unchanged loss and 1/2/4/8-pass pressure residual using the
captured metric owner; no gradients or optimizer updates on the pressure network.

Evaluate each distinct parameter vector once; aliases retain policy/fraction
identity and point to that exact result. Zero fractions must be exact pre-state,
observed fraction 1 exact post-state. Retain candidate vector hashes, not arrays.
Calibrate CPU/GPU metric agreement against observed pre/post values, retaining
all gaps. An empirical envelope is twice the maximum observed pre/post absolute
gap for that case/metric. It is a directional caution, not a mathematical bound
or a relaxed physics target. Report strict guards separately: no case worse at
1/2 passes, no previously qualifying 4/8-pass case lost, weak-case loss and
four-pass residual improved, and worst four-pass residual lower. Require weak
improvement exceeding the empirical envelopes before recommending an arm. Never
hide small regressions behind calibration tolerances.

Run at most three independent one-thread/core workers, queueing five snapshots.
Admit only with 12 GiB host reserve plus 2 GiB per active worker available; monitor
host reserve and each worker RSS, with a global 120-second wall deadline. On
failure/cancellation terminate all owned groups, share a ten-second grace period,
kill remaining children and reap every process. Record commands, runtime/source
hashes, concurrency, timing, memory and failures. No model weights/data enter Git.
Publish activity to the existing Tuldok-linked workflow record and link the
completed offline report from the cohort page. No separate catalog is created.

Select a subsequent proposal from measured tradeoffs. CPU success alone does not
qualify CUDA training; do not automatically train or combine moment, loss,
capping and schedule changes. Preserve historical failures and existing dirty
chart/THIRD_PARTY work. Private experiment branch based on be22323; retain its
source/result proposal for lab-user review until integration/explicit retirement,
with append-only synchronization and no automatic push/merge/history rewrite.
