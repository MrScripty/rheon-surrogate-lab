# Next proposal: qualify optimizer stability before another full coarse fit

The unconstrained spatial coarse candidate did not complete optimization under
the matched recipe. Its last logged completion was 1,300 of 5,000 updates; the
existing finite-gradient guard then stopped the Run. No trained final checkpoint
or candidate accuracy result exists. Leave all eight unused scenes untouched.
No replay or replacement fit has started.

First run one bounded managed CUDA diagnostic with the same initialization,
inputs, stream order, losses and coarse architecture, capped at 1,500 updates.
Its purpose is to locate the numerical failure, not to select a pressure model.
Preserve float32 network tensors, original physical arithmetic, clipping policy
and all checks. Emit bounded update/stream and scalar failure diagnostics into
the existing retained custodian logs; no dataset or large tensor dump is needed.

At the first problematic clipping operation, record:

- The exact update, learning rate and sample identities for every active stream.
- Whether every gradient component is finite; maximum absolute component by
  parameter group, and the ordinary float32 total norm.
- A diagnostic reference norm computed with stable scaling or a wider scalar
  reduction, without changing model tensors or the optimizer's decision.
- Finite output/loss checks and relative coarse/local contribution and residual
  sizes at each of the two training corrections.

The retained error proves a nonfinite total norm, not which tensor caused it.
If all components are finite and only the reduction overflows, qualify a stable
norm calculation at the optimizer owner. An independent scaled 3–4 vector can
exercise a finite known norm where squaring overflows float32. Keep rejection of
actual nonfinite gradients; do not disable error_if_nonfinite or relax physics.

If components themselves become nonfinite, use the captured stage/sample to
test one controlled change to the coarse path's gain or update scale. A small
fixed branch scale or separate coarse learning rate is a hypothesis, not an
assured fix. Record the successful and failed stability screens before choosing
a recipe. Do not infer steady loss growth from sparse logs on different samples.

Only after a reproducible finite short screen should another matched full CUDA
control/candidate comparison start. Preserve or explicitly preregister any
changed optimization schedule; never describe a changed-gain/rate experiment as
the identical earlier recipe. Require exact fresh CPU restore, first-frame
qualification and all 74 development cases with every original population gate.
Keep raw-forward latency separate from paired diagnostic worker cost. Freeze
one eligible package before any untouched-scene evaluation; no automatic promotion.

The existing Tuldok handoff performance lead remains separate application work.
This experiment made no Tuldok or Rheon source changes. Diagnostic logging and
failed-attempt receipts should remain small so they can be reviewed without a
large successful-checkpoint download.
