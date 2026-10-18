# Issues

No open blockers. A negative loss-gradient cosine is a local first-order
observation; Adam moments, finite step sizes and the changing trajectory are not
measured by it. The same C66 objective is required for both checkpoint states;
using C53's historical training-loss implementation would confound the comparison.
Retained C66 is a one-seed control. The new schedule changes exposure at each rate,
not the total example count. These are manufactured fitting cases, not new CFD
simulations or a deployment qualification.

The first two CPU diagnostics reached the vector-bank guard before backward:
storing eight double vectors would use about 22.6 MiB. The implementation now
retains native float32 gradients (~11.3 MiB), with bounded float64 temporaries
for dot products. The 16 MiB stored-vector cap is unchanged. Both original logs
are retained; no optimizer or CUDA job ran in those failed preflights.

The first C67 match check rejected the expected captured config.json hash change.
That file is now compared semantically with only schedule and experiment ID
excluded; all numerical source files remain byte-identical. No qualification
was published and no managed mutation occurred in that failed preflight.


Sequence closure: C67's restored CPU four-pass and early-pass gates failed.
This is a retained scientific result, not an execution blocker. Conditional C68
was skipped and its preparation guard was exercised against the actual failed
decision. Both managed Runs completed/drained, all published artifacts verified,
and every owned controller, monitor, CPU worker and headless browser exited and
was reaped. No unresolved custody, resource-cap violation or recovery remains.
