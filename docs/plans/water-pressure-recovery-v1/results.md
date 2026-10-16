# Pressure accuracy recovery

Measurements in progress. No replacement selected or deployed; eight reserved
independent scenes remain unused. [Plan](plan.md), [methods](reports/methods.json),
[execution ledger](execution-ledger.md).

## Saved-input convergence

Eighteen retained sparse states from consumed scenes17,60,69,73; both models see
the exact same RHS, base/history and physical settings. These are local mean-free
same-RHS pressure errors against the independently checked sealed-Neumann oracle,
not pressure differences between independently advanced trajectories.

| Checkpoint | Pressure target | Mean calls | Max calls | Worst observed pressure relative L2 |
|---|---:|---:|---:|---:|
| Harder-data control |1e-6|7.8333|11|5.8327e-7|
| Harder-data control |3e-7|8.8333|12|1.7124e-7|
| Harder-data control |1e-7|9.2778|13|7.8348e-8|
| Prior late |1e-6|9.2222|16|5.5122e-7|
| Prior late |3e-7|9.8889|18|1.7343e-7|
| Prior late |1e-7|11.0556|20|5.4709e-8|

The harder-data checkpoint retains a work advantage on these supplied inputs at
each matched target. Stricter stopping buys local accuracy; autonomous accuracy
and the full comparative work gates are separate pending measurements.

No optimizer or new simulation:36 fresh-process replays,0..32 calls each,45.978s
total command time. Peak individual RSS789880832B. Full curves, source/input/model
identities and per-band spatial signal/error energies are retained under
[frontier](reports/frontier/) and [summary](reports/frontier-summary.json).
Spatial cosine modes and per-state amplitude bins are not temporal frequency
bins. Zero/near-zero signal is reported separately; relative error alone does
not establish an input or arithmetic noise floor.

## Runtime qualification

New project assembly:3.216s smoke,3.266s weight0.1,3.565s weight0.25. All capture
the existing v9 trainer at3a6ec2d. CUDA smoke plus fresh CPU restore completed and
drained in592.978s managed wall time; the actual recipe times and CUDA allocation
are in [smoke receipt](reports/smoke-result.json). The managed wall includes
runtime attestation, input custody, approval and artifact retrieval; it is not
optimizer time. No runtime/portability changes were made.

## Consumed trajectory comparisons

Pending: both stricter inference policies and both gentler-feedback CUDA fits
on all74 consumed scenes, with original58/amended8/recent8 reported separately.
Existing physical limits remain unchanged. No final-informed tuning or relaxed
replacement gate is permitted.

### Stricter stopping, same harder-data checkpoint

All148 AI trajectories and74 native controls qualified in2185.8s concurrent correctness wall. No policy is eligible under the frozen combined accuracy/work gates.

| Population | Target | Mean calls | Max calls | Worst paired trajectory pressure relative L2 |
|---|---:|---:|---:|---:|
|original58|late-pressure|7.9373|12|3.1447947e-06|
|original58|tight-3e-7|8.5047|13|3.2255102e-06|
|original58|tight-1e-7|9.1844|14|3.9575718e-06|
|late-amended8|late-pressure|9.5619|12|1.3489549e-06|
|late-amended8|tight-3e-7|9.7378|12|1.0261382e-06|
|late-amended8|tight-1e-7|10.5420|13|1.3850956e-06|
|minimum8-failed8|late-pressure|9.0796|17|1.4151669e-06|
|minimum8-failed8|tight-3e-7|8.9900|12|9.5384598e-07|
|minimum8-failed8|tight-1e-7|9.6538|13|1.5042394e-06|

Target3e-7 recovers pressure accuracy on the two newer populations, with extra work. Original58 still narrowly regresses against prior late pressure accuracy, and mean/maximum work gates also prevent replacement. Target1e-7 is not monotonically better on paired autonomous trajectories. Same-input convergence alone would not reveal that. Full per-step data, sparse oracle checks and failed comparative gates are in [policy reports](reports/policy/).
