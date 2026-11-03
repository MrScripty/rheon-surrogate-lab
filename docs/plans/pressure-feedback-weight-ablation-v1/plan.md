# Smaller pressure-feedback weights

Status: Accepted. Operation: complete. User authority: “continue with the next proposal”,
referring to weights0.01and0.03 in the completed preceding results. Canonical
plan: docs/plans/pressure-feedback-weight-ablation-v1/plan.md. Owner/integration
owner: root. Branch experiment/pressure-feedback-weight-ablation-20261017, based
on a4ae4e9cdff97f8b0e0f37472677dc96c8915dc1. Private local experiment branch
retained for review/subsequent integration; no merge, rewrite or cleanup.

Objective: measure whether smaller low-frequency feedback weights preserve the
call improvements without the rejected0.1candidate's worst-pressure regressions.
Reuse the established trainer, current catalog and lifecycle owners. The matched
control and0.1candidate remain immutable historical evidence. Refresh one
control frame/all-array equivalence before explicitly labelled reuse of its
74-case campaign; no full control campaign is repeated.

Current slice S1: freeze two5000-update CUDA Studies with weights0.01/0.03,
identical actual initialization/input/normalization/runtime/sample order, schedule
and scientific source. Change only feedback weight scientifically. CUDA training,
fresh CPU restores, float32network, cubic spacing, sealed water physics,15native
warm-up,32maximum network calls and physical/stopping gates remain fixed.
The existing model-only1800s CUDA deadline and CPU600s deadline remain unchanged.
No runtime prototype, network architecture or new training data is admitted.

S2 depends on S1 artifacts: package each fresh CPU result; qualify one frame
before its full74development episodes. One bounded16-worker one-thread CPU pool
may overlap the other CUDA fit, excluding desktop/GPU/restore cores,2GiBaccounting
reservation per worker plus8GiBavailable reserve. Do not run two16-worker pools.
Classify/reap infrastructure failures and halt rather than treating them as model
failures. Retain physical rejections explicitly.

S3 depends on complete campaigns: fixed-call manufactured diagnostics and
per-population gates against reference-current, prior accurate late model and
harder-data control. A candidate must meet every unchanged accuracy and mean/max
call criterion in all three populations before eligibility. Freeze an eligible
package before consuming the unused eight whole scenes once. If both qualify,
select the smallest worst normalized pressure ratio across the three populations
(candidate worst error / smallest retained comparator worst error per population);
then smaller overall maximum calls, then mean calls, then lower weight. This rule
is fixed before candidate CPU regression results. Otherwise leave those scenes
untouched. No automatic promotion. Serial CPU timing follows broad
qualification drainage, same consumed grids and exact repeated physics/calls.

Acceptance is completion and honest measurement, not guaranteed improvement.
Claims: frozen matched materials (artifact/hash evidence); actual CUDA5000updates
and fresh CPU parity (current catalog, environment-qualified); first-frame and
all74trajectories/oracles with child drainage (CPU executable evidence); unchanged
population decision/conditional holdout disposition (independent analysis);
recorded clocks/memory/failures and original Project/Rheon source observation.
Frozen-material, both CUDA/CPU parity and both first-frame claims satisfied.
Both candidates completed all74physical trajectories and233independent checks
each, with all child processes drained, but failed unchanged population promotion
gates. All six serial timing batches completed, with exact repeated physical outputs
and call sequences. Measurement claims satisfied; independent final review and
record closure complete this slice. Both candidates are ineligible; the eight
unused whole scenes remain untouched. No promotion or subsequent fit.

Write set: this plan directory; two new config files in the existing experiment;
existing observer only to select the two actual Study labels; compare_training.py
and its focused test for reusable exact artifact/matching evidence; trace_response.py
for post hoc analysis of already recorded consumed-case corrections. The latter
advances no simulation, optimizer or holdout and changes no scientific inputs. Reuse all other
experiment owners; adjust generic reporting only if measurement requires it.
Raw artifacts live under runs/pressure-feedback-weight-ablation-v1, outside Git.
No weights, datasets, credentials, previous plans/results, Tuldok/Rheon source
or unrelated root changes are included.

Standards MCP snapshot:v1:2e64b09d-e0ad-4f7f-a02f-0b0a61be561e. Independent
standards/spec review covers the bounded source/configuration and result changes.
Replan on altered scientific/custody semantics, runtime mismatch, insufficient
resources, uncertain mutations or invalid control equivalence; never auto-replay
admission or resume an unknown execution. Retain branches/checkout and catalog.
