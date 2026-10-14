# Complete corrective development diagnostic

The fixed final200 GroupNorm candidate did not demonstrate solver usefulness.
All32 raw predictions failed original-RHS residual, predicted divergence and
actual f32 divergence gates. All32 learned PCG warm attempts reached their
32-iteration limit without qualifying. Every policy then qualified through a
fresh previous-pressure reset, with exactly the same final quality/iterations
as the primary previous-pressure-only solve. This candidate is not a qualified
accelerated solver. Representation/loss improvement from training did not
establish native physics improvement.

The one corrective invocation completed with exit0, complete publication and
unchanged source/input/native identities. It used corrected source
`135ae25786f72ff37848b768aafa27d331164596`, source freeze
`ebcbdcd6f5f461fee921694a268d059c8b5bfaa54fd4ac9313e013e4f2d94ffb`,
scientific source `fd3a6009e5805978a29ed1df7389ad226a40a914`, unchanged
weights `4adf55ca…` and cached predictions `bb0290ca…`. It followed explicit
owner approval, both renewed independent reviews, publication of draftPR8 and
three terminal-success CI workflows at head `10c2212`. No optimizer, model
import, new prediction, simulation advance, Tuldok grant/Run or final-array access
occurred.

## Complete outcomes and work

| Method | Qualified /32 | Known completed iterations | Warm iterations wasted | Fresh resets |
| --- | ---: | ---: | ---: | ---: |
| GN warm32 + previous-pressure reset | 32 | 6,250 | 1,024 | 32 |
| Full previous-pressure-only PCG | 32 | 5,226 | 0 | 0 |
| Short previous-pressure32 + reset diagnostic | 32 | 6,250 | 1,024 | 32 |
| Full zero-start PCG | 32 | 5,393 | 0 | 0 |

There are no unknown completed-iteration counts. GN performs1,024 additional
iterations relative to the primary baseline, about19.6% more work, and still
uses the full same-baseline reset. This is an iteration-work comparison, not an
end-to-end timing ratio. Native wire method names retain the historical
`learned_jacobi`/`temporal_jacobi` keys; the actual qualified native algorithm is
Jacobi-preconditioned conjugate gradient.

Train24/24 and validation8/8 cases required GN fallback; every raw train and
validation proposal failed the same three gates. All16 frame0 and16 frame23
cases required fallback. Reference fields passed32/32; raw previous-pressure and
raw zero fields each passed0/32. The entire population, percase response,
split/frame/whole-parent summaries and pressure-error observations are retained
in [exact evaluation](corrective-evaluation.json) and [journal](corrective-cases.jsonl).

Across all32 accepted GN/reset results, maximum original-RHS relative residual
is9.95980381547903e-9 (gate1e-8), absolute residual L2
1.2946076357376312e-10m³/s², predicted divergence
1.3681406627978158e-9s⁻¹ and actual f32 MAC divergence maximum
2.8808911645139545e-7s⁻¹ (gate1e-5). Actual divergence RMS maximum is
1.4929094390864966e-8s⁻¹; derived L2 is RMS×64 for4096cells. Maximum absolute
net flux is1.2334493777917122e-19m³/s and Courant0.11476040681203206.
All accepted fields are finite, gauge-fixed and have exactly zero wall-normal
flux. Full previous-pressure results match GN's reset quality on every case.
Raw cached GN pressure versus reference has median relativeL2 error0.4089309811
and maximum0.9045947757; lower pressure approximation error does not override
the failed native gates. All physical gates/budgets are unchanged.

## Actual cost scopes

| Measurement | Corrective32 |
| --- | ---: |
| Workflow through publication, terminal CLI | 6.974795055997674s |
| Workflow before final publication, artifact | 6.9607625949938665s |
| Final report publication interval | 0.006706924003083259s |
| All-method native elapsed sum | 2.090727053s |
| All-method case process-wall sum | 5.215119243039226s |
| Initial/final identity process-wall sum | 0.055295327001658734s |
| GN corrected-path sum, excluding initial raw checks | 0.569754719s |
| Historical cold-restore inference, separately recorded | 0.32481544799520634s |

These measurements overlap; do not add nested native/process/workflow timers.
The workflow starts after interpreter/stdlib startup and includes frozen-source
admission, artifact checks, native calls/drainage, final identity/postchecks and
publication. Terminal stdout delivery is outside its clock. Native corrected-path
timers omit initial raw-check/setup work; standalone candidate enclosing wall
is unavailable. Earlier restore inference is a different historical execution,
with no new inference performed here. No matched cold/resident end-to-end
speedup follows from these scopes.

Corrective known work across all four methods is23,119iterations. The preserved
failed prefix adds644, giving23,763 total diagnostic iterations,32 unique cases
and33 native case observations. The repeated first case agrees exactly in all
scientific fields/quality/outcomes/iteration counts, excluding explicitly named
timers and process metadata. Both timings remain separate observations, with no
favorable selection. Prefix native elapsed0.057277083s and case process wall
0.14573508100147592s add to respective all-episode totals2.148004136s and
5.360854324040702s. Prefix workflow1.4741860560025088s is before failure
publication; its terminal publication time is unavailable and is not zero.
Complete lineage/counts and non-overlapping ledger scopes are in
[cost ledger](corrective-cost-ledger.json).

Corrective declared filesystem reads are224,882,615bytes, including1,616,384
source bytes, below256MiB. New scientific reads223,266,231 plus the failed
prefix111,687,900 total334,954,131 across separately admitted episodes. Earlier
offline admission, independent audits, provisioning and interpreter/bootstrap IO
are separate qualification work, not a measured total; failed-prefix source IO
is unknown. Per-action budgets do not claim to bound combined engineering cost.

## Closure, retained failure and limits

All34 corrective native children (32case +2identity calls) returned0, completely
drained, closed streams, reached ECHILD/group-empty and released ownership.
No truncated streams, cleanup errors or observed escaped descendants occurred.
Maximum observed parent/child RSS is221,609,984bytes, below the observed4GiB
policy; fork-inherited high-water is included. This is not OS memory containment.
Output totals923,583bytes, below32MiB, with no failure/pending/incomplete report.
Identity matched before/after and all declared inputs/sources passed unchanged
postchecks before final publication. [Terminal observation](corrective-terminal-observation.json)
binds exact artifact hashes, command exit and prelaunch CI.

The initial failed prefix remains byte-identical. Its cause was the receiver's
Python3.12 compensated sum versus native serial norm, not serialization, altered
operator or an unexplained physics defect. Corrected admission uses serial norm
with exact equality. Its real raw/warm numerical failures remain model failures;
the adapter correction does not erase them. No subsequent scientific pass is
needed or admitted by the corrective transition.

This is teacher-state development evidence:24 train/eight held-out validation
snapshots from16 whole trajectories, frames0/23. It is not an autonomous rollout,
new held-out geometry/regime, final-suite exposure or engineering-flow result.
Post-PCG pressure/velocity arrays are unavailable under native6; tracer,
momentum/energy conservation, rollout stability and GPU metrics remain pending.
No zero-residual case exists in this population; that edge has construction and
fixture evidence only. Original small-CNN0/16 and global correction0/32/384-row
collapse negatives, all GN checkpoints and old reservations remain unchanged.

The next local-development target is a new manifest/packer/Tuldok bootstrap and
a separately qualified model hypothesis, with independent whole-parent splits,
restore and full fallback cost. [Local handoff](../local-handoff.md) identifies
current clean-start and CUDA gaps. Failed learned warm32 usefulness does not
admit full/GPU training, extra updates, checkpoint selection or opening final
reservations.
