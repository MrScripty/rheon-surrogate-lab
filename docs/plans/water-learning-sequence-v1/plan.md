# Water pressure learning sequence

Status: Completed experiments; model accuracy remains unqualified. Canonical operation: execute the user's approved sequence
and retain measured results. Serial owner: root. Lab branch
experiment/water-learning-sequence-v1 from a842606. No push or merge. Historical
experiments, accepted Rheon comparison branch and dirty original checkouts remain
outside the write set. Authority: user explicitly approved the proposed sequence.

Write set: experiments/water-learning-sequence-v1, this plan/results/reports,
root README and tuldok-project pointers. Reuse frozen seven-channel float32 model,
unchanged loss, qualified installed CUDA runtime, native water capture generator
and water-material independent pressure evaluator. No architecture, precision,
physical-gate or Tuldok runtime changes. Capture copied source explicitly; never
replace historical hashes or results. Source commits precede managed execution.

Question: can more optimization accurately fit known data, and does increasing
independent physical simulations improve generalization at matched compute?

Sequence (all optimizer work CUDA, all deployment checks CPU):
1. Fit one existing 16-cubed water training snapshot, native index 15. One train
   parent/frame only; original 64 validation parents remain held out. Batch 1,
   10,000 updates, maximum 600 seconds. Record fitted-case physical qualification.
2. Train the unchanged model longer on all original 256 training simulations and
   both saved states. Batch 2, 10,000 updates, maximum 600 seconds.
3. Matched data-size comparison: 256 versus 1,024 training parents, one saved
   state each, alternating native indices 15 and 63 by parent ordinal. Shared
   validation: original 64 parents, both states. Same initialization/sampling seeds,
   optimizer/schedule, two-sample updates and 10,000-update/600-second budgets.
   This is a separate paired control; it is not compared to step 2 as data-only.
4. Choose among long-256 and both matched data-size candidates using validation only, freeze the
   declared final checkpoint, then evaluate on 64 fresh reserved simulations.
   The historical 64 test parents are not reused as a fresh test set.

All optimizer schedules: Adam 1e-3, 1e-4 at update 1,000, 1e-5 at 6,000. Save states
0, 2,000, 10,000; report progress every 100 updates. Final reached declared state
is the candidate, not a lucky intermediate checkpoint. One sampling/init seed per
trial is a bounded first comparison, not a statistical confidence claim. A budget
or accuracy failure is recorded honestly. Pipeline/setup failures must be repaired
before dependent execution; a one-case learning failure does not authorize model
changes and remains evidence when running the already-approved bounded controls.

Dataset design: nest the original 256 training parents within the 1,024 cohort;
add three independent 256-training-parent native reservations (seeds 2026101601,
2026101602, 2026101603). Old generator also creates auxiliary validation/test
parents; those are excluded except the third cohort's predeclared 64 fresh test
parents. Preserve full original captures and physics qualification. Qualify a
one-frame fixture and four-scene pilot for each new reservation before full capture.
No cold-start AI testing: retained states already include preceding solved pressure.
Filled sealed water, cubic spacing and proportions/forcing distribution stay fixed.

Discovery decision: an uncompressed 1,024-parent/two-snapshot training packet is
larger than the managed runtime's 64 MiB total capture contract. Byte compression
on the existing packet saved only 15%, insufficient. The matched scaling pair
therefore uses one snapshot per parent, with bounded two-part lossless packets;
all captures retain two states. Do not silently increase generic runtime bounds or
reduce voxel precision. Runtime preflight must prove complete captures including
restore inputs fit. Keep numerical parent/frame provenance and source hashes.

Composition: native generator owns physical scenes; new lossless training views
own selected parents/frames and namespacing across measured captures; the copied
versioned trainer owns sampling and optimization; Tuldok owns material, runtime,
Run/artifact custody and teardown; CPU inference/native oracle own measured gates.
The new view decoder validates closed inventories, source reservations, cubic
spacing, exact numeric hashes, split separation and whole source frame selection.
Never mix reserved arrays into normalization/optimization or selection metrics.

Evidence: fixture/pilot/corpus reports, exact view round trips and negative boundary
regressions, CUDA parameter/batch/gradient/Adam evidence, train/validation history,
independent fitted/validation pressure gates, managed CPU restore parity, fresh
reserved native evaluation for the selected final candidate, catalog IDs/artifact
hashes and a durable result table explaining limits. Publish real activity through
the existing Tuldok project and native Rheon observer; collection records use the
existing catalog. No private catalog or unrecorded local optimizer.

Resources: four sequential CUDA trials, each at most 600 seconds; observed earlier
2,000-update baseline took 81 seconds. One native capture worker, existing cached
build target, maximum 1 GiB additional experiment data with at least 1 GiB reserve.
Current repo filesystem has 5.2 GiB free; /tmp has 50 GiB if disposable build work
requires it. Retain all scientific outputs. No destructive cleanup.

Standards: Coding Standards MCP route selected implementation, planning/discovery,
verification/oracles, commit, code design, contracts and persistence/interop.
Snapshot snapshot:v1:342ae35f-bc16-43e7-84d4-9527626d95e2. Apply exact staged reviews,
closed measured inputs, normal Run custody and local conventional atomic commits.
Exactly one next slice: qualify versioned training views and the one-case managed
CUDA/CPU run, while independently generating the newly reserved native captures.

Candidate ordering is predeclared before new optimizer execution: most qualified
validation proposals, then lower median equation residual, then lower worst
residual; exact ties use long-256, scale-256, scale-1024 in that fixed order.
The one-case model is a diagnostic, not a generalization candidate. A two-update
managed smoke (separate initialization/Run, not a scientific candidate) qualifies
the new selected-frame packet and CUDA-to-CPU recipe/custody path first.

The schedule retains the historical 1e-4 transition at update 1,000, then adds
1e-5 at 6,000. Thus long-256 preserves the original first 2,000-update optimization
prefix. Compare its saved update-2,000 tensor inventory to the frozen baseline;
any mismatch must be reported before claiming an update-only continuation.

User viewer correction (while training proceeds): current observer cycles between
sparse saved states from distinct domains, hiding each complete simulation. The
user also requests native translucent voxels and a force widget with geometry,
location, magnitude and swirl/fluctuation parameters. Add a separate observation
slice: probes/water-preview-v1 and preview support here, plus isolated Rheon branch
feat/live-training-force-preview in runs/water-learning-sequence-v1/Rheon from
66e5038. Do not alter frozen captures, training views or in-flight managed sources.
Replay 64 native steps for representative training scenes, emit all states without
changing numerics, and require byte-exact parity with original retained fields.
Use one pinned simulation timeline with playback, not mixed-domain publications.
Share Rheon's native voxel geometry/color/mesh path, preserving ordinary viewport
behavior; add bounded opacity and force overlays and exact parameter labels.
The old serving viewer's dirty source remains untouched. Native numerical tracer
is not static: original train-0000 changes in all 4,096 cells between retained
states, maximum difference 0.04560749; this is evidence about arrays, not proof the
current renderer shows transport clearly. The new view defaults to water velocity
voxels, with passive tracer and pressure as separate inspection fields.


Completion evidence: all four 10,000-update CUDA trials and fresh CPU checks
completed. The matched 1,024-parent candidate reduced median validation residual
15.5% and worsened the worst residual 9.9%; it qualified 0/128 fresh reserved
proposals. The one-case fit also failed its pressure gate. See results.md and
acceptance.json; no threshold, model architecture or loss was changed. All
historical results remain. Original first-2,000-update tensor inventory parity is
exact. All 13 scientific managed stages and the two smoke stages completed/drained.

Observation accepted separately: native transport's analytical one-cell test
passes on all axes; the first one-second scene moves only 0.227 voxel. During
optimizer updates the observer shows saved inputs rather than advancing fluid.
The corrected viewer pins one physical-time trajectory; longer native-only
continuations show 2.056 and 4.244 voxel travel and preserve every original saved
field exactly. See observation.md and its measured reports. Semi-Lagrangian
tracer diffusion remains explicitly identified. New Rheon commit fe9d586 on
feat/live-training-force-preview is retained/protected for the active viewer and
Tuldok source references, owned by this task for Jeremy; retire its clean
worktree after accepted integration and consumer migration. No push/merge or
cleanup of other worktrees occurred.

All 768 new training parents and two full preview sequences are in the original
Tuldok catalog; auxiliary validation/reserved fields were excluded from that
publication. Project snapshot 179bf8ae04184187b1e5d1895a4faf23 carries the current
source, activity and result references while earlier snapshots remain intact.
Local experiment artifacts occupy under 1 GiB; the filesystem retains over
1 GiB free. The next scientific question is the one-case fitting/conditioning
gap, before treating more data as sufficient for a qualified pressure solve.
