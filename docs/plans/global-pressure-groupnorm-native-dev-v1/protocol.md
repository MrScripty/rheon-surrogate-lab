# Fixed final200 native development protocol

**Status:** proposed protocol; planning review only. Scientific execution is pending separate authorization.

This protocol owns the proposed 32-case teacher-state projection comparison. The [plan](plan.md) owns admission, sequencing and the next slice. This document authorizes no implementation, model import/forward, native solve, additional training, access/session, Graph, preflight, grant or Run. The planning branch is stacked on the accepted PR6 source; PR6 promotion/merge remains a separate parent-owned boundary.

## Question and evidence boundary

Does the single frozen GroupNorm final200 pressure correction help a native Jacobi PCG projection on the retained development snapshots when compared with one full-budget solve from previous accepted pressure? Measure proposal validity, complete correction/fallback work and descriptive cost. The final200 candidate is fixed before access; intermediate states remain observations and cannot be selected, resumed or substituted.

Every method uses the same original preprojection MAC velocity, geometry, density, timestep and original RHS within a case. Previous pressure is the previous **reference** accepted pressure from the captured teacher history. No method feeds its result into the next case; no source state advances. This is one-step development evidence, including training states and observational validation states. It proves neither autonomous stability, final-suite generalization, tracer transport conservation nor operational acceleration.

The prior global model passed 0/32 raw projections; all 32 short warm attempts failed and then qualified by resetting from previous pressure. Its 1,024 wasted warm iterations and 5,226 reset iterations remain immutable negative evidence. [Prior physics report](../global-pressure-training-v1/reports/physics.md) records descriptive learned-plus-inference cost above the fair temporal baseline. A successful GN fallback must not be reported as a successful learned proposal or as acceleration.

## Frozen candidate and input custody

The future adapter must admit the actual GN restore closure, preserving its originating train closure:

| Item | Exact identity |
| --- | --- |
| Scientific source commit | `fd3a6009e5805978a29ed1df7389ad226a40a914` |
| Originating training Run | `d73bc2976c4f442f8aefc8f7242bd252` |
| Completed cold restore Run | `265c9574c06048d8bc91daa1cf8eab02` |
| Final weights and checkpoint200 SHA-256 | `4adf55ca940c2013b6352edc4a6716492e0a697484331c65cdf80495312db147` |
| Predictions archive SHA-256 | `bb0290ca1855f841d3fe99260790d9cfdd39e8b7982e054ea01791b3a848ee86` |
| Training receipt SHA-256 | `88abf1b2eaff46da583588c422166f0db4e63eb8aaa7790ed585f88d1a5af1ed` |
| Restore receipt SHA-256 | `f83242bf89a3db7da305781c5a880c7b08d4e7a0090e8469276b67ee88cc0a0a` |
| Original capture manifest SHA-256 | `dee7811a95b85a4c9fd2a228163ab39288d85bdeee804d09804c24eb7c92663d` |
| Retained splits SHA-256 | `fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a` |
| Retained train-only normalization SHA-256 | `8a68445ce25b46645b50939b13f67b974d60f1c44d8e69ce0a078f24dc81458f` |

[Training](../global-pressure-groupnorm-v1/reports/training.json) and [restore](../global-pressure-groupnorm-v1/reports/restore.json) retain full source/config/runtime/data/custody identities. The actual restore has zero new updates, 200 trained updates and source-enforced closed-key/dtype/shape/raw-byte prediction parity. The read-only planning inspection verified the local retained weights, prediction and receipt bytes against these identities. No new inference was performed.

The predictions archive is 1,049,356 bytes, containing only C-order `pressure: <f8[32,16,16,16]` and `row_indices: <i8[32]`. Pressure is the full physical candidate `p_base + gauge-fixed correction`, not a residual, normalized correction or solution to a shifted RHS. Future admission must validate bounded regular files, exact bytes, archive/header/dtype/shape rules before allocation, finite pressure and fixed row correspondence. It must check exact receipt schema `rheon-global-pressure-groupnorm-training-receipt-v1`, complete restore mode, exact typed 0/200 update counts, artifact inventory/raw hashes, source/config/normalization/splits/origin relationships and originating Run/custody. Unknown schemas, bool/float update-count substitutions, duplicate/nonfinite JSON, unexpected archive entries or corrupt source/data identity fail before numerical execution. They never trigger numerical fallback.

Use the existing lossless packet2/original-capture lineage; do not generate, relabel, crop or augment data. Source pressure, RHS and pre-MAC arrays must be checked against the original manifest before use and again before successful publication. The packed native residual is derived input provenance, not a substitute for rebuilding original RHS from the captured preprojection velocity. Original final reservations and previous failed artifacts remain unchanged. Absolute historical paths are opaque provenance unless explicitly admitted as verified local inputs.

## Fixed population and ordering

Use exactly the existing `case_population()` contract in [evaluate_native.py](../../../experiments/global-pressure-v1/training/evaluate_native.py): 12 train trajectories `smoke-train-100000` through `smoke-train-100011`, followed by four validation trajectories `smoke-validation-100100` through `smoke-validation-100103`. For each trajectory take frame 0 then frame 23, in that order. Case IDs use `trajectory_id:frame-000` and `trajectory_id:frame-023`. With trajectory ordinal `t=0..15`, original row indices are `24*t` and `24*t+23`; first row is 0, last is 383.

All 32 cases are declared before evaluation. Each case reads current reference pressure, original RHS and `pre_u/pre_v/pre_w`; `p_base` is initial pressure at frame 0 or frame22 accepted reference pressure at frame23. Exact geometry and timestep come from the original trajectory/reservation entries: 16³ cells, positive physical spacing, density 1.25 kg/m³, timestep 1/60 s. Native RHS reconstruction must equal original captured RHS. Report train and validation separately and report frame0/frame23 separately. Frame0 temporal and zero starts may coincide; retain their equality explicitly, without counting them as independent baseline evidence. Two frames from one trajectory are paired diagnostics, not independent trajectories.

## Source-mapped policies

The preserved [native evaluator](../../../probes/global-pressure-v1-training/src/main.rs) exposes the following policies. Its `temporal_full_budget` is the required previous-pressure-only numerical warmstart and the existing full-budget temporal baseline: these are **one identical comparator**, not two independent methods. Raw `p_base` is the same pressure before solving and is a separate zero-PCG diagnostic.

| Fixed method/report field | Start and work | Role |
| --- | --- | --- |
| `reference` | Captured current reference pressure; common physics checks | Required case qualification; no new solve |
| `raw_learned` | Frozen GN final200 full pressure; common checks | Proposal diagnostic before PCG |
| `raw_temporal` | Captured `p_base`; common checks | Existing raw baseline; no PCG |
| `raw_zero` | Zero pressure; common checks | Unchanged raw zero context |
| `learned_jacobi` | Eligible GN candidate, at most 32 warm iterations, then at most one fresh `p_base` reset with limit 3000 | Candidate operational policy |
| `temporal_full_budget` | One fresh Jacobi PCG solve from `p_base`, limit 3000, no preliminary 32 attempt | Primary fair previous-pressure-only comparator; existing temporal baseline |
| `temporal_jacobi` | `p_base`, at most 32 warm iterations, then at most one fresh `p_base` reset with limit 3000 | Fixed short-budget/fallback diagnostic; never substitute for primary comparator |
| `zero_jacobi` | One fresh zero-start Jacobi PCG solve, limit 3000 | Unchanged native zero-start comparator |

Each attempt owns fresh scratch; attempts never reuse Krylov directions, preconditioned residuals, failed pressure or partial corrected velocity. All starts use identical original-b stopping scales and Jacobi preconditioning. Record the actual initial-guess hash and iteration limit. Reference failure, reconstructed-RHS mismatch or authoritative input corruption stops qualification; no case is dropped or replaced. Numerical proposal rejection and solver failures retain their distinct outcomes and all work.

The existing native evaluator does not implement analytic/cosine starts or learned SGS in this action. Preserve their historical policies/evidence without inventing a comparator or expanding this slice. The original correction design's broader baseline/rollout suites remain pending separate admission. No original failed model is rerun or selected here; its retained results provide historical context, not a fresh paired benchmark.

## Common projection gates

These are the unchanged [accepted projection criteria](../global-pressure-correction-v1/acceptance.md). Let `V=h_x*h_y*h_z`, `r=b-Ap` with native **full ungauged operator action including cell0**, and `d` be independently recomputed divergence from the actually corrected stored f32 MAC faces. Norms/sums are f64 over the full field in native order. The residual sign does not affect these norms; do not eliminate gauge row or use normalized/preconditioned residual for acceptance.

| Gate | Required bound |
| --- | --- |
| Pressure and face contract | Exact native shapes; all finite; pressure gauge `p[0]=0`; closed walls |
| Full original-b residual | `||r||2 <= max(1e-12, 1e-8*||b||2)` |
| Predicted divergence | `dt*||r||inf/V <= 1e-7 s^-1` |
| Actual stored-f32 divergence | `||d||inf <= 1e-5 s^-1`; also report absolute RMS/L2 |
| Wall-normal volume flux | Every stored f32 wall-normal face is exactly zero |
| Recomputed net volume flux | `abs(V*sum(d)) <= max(1e-12 m³/s, 1e-8*V*sum(abs(d)))` |
| Courant | `sum_axes(dt*maxabs(corrected_face_axis)/h_axis) <=0.5` |

Every accepted raw/corrected result must pass all gates, independently of solver status. A solver error cannot be converted into success merely because diagnostic scratch happens to pass checks. Report absolute residual even when `||b||2=0`; relative residual is unavailable in that case. Never rescale tolerance by the proposal's smaller initial residual. Correction must use native velocity correction into disposable f32 faces, followed by fresh physics checks. Preserve captured teacher state throughout; no commit/advance occurs even for qualified results.

Volume-flux conservation is the projection metric. This action performs no tracer advection and establishes no tracer-mass, momentum or energy conservation claim. The original autonomous tracer/error/drift bounds, forced dissipative-flow interpretation and engineering-flow limitations remain unchanged and untested by these snapshots.

## Proposal and fallback state machine

Authoritative cached-artifact admission is distinct from numerical proposal handling. A bound archive/header/hash/shape/dtype violation, nonfinite stored pressure or nonfinite/duplicate JSON fails before native execution. The frozen actual archive is finite. Such corruption is never converted into `candidate_error` to obtain a fallback. For an admitted numerical request, an invalid gauge, native nonfinite operator quality or an explicitly supported numerical `candidate_error` rejects the proposal and records its cause. Isolated fixtures may exercise malformed/nonfinite numerical outputs without weakening actual cached-artifact admission; they do not represent an observed failure of these frozen artifacts.

For an admitted candidate, compute complete `r_base` and `r_guess`. Reject a finite proposal whose residual L2 exceeds base residual L2. If both residual norms are zero, retain `p_base` as the eligible pressure and record base retention; do not accept a different GN pressure or run a warm attempt from it. Accept an eligible pressure directly only after all f32 correction/common gates pass; an eligible but unqualified pressure receives one warm attempt of at most 32 iterations (from `p_base` in that zero-residual case). Any warm solver error or failed common gate discards its scratch and invokes exactly one fresh full-budget reset from original `p_base`, at most 3000 iterations. Rejected proposals go directly to that reset. No repeated inference, recursive retry, tolerance relaxation or use of zero pressure as reset is allowed.

A failed full reset is a terminal numerical failure. Retain its case, cause, attempted/completed iteration information, cost and partial diagnostic evidence; do not advance state or label the full 32 qualification successful. Bounds/resource/custody failures terminate the evaluation with an incomplete/failed report, not a numerical rescue. Later cases may be observed only under a separately specified failure-collection policy that cannot turn the failed batch into success; this protocol selects fail-stop after durable failure recording.

**Zero-residual conformance obligation:** existing native `corrected()` has no explicit both-residuals-zero/base-retention branch. Preserve the qualified old source and reports. A minimal new adapter/composition may satisfy the obligation by construction without changing or rebuilding the old native evaluator: the frozen GN source bypasses its correction when its native-input residual is zero, returning `p_base`. The source-mapped proof must bind packet2's native-residual attestation to the identical full operator/base/RHS action, geometry, density and original raw lineage used by native6, establish the zero-condition correspondence on the admitted cases, and require zero-input cached prediction bytes equal base bytes before execution. An unequal prediction is a cached GN contract failure and aborts admission; it is not a numerical rejection rescued by fallback. A recomputed native base-residual mismatch with the attested input likewise fails the source/data/operator closure. Do not accept a mutable self-declared residual hash as authority.

Where that proof holds, passing the required base-equal prediction to unchanged native6 preserves base retention by construction. Focused synthetic zero-input/base-retention and mismatch regressions must demonstrate this composition, including all subsequent common physics gates. If the proof or routing capability is unavailable, conformance remains blocked for the new adapter; do not silently claim it, edit native6, or infer a governing-contract amendment. Any genuinely required native or contract change needs a separate owned disposition. This protocol does not execute the proof fixtures.

**GN admission blocker:** the old Python wrapper accepts only `rheon-global-pressure-training-receipt-v1`; GN uses the distinct schema above. A newly authorized isolated adapter must validate actual GN final200/restore identities without rewriting old receipts, teaching the original wrapper permissive schema aliases or importing scientific training code. It should compose existing qualified native operator/solver behavior. Exact build/source closure must be frozen and independently reviewed after any explicitly authorized conformance work; this planning report supplies no new qualified binary.

## Complete work and cost accounting

Retain a case journal and machine-readable aggregate with these non-interchangeable counts: admitted proposals, numerical rejection by reason, raw physics failures by gate, direct accepts, warm corrections, any-PCG use, full resets, terminal failures and complete observed case count. Record warm/reset completed iterations separately, total attempted work, wasted warm work on reset cases and zero-iteration solves. Native breakdown's `iteration` is a failure index, while limit/cancellation counts denote completed iterations; preserve error variant/field meaning and report completed count unavailable where it is not measured. Never present a breakdown index as exact completed iterations.

For each method/case retain actual initial-guess/source hashes, warm solve/check time, reset solve/check time and available native stage intervals. Existing native `solver_s` includes fresh workspace allocation and initial-guess hashing; label it accordingly, and do not add setup again. Existing `corrected.elapsed_s` begins after initial raw checks. Native6 supplies no isolated initial proposal/reference/base check or operator-setup timer; these timings and a complete candidate-specific enclosing wall are **unavailable** under the existing response. Their work remains included in the enclosing case/process interval with every comparator. Do not invent separate times, subtract other methods to estimate them, or imply the new adapter can expose unmeasured internal stages. Preserve process startup, JSON/IPC, correction, verification, timeout cleanup and artifact publication costs at their measured enclosing scope.

The frozen restore receipt records 32-case inference totaling **0.32481544799520634 s**, separately from training's **0.30471505499008344 s**. Use restore per-case times in the primary recorded candidate-cost ledger and identify their earlier execution scope. They exclude native work, imports/loading/setup and generic custody. Do not select the faster training times or rerun inference in this slice. Optional sums of recorded inference and new method stages are clearly labeled descriptive cross-execution sums, with exclusions listed; they are not enclosing timing samples.

The existing enclosing native subprocess evaluates several methods. Its wall time and `native_elapsed_s` are total experiment cost, not standalone GN method wall. Include every comparator and failed attempt in total diagnostic cost; do not charge model import/inference to numerical comparators. Retain separate historical build/provisioning, capture/reference verification, packet packaging, scientific training, cold restore, model initialization/load, source admission/custody and publication costs with their actual scope. Missing stage timings are unavailable, not zero. A future acceleration claim requires the separately admitted complete resident/cold paired benchmark from the original acceptance design; this single fixed pass grants none.

Summarize all 32 paired cases, split/frame strata and trajectory pairs, not only successes. Native6 returns residual, predicted divergence, actual f32 divergence maximum/RMS, gauge, net flux, wall and Courant diagnostics. It returns neither solved pressure nor corrected MAC arrays. Raw cached pressure-versus-reference diagnostics are available from the frozen input arrays; post-PCG pressure-error and corrected-velocity-error norms are **unavailable** under the current wire contract. If reporting actual divergence L2, label `native_RMS*sqrt(4096)` as derived rather than an independently returned native measurement. Do not reconstruct unseen final fields, add a hidden re-solve or imply an API expansion to fill these gaps.

Report available physics diagnostics, gate failures, total warm/reset work and descriptive timing min/median/max with denominators. Keep per-case originals so aggregates cannot hide failures. Iteration/work comparisons are primary; no candidate-specific total-wall speedup is measurable here. A descriptive stage ratio is interpretable only when both compared methods qualify under identical gates and its measured scope is stated; retain failed comparisons without a favorable ratio. Cached earlier inference plus native stages remains a cross-execution sum, never a new matched end-to-end measurement. No proposal rejection/fallback threshold, checkpoint selection or altered budget is tuned after seeing these cases.

## Bounded future execution and acceptance

The proposed future action is one serial fixed 32 pass with one native thread, one operator configuration and no repeated scientific solve beyond each declared method's attempts. Maximum per-case PCG caps are 3032 for GN, 3000 for primary temporal, 3032 for short temporal diagnostic and 3000 for zero: at most 12,064 capped iterations/case, 386,048 over 32; direct acceptance/rejection can reduce actual work. All warm work remains charged. These are work ceilings, not projected observations.

Retain existing native-wrapper limits: 600 s total and at most 60 s/case subprocess within remaining budget; bounded request 1 MiB, response 512 KiB and predictions 8 MiB. Preserve the accepted one-thread CPU runtime and observed 4 GiB process high-water policy, expressly not OS containment. Before future execution, implementation review must measure/declare complete flat output inventory, per-file and aggregate caps, input/capture closure and cleanup ownership. No future Run/preflight/credential recipe is specified or authorized by this protocol. If these ceilings cannot support all fixed cases/policies, stop and review feasibility; do not crop cases, omit costly methods or tighten the time window after observing failures.

Protocol acceptance is independent review of the frozen population, source-mapped comparator equivalence, GN custody/schema admission, zero-residual disposition, all common gates, failure lifecycle and honest full-cost bookkeeping. Executable acceptance later requires focused evidence for wrong receipt/source/data, malformed numerical proposal, rejected/warm/direct paths, fresh nonzero-base reset, zero/tie behavior, original-b/gauge-row checks, failure/timeout cleanup and bounded publication, followed by separately authorized complete 32 observations. Planning link/source review proves the proposed contract only. Native candidate qualification, autonomous rollout, sealed final geometry/regime/trajectory suites and GPU/full training remain pending separate reviewed transitions.

Adopted standards pin is `188beda1fa477d21d576c233dd8c7f4c4c267d23`: Core/Contracts preserve authority and proof lifetime; Planning owns the distinct implementation/execution gates; Verification requires evidence at the actual boundary; Documentation owns this protocol's stable interpretation. This document changes no numerical policy or executable source.
