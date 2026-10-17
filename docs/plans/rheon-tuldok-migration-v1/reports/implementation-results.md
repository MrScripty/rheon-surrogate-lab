# Current-dependency migration results

Status: Completed. A0–A11, isolated Q2, original M3 and offline converter/test retirement
verification passed in their recorded scopes. Code/evidence commit identities
belong to Git; final resource/selection identities are retained separately as
ignored operations witnesses, without an own-commit hash in this document.
Original conversion, exact reconciliation, prelaunch read-only verification and
clean service launch passed. Runtime and current Project registration passed.
The single authorized live CUDA/CPU pipeline completed/drained with machine
qualification in 225.243 s; actual browser07 and independent combined A10/A11
acceptance passed. The root completion receipt marks M3 complete and ordinary
submissions open by operator policy. Converter/test retirement is verified and
committed as a separate offline slice. No model promotion is asserted.

## Method and invariants

Start the lab integration from the complete water-training lineage and merge the
accepted migration plan without rewriting either history. Integrate fetched
Rheon `bc2f472787060d9a69cb619ed2c2fea10fd754b6` and Tuldok
`300f12df6f48e028fd9379911703f7c89337952a` in separate worktrees/branches.
Select source, build artifacts and installed runtime independently through the
reusable configuration. Record dirty source bytes as such, not as the unchanged
upstream commit. Preserve original weights, normalization, numerical sources and
scientific results.

Qualify one captured frame before whole trajectories. Reproduce all 74 consumed
baseline cases, then compare the current build on the same cases with unchanged
physical/stopping limits and independent sparse pressure checks. The 26b residual
checkpoint is the regression control; the desktop keeps the original aa82 model.
The latter's known pressure failures remain visible. No untouched scenes or
checkpoint selection are part of this migration.

CUDA float32 training and CPU float32 neural inference remain separate. The
pre-cutover execution smoke performs one optimizer update; its fresh CPU restore
performs none and does not initialize CUDA. The later live typed Trial/restore
is a second separately gated smoke. Neither replaces deployed weights.

## Completed evidence

| Check | Observed result | Retained evidence |
| --- | --- | --- |
| One captured frame | 15 native prefix steps, one AI step, six actual network calls; input/output bytes match the reference | [Probe receipt](evidence/lab-probe.json) |
| Saved-input and operator parity | Exact model preprocessing/reconstruction and zero-residual behavior; independent operator/gauge checks pass | [Scientific parity](q4-scientific-parity-v1.md) |
| Original 74-case reproduction | All 74 cases and 8,523 AI steps reproduce recorded calls and pressure metrics; no native fallback/refinement | [Original reproduction](q7-old-reproduction-v1.json) |
| Current 74-case qualification | All 74 current cases, 74 native controls and 233 independent sparse same-RHS samples pass; every per-case pressure maximum and actual call sequence equals the reproduced baseline | [Current scientific qualification](q7-scientific-qualification-v1.md) |
| Tuldok CUDA/CPU pipeline | One CUDA update and fresh zero-update CPU restore, both completed/drained; checkpoint and weight bytes match | [Full pipeline receipt](evidence/q3-pipeline.json) |
| CUDA lifecycle | Five real hashing/training cancellation, deadline, exception and cold-recovery cases pass with zero updates | [Lifecycle receipt](evidence/q3-lifecycle.json) |
| Rheon lifecycle | Real child startup/cancel/reap/restart/shutdown; restarted 16-step run makes one prediction after 15 native steps | [Lifecycle receipt](evidence/rheon-q5-real-lifecycle-03.json) |
| Visible desktop rehearsal | Native pressure and signed AI-minus-native voxels/legend/probe work; model rejection preserved; schema-3 saved result opens with package unavailable | [Desktop receipt](evidence/rheon-visible-desktop-rehearsal.json) |
| Post-A7 desktop enablement | Same bounded 16-step comparison and package-hidden cold inspection pass; current read-only observer remains visible | [Enablement receipt](evidence/rheon-post-a7-enablement.json) |
| Schema-only restore | Exact original 31 tables, sequences and 38,598 managed entries retained through restore and explicit 8-to-10 upgrade/re-entry | [Schema rehearsal](evidence/schema-rehearsal02-receipt.json) |
| Corrected isolated domain conversion | Attempt04 published and exactly reconciled; independent source/accounting/recipe-policy review accepted the isolated rehearsal | `ops/independent-q2-final-review04.json`, SHA `751fbcd10c314cde30f43ea9b4f9fe55cf922390fb1bae5c50174022b91a6515` |
| Source and restored current cold reads | Each inspected 874 Projects, 47,773 evidence previews and 38,574 available data references; all four retained Runs and 72 artifact files / 73,528,211 bytes verified; read-only database counts unchanged | `ops/cold-current-read04-final-receipt.json`; `ops/cold-restored-current04-receipt.json` |
| Completed-current backup/restore | All 43 current tables, sequences and 38,599 managed entries equal; the restored destination adds only its exact validated owner marker; external stage retained separately | `ops/completed-current-roundtrip04-receipt.json` |
| Actual clean-deployment browser | Chromium opened the new current Project, external activity and evidence from an actual archived completed Run; no page errors or Run admission; temporary server exited cleanly | `ops/browser-rehearsal-04/browser-receipt.json`; `ops/browser-rehearsal-04/server-receipt.json` |
| Actual original conversion | Fresh original stage01 prepared, independently reviewed, published and exactly reconciled; no clone stage reuse | `ops/m3-domain-prepare.json`; `ops/independent-m3-actual-prepared-review.json`; `ops/m3-domain-publish.json`; `ops/m3-domain-reconcile.json` |
| Actual original prelaunch inspection | 874 Projects and 47,773 evidence previews; four retained Runs and 72 artifacts / 73,528,211 bytes verified read-only without Dataset or workers | `ops/m3-verify-current.json` |
| Clean service switched | Current-03 serves original catalog identity on 8091; exact process, source and project-root bindings captured; original cutover receipt preserved | `ops/m3-app-launch.json`; immutable `ops/m3-cutover.json` |
| Supported runtime registration | Existing-owner CLI authentication and current registry registration/verification passed; zero optimizer updates | `ops/m3-runtime-registration.json` |
| Current Project04 registration | Normal registration/readback passed without admitting Runs | `current-project-04-registration-01/receipt.json`, SHA `9e695c9602eaedd8b0e6fd9b6f7d39de1de1c1197a010e43bef506a8990df82d`, under the primary migration artifact root |
| Post-cutover typed pressure machine qualification | Exactly one CUDA float32 update and ordinary CPU restore with zero updates, both completed/drained; immutable readback matches pre-cutover Q3 weights; browser07 separately passed | `typed-pressure-live-01/receipt.json`, immutable machine-stage status `machine_qualified_browser_pending`, SHA `9595933635cbde2041acefe3fb076ed9ced673b7b423610f3612c2c7f2320832`, under the primary migration artifact root |
| Actual observation health | Watcher04 published exact frame/source evidence and a valid 461-character activity message; actual current Project GET returned available, not stale, 74/74 | `receipts/live-migration/watcher-04.json`; `receipts/live-browser05-actual-consumer-health.json`, SHA `81d48e3e04cec692a2a856d2b9adcdf5bf369c53cc87ed5317959d9b5d98a99d`, under the primary migration artifact root |
| Actual full browser07 | Nine screenshots and exact field readback cover current docs/activity, existing collection canvas, Study/Trial, native pressure pair/probe/scalar table and ordinary CPU restore; zero imports/mutations/extra Runs; child and four descendants reaped | `live-browser-07/browser-receipt.json`, SHA `e54e404630cd6e8ce72670fc5d09913faab797b2f9788e283dc74c5f272f6064`; launcher SHA `c4e1de262bfae227d4ab8a59f988272bd71d6f548bb73ec1d38e6284a3d38a85` |
| Final combined A10/A11 review | Actual original-catalog machine/browser acceptance passed; scientific promotion and overall M3/retirement decisions excluded | `ops/live-machine-browser-independent-review.json`, SHA `ec374853835f3a4fe30e735458a353f2e5b6a52ba9582000a969c546db504528`, `accepted=true` |
| Root M3 completion | Completed, service switched, ordinary submissions open by operator policy; no scientific/model promotion | `ops/m3-complete.json`, SHA `007143855647efbbb2a38436a8a96bc63ec9be878835d28f6b433c70d7046ca1` |
| Final normal Project visibility | Actual GET available, not stale, external 74/74; 520-character activity explicitly includes final M3 completion and ordinary-open policy; completion bytes unchanged | `receipts/live-browser07-final-completion-visibility.json`, SHA `6a0889a8ae7e4b944e3804979409c0e293f7c100ad24b4baaae1ae8d1c67e308` |
| Offline converter/test retirement | Exact source/test move preserves converter, both fixture class ASTs and all 80 application Python files; 16 offline fixtures and normal alias pass; normal CI imports no offline modules | `receipts/post-cutover-retirement.json`, SHA `0af19b52ba079a4f376d9b49d5743526383fa1dd462aa479e72579ec8ba6dd5a` |

Paths beginning `ops/` in this report are under the primary artifact root
`runs/dependency-migration-v1/`, not nested under the lab worktree. These exact
small original receipts remain there; root owns their retained-copy collection.
This documentation update does not change the evidence index.

## Timing and resources

These are different workloads; their elapsed times must not be added as a serial
project duration or interpreted as training throughput.

| Workload | Elapsed time | Scope |
| --- | ---: | --- |
| CUDA update plus fresh CPU restore | 271.887 s | Includes runtime verification, current material/custody setup, execution and artifact checks |
| Schema-8 restore and schema-only upgrade/re-entry | 329.666 s | Restore itself 316.815 s; peak RSS 1,234,776 KiB |
| Corrected Q2 source current cold read | 152.930 s | Final frozen reader; peak RSS 308,672 KiB; no application/worker or database writes |
| Completed-current backup plus independent restore | 686.805 s | Backup 298.536 s; restore 383.678 s; final equality checks included in total; peak RSS 866,780 KiB |
| Corrected Q2 restored current cold read | 142.347 s | Same frozen reader and preserved state; peak RSS 308,516 KiB |
| Actual isolated Chromium/server inspection | 2.878 s | Server lifecycle including 1.579 s browser inspection; clean deployment and owned server exit verified |
| Original M3 quiescence | 8.654 s | Exact selected original process stopped; original port closed |
| Original fresh stopped bookend | 7.520 s | All 31 tables / 6,870 rows, sequences and 38,598 managed entries including hashes/modes equal complete backup02 |
| Original schema upgrade/re-entry | 9.476 s | Explicit schema 8-to-10 upgrade and verifying already-current re-entry; no service launch |
| Original domain preparation | 855.113 s | Fresh actual-original stage01/spec; separate connection and identity from rehearsal |
| Independent actual prepared review | 4.024 s | Exact source accounting, historical recipe policy and 72 retained artifact payloads; no DB/application/converter imports |
| Original publication | 69.431 s | Exact reviewed current conversion published once |
| Original exact reconciliation | 48.125 s | Verified the already-completed publication; no republishing |
| Original prelaunch current read-only verification | 179.314 s | 874 Projects / 47,773 evidence previews / four retained Runs / 72 artifacts; no Dataset/workers |
| Current service launch | 44.554 s | Original catalog and 8091, clean current-03, captured process/root/API identity; smoke was pending at this phase |
| Supported runtime CLI registration/verification | 14.112 s | Current measured manifest, legitimate authentication/registry audit writes, zero optimizer updates |
| Current Project04 registration/readback | 3.154 s | Normal current API registration, no Run admission |
| Post-cutover typed CUDA/CPU machine qualification | 225.243 s | One CUDA float32 update, zero-update ordinary CPU restore, drained Runs, immutable artifacts and fresh field/API/CLI readback; browser excluded |
| Live browser01 failure | 0.165 s | Node18 `require` failed before API access; no import/admission |
| Live browser02 partial inspection | 21.033 s | Current Project/activity and three evidence documents inspected; one import dispatch had no retained acknowledgement; final browser acceptance not established |
| Live browser03 read-only failure | 50.721 s | Timeout in existing-field phase; no imports or mutations, five Project/docs/activity screenshots retained, all children reaped |
| Actual read-only DOM diagnosis | 4.637 s | Closed Filters disclosure identified, exact existing field/scientific canvas read, zero writes |
| Same DOM helper red/green | 1.478 / 0.677 s | Only a visible-summary click changed; green issued one exact GET and read the existing record/canvas |
| Actual full-log reproduction | 0.14 s | Identical regular-file stdout / 2 MiB file-size limit reproduced `EFBIG` being mislabeled as a polling failure |
| Publisher log-limit tests | 1.36 / 3.483 s | Two actual file-size-limit tests red, then 18 tests passed after quiet continuous publication / separate once-only printing |
| Actual activity-consumer tests | 3.545 s | 19 configured tests passed, zero skips; unchanged current decoder accepts 461 characters |
| Watcher04 health | 3.353 s | Exact pressure/source evidence, current reader available; captured PID 536521 / start ticks 6106702 |
| Actual current Project activity GET | 0.079 s | Available true, stale false, 461 characters, 74/74 previously qualified cases |
| Live browser04 failure | 16.071 s | Activity invalid before page interaction: 1,875-character message exceeded current 1,000-character limit; no imports, cleanup passed |
| Live browser05 failure | 88.225 s | Timeout clicking the actual train Run before field API access; seven screenshots, no imports/mutations, cleanup passed |
| Run-latency diagnostic01 | 11.805 s | Both five-second probes failed; read-only, zero extra Runs |
| Solo actual Run list | 14.509 s | Six Runs / 4.18 MB response; no inference or speedup claim |
| Run-latency diagnostic02 | 40.234 s | Actual native view ready 34.385 s after visible Trial route; route alone is not claimed faster |
| Live browser06 failure | 90.317 s | Native field GET 200; direct read-only POST observed 403 because Origin missing; zero mutations |
| Actual full browser07 | 146.291 s | Nine screenshots, exact machine/field readback, no imports/mutations/extra Runs; no retained difference array or physical timeline |
| Browser07 launcher/drain | 146.559 s | Child and four descendants reaped; no errors; includes browser interval |
| Browser07 train / restore UI navigation | 28.346 / 30.202 s | Visible saved-Run/ordinary restore read paths; these are UI latencies, not model-compute timing |
| Final normal Project completion visibility | 0.0346 s | Available true, stale false, external 74/74 / 520 characters; completed M3 / ordinary-open policy visible, zero new Runs |
| Retired offline fixture suite | 39.803 s | 16 tests executed using isolated PostgreSQL/fresh fixtures; 40.036 s command wall time; no original service/catalog actions |
| Permanent runtime-alias regression | 0.003 s | One test executed; 0.039 s command wall time |
| Clean normal-CI import-only discovery | 0.725 s | 1,503 tests discovered, zero methods executed, loader errors or offline imports/modules |
| Initial app-environment import-only discovery | 0.651 s | 1,495 discovered, zero methods/offline imports; one unrelated missing-NumPy loader error retained |
| Visible 16-step teacher comparison | 13.429 s | Model startup 11.090 s and one prediction 0.031 s included; GUI interaction time excluded |
| Independent saved-field pressure oracle | 0.044 s | 4,096 cells; no model or simulation rerun |
| Matched advisory timing | 540.83 s | Twelve serial measurements, two reversed-order repetitions; independent raw quality checks pass |

Use at most two one-thread simulation workers, one GPU optimizer and the measured
CPU affinity, retaining the 8 GiB available-memory reserve. Full installed
numerical-runtime verification binds 25,765 files, 1,047 symlink aliases and
5,744,383,179 file bytes. Runtime verification/startup are recorded separately
from steady-state inference. [The timing report](q7-advisory-timing-v1.md) records
first versus steady calls, spread, native/model/full-step scopes and uncontrolled
warm-cache limitations; it makes no performance acceptance or speedup claim.

## Historical preservation and failures

The source archive has 120 Projects and 160 Runs. Selected external closure is
3,314 files / 1,481,169,445 bytes. Thirty-six exact pinned files were recovered
from retained evidence; 137 pre-existing unavailable references remain explicitly
unavailable, never replaced with similar current bytes. Large historical JSON is
exposed as lossless bounded fragments through current evidence readers.

Prepare02 stopped before publication on a terminal interrupted Run whose original
phase remained `prepared`. The original owner permits this interruption before
GO, with no worker or custody identity. A narrowly qualified archive-only
disposition preserves that exact state; unknown custody still blocks conversion.
The failed stage remains intact; retry03 uses a fresh stage/spec.

Attempt03 then published and reconciled its exact accounting successfully. A
deeper current recipe check found four captured-CPU declarations unsupported by
the selected Project-v2 contract and 231 inspectable managed declarations pinned
to the obsolete runtime descriptor. The owned migration policy exposes all 235
as historical evidence, with exact bytes and pins preserved. The correction
passed 15 real PostgreSQL tests; fresh restore03 and stage04 then qualified it
without patching published state. All 235 definitions are historical evidence,
with no active recipe references in the 83 converted Projects. Attempt03 remains
a superseded rehearsal; corrected attempt04 supplies final isolated Q2 acceptance.

The corrected source mapping accounts for all 49,192 source keys and preserves
the original 120 Projects / 160 Runs. It exposes 83 converted Projects plus 791
historical evidence pages. Four supported completed Runs remain current with
their 72 artifacts; the other 156 Runs retain exact historical state and bytes,
including failed and pre-GO interrupted results. Independent review reconstructs
all 6,870 source rows and 10,456 bounded fragments and checks the exact historical
reference and data sets, not just their counts.

The two final cold reads used reader SHA
`7ea8dfc10fbc206307707be6d823a3edd58ae90837c9d1599eeead426f0d58a3`,
verified unchanged at startup/completion. Process-local filesystem/spawn guards
denied the original catalog, selected original file payloads, original backup,
old application, converter and installed numerical runtime. The restored read
additionally denied its completed-current backup. Shared ancestor directory
traversal remained possible to reach current
descendants. Each read recorded only its eight deliberate denial probes, zero
spawns, current owner-module origins and unchanged read-only database counts.
Twenty-six data references are available but unverified within the API's bounded
hashing limits; separate complete archive and independent byte proofs cover their
exact content. They are not reported as API-verified scientific results.

An earlier read-only cold attempt overlapped the complete backup owner and
stopped with `LockNotAvailable` when that owner held the catalog advisory lock.
No database write or scientific failure occurred. Its elapsed time was not
captured: the retained negative record uses `elapsed_seconds: null`, not an
estimate, in `ops/cold-current-read04-failed-attempts.json`. Earlier guard false
positives are retained there too. The authoritative reruns used the frozen final
guard; source cold inspection overlapped only restoration on a distinct database,
and restored cold inspection followed its owner's completion.

The browser deliberately registered the fresh current Project only in the
restored clone after the exact backup/restore and read-only proofs. This added
exactly one Project and one registration audit record (874→875 and 3,949→3,950).
Run, actor and credential counts stayed 4, 1 and 328. Historical evidence came
from completed Run `00b82369780c47cdae4db1b416eb5cdf`; its bounded preview remains
external evidence with scientific verification not performed. The current
Project and activity are prospective observations, not invented historical Run
lineage. Three screenshots and the server's clean exit bind the actual browser
check to deployment manifest
`09400b61cee86dde0014a71ade0a1d8e4f80372684b53950cd9a1cae31873c5a`.

Other retained failures include parent-directory timestamp false conflicts before
Run admission, early lifecycle cold-prefix clock disagreement and the first
GUI evidence decoder attempt. Their fixes and successful attempts are linked
from the owning receipts. No failed Run or rejected pressure is relabeled passed.

The authorized post-cutover typed Trial has now completed its machine phase in
225.243 s. Training Run `6b75f3ce89e64c8fb45ac514d0f386ae` performed exactly
one CUDA float32 optimizer update; ordinary CPU restore Run
`6f0d80863535407e86f10f625a0975a6` performed zero updates. Both completed and
drained. Downloaded weights/checkpoint SHA
`bf3f77666a69a4906a3733c20eea46ebeac6c9652bb1029ee1f05c96b628019c`
matches the pre-cutover Q3 result; original normalization remains unchanged.
Fresh CPU pressure and field API/CLI readback are exact. Mean-free same-request
pressure relative L2 is 0.04392560314266078. This is a diagnostic, not a model
promotion, scientific acceptance or replacement of the frozen trajectory control.

Actual live browser01 stopped in 0.165 s on Node18's `require` failure before
any API access. Browser02 inspected the current Project, its activity and three
evidence documents, then dispatched one field import without retaining a response
acknowledgement; its launcher stopped after 21.033 s. The consumed import claim
and failed attempts remain intact. Read-only exact-source reconciliation
`live-browser-02/reconciliation-exact-source.json`, SHA
`031dd1290b1913359612e4ab44a060cf4e97316feaec84426c354082bfb46d73`,
found the single pre-existing field record
`77971da0b73d45a29c373a8461133761`, created October14, with exact source SHA
`d91c7e7136a5d1a4589db02f0ad86cca904571e500c9ab76ba99104e5d3a42b5`.
No new field records were created by attempt02. The normal owner's HTTP409
duplicate conflict is inferred from that exact record and admission contract,
not observed: the original response status/body are missing. No request is
repeated to obtain them. Read-only browser03 reuse of that exact record was
independently reviewed and root-triggered, then failed after 50.721 s with
`TimeoutError` in the existing-field phase. It performed no imports, recorded
`mutations: []`, retained five Project/docs/activity screenshots and reaped all
children. Raw failure `live-browser-03/failure.json` has SHA
`c9494b7791db08e151f32ddc7b0dc2f940930f07498b0643a8a335df1b7db698`;
`live-browser-03/launcher-receipt.json` has SHA
`eacddddcc7fc7470234bc563fbee02e8dbe297a34840a6524546b5c5003bcbd4`.
The authorized actual DOM diagnostic then completed in 4.637 s. It found the
collection wrapper inside a closed Filters disclosure in current
`workspaces-data.js` lines 191–196. The unchanged helper failed red in 1.478 s;
clicking only the visible summary made it green in 0.677 s, issuing one exact
GET and showing the existing record/scientific canvas without writes. Receipts
under `live-browser-diagnostic-01/` and the independently reviewed disclosure
fixture retain that limited result; it was not full browser acceptance.

Browser04 subsequently failed in 16.071 s before page interaction because the
actual current Project API reported invalid activity: the producer message was
1,875 characters, above the unchanged consumer's 1,000-character limit. Its exact
GET diagnosis is retained in `live-browser-04/activity-projection-diagnosis.json`.
It performed no imports, recorded no mutations and completed cleanup. Browser05,
after the observation corrections below, reached healthy activity, the collection
and Study/Trial views, retaining seven screenshots. It then timed out after
88.225 s on `TRAIN.click`, `live_readback.cjs:170:87`, before field API access.
It likewise made no imports, recorded `mutations: []` and completed cleanup.
`live-browser-05/failure.json` has SHA
`d5b4d4d563ae84136ce4471fd6f233ce732139ea4109816fff76b536a792a249`.
The combined review at that point was `accepted=false`, SHA
`2fb9fe9cffa3c2e86b40612811994bb23c0e90d13c050f7c947ff1c71a632e4b`;
that pending review remains retained as historical evidence.

Read-only Run-latency diagnostic01 took 11.805 s, with both five-second probes
red. A solo actual list took 14.509 s for six Runs / 4.18 MB. Diagnostic02 took
40.234 s; the real native view became ready 34.385 s after the visible Trial
route. These observations do not establish that the route alone is faster.
The helper uses bounded 90 s train/restore readiness and a 360 s outer deadline;
local actual-DOM tests qualified that adjustment while read/admission guards
remained unchanged.

Browser06 failed in 90.317 s. Native field GET returned observed 200, but direct
read POST returned observed 403 because Playwright's direct request lacked
Origin. This observed response is distinct from attempt02's inferred 409.
The minimal helper fix supplies the pinned Origin only on POST read requests,
preserving normal origin policy and all read/admission guards. Eleven tests
passed and independent review `183877…` accepted the scoped correction; neither
application code nor model execution changed. All six failed attempts remain
preserved; none caused another import or training admission.

Actual browser07 passed in 146.291 s, with launcher/drain 146.559 s. Nine
screenshots show current documents, healthy 74/74 activity with sequence 618→619,
the exact existing collection record/canvas, Study/Trial, native pressure
prediction/reference/probe/scalar table and ordinary CPU restore artifacts.
The browser reused `77971da0b73d45a29c373a8461133761`: zero imports, mutations or
extra Runs, no errors, and the child plus four descendants reaped. Actual field
API bytes equal retained machine CPU pressure, declared Pa / `<f8` with grid
units metres. The visible scalar pressure L2 remains 0.04392560314266078,
diagnostic rather than model promotion. Train and restore UI navigation took
28.346 and 30.202 s respectively; these are observed read-path latencies.

The native SDK view retains no difference array and explicitly reports
difference unavailable; this standalone field has no physical timeline. The
ordinary CPU restore is outside the bound field profile: equality is established
by its retained artifacts and the fresh machine CPU probe/API/CLI chain. Rheon's
signed-error Q5 view is separately qualified. Browser07 does not claim those
additional visual products. Final independent combined review is now passed,
`accepted=true`, SHA
`ec374853835f3a4fe30e735458a353f2e5b6a52ba9582000a969c546db504528`.
A10/A11 machine/browser acceptance is established and the root M3 completion
receipt is durable. Converter/test retirement verification also passed and its
offline slice is committed, with no further migration training authorized by
this smoke.

## Observation corrections and actual consumer health

Two separately committed observation fixes preserve numerical fields and their
recorded source evidence.
First, a full regular stdout log at the identical 2 MiB file-size limit reproduced
`EFBIG` in 0.14 s. Printing inside the broad polling catch mislabeled a healthy
observation as failed. Commit `d8e95ac184f32b0f84d37c1be9e63e00465ff7df`
makes continuous publication quiet and moves once-only printing outside that
catch. Two real file-size-limit tests failed red in 1.36 s, then 18 tests passed
in 3.483 s, including exact pressure/source and completion-receipt checks.

Second, commit `01b9e5bd5b8603204979dda656e189b6938582ad` bounds the activity
summary while retaining full structured status details separately. Producer SHA
`ab6d937d3c4723f4d48d738306d39dfff70f6a394e7d856c58b9d5702004481b`
passed 19 configured tests with zero skips in 3.545 s. The actual unchanged
current reader (`8b6cab…`) accepts the 461-character result using nine real M3
and 14 journal-status projections plus 74 fixture counts. Those fixture counts
are not new scientific runs; the 74-case scientific qualification is unchanged.

Watcher04 health passed in 3.353 s, capturing PID 536521 / start ticks 6106702,
current-reader availability and exact pressure/frame/source evidence. An actual
normal current Project GET passed in 0.079 s with `available=true`, `stale=false`,
461 message characters and progress 74/74. Its receipt SHA is
`81d48e3e04cec692a2a856d2b9adcdf5bf369c53cc87ed5317959d9b5d98a99d`.
This establishes observed publisher/consumer health at that read, not final live
browser, scientific or M3 acceptance.

After root completion, a final normal Project GET passed in 0.0346 s with
`available=true`, `stale=false`, external 74/74 and a 520-character message
including `Final M3 completion: status=completed, m3_complete=true,
ordinary_admissions_open=true`. Its receipt SHA is
`6a0889a8ae7e4b944e3804979409c0e293f7c100ad24b4baaae1ae8d1c67e308`.
Completion receipt SHA `007143…` stayed identical before/after. The projection
retains the original cutover's incomplete flag and immutable machine-stage
pending status alongside the separate final completion, preserving their
historical authority rather than rewriting those receipts. Zero new Runs were
created by the visibility read.

## Offline test retirement

Root authorized and the converter owner applied the exact prepared retirement
patch after M3 completion. Sixteen offline migration/scientific-inspection tests
passed in 39.803 s using isolated PostgreSQL/fresh schemas; the permanent runtime
alias test passed in 0.003 s. The source converter remains exact `e4ab924f…`,
both preserved fixture class ASTs match, and all 80 application Python files are
byte-identical to their before snapshot and host commit `824665ae…`.

Normal CI validation was import-only discovery. The initial application-env scan
discovered 1,495 tests, executed zero methods, imported no offline modules and
retained one unrelated missing-NumPy loader error. A clean recheck used existing
qualified primary `ROOT/.venv` with NumPy: 1,503 tests discovered, zero methods
executed, zero loader errors and zero offline imports/modules. No package was
installed. The full reference scan found zero normal-CI dependencies on retired
fixtures. This does not claim 1,503 executed test methods. Receipt SHA
`0af19b52ba079a4f376d9b49d5743526383fa1dd462aa479e72579ec8ba6dd5a`
records the actual results and the initial environment limitation. Retirement
verification passed without original catalog/service actions or app-byte changes.
The separate offline slice is committed at
`8af5a6a1e288e5b2f86a75b53cab93ca71222a04`, tree
`b92e14e8ac8c42543ab6d4a40247d232b1ec8463`. Independent Standards (`5b1f839e…`)
and Spec (`e87ed244…`) reviews have zero findings. The actual deployed execution
source remains `824665ae…`; all 80 application Python and 192 deployed payload
files remain unchanged.

## Evidence custody and final integration

[The evidence index](evidence/index.json) hashes small exact original receipts
retained in Git. Full command logs, SQL backups, source inventories, raw arrays,
model bundles and datasets remain under `runs/dependency-migration-v1/` or their
original historical paths. Credentials and large scientific assets are excluded
from Git. A receipt that names a pending check is not evidence of its completion.

Corrected isolated Q2 is complete. Its independent final review SHA is
`751fbcd10c314cde30f43ea9b4f9fe55cf922390fb1bae5c50174022b91a6515`;
it binds 16 proofs to exact prepared, source, application and clean deployment
identities, and explicitly excludes original-service cutover qualification.

M3 has completed on the original catalog, without reusing the cloned stage.
`ops/m3-quiesce.json` records stopping the old service in 8.654 s;
`ops/m3-fresh-bookend.json` records 7.520 s and exact full equality to immutable
backup02, SHA `3e1730ce8dfda77330ca772136b2543944339940d2fc18bda57f90dfa790f6e0`.
`ops/m3-schema-upgrade.json` records the successful 9.476 s explicit upgrade and
re-entry. `ops/m3-domain-prepare.json` now records successful fresh original
stage01 preparation in 855.113 s, canonical identity
`51eff8892e179af1238b6c0b6f95fd6322997270774933f1c15adf4e7d368613`
and exact prepared-file SHA
`907566bbfd92654b021961cb94016a48972847f9689196ad35eb0d4f1d4e67d4`.
Independent actual-original review passed in 4.024 s, receipt SHA
`fcaf5ac25f82d9891487b6bea6780cd98fad49be6b498e5238adae0532dfef89`.
Publication passed in 69.431 s with receipt ID
`ec99c624a203d868749e2be8593da90c`; exact reconciliation passed in 48.125 s.
Prelaunch read-only current-owner verification then passed in 179.314 s with
874 Projects, 47,773 evidence previews and all four retained Runs / 72 artifact
files / 73,528,211 bytes, without constructing Dataset or workers.

`ops/m3-app-launch.json` records the successful 44.554 s launch of clean
current-03 at `http://127.0.0.1:8091` against original catalog ID
`a7da4a8de53541a2a0e2aad08005bb71`. It captured PID 516904 with start ticks
5889611 and source identity
`0b0ca0dcf29d87a107bd810538125fd9a0d5f6230e4bf66ccee571e11aec43c1`.
Historical and lab aliases point to actual original-stage-01 and its external
lab closure; current-lab and pressure aliases point to the primary lab root.
The original API identity and exact process ownership were verified.

The initial immutable cutover receipt `ops/m3-cutover.json` has SHA
`7bd1c17f52b29469893711ca2d298d6e5b723317b94dab5e05fe864c0c82a33a`.
It recorded `m3_service_switched=true`, `m3_complete=false` and
`ordinary_admissions_open=false` at the structural service switch. That closure
was operator policy, not a new transport enforcement mechanism. At that stage
only the bounded one-update typed-pressure smoke was authorized; the structural
switch alone asserted neither scientific acceptance nor model promotion.

Supported runtime CLI registration/verification passed in 14.112 s using the
measured current manifest; its legitimate authentication and registry audit
writes are explicit. Fresh Project04 then registered/read back through the
normal current API with no Runs. The single authorized live CUDA training/fresh
ordinary CPU restore completed/drained with machine/artifact/API/CLI checks
qualified in 225.243 s. Its immutable stage receipt still says
`machine_qualified_browser_pending`; separate browser07 and final independent
review now establish passed A10/A11 acceptance. Browser01–06 failures and exact
existing-field reconciliation remain preserved. Observation fixes and actual
watcher/current Project health passed. Root `ops/m3-complete.json`, SHA
`007143855647efbbb2a38436a8a96bc63ec9be878835d28f6b433c70d7046ca1`,
now records `status=completed`, `m3_service_switched=true`, `m3_complete=true`
and `ordinary_admissions_open=true`. This is operator-policy authority, not new
transport enforcement. The raw cutover and machine-stage receipts are unchanged
historical gates; the new completion receipt binds actual browser07 and the
accepted independent review. Scientific acceptance and model promotion remain
not asserted. Root-authorized offline converter/test retirement passed its 16
fixtures, normal alias and import-only CI/reference checks, with its separate
offline slice committed. Code/evidence revision identity is the Git record;
resource/selection identity is separately retained in ignored operations
witnesses. No additional migration import, write or
training is authorized by this documentation.

## Source commits and active resources

Lab branch `integration/current-dependencies-20261016` has permanent workflow
commit `2fd327ae0ffe79d315e9f3010bc9b07b9238975e`, followed by observation commit
`1ea12f731c7e74d62058d6da1a9f21824d61cf06` and observation fixes
`d8e95ac184f32b0f84d37c1be9e63e00465ff7df` /
`01b9e5bd5b8603204979dda656e189b6938582ad`. These status documentation edits
follow those source commits.

Rheon branch `integration/lab-current-contracts-20261016` is committed through
`759d928448dec8b160f4c113e35bf2bf2bd1b9d3`; its public diagnostic integration is
`054e4cc1ded96e1cd22d80f6473a678a33482f17` and observation UI is a separate
commit. Tuldok's permanent host integration is
`824665aefdc68a458b5510a0b33d6cae1eb4887d` on
`integration/rheon-lab-current-20261016`. Its separate offline retirement tip is
`8af5a6a1e288e5b2f86a75b53cab93ca71222a04`; the actual deployed host identity
remains `824665aefdc68a458b5510a0b33d6cae1eb4887d`. Rheon and Tuldok worktrees
were clean at the recorded tips. No push or history rewrite was performed.

The three integration worktrees under `runs/dependency-migration-v1/` remain
protected active sources owned by this migration task for Jeremy. Their branches
retain their commits. Synchronization is explicit after assessment, never an
automatic pull into qualified source. Keep these paths while live Project
references, review and deployment depend on them. Their next disposition is
review/merge, relocation of active references/deployment, then a separate clean
task-owned worktree retirement assessment with commit protection verified.
The exact post-commit heads and reachability are retained in the operations
receipts. The original primary checkouts and unrelated files remain preserved.
