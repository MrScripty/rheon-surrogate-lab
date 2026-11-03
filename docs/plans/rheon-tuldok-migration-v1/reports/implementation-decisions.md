# Implementation decisions

## Indexed fixed-step simulation time

The user selected the original fixed-step clock arithmetic during implementation:
"the old path is better." Fixed-step diagnostic time is calculated from the
integer step index and the unchanged dt=1/60 s, instead of accumulating dt.
Force sampling, frame times and pressure-history interval endpoints use that
same authority. Adaptive accepted Sessions retain their own timestep authority.

An explicitly supplied interval end may differ from start+dt by at most one
floating-point ULP. Exact previous-end/next-start continuity still applies;
materially inconsistent intervals must reject. The current diagnostic recording
must store and hash the actual endpoint, rather than reconstructing a different
endpoint by addition. This requires coordinated current context, recording,
adapter and oracle changes; no historical recording reader is added. Original
recordings remain archival evidence. Numerical model/weights/loss are unchanged.

## Public diagnostic ownership

The selected current Rheon source originally kept the needed numerical engine
and scene types inside the desktop binary. Move those owners behind a narrow
public diagnostic session and reuse them from desktop and lab. Extend M2-R's
concrete write set to the diagnostic module, pressure-context endpoint validation,
diagnostic recording and directly affected tests/oracle/docs. Do not copy those
owners into the lab or expose the child-process lifecycle as caller-owned work.

## Runtime aliases

Independent review demonstrated that recording only resolved target files misses
retargeting an import alias between two already-inventoried files. The current
managed runtime inventory must also bind symbolic-link paths and targets.
Regenerate the measured current descriptor and package identities for the same
installed numerical assets, then repeat saved-input qualification. Preserve the
original manifest as historical evidence; do not add a permissive legacy reader.

## Independent review findings

The migration must also account for graphs referencing replaced Project IDs and
make large archived result documents inspectable through current bounded evidence
readers. Tests and final dispositions belong in the retained implementation
receipts; this decision record alone is not an acceptance pass.

## Bounded external evidence and pre-existing unavailability

The actual 120 Project records bind the whole primary lab as their root. Copying
that root recursively would include unrelated Git/cache/generated directories and
the migration staging itself. Replace the initial full-root inventory proposal
with a bounded union of actual Project manifest, source, recipe and evidence
references, independently hashed before and after staging. Preserve root handle
identity and reject path escapes or changed selected sources. Staging must never
become part of its own input inventory.

The initial read-only source inventory found 2,181 unique referenced paths, with
137 already unavailable in the primary planning checkout. Attempt exact-hash
recovery from the retained water lineage and Git evidence first. A historical
Project with unrecoverable references is archived-visible with its original row
and explicitly unavailable evidence, rather than made runnable or filled with
different current bytes. Pre-existing unavailability is a preserved source fact;
new loss or an unaccounted record still blocks publication. Report recovery
coverage separately; never claim unavailable source bytes were backed up.

Publish the current live lab project prospectively with an explicit mapping from
the selected integration source to primary artifact paths. Historical imports
must not imply that the old Runs executed from this new source.

## Build and capture closures

Whole-checkout fingerprints remain provenance. Native build admission validates
the actual compiled library/probe sources, Cargo graph and toolchain; unrelated
desktop bins, documentation, Tuldok changes or new experiment orchestration do
not invalidate an unchanged native binary. Capture the actual copied trainer,
model, wrapper, runtime and data bytes before and after preparation. No source
guard is removed from the files that participate in the selected computation.

An actual one-update admission exposed directory size/mtime/ctime changing when
other workers created unrelated siblings. Retain anchored no-follow directory
traversal and device/inode/type/mode/owner identity, plus the selected file's full
stat/read/hash/size proof. Ignore only parent entry timestamps and size. Threaded
sibling-write positives and parent/file replacement, symlink and mutation
negatives qualify this correction. Earlier failure was before Run admission.

## Preserve live observation in current Rheon

The fetched Rheon lacks the local read-only experiment observer. Add a current
bounded observer to its desktop owner, using the existing camera and latest
actual pressure frame. Extend the concrete UI write set to desktop main, the
new live-project module and its tests/docs. No second solver or process stack.
Use a dedicated migration observation root to avoid modifying historical files
in the conversion inventory. Explicitly report missing tracer; never generate a
static sphere or substitute a synthetic flow. Force parameters are exact scene
metadata with units and nominal/base labels, not a measured force-field claim.

## One live smoke also qualifies the typed pressure Trial

Keep the pre-cutover Q3 one-update CUDA/CPU execution qualification train-only.
After all Tuldok precutover gates pass, switch the schema/service with ordinary
submissions still closed. The one-update live smoke in Q8 is the actual typed
pressure Trial and fresh ordinary CPU restore specified by Q10, using only
already-consumed original validation data. This second update jointly proves
the A10/A11 observables instead of performing two equivalent live fits. No
untouched scenes, model promotion or numerical gate relaxation is allowed.

The root-owned cutover receipt distinguishes `m3_service_switched=true` from
`m3_complete=true`. It explicitly authorizes only the bounded typed smoke while
`ordinary_admissions_open=false`. The driver requires that exact receipt/hash,
endpoint, catalog and project-root binding. Mark M3 complete and reopen ordinary
admission only after CUDA/CPU drainage, immutable artifact readback and current
API/CLI/browser field inspection pass. A failed smoke preserves the new state
and receipts; it does not authorize a blind rollback or repeated admission.

## Restore ordering and fresh cutover bookend

The real schema 8 backup/restore, exact 8-to-10 upgrade and verified re-entry
prove the schema-only preservation contract. The completed converted schema 10
backup/restore then proves current-state restoration. Do not make an additional
full backup of the intermediate unconverted schema 10 copy; retain both existing
proofs and the exact row, sequence and file comparisons.

At final cutover, stop all writers and capture a fresh original-source bookend.
The already verified immutable full backup may be reused only if the current
catalog identity, schema, all tables and sequences, and every managed file's
bytes and permissions exactly match it. Record that independent revalidation.
Any difference requires a new full backup; no partial refresh is allowed. This
changes redundant copying, not the preservation or recovery requirements.

The original M3 window has now applied this rule: quiescence passed in 8.654 s,
and the fresh stopped bookend passed in 7.520 s with exact equality of all 31
tables / 6,870 rows, sequences and 38,598 managed entries including hashes and
modes. `ops/m3-fresh-bookend.json` under the primary migration artifact root has
SHA `3e1730ce8dfda77330ca772136b2543944339940d2fc18bda57f90dfa790f6e0`.
The explicit original schema upgrade/re-entry then passed in 9.476 s. Fresh
original-catalog preparation passed in 855.113 s against a new stage/spec; no
isolated stage was reused. Independent actual preparation review passed in
4.024 s (receipt SHA
`fcaf5ac25f82d9891487b6bea6780cd98fad49be6b498e5238adae0532dfef89`).
Publication passed in 69.431 s, exact reconciliation in 48.125 s and prelaunch
current read-only verification in 179.314 s. The latter inspected 874 Projects,
47,773 evidence previews and four retained Runs / 72 artifacts / 73,528,211 bytes
without Dataset or workers. These original-catalog proofs are separate from the
isolated Q2 receipts.

## Service switch is partial acceptance

Clean current-03 launched against the original catalog/8091 in 44.554 s. The
launch captures PID 516904 / start ticks 5889611, original catalog identity and
exact current source/root bindings. The immutable cutover receipt
`ops/m3-cutover.json`, SHA
`7bd1c17f52b29469893711ca2d298d6e5b723317b94dab5e05fe864c0c82a33a`,
records `m3_service_switched=true`, `m3_complete=false` and
`ordinary_admissions_open=false`. Ordinary closure is operator policy, not a
claim of transport enforcement. The receipt authorizes only the bounded
one-update typed-pressure smoke and asserts neither scientific acceptance nor
model promotion.

Supported current runtime CLI registration/verification passed in 14.112 s;
legitimate authentication and registry audit writes are recorded. Fresh current
Project04 registered normally without admitting Runs. The single authorized
live CUDA train/fresh ordinary CPU restore completed and drained in 225.243 s:
exactly one CUDA float32 optimizer update, zero restore updates and exact machine
artifact/field API/CLI readback. Weights match pre-cutover Q3's `bf3f7766…` identity.
Mean-free same-request pressure relative L2 0.04392560314266078 is diagnostic,
not model promotion. The immutable machine-stage receipt remains
`machine_qualified_browser_pending`; it is not rewritten after browser acceptance.
Actual full browser07 subsequently passed in 146.291 s (launcher/drain 146.559 s),
and independent combined review accepted A10/A11, SHA
`ec374853835f3a4fe30e735458a353f2e5b6a52ba9582000a969c546db504528`.
New root `ops/m3-complete.json`, SHA
`007143855647efbbb2a38436a8a96bc63ec9be878835d28f6b433c70d7046ca1`,
separately records completed M3 and ordinary submissions open by operator policy,
without scientific/model promotion. Keep the original raw cutover, machine stage,
failed operations and pending-review snapshots intact. Offline converter/test
retirement verification has passed, receipt SHA
`0af19b52ba079a4f376d9b49d5743526383fa1dd462aa479e72579ec8ba6dd5a`.
The offline retirement slice is separately committed at
`8af5a6a1e288e5b2f86a75b53cab93ca71222a04`. Git owns code/evidence revision
identity; ignored operations witnesses separately bind final resource/selections.

## An unacknowledged import requires reconciliation, not replay

Live browser01 failed in 0.165 s on Node18 `require` before API access. Browser02
inspected the current Project, activity and three evidence documents, then one
field import dispatch lacked a retained acknowledgement; the launcher stopped
after 21.033 s. Its consumed claim remains preserved. Do not repeat an import
to learn the outcome of the first request.

Exact read-only source reconciliation, SHA
`031dd1290b1913359612e4ab44a060cf4e97316feaec84426c354082bfb46d73`,
found the single pre-existing record `77971da0b73d45a29c373a8461133761`, created
October14, with source SHA
`d91c7e7136a5d1a4589db02f0ad86cca904571e500c9ab76ba99104e5d3a42b5`.
Attempt02 created zero new field records. The normal owner would reject that
exact duplicate with HTTP409; this is an inference from the admission contract
and retained record, not an observed reply. Original status/body are missing and
must not be reconstructed as raw evidence. Read-only browser03 reuse of the exact
existing record was independently reviewed and root-triggered, then failed in
50.721 s with `TimeoutError` in the existing-field phase. Its `mutations: []`,
zero imports, five retained Project/docs/activity screenshots and complete child
reaping remain negative-result evidence. Raw failure SHA is
`c9494b7791db08e151f32ddc7b0dc2f940930f07498b0643a8a335df1b7db698`;
launcher receipt SHA is
`eacddddcc7fc7470234bc563fbee02e8dbe297a34840a6524546b5c5003bcbd4`.
The actual read-only diagnostic then completed in 4.637 s. Current
`workspaces-data.js` lines 191–196 place the collection wrapper inside closed
Filters details. The same helper was red in 1.478 s and green in 0.677 s after
only clicking the visible summary: one exact GET, the existing record/scientific
canvas and no writes. This qualifies the narrow disclosure correction, not the
full workflow.

Browser04 failed after 16.071 s before page interaction: actual Project activity
was unavailable because its 1,875-character producer message exceeded the
unchanged 1,000-character consumer bound. After observation corrections and
healthy watcher04, browser05 failed after 88.225 s on the real train Run click
(`live_readback.cjs:170:87`), before field API access. Seven screenshots retain
healthy activity, collection and Study/Trial; no imports/mutations occurred and
cleanup passed. These five attempts remain negative results, with original
timings: 0.165, 21.033, 50.721, 16.071 and 88.225 s. Remaining read-only diagnosis
and normal Run-list navigation iterations were authorized without imports or
application writes. Read-only diagnostic01 took 11.805 s with both five-second
probes red; a solo actual Run list took 14.509 s for six Runs / 4.18 MB.
Diagnostic02 took 40.234 s with actual native view readiness 34.385 s after the
visible Trial route; this does not establish that route selection alone is
faster. Bounded train/restore readiness was set to 90 s and the outer deadline
to 360 s, retaining existing read/admission guards and actual-DOM tests.

Browser06 failed in 90.317 s: native field GET returned observed 200, while a
direct read POST returned observed 403 because Origin was missing. The minimal
POST-only pinned-Origin correction passed 11 tests and independent review
`183877…`, keeping normal Origin policy and all read guards. This observed 403
must remain distinct from attempt02's inferred duplicate 409. All six failures
are preserved; no further import or training admission was performed.

Browser07 then passed the full read-only workflow. Nine screenshots show healthy
74/74 activity (sequence 618→619), current documents, the existing collection,
Study/Trial, actual native pressure pair/probe/scalar table and ordinary CPU
restore artifacts. Train/restore UI navigation took 28.346/30.202 s, not model
compute timings. Exact field API bytes equal the retained machine result in Pa /
`<f8` / metre grid units; child and four descendants were reaped with no errors,
imports, mutations or extra Runs. The native pair has no retained difference
array and no physical timeline; ordinary restore remains outside the bound field
profile. Its equality rests on retained artifacts and the fresh machine
CPU/API/CLI probe. Rheon signed-error Q5 is a separate prior pass. These honest
view limits are not new scientific qualification or model promotion.

## Observation failures stay separate from scientific state

The identical 2 MiB regular-log/file-size-limit reproduction took 0.14 s and
confirmed stdout `EFBIG` was caught as a polling failure. Commit
`d8e95ac184f32b0f84d37c1be9e63e00465ff7df` makes the continuous publisher quiet
and puts once-only printing outside the polling catch. Two actual file-size-limit
tests were red in 1.36 s; 18 tests then passed in 3.483 s, including exact
pressure/source and completion-receipt checks. This fixes false failure reporting
without changing simulation or model acceptance.

Commit `01b9e5bd5b8603204979dda656e189b6938582ad` separately compacts activity
messages while preserving full structured details in their source status
receipts. Nineteen configured tests passed with zero skips in 3.545 s; the actual
unchanged current reader accepts 461 characters. The test projects nine real M3
and 14 journal statuses and 74 fixture counts, not fresh scientific runs.
Watcher04 health passed in 3.353 s (captured PID 536521 / start ticks 6106702),
with exact pressure/frame/source and current-reader availability. Actual normal
Project GET in 0.079 s confirmed `available=true`, `stale=false` and 74/74;
receipt SHA is
`81d48e3e04cec692a2a856d2b9adcdf5bf369c53cc87ed5317959d9b5d98a99d`.
That observation health does not convert browser05's failure into acceptance.

After actual browser07 acceptance and root completion, the final normal Project
GET passed in 0.0346 s: available, not stale, external 74/74 with a 520-character
message explicitly showing completed M3 / `m3_complete=true` / ordinary-open
operator policy. Receipt SHA
`6a0889a8ae7e4b944e3804979409c0e293f7c100ad24b4baaae1ae8d1c67e308`
also binds unchanged completion bytes before/after. Keep original cutover and
machine-stage pending values visible as historical gates alongside the new final
completion; do not overwrite them to make every old status look completed.

## Historical interruption before child startup

The actual source contains one terminal interrupted external-recipe Run whose
custody phase remains `prepared`. Its original owner explicitly classified this
as interruption before GO; identity and worker are null, a finish time exists,
and the final event is interrupted. Preserve that Run only as historical visible
evidence with its exact original phase and bytes. Do not call it drained or
create a current executable Run. Unknown custody, GO intent and other undrained
phases still block conversion. Retain the failed rehearsal and qualify this
narrow classification with the real PostgreSQL fixture before a fresh stage.

## Current browser execution review

Final composition review found two browser consumers that did not follow the
current backend contracts: all schema-2 recipes were routed to the bound-model
inspector, and the execution-review provider check allowed only the captured CPU
runtime profile. An ordinary current CPU restore recipe must use the existing
generic inspector/capture path; a bound managed Torch Trial must retain exact
binding, material and snapshot checks while reaching review through its supported
profile. Extend the current UI write set to `static/workspace-service.js` and its
focused browser fixture, alongside the already selected project workspace files.
No host Python, runtime, optimizer or numerical change is needed. Publish a fresh
clean deployment artifact after the actual browser regressions pass; keep the
previous artifact and its receipts unchanged.

The real browser fixture also exposed immutable policy numeric tokens changing
from `0.0` to `0` during ordinary JavaScript JSON serialization. The current
bound-policy owner compares its encoded representation exactly. Preserve the
unchanged inspected numeric tokens at that authoring transport boundary using
native JSON source context and raw-number support. Do not weaken backend policy
identity or guess which numbers should be floats. Edited values retain their
edited representation; unsupported browser capability fails explicitly before
authoring. Exercise the actual Model/closed-Trial browser path before acceptance.

## Historical recipe admission is deeper than syntax

The actual converted set initially retained 235 generic recipe references. Four
captured-CPU recipes across two Projects pass the general definition decoder but
are unsupported by the selected current Project-v2 inspection contract. The
other 231 managed Torch definitions are syntactically current and inspectable,
but hard-pin the original registered runtime descriptor. That descriptor lacks
the current required link inventory; fresh material preparation cannot admit it.
This is distinct from an opaque captured model/trainer input that happens to
describe its original numerical runtime.

For this migration, expose all 235 historical recipe definitions as ordinary
historical evidence, retaining exact bytes, original pins and labels. The
managed declarations are not malformed; preserving them as unavailable execution
templates would be possible, but is not the selected treatment. Fresh current
producers create separately identified executable recipes/material and approvals.
Do not silently repin historical recipes to the measured current runtime.

The converted Projects keep their data and other evidence. The known publication
and exact reconciliation of attempt03 remain preserved. The converter correction
passed 15 real PostgreSQL tests and corrected attempt04 passed publication,
exact reconciliation, both cold reads, completed-current backup/restore and
actual clean-deployment browser inspection on a fresh destination/stage/spec.
Never edit a published stage or patch its accounting receipt. All 235 historical
recipe bytes and pins are evidence, with zero active recipe references in the
83 converted Projects. Current host Python, model math and runtime assets are
unchanged by this offline classification correction.

Independent final Q2 acceptance is `ops/independent-q2-final-review04.json`, SHA
`751fbcd10c314cde30f43ea9b4f9fe55cf922390fb1bae5c50174022b91a6515`,
under primary `runs/dependency-migration-v1/`. Its scope is isolated rehearsal,
not original-service cutover. Source and restored cold reads took 152.930 and
142.347 s; completed-current backup/restore took 686.805 s, including 298.536 s
backup and 383.678 s restore. The actual Chromium/server check took 2.878 s.
The original service was stopped and upgraded only after these gates passed;
fresh original publication/reconciliation and service launch have since passed
under their own identities. A10/A11 machine and actual browser07 readback have
since passed accepted independent review and root M3 completion. Offline
converter/test retirement verification has also passed.

## Cold inspection authority and catalog serialization

The final cold reader binds its exact source bytes at startup and completion:
`7ea8dfc10fbc206307707be6d823a3edd58ae90837c9d1599eeead426f0d58a3`.
Its process-local guards deny original selected file payloads, original catalog,
original backup, old application, converter and numerical installation. The
restored read also denies its completed-current backup. Shared ancestor directory
traversal is permitted only so current
descendants remain reachable. Directory-FD and ordinary file reads are checked
against their actual targets; the nested CPython audit event omits `dir_fd` and
must not reinterpret an already-checked relative name against the old cwd.
Current owner-module origins and bytes are recorded independently of the guards.
Each authoritative read records eight deliberate denial probes, zero subprocess
starts and unchanged read-only database counts.

Current API inspection leaves 26 data references available but unverified within
its bounded hashing limits. Full archive and independent byte proofs establish
their exact preservation separately. Preserve that distinction rather than
claiming API hashing or scientific verification beyond the current contract.

An earlier read-only cold inspection overlapped the complete-backup owner's
advisory lock and failed with `LockNotAvailable`. This is a retained operational
negative result, not scientific failure or permission to weaken the lock. No
database writes occurred. Its total elapsed time was not captured and is recorded
as `elapsed_seconds: null` in `ops/cold-current-read04-failed-attempts.json`.
After the source backup owner closed, authoritative source cold inspection ran
alongside restore only on a distinct database; restored cold inspection followed
restore completion. Remaining same-catalog Metadata operations are serialized.

The actual browser's one new current Project and registration audit belong only
to the restored clone, after its exact roundtrip and read-only proofs. Counts
changed 874→875 Projects and 3,949→3,950 audit records; Runs, actors and credentials
stayed 4, 1 and 328. The owned temporary server exited. Those isolated writes do
not establish original publication; the fresh actual-original proofs above do.
They also do not establish live smoke acceptance. The offline
converter source witnesses remain exact archived evidence. After root M3
completion, offline converter/test retirement passed 16 isolated migration and
scientific-inspection fixtures in 39.803 s, plus one normal alias test in 0.003 s.
Converter bytes, both fixture class ASTs and all 80 application Python files are
preserved; application files also match host commit `824665ae…`. The full source
tree changes with fixture placement, while the deployed app witness remains the
actual qualified host/artifact identity.

Import-only normal CI discovery used the existing qualified primary `ROOT/.venv`:
1,503 tests discovered, zero methods executed, loader errors or offline imports/
modules, and a full reference scan with zero retired-fixture dependencies. The
initial app-env scan's 1,495 discovered tests and one unrelated missing-NumPy
loader error remain recorded; it executed zero methods and imported no offline
modules. No package was installed. Do not call that import-only result 1,503
executed tests. Receipt SHA
`0af19b52ba079a4f376d9b49d5743526383fa1dd462aa479e72579ec8ba6dd5a`
binds retirement verification. A0–A11, M3 and retirement verification are passed
within their actual scopes. The separate offline commit
`8af5a6a1e288e5b2f86a75b53cab93ca71222a04`, tree
`b92e14e8ac8c42543ab6d4a40247d232b1ec8463`, passed independent Standards and
Spec reviews with zero findings. All 80 application Python and 192 deployed
payload files remain unchanged, with the actual deployed execution source
retained as `824665aefdc68a458b5510a0b33d6cae1eb4887d`. Code/evidence commit
identities belong to Git; final resource/selection identities are separate ignored
operations witnesses. Do not create a documentation manifest that must predict
its own commit hash. No additional migration training is authorized by these updates.
