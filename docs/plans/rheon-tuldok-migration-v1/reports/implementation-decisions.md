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

The converted Projects keep their data and other evidence. Preserve the known
publication and exact reconciliation of attempt03, then qualify the converter
correction and rehearse it on a fresh restored destination/stage/spec. Never edit
a published stage or patch its accounting receipt. The original service remains
on schema 8 until the corrected rehearsal, cold reads, completed-current restore
and browser checks pass. Current host Python, model math and runtime assets are
unchanged by this offline classification correction.
