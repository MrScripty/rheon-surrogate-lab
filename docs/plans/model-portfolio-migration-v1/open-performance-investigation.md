# Make the full project practical to open

Owner: /root. Status: installed f8ba4ce passes actual Open/custody/UI checks;
near-instant opening remains unresolved. Canonical plan: [plan.md](plan.md).

Latest actual postboot Open takes 712.675 native /716.216 wrapper seconds,
about11.9minutes, at checkpoint48b888b. The previous preboot observation was
634.114 native /636.389 wrapper seconds at47c0d7b. Earlier d3b8f92 took574.414native seconds on an earlier
checkpoint; these observations are not a controlled statistical comparison.
Current metadata totals1,767,615,399bytes across53,930canonical records after the
new organization note. The new feature improves navigation/readability, not Open
latency. Actual card reads remain around5seconds. Full validation/history and
projection reconstruction still run before availability; smaller metadata-only
profiles do not allocate the complete actual Open cost. Native guide UI/custody
is accepted in reports/native-server-prior-boot-recovery-02.json. Do not reopen
solely to repeat this observation while model review/retention advances.

The following investigation entries are chronological. Their then-pending
rollout/UI statuses are superseded by the current evidence above.

The user expects Open to be nearly immediate. The actual fresh combined-project
Open under source dd1feca remains CPU-active after 37 minutes, with no lifecycle
acknowledgement. The original request/driver/controller remain live; observation
timeout does not authorize replay. The project has 53,873 canonical records,
8,705 payloads and 62,792 portable files, totaling 18,065,071,125 bytes. About
1.8 GB is portable metadata. This observed workflow is not interactive enough.

The portable store's snapshot path demonstrably reloads all canonical records,
validates their bytes/JSON and receipt digest, then deep-copies the result.
Borrowing a root also requests snapshots. These are source facts, not a measured
allocation of this live Open's CPU time. Earlier profiles used older code and a
smaller source project; do not present those as a profile of the current run.

Selected next evidence: use the existing bounded metadata-only profiler with
explicit source/project arguments on the exact combined checkpoint. It must
read and hash existing canonical bytes in place, exercise the actual native
validator, retain phase/self/cumulative CPU data and stop its own diagnostic
after 90 seconds. No native ProjectStore/Session, SQL, source capability, model
runtime or payload is opened by this probe. This identifies contributing work;
it cannot qualify final Open latency or replace the original cold lifecycle.
Also record current process I/O/CPU counters without dumping memory or locals.
The first observation shows 27.4 GB of logical reads but only about 12 MB of
physical reads, so disk payload transfer alone does not explain the elapsed
time. Live inspection tools are unavailable/restricted (py-spy absent,
perf_event_paranoid=4); do not change those system policies or install a signal
stack dumper in the running application.

Original diagnostic write set: extend profile_current_validation.py with
explicit source/project selection and phase coverage; retain its report and
ordinary planning/state evidence. Source dd1feca and the open cold project remain
unchanged. Do not implement a cache, weaken validation, choose a guessed timing
budget or claim an end-to-end improvement before evidence selects a mechanism.
Any fix requires an isolated source branch, a focused reproducer, exact mutation/
tamper/freshness guarantees, representative measurements and the actual native
UI Open path. Keep the user UI serving throughout.

The exact combined-checkpoint profile is now retained in
[combined-native-validation-profile-01.json](reports/combined-native-validation-profile-01.json).
It ends at its own 90-second bound without completing domain validation.
Composition membership decoding accounts for 88.6 profiled seconds; repeated
canonical envelope encoding/hashing dominates that interval. This establishes
a contributor, not the complete cost of the live Open.

The resulting first intervention is admitted separately in
[verified-project-read-cache.md](verified-project-read-cache.md). It retains
decoding only inside one Store after an exact successful validation and still
reads and hashes every current canonical file on each check. The original cold
controller and its source remain unchanged.

The same closed 51,396-record, 1,110,965,915-byte metadata checkpoint was measured
in place on baseline and candidate source, with no project/payload fixture copy.
Both returned the same independently fingerprinted public snapshots and retained
all 51,396 actual canonical reads per check. Verified root borrowing performs
two checks: wall time fell from 154.927 to 1.779 seconds. Snapshot time fell
from 74.944 to 16.829 seconds; its required independent deep copy remains.
Initial store loading remains about 60 CPU seconds. See the
[baseline](reports/portable-store-read-baseline-01.json) and
[candidate](reports/portable-store-read-candidate-01.json).
These are component measurements on the smaller retained metadata, not a
complete 1.8 GB combined-project Open or a guarantee of near-instant UI startup.

All 41 real-filesystem store checks pass, including same-length corruption with
restored mtime and independent public snapshots. The first 173-test native/SQL
fixture had three failures because artifact tests counted public snapshot calls;
root borrowing intentionally no longer creates those unused copies. Their
observers now count the verified current-state checks, still requiring exactly
two and retaining every literal payload/integrity assertion. The first failure
evidence remains; all 173 checks pass in
[SQL-02](reports/verified-project-read-cache-SQL-02.json), with stable source
fingerprints and clean PostgreSQL shutdown. Candidate commit 1f1e3bc is retained
for later serial integration. No fix has been installed into the live user UI
or active cold controller. Both stopped synthetic qualification databases were
retired, retaining their reports/logs; no scientific payload was deleted.

The next measured composition digest cache was rejected and withdrawn. It
passed the minimized full decode regression and 185 native/SQL tests, and
reduced 127,094 envelope hashes to 53,655 on the actual combined checkpoint.
However, two baseline calls took 39.316/40.366 CPU seconds and the candidate took
40.236/39.853. Encoded bytes only fell from 1.934 GB to 1.841 GB; large unique
envelopes dominate. There is no demonstrated runtime benefit to justify that
extra cache. The proposal is retained only as a small exact patch and rejected
measurement record, and the worktree is clean at the already-qualified 1f1e3bc.
See [disposition](reports/composition-membership-cache-disposition-01.json).

Next diagnostic, admitted before execution: extend the existing metadata-only
profiler with an explicit phase-only mode and a maximum 600-second budget.
Default cProfile/90-second behavior remains unchanged. Exercise the complete
actual codec on the same combined metadata without profiler overhead to locate
work after membership decoding. Preserve source/header identity and per-phase
CPU observations, and report missing artifact access or the bound as incomplete.
This opens no Store, SQL or payload and cannot establish native Open completion.
The live cold request, source and project remain unchanged. Do not select a
second optimization before this evidence identifies the next material cost.

The phase-only measurement now identifies 19.423 CPU seconds in
operation_records. Its source uses active_records on every family and only then
selects project_operation receipts, cloning unrelated retained Run bodies.
Admit the next isolated slice at
`runs/model-portfolio-migration-v1/sources/Tuldok-active-operation-record-selection`,
private branch codex/active-operation-record-selection-20261024, base 1f1e3bc.
Owner/integration owner: /root. Target the verified-read-cache proposal, then the
composition source only after its active cold owner is stopped/reconciled.
Write set: project_fork.py, project_session.py and their existing focused tests;
bounded metadata measurement tooling and reports/state only. Add family selection
inside the existing active-record owner before copying, preserving complete fork
validation, default independent copies, receipt validation and archive rules.
The tight operation_records reproducer must first demonstrate copying an
unrelated envelope. Require literal values, independence, archive/malformed-fork
negatives, appropriate native integration and the actual retained-metadata
baseline/candidate before accepting. No source used by a live app or profiler
may change. Retain the worktree protected until serial acceptance or rejection;
then remove it cleanly with its head reachable from a recorded retained ref.

The operation-selection regression first observed an unrelated envelope copy.
The narrowed selection passes the focused semantic tests and all 226 affected
filesystem/native/SQL tests with stable source and clean PostgreSQL shutdown.
On the actual 53,873-record checkpoint, the baseline makes 52,513 irrelevant
active-envelope copies and returns zero active receipts in 21.774/21.352 CPU
seconds. The candidate makes zero such copies and returns the exact same values
in 0.527/0.524 seconds. Both still validate all historical receipts and complete
fork structure. Commit bf2746e was fast-forwarded into the isolated Store-cache
proposal, leaving live dd1feca and user UI07 unchanged. The operation worktree
and branch are removed-reachable, protected by the retained cache branch.
See [qualification](reports/active-operation-selection-qualification-01.json).

The first phase-only complete fixed-metadata codec run finishes successfully
without opening any artifact, Store, SQL or neural runtime. Besides the costs
above, history decoding takes 98.423 CPU seconds and receipt validation causes
many repeat fork decodes. These phase costs identify future investigation;
they do not describe complete native Open, which also owns actual payload
access and projection reconstruction. The second complete fixed-metadata run
uses bf2746e under the same 600-second diagnostic bound; its exact process and
report are recorded in work-state47. No full-operation speedup is claimed yet.

The fixed-source real cold Open remains CPU-bound after 26 minutes, so the
component results do not qualify interactive startup. Next bounded diagnostic:
profile the real MeasurementTable constructor on an original prepared pressure
packet whose exact bytes occur in the already-passed portable inventory. Read
that preparation outside the active cold root, without another Store/Session,
SQL, model, payload copy or retained root handle that could obstruct restoration.
Keep the active source immutable. A fast constructor rules this contributor out;
a slow constructor identifies its actual work before any new optimization.
The earlier metadata-only diagnostic explicitly omitted these packet bodies.

That representative 7,659,232-byte pressure packet validates in 0.652 CPU seconds
even with cProfile, so it does not explain the current startup cost. The next
diagnostic compares the actual pure Store snapshot-value projection with a
compact JSON round trip on pinned closed current-history metadata. Neither
constructs a Store/Session nor opens SQL/artifacts. All canonical bytes are
freshly hashed, and independent value fingerprints and caller-isolation checks
must match. Record garbage-collection pauses to distinguish cloning/allocator
work from repeated validation. This is a temporary mechanism experiment, not
permission to alter live source, weaken freshness or claim native Open speed.

### Fixed-source cold lifecycle completes; opening latency remains unresolved

Both no-copy native cold lifecycles are complete. Fixed-source run02 under
bf2746e obtained HTTP200 Open/Save/Close acknowledgements, stopped its own app and
PostgreSQL cleanly, and restored the original published project and local
continuation. All62,792 portable files /18,065,071,125 bytes and8,705 payload
inodes remain exact. Payload copies, neural forwards and actual-app audit
violations are zero. Original driver/controller/app PIDs are absent; user UI07
at32989 still returns HTTP200 on its separate old source.

The measured owner-wrapper times are Open2367.126 seconds (~39.45minutes),
Save197.785 seconds and Close191.913 seconds. These include SDK authentication,
status/preconditions and acknowledgement under verification-only cold auditing;
they are neither inference timings nor qualified ordinary-UI timings. Accepted
component fixes do not establish acceptable startup. Audit overhead remains
unmeasured. Inspection shows a Python audit hook for file/import/exec events,
with an armed-file existence check; it does not perform the initially suspected
Path.resolve/stack inspection on every file. Do not assert that auditing explains
the entire delay.

The pure snapshot mechanism probe completed: actual native projection16.111 CPU
seconds versus compact JSON prototype8.785 seconds, with equal full-value hashes
and independent caller values on pinned1.11GB metadata. Neither constructs a
Store/Session or qualifies native Open. Its GC counters also include the
outside-timed fingerprint/isolation oracle, so they do not prove GC domination.
The representative7.66MB pressure packet constructor completed in0.652 CPU
seconds. No further native source change is installed. Work-state49 supersedes
state48's pending-operation status; interactive UI acceptance remains open.


### Ordinary Open measured; basic rendered view passes

The original no-audit Open completed at the expected project/checkpoint in
1153.234SDK-owner-wrapper seconds. Real Chromium subsequently navigated the
already open project, asserted its complete name/ID and enabled controls,
captured the rendered page and stopped without runtime errors. This proves
the basic native view; deeper history/tree/chart workflows remain unqualified
and startup is still unacceptable. The cold and ordinary runs differ in
catalog/root/local continuation and observation conditions, so their timing
difference does not measure audit overhead alone.

The freshly bound inventory attributes1,568,836,148of1,756,105,511canonical metadata
bytes to460Run records. A rehashed original Run retains runtime inventories in
spec/material/payload edge structures. These are volume/structure facts, not
a complete CPU profile. Source inspection confirms history readers already
construct a fixed MaterialSnapshot view rather than redecoding material through
a live Session per Run; do not reintroduce that rejected hypothesis as the cause.
[Normal Open/basic UI](reports/normal-project-Open-and-basic-UI-01.json),
[metadata sizes](reports/combined-metadata-size-breakdown-01.json).

### Admitted proposal: avoid full graph copies for scalar metadata reads

Owner/integration owner: /root. Canonical plan: plan.md. Isolate a private
proposal at `runs/model-portfolio-migration-v1/sources/Tuldok-project-summary-reads`,
branch codex/project-summary-reads-20261024, basebf2746e. Target the composition
source only after its current native note-adoption owner is reconciled and the
app is cleanly stopped. Keep the old UI serving. Never edit live source.

The source shows Store snapshots used only for project ID/name/checkpoint in
workspace status, selection and post-dispatch envelopes; Session snapshots are
similarly copied to obtain scalar metadata in domain owners. Existing actual
snapshot-copy measurements establish the mechanism cost. The narrow complete
change adds a scalar summary at those canonical owners and updates all directly
reachable scalar-only project-owner consumers, preserving full snapshots for
callers that need records. Store summary still calls the exact current-state
owner and reads/hashes every canonical file; Session summary has the same cached
checkpoint/lock semantics as its existing snapshot property and exposes only
independent immutable scalar values. There is no timestamps shortcut, persisted
cache, change to wire schemas or deferred semantic validation.

Write set: project_store.py, project_session.py, project_workspace.py and
directly affected scalar-only project_*.py consumers; their existing focused
filesystem/native tests; bounded component measurement tooling and task
plan/report/state. First drive real workspace.status and demonstrate the unused
record-graph snapshot copy. Require same literal metadata, independent returned
dicts, same-length/restored-mtime corruption and root-replacement failures,
unchanged public snapshot isolation, affected native/SQL paths and representative
metadata-only baseline/candidate measurements. An actual UI/native operation
remains required before claiming an end-to-end speedup or interactive startup.
Retain this task-created worktree protected through acceptance/rejection and
record its exact head reachability before any terminal retirement.

### Scalar-summary component result; actual Open still unresolved

The isolated proposal is implemented but uncommitted/uninstalled. Equal scalar
values and complete canonical checks measure18.36/18.70s for snapshot-then-select
versus1.01/0.93s for summary. Initial Store load remains about a minute on that
pinned1.11GB metadata. Ordinary Open remains1153.234s; no native Open speedup
has been established. The broad900s gate did not pass; isolated real-SQL
session/persistence15tests pass with stable source and clean shutdown. Complete
remaining affected gates before integration and actual Open/UI timing.

Actual post-adoption Chromium loads C126 correctly, but status/list/records/note
requests take23.65/121.23/118.26/101.67s. These whole API times do not allocate
CPU phases. Eager history materialization and repeated copies/validation are
evidenced contributors, not an accounting of every second. Near-instant Open
cannot be promised from these component improvements.

### Remaining freshness-result copies (separate contract)

Source inspection finds seven consumed and55 discarded check() results across
project owners. ProjectSession.check still builds a full Store snapshot, checks
its checkpoint, and returns it. The current frozen summary proposal deliberately
preserves that return contract: native file reads and mutation paths consume
records. An optimized freshness-only path therefore needs explicit ownership
and consumer disposition, not a blind replacement with three scalar fields.
[Caller inventory](reports/session-check-copy-followup-01.json) is source-only
evidence, not allocation profiling or another accepted implementation. Revisit
it after actual Open/UI measurements of the currently qualifying proposal.

### Qualification fixture correction admitted after baseline reproduction

The12-module execution group reaches all169tests:168 pass and the interpreter
alias-drift test fails because this host's configured venv and /usr/bin/python3
both resolve to /usr/bin/python3.12. The exact same test fails on clean accepted
bf2746e, with no summary proposal imported. Production native-process code is
unchanged. Add tests/test_project_native_process.py to this slice's coordinated
fixture write set: replace its assumed alternate system interpreter with a
private copied interpreter at a guaranteed distinct resolved path, assert that
precondition, and retain any unexpectedly launched handle through existing
fixture cleanup. This tests a real alias change with identical executable bytes;
no native process validation is relaxed. Rerun the complete stdlib process
fixture, retain the original169-test result, and run remaining22public modules
under a fresh complete source manifest. Prior381storage/168execution passes
cover unchanged production/source test paths; the new test-source delta must
be explicit in final qualification. No live source or lab model is changed.

### Source acceptance and same-catalog rollout

Accepted commit0adcc5bea0bcc2cbc5bcf41a0627677582a52335 is installed in the
primary source after acknowledged original Close and clean app/PG termination.
Storage381 and public129 tests pass; execution168/169 plus corrected full26-test
stdlib process fixture qualify the selected paths. The lone original failure
reproduces on unchanged baseline and assumes two Python aliases resolve to
different binaries; its fixture now guarantees a distinct path. Production
process validation is unchanged. Three opt-in CPU-runtime integrations remain
excluded. See reports/project-summary-source-qualification-01.json.

Both UIs remain available. A real timed Open from empty workspace is live as
session99822; final times and actual UI measurements remain pending. The earlier
18.5→1s scalar mechanism result does not establish whole Open latency. External
same-drive writing was observed, so any wall-clock comparison must retain its
resource-condition limits. No source change is allowed while the app uses it.

### History cache rollout; optional Unicode change rejected

The bounded first-read experiment now binds agent_api.py as well as the four
history/material/session/store files, against checkpoint327b02f and all53,906
canonical records. A native strict-UTF8 text guard preserves the literal public
result and passes202native tests, including an exhaustive Unicode-scalar oracle.
It does not improve the representative workload: baseline97.486881390CPU seconds
versus candidate100.632400958. The older per-character profile is not sufficient
to attribute the remaining cost on the current source. The optional change is
not adopted. Its complete source patch and reports remain; the disposable
233,136,495-byte PostgreSQL fixture and temporary source worktree were retired
after verified clean stop and exact patch/source preservation. No model/data
bytes were removed. See reports/history-unicode-decision-01.json.

The already-qualified history cache9ff9899 is now installed via clean
fast-forward after the original native Close acknowledged in77.257947871s.
Only the verified owned launcher was asked to stop; its application and own
PostgreSQL stopped cleanly. The same catalog/port resumed with keep_open=true.
The original new Open request is in flight; neither an observation timeout nor
a component measurement establishes completion or a native speedup. Follow
runs/model-portfolio-migration-v1/project-read-performance-live-integration-03.
All source files remain immutable while that owner runs.

### Installed cache: actual Open and corrected progression workflow complete

Original Open2201074/session55768 acknowledges771.852895996s. No replay or
observation-timeout restart. Before/after audit verifies every53906canonical
body and exact project-root/payload membership/inodes/sizes/modes; payload bodies
were not rehashed. No data/weight copy. Same catalog/port and keep_open remain.

Corrected browser88391 passes61actions with17complete200API requests and no
runtime errors. It loads all94Trials, inspects original Study and Trial with
fresh context between them, and explicitly shows no recorded C53 cohort. Four
screenshots were visually inspected. First history origins100.733s; records
52.552s. Seven graph queries2.869–3.062s; the preceding same-project graph pages
were104.6–114.6s. Actual request times are not inference, controlled speedup or
near-instant startup proof. The user UI remains available. See
reports/history-cache-progression-native-UI-01.json and
reports/project-read-performance-native-Open-observation-03.json.

Source revalidation also recovers a dormant uncommitted proposal at
sources/Tuldok-json-snapshot-copy, based onbf6b39b, changing Store/Session cloning
and its Store fixtures. Old119native tests/61MBcopy evidence did not establish
adoption or current-source/full-population performance. Current9ff9899 still
uses deepcopy in Store._snapshot, Session.snapshot and Session.adopt. The text
list snapshots the whole Session and _tag takes another whole snapshot for two
scalar metadata fields. Preserve the existing patch; reconcile and qualify it
against current1.76GBmetadata before choosing another rollout. Do not infer
current phase allocation from the older Unicode profile or claim copying alone
accounts for all771.853s. Live source stays immutable while open.

Current execution state: [work-state-70.json](work-state-70.json).

### Current-checkpoint copy reconciliation

The original two frozen-GPU timing imports have acknowledged completion at
checkpoint760a464. The after-custody audit rehashes all53,915canonical records
and reconstructs the exact previous header; no old payload changes or scientific
data copies occur. Read-only chart qualification may run while isolated source
work proceeds. The live9ff9899source remains immutable.

Owner/integration owner: /root. Reconcile the preserved, uninstalled
bf6b39b JSON-copy proposal on current9ff9899 in the private worktree
`runs/model-portfolio-migration-v1/sources/Tuldok-current-snapshot-copy`, branch
`codex/current-snapshot-copy-20261025`. Preserve the old dirty worktree and its
three exact source hashes. This is one serial proposal, not a second live app.
Write set: project_store.py, project_session.py, project_projection.py and their
focused fixtures; task measurement/preparation scripts and evidence/docs.

Ranked falsifiable contributors: (1) repeated graph copies: replacing a copy
mechanism must reduce measured copy time on the exact current checkpoint while
preserving scalar bits and caller isolation; (2) eager history reconstruction:
copy improvements alone will leave material first-read reconstruction cost;
(3) repeated validation: canonical freshness reads and codec time remain even
after copies improve. No single component result allocates all771.853seconds.

Run the existing pure clone probe against the exact current1.76GBmetadata
without Store/Session construction, SQL, payload reads or a project fixture.
The earlier proposal's tests do not qualify this base. Before applying its copy
changes, require the current scalar/float-bit/isolation fixture to pass on the
baseline and reproduce the extra _TextView._tag full snapshot with a tight test.
Then qualify real Store/Session and mixed native-projection paths against a
disposable bounded database, retaining native corruption/freshness negatives.
Measure actual current Store/Session copy seams with source/header fingerprints,
full-value hashes, scalar-bit/type and complete mutable-container independence
oracles outside timing. Full lifecycle performance needs a later actual native
Open after original UI requests finish and a clean owned Close/stop; no restart
or end-to-end success claim is authorized by a component-only result.

Current-checkpoint evidence now confirms the copy mechanism gain. On the same
53,915records/1,760,975,751canonical bytes, actual Store._snapshot costs22.368CPU
seconds on9ff9899 versus8.304on the reconciled candidate; actual Session.snapshot
costs23.542versus7.978. The independent streaming oracle verifies complete
JSON scalar types, every float bit and all10,460,498mutable-container identities.
Both full-value hashes agree, every returned container is independent of the
source, and source values remain exact after caller mutation. The process peak
RSS rises from8,883,680to11,365,212KiB; this is a measured memory tradeoff, below
the explicit18GiB diagnostic address-space bound, not a whole-app memory bound.
See reports/current-native-copy-baseline-01.json and
reports/current-native-copy-candidate-01.json.

The tight text-tag baseline fails because _tag requests a whole Session
snapshot; its candidate uses the existing scalar summary. A second red fixture
shows text listing copying an unrelated active document. Selecting the two text
families inside the existing active-record owner avoids that copy while keeping
complete fork validation. The independent snapshot scalar/float fixture passes
on both mechanisms. Current45Store/text tests,96native Session/projection/
history/material/fork/workspace tests and91extraction/data/files/measurement/API
tests pass (230distinct checks, two text fixtures repeated across groups).
The first native96-test run had one new-fixture expectation error: import_text's
compact response is not the complete canonical current record. The corrected
fixture compares the exact captured current values and known note identity;
production values were not changed to satisfy the fixture. A separate test
selection typo refused before PostgreSQL initialization; it is not a pass.
All actually started fixture databases stopped cleanly. Live source remains
9ff9899while the eight-note native adoption is in flight. Whole Open, first
history, measurements and actual text-list API timing remain unqualified on
the candidate; I36 stays open.

### Actual copy rollout; next phase measurement

The current copy proposal is installed as0490cb1after acknowledged Close and
clean owned app/PostgreSQL stop. The same catalog/port resumes keep_open.
Actual Open acknowledges675.104owner-wrapper seconds (673.215native-call
seconds), versus the preceding771.853observation. Conditions/checkpoints differ;
this is not a controlled speedup or acceptable opening. Before/after audit
freshly hashes all53,916canonical bodies and preserves the exact header, project
directory and all8718payload identities. No project/dataset/weight copy.

Actual62-action UI acceptance reads all six new cards and the explicit temporal
split correction. All18API requests complete200. Records cost14.304seconds
versus the previous52.552observation; file pages6.593–6.690s and individual reads
4.453–4.986s. A further54-action chart campaign passes11API reads and all three
rendered comparisons, with source-list40.07s and comparison calls19.97–20.44s.
All ten screenshots were viewed. The user UI stays open. See
reports/current-copy-native-Open-and-reviews-UI-01.json and
reports/current-copy-native-chart-UI-01.json. I36 remains open.

Selected next uncertainty reduction: one600-second metadata-only phase probe
against the exact current0490cb1/aeeb838checkpoint, after the actual UI and
representative GPU worker qualification are terminal. The versioned
profile_validation_phases.py retains the original profiler, pins all tracked
source bytes/header, hashes every canonical body in place and records total
codec time plus aggregate receipt-validator calls/CPU durations. This directly
tests remaining domain/history/receipt-validation work without another
production optimization. The16GiBaddress-space bound remains. No Store/Session,
SQL, payload body or model runtime is opened; artifact-dependent validation may
end the probe as incomplete. Its result cannot allocate every second of actual
Open or qualify an alternate UI loading lifecycle. Do not run broad GPU latency
collection concurrently with this heavy CPU/memory diagnostic.

The original probe87265/timeout2283846/child2283847is terminal0. Complete
metadata validation costs264.630CPU/264.924wall seconds, with all2505tracked
source files and the canonical header unchanged. Initial composition decoding
costs29.775CPU seconds and history70.535. Receipt validation adds17.612source,
80.278measurement and55.473working-file CPU seconds:153.363seconds/57.95%of
the codec interval. These owner totals include nested fork checks; do not add
the307phase observations as independent costs. Native SQL, Store/Session
construction and artifact bodies are excluded, so this does not explain every
second of675.104second Open. See reports/current-copy-validation-findings-01.json.

Source confirms the repeated work: source receipts reconstruct material records;
each measurement-import receipt redecodes all measurement/material records;
each item in a batch file-adoption receipt redecodes the complete file inventory
and fork structure. This selects the next production target: preserve every
exact receipt/revision/reference check while reusing owner-validated candidate
records within one validation pass. Build the red codec fixture first, then
qualify negatives and representative metadata cost. No further live source
change has been made. The accepted0490cb1proposal worktree is retired after
exact2505-file/50,412,505-byte keeper comparison, clean status and actual UI.
The accepted head remains reachable from the installed composition branch;
the older dirty proposal remains unchanged.

### Admit one-pass receipt ownership

Previous goal turn is progress: measured/installed0490cb1, actual Open/UI/custody,
the complete264.630smetadata profile and live original GPU timing job. Revalidate
that job and keep its frozen numerical sources untouched. Owner/integration
owner: /root. Private proposal at
`runs/model-portfolio-migration-v1/sources/Tuldok-candidate-receipt-validation`,
branch `codex/candidate-receipt-validation-20261025`, base0490cb1, target the
composition source after clean native Close/owned stop. Never edit live source.
Write set: project_files.py, project_sources.py, project_measurements.py,
project_projection.py and their focused codec/native fixtures; task evidence/docs.

Use a small owner factory that decodes/copies one candidate and returns its
receipt validator. Keep exact result shape, revision, cross-owner reference,
artifact and unknown-operation checks. Contexts are local to one validation
pass, own their decoded values, and carry no publication capability. Existing
standalone validation keeps its interface and fresh per-call semantics; batch
items share only that call's decoded records. No process-global cache or input
flag can suppress validation.

First reproduce repeated decoding through the actual full-candidate validator
and standalone batch seam on a tiny valid mixed fixture. Preserve malformed,
forged, absent-reference, cross-candidate mutation, deletion and unknown-operation
negatives. Ranked predictions: repeated per-receipt/item decoding should fall
to one owning decode per pass; initial history/composition cost should remain;
filesystem/SQL/artifact costs remain outside the pure fixture. Measure the exact
current metadata only after broad GPU timing finishes, avoiding a simultaneous
heavy CPU/memory probe. Require affected native/SQL checks, then actual Open/UI
and exact project custody before claiming a loading improvement. Retain this
worktree protected until acceptance/rejection, then retire it with exact head
reachability/keeper evidence; do not delete scientific data or the older dirty
proposal.

### Receipt-ownership proposal qualified; live UI unchanged

The small mixed-candidate regression reproduces repeated decoding on the
installed baseline. The proposal creates one fixed, privately owned validator
per domain and validation pass. Standalone callers retain fresh per-call
semantics and result-shape/error ordering. Batch file items share only their
call's decoded inventory. No global cache or schema change is introduced.

All298distinct selected native checks pass:137file/source/measurement/session/
history/fork checks and161extraction/composition/HTTP/client/artifact/store
checks. Both disposable SQL owners stop cleanly. Negative cases cover forged
late receipts, unused malformed records, missing packet references, unknown
operations, caller mutation across candidates and fresh same-sized artifact
corruption. Source bytes remain equal to the tested proposal.

On the exact53,916-record aeeb838checkpoint, the same metadata profiler launched
on CPU8 observes359.022CPU/359.254wall seconds on0490cb1and
155.714CPU/155.753wall on the proposal. All35source,31measurement and12file
receipts remain checked, with one owning decode per domain. Peak process RSS
is8,481,892/8,483,320KiB. This is one paired observation, not statistical latency
qualification or a whole-Open speedup. The profiler excludes SQL construction,
Store/Session, real artifact bodies and UI. It does not self-record affinity;
both original taskset commands are retained in the execution history.

The explicit proposal write set is four owner/projection modules and
tests/test_project_receipt_validation.py. Source qualification:
[candidate-receipt-validation-source-qualification-01.json](reports/candidate-receipt-validation-source-qualification-01.json).
Commit this qualified private proposal before serial rollout. The installed
0490cb1UI remains open and immutable while GPU timing is active. Later actual
Open/UI and custody checks are still required; I36 remains open. Do not start
another loading optimization before testing this proposal in the real workflow.

### Receipt ownership installed; real Open remains slow

The original GPU timing continuation has completed all 254 process blocks and
its independent reduction passes before this rollout begins. Native Close
acknowledges in 59.851 seconds (63.066 seconds including owner setup). Only the
verified owned launcher is then stopped; its application and PostgreSQL stop
cleanly. The complete accepted proposal fast-forwards into the live composition
source as `d3b8f92ed99ad005e2bb09f6e963c3aebc24d9d0`.

The same catalog and port resume with keep_open. Actual Open acknowledges in
574.414 native-call seconds, or 576.609 owner-wrapper seconds: **9 minutes
37 seconds**, versus the preceding 675.104-second observation. Both use the
same canonical checkpoint. This observed 98.494-second reduction is not a
controlled statistical result, and opening remains unacceptable. The smaller
metadata-only phase does not account for the complete loading path. I36 stays
open; do not claim near-instant startup.

Before/after custody freshly hashes all 53,916 canonical bodies and retains the
exact checkpoint/header, project directory identity and all 8,718 payload file
identities/sizes/modes. No project, data or weight copy occurs. The original
timed requests are retained under
`runs/model-portfolio-migration-v1/project-read-performance-live-integration-05`.
Actual rendered chart acceptance passes54actions /11complete HTTP200 reads and
all four viewed screenshots. The redundant source worktree is retired after
exact installed keeper/ref checks, reclaiming50,427,316logical bytes. Keep the UI
open and the installed source immutable. Complete model organization and
benchmark publication before admitting any further loading investigation.


### Read-only chart source lookup: bounded continuation after coverage195

Owner/integration owner: /root. Canonical plan: plan.md. The latest user asked
why Open is slow and then to continue. With all current GPU owners terminal,
advance the narrow chart-read diagnostic while the remaining ten component
benchmarks retain their required place in the overall goal. This supersedes the
earlier sequencing preference to finish all benchmarks before another read-path
investigation; it does not assert that this read path explains initial Open.

The real ProjectMeasurements._documents → ProjectSession.snapshot regression
is red on installed f8ba4ce: five fully codec-validated records, one measurement,
one unrelated artifact envelope copied. Literal documents and caller isolation
already pass. The first attempted fixture used an invalid workbench envelope
and failed before the intended seam; that attempt is not bug evidence. The
corrected fixture is valid under the complete candidate validator and reaches
the exact unused-copy assertion in under a second.

Ranked predictions: (1) using the existing checked family read removes unrelated
graph copying; (2) reusing already adopted measurement documents avoids material
redecoding in read-only lookup; (3) canonical verification still costs time and
will remain mandatory. Original packet validation and full-candidate decoding
remain at their current publication/adoption owner. No cache or wire schema is
added. This is an isolated mechanism claim until representative and actual
native read checks qualify the consumer benefit. Whole Open remains unresolved.

Private source proposal: runs/model-portfolio-migration-v1/sources/
Tuldok-measurement-source-reads; branch codex/measurement-source-reads-20261026,
base f8ba4cedd22580f273f8c4fcbc06bca53797a7a7. Production write set:
project_measurements.py only. Focused evidence: new tests/
test_project_measurement_reads.py, existing measurement/session/native fixtures,
experiments/model-portfolio-migration-v1/reproduce_measurement_read_copies.py,
one read-only fixed-metadata measurement harness, reports and plan/state/ledger.
No live source edit, model/dataset/project copy, scientific payload access,
unique-weight deletion or UI interruption. Keep candidate protected until
accepted/rejected and record its exact retained ref before retirement.

Acceptance requires literal result identity, no unrelated graph copy, returned
value isolation, stale/unavailable and unselected-family tamper detection,
permission/revocation/artifact failures, original import/replay/fork/receipt
validation, representative exact-checkpoint baseline/candidate measurements and
later actual native chart reads. Full Open and final lifecycle are separate.


### Chart-read proposal qualified; actual baseline retained, not installed

Private commit00a02ee25cb30d254c059a9b372036f9f6f36696 changes only the
measurement lookup owner and its six new focused tests. All122distinct selected
checks pass across two retained fixture runs: the first has121passes and one
explicit missing-cold-role precondition failure; the corrected projection group
passes allfive tests with a distinct role that cannot CONNECT to the source
database. Both fixture PostgreSQL owners stop cleanly. Their generated clusters
are retired (215,407,016logical bytes); all logs/reports remain.

On the exact current checkpoint4c6da42c,53976records/1789673543canonical bytes,
two fixed-record observations take20.2491/19.1310CPU seconds on the installed
source and0.002945/0.002866seconds on the candidate. Baseline makes one entire
graph copy and material decode per call; candidate makes neither. All31document
values/hashes agree with literal canonical records. Process peak RSS is
9773676/4457724KiB. These exclude canonical freshness I/O, SQL, artifact bodies,
transport and rendering; no native or statistical speedup is established.

The actual native baseline owner is terminal:31source listing79.3722seconds,
first packet metadata28.7613seconds, same accepted checkpoint and exact source
identity throughout. Source list and metadata are different requests; their
relative times do not isolate a warm-cache effect. One later app-resource
observation shows resident23778120KiB and high-water37754384KiB, with188908KiB
swap; it is not a continuous resource trace or an attribution of the API delay.
No candidate endpoint or rendered UI has been measured. Whole Open remains
unresolved; this read-owner change is not on the initial reconstruction path.

Next: guarded serial source rollout, same-port/catalog keep-open resume, one
original timed Open, exact before/after canonical/payload custody, same two
native read requests and actual charts. Native baseline and all numerical
workers are terminal. Live sourcef8ba4ce is clean, user UIHTTP200/open. All ten
remaining component benchmarks,524state/variants,new-Rheon/reproduction,retention
and final lifecycle remain required. Qualification and commit reports are
reports/measurement-source-reads-source-qualification-01.json and
reports/measurement-source-reads-source-commit-01.json.


### Measurement lookup installed and scoped native UI accepted

The exact53976canonical bodies/header/root and8734payload identities survive
the acknowledged source-only restart unchanged. OriginalClose324.579seconds;
originalOpen1373.731seconds. The user reports all-core contention during Open
and later confirms it ended;this total spans both conditions and is not a clean
comparison. The application/project remain open on the same catalog/port.
All original lifecycle/read/browser owners are terminal.

Under post-notice conditions,the same native source-list/packetmetadata reads
observe74.52484/12.05796seconds versus79.37216/28.76131earlier. All31source
documents and chosen metadata match literally. These single observations do
not establish controlled speedup. Real browser passes68actions/11completeHTTP200
reads/three original chart selections/14literal numeric assertions. All three
chart images actually inspected;native-project.png is not claimed viewed.
Existing near-zero axis/table wrapping issues remain. See actual acceptance
reports/measurement-source-reads-native-acceptance-01.json.

Source listing still calls ProjectArtifacts.inspect once per packet. Each
borrows/verifies the canonical root;the existing inspect_many already provides
a bounded before/after-verified page and is used by history.Run. This is source
evidence plus observed slow API,not a measured allocation of every second.
I60requires a real-caller root-scope regression and exact pagination/availability/
permission/corruption checks before selecting that next intervention. Do not
edit live source or repeatedly restart solely to retime. Complete pending model
benchmarks meanwhile;combine a later qualified source rollout with useful native
publication work. Whole Open remains I36open.

The accepted private proposal is retired after all2513candidate files/50487455
bytes/modes match the live keeper and the exact head remains on retained refs.
First strict file-set equality rejected six extra live bytecode files;none were
proposal-only. Correct complete-keeper comparison preserves all six live caches
and removes only the redundant proposal. No scientific artifact is deleted.
Read-only storage audit finds nine SQLprojection schemas totaling8118272bytes
at that observation;they do not explain the gigabytes of host disk growth. No
schema is dropped. Accessible-process I/O observations are retained without
command lines,credentials or process termination;they do not identify duplicate
data. All remaining portfolio/retention/reproduction/new-Rheon/drift/lifecycle
requirements remain active.
