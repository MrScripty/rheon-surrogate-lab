# Native retained measurements

Status: the source format, native admission/query endpoints, chart and real packet
portability/browser workflow are qualified. Tested candidate `caee533` is
integrated locally as `916f848` with an identical complete source tree. The live
host is unchanged. Full portfolio coverage and a unified project remain pending.
The graph gate is complete in [its actual-history report](reports/native-progression-original-history-01.json).

## Data authority

`retained_measurements.py` owns a bounded scalar representation, exact retained
source bindings, complete subject/case/metric membership and fixed query behavior.
The producer owns scientific metric definitions, case pairing, reference epoch,
exposure classification and interpretation. The source format does not establish
scientific validity, independent validation or current permission.

Each packet retains original source bytes with SHA256 and explicit JSON pointers.
Every measured value must equal its source value as binary64, including signed
zero. Metric and comparison definitions are separately hash-bound source
documents. Failed, unsupported and unavailable observations have null values and
explicit reasons; opaque original failure evidence can be retained. Each matrix
combination has exactly one result, so omission cannot improve a summary.

Dimensions have explicit owners: subject attributes hold execution device,
precision or weight identity; case attributes hold grid density, domain shape,
timestep and trajectory identity; observation attributes can hold actual internal
work that varies by model and case. Unknown values remain null. Filters use only
declared typed values. Query summaries include every status and label their
numeric reduction as a maximum; they do not invent an application criterion.

The first packet is
`runs/model-portfolio-migration-v1/native-measurements-packet-05/measurements.json`.
It contains 1,024 unchanged scalar values for two subjects, 128 cases and four
accuracy metrics, plus 256 explicit unavailable latency observations. Source
captures have `latency_claim=false`; diagnostic job duration is not substituted
for call latency. Grid densities are 8/16 cells per metre, with separate XYZ
domain counts and input dt=1/60 s. Normalization denominators/floor flags and
original case/input/reference identities are retained. No physical timestamp or
simulation advance is inferred from the native frame index. This is a pressure
proposal comparison, distinct from a complete multi-output simulation generator.

Reproduce the packet with the task application Python and a fresh output:

```bash
runs/dependency-migration-v1/app-env/bin/python -B \
  experiments/model-portfolio-migration-v1/prepare_native_measurements.py \
  --output runs/model-portfolio-migration-v1/native-measurements-reproduction
```

The script verifies original result/archive hashes, subject identities, metric
definitions, paired cases and physical scalar input hashes. It retains source
JSON byte for byte and emits a validated `source-project.json` for the existing
source-registration workflow. It runs no model, simulation or numerical runtime.

## Composition admission

1. Scientific definition/interpretation, portable byte custody, current access/
   publication and visual selection have separate owners. The scalar codec
   neither reads paths nor launches producers.
2. Observation identity, source scalar and case membership must stay coupled.
   Current credentials, native project selection and renderer state must not
   become scientific provenance or imported execution authority.
3. Consumers supply a byte snapshot and select declared identities/dimensions.
   They need no source JSON traversal, scalar equality or missingness knowledge.
   The native wrapper will supply current artifact/access/session lifetimes.
4. Metric-policy changes belong to the producer and a new exact definition.
   Format/query changes belong to this owner. Actor/checkpoint changes belong to
   the native wrapper; plot interactions belong to the existing visual host.
5. References carry identities and hashes, not live runtime objects, SQL,
   expressions, filesystem capabilities or caller-provided chart code.
6. Nine decision tests and the actual source packet qualify the codec separately.
   Native import, role/epoch failures, publication/reopen and browser rendering
   still require their own actual proofs.
7. Deleting the codec would distribute duplicate scalar/source/missingness rules
   across import, query and display. It adds no plugin registry or second Run
   authority. Native custody/publication must reuse ProjectArtifacts and
   ProjectSession rather than create alternate file/receipt machinery.
8. Necessary complexity is bounded explicitly: 32 MiB packet bytes, 16 MiB total
   decoded sources, 8 MiB per source, 100,000 complete observations and bounded
   selections/responses. No model count is silently truncated. Larger portfolios
   must use explicitly indexed packet partitions and paged discovery; cross-packet
   comparison must match scope/metric definition hashes and case identities.

## Native ownership and qualified behavior

The isolated source `runs/model-portfolio-migration-v1/sources/Tuldok-measurements`
now contains native owner commit `457f9f2` and API commit `159bd19`.
`measurement_source` retains one immutable packet artifact, original source
snapshot/reference facts and attributed admission. Its identity stays scoped to
the original project on a fork. Original measurement bytes have unique custody,
including when a runtime dependency also refers to those bytes. No Workbench
record, Run, Trial, dataset release or present execution permission is invented.

The fixed public operations are `measurement_import`, `measurement_sources`,
`measurement_metadata`, `measurement_subjects`, `measurement_query` and
`measurement_inspect`, under `project_files`. They use the existing rooted
source capture, ProjectSession publication, ProjectArtifacts verification,
current permissions and selected binding/checkpoint. Typed SDK, CLI, MCP and
generated browser routes use the same registry. Scalar queries are read-only
despite their POST transport. No caller expression or rendering code is accepted.

The focused owner/source/fork fixture passes 39 tests. The API/client/HTTP fixture
passes 52 tests, including actual import, replay, stale selection, role refusal,
revocation and independent cold reopening. These overlapping suites are not
91 distinct cases. Failed first attempts and their exact sources remain retained:
one permission fixture asserted at the wrong boundary, and the initial API
registry crossed an incorrectly global operation bound. Capability discovery
now bounds the actual advertised profile, preserving the installed 256-operation
and 1 MiB limits. Oversized-profile refusal is tested. See
[owner qualification](reports/native-measurements-owner-SQL-fixture-02.json) and
[API qualification](reports/native-measurements-contract-SQL-fixture-02.json).

The actual packet is imported into the separate rehearsal project
`projects/rheon-measurements-rehearsal-01`, ID
`46326a529c2042acb7be4437e8de2345`. All 1,024 measured values match binary64,
all 256 unmeasured timing entries remain explicit, and ten original case/scalar
inspections match the codec. Native Save/Close pass. A new physical host opens an
independent project copy with only the copied root configured and a cold audit
armed before Open; it returns the same full data and passes Save/Close. All ten
portable files remain exact in bytes, modes and hashes with independent inodes.
Both hosts and their PostgreSQL instances stop. No protected audit violation,
new model forward, reference generation, original service access or UI acceptance
is claimed. See [actual qualification](reports/native-measurements-actual-qualification-01.json).

## Native scalar view qualification

The native view uses the shared pinned Vega renderer, extracted from the existing
retained-analysis view. It retains specification SHA verification, interpreter/CSP
rendering, disabled URL loading, resizing, disposal and stale-read fences. The
existing classifier renderer passes its twelve browser check groups after this
extraction. The scalar view has explicit source/subject pages, typed exact filters,
maxima and status counts over all matching cases, observation paging and exact
case/scalar inspection. Editing a selection clears its previous result. A source,
checkpoint, credential or host epoch change aborts pending reads and clears data;
loss of access also clears state without needing a new context key. Focus is
restored after loading and when a disabled pending control loses its context.

The final controlled browser fixture passes 23 checks, including 41-subject
paging, source population consistency, keyboard/pointer selection, separate grid
density/domain filters, empty intersections, failed/unavailable timings, stale
responses, revocation and narrow layout. The native SQL/API/contract fixture
passes 55 tests. Two small record-controller fixture programs also pass; their
view stubs do not establish rendering. Initial controlled-fixture failures retain
their sources: reversed vendor loading, missing top-level async wrapping and a
missing fixture UTF-8 declaration were corrected. A later focus-loss failure
exposed and fixed product behavior. See
[view qualification](reports/native-measurements-view-qualification-01.json).

A fresh native host opens an independent copy of the real C106/C107 scalar
project under an armed cold audit. All 59 browser actions pass: exact pressure
maxima, original C107 case/scalar inspection, independent density/domain filters,
explicit unavailable timings with no plotted values, Save, reload and Close.
Screenshots are visually inspected. All ten portable files remain exact in
bytes/modes/hashes with independent inodes afterward. Browser, controller and
PostgreSQL stop; no protected audit violation or new model/reference execution.
This is a two-subject scalar project, not the complete historical portfolio.
See [actual browser qualification](reports/native-measurements-browser-qualification-01.json)
and [local integration](reports/native-measurements-integration-01.json).

## Next portfolio slice

Expand native ingestion to the reconciled portfolio. Build a machine-readable
index that distinguishes model/fit/checkpoint/execution variant and preserves
aliases, original identities, unknown ancestry and missing evidence. Use existing
native definitions, adopted notes and retained sources before adding another
authority. Start with all 127 available common-pressure subjects and the retained
fit-card drafts, then include the other prototype/full-state variants. Combine
selected original history and scalar/model information into one native project.
No additional small-pilot qualification substitutes for that coverage.

Retain complete original per-case data with explicit packet partitions where
needed. A logical cross-packet comparison must verify the scope/metric definition
hashes and exact case identities; an aggregate view must explicitly describe its
support and preserve links to the full observations. Do not relabel a single
aggregate row as 128 separately stored observations, silently cap model count,
invent timings/independence or create Runs/Trials for direct captures. Trial-free
prototype/model relationships need an honest curated representation. The
completed ownership constraints below continue to apply.

Add a first-class native measurement-source record around the immutable packet
artifact and a small validated descriptor. Do not persist each measurement as a
separate canonical project record, encode scalars as voxel fields, or invent a
Run for a direct CLI capture. Compose its closed codec and artifact declarations
into the existing project validator. Use the existing rooted source registration/
capture owner for admission and the existing session mutation/receipt owner for
publication; imported provenance grants no present access or execution rights.
Delegate any Run-backed capture/disclosure to its existing owner. Scientific
population labels alone must never grant current permission.

Expose bounded source discovery, metadata/subject pages, selection and exact
observation inspection against binding/checkpoint/current actor. Preserve source
SHA, case identity, metric definition, epoch/population and missing statuses in
the response. Verify current artifact bytes and permission before/after reads.
No arbitrary SQL, expression, source path or rendering specification is accepted.

Use a host-owned pinned chart recipe and the existing renderer/lifecycle for
per-subject maxima and case drill-down. Expose grid density and XYZ dimensions
independently. Device, precision, task and actual-work distinctions remain
visible. A latency-only selection currently has no measured points and must show
its unavailable counts. A speed/accuracy frontier requires qualified compatible
timings. Add model-card/native-reference links without changing original facts.

Prove actual import of this packet, exact native summaries and drill-down,
Save/Close, independent copied-root cold reads and real browser interaction
before integrating the source/view. Then expand to the reconciled portfolio,
including missing and unsupported states, remaining frozen-old-reference
benchmarks, Trial-free provenance and reproduction. New Rheon remains a later
separate physics epoch.
