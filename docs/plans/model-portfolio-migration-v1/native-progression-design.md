# Native Experiment progression

The selected original-history browser gate passes on local Tuldok `e9b2819`:
independent Open, 26 browser actions, Save, Close and exact post-Close portable
files. See [qualification](reports/native-original-history-browser-01.json).
This does not yet provide the rich progression or scalar comparison views.

## Current implementation boundary

The isolated `Tuldok-progression` worktree extracts the existing Experiment
progression layout and canvas into `static/experiment-graph.js`. Workbench uses
that component through an explicit Trial-to-display projection. The component
accepts bounded display entries with identity, title, explanatory lines and
declared parent/baseline identities. It owns layout, escaping and presentation;
it contains no catalog, Trial, authentication, API or execution workflow.

Existing Workbench selection, pending requests, branch/admission workflows,
credentials, detail inspection and navigation fences retain their original
owner. The shared canvas uses the same buttons, styling and selected state.
The old test-only global layout alias is removed; layout decisions now have
their own direct tests. Four existing/new Node fixture programs pass. A real
Chromium test exercises the actual Workbench controller, shared canvas and CSS
with synthetic service transport: parent versus baseline, escaped names,
Tab/Enter selection, visible focus retained across rendering, pointer selection,
unknown ancestry and an explicit inspect-only agent. This does not qualify a
native-project backend or production Workbench training.

## Composition and change ownership

1. Retained history owns original facts and provenance. Workbench owns current
   authoring workflows. The graph owns their common visual representation.
   Native project binding, permission and checkpoint lifetimes remain with the
   native controller, which is the second required host.
2. Display identity and declared edge identity must remain coupled; chronology,
   accuracy and execution retry cannot create an ancestry edge. Authentication,
   transport, mutation journals and saved metrics remain outside the renderer.
3. Hosts supply display values and consume selected identities. They need no
   coordinate, edge-path or escaping knowledge. The page composition loads the
   shared script before its host. Existing CSS is shared rather than copied.
4. Layout changes touch the graph owner and its focused tests. Trial intent
   policy changes touch the history/Experiments owners and host projection.
   Credential or project switches touch their host lifecycle. New model types
   supply truthful display entries without pretending to be Trials.
5. The graph interface carries stable display values, not live domain owners,
   worker objects, callbacks with hidden authorization, or mutable catalogs.
6. Pure graph decisions and actual DOM interaction are independently tested.
   Each host still requires its own complete lifecycle and real service proof.
7. Deleting the shared renderer would put duplicate coordinate, relationship
   and escaping logic back into both hosts. No registry, plugin loader, second
   history store, new persistence format or custom interaction mechanism is
   introduced by this extraction.
8. Required complexity remains with each owner: provenance and immutable intent
   in retained facts, async authority/selection in hosts, visual bounds and
   explicit edges in the renderer. The component caps its window at 100 entries;
   native/API pagination must expose remaining entries explicitly.

## Next native integration

Add a bounded read-only projection of original history, including origin
selection, original Trial/material/Study references, declared intent, original
Run statuses and exact navigation references. Paginate against the selected
project checkpoint. Do not reuse the Workbench mutation controller or recreate
historical admissions. Use composite origin/identity display keys where needed;
preserve original IDs in the domain documents. Saved numeric results require
the existing disclosure owner and exact metric identities; graph status must
not become an alternate way to expose protected metrics.

C106/C107 have no recorded Trial ancestry. Show them as separate unknown-ancestry
entries. Trial-free checkpoints, assemblies and execution variants still need
their own truthful provenance representation; this Trial view is not the full
portfolio denominator. Do not create dummy Trials to fill it.

The native host must invalidate late reads on project, checkpoint, credential
or selection changes and clear stale visual state on refusal. Prove actual
origin selection, inspection, keyboard/pointer operation, Save/Close and an
independent cold reopen before integrating the view into the qualified branch.
The supported browser interaction evidence is deliberately scoped to the
tested Chromium environment and keyboard/pointer tasks; no broader conformance
claim is inferred.

Then integrate the separate typed retained-measurement comparison described in
[workflow findings](native-workflow-findings.md), beginning with the real
C106/C107 common-request pressure and equation results. Preserve the complete
old-reference benchmark and reproduction scope throughout this UI work.


## Native host implemented, real history gate pending

Local isolated commit7f533f0 adds two closed read-only API operations:
project_files.progression_origins discovers actual history namespaces in pages
of40; project_files.progression returns at most20 original Trial nodes, five
saved Run statuses per Trial and explicit related references. It preserves
original Trial/material/Study/lineage facts. Numeric validation/evidence fields
are absent from the status projection, leaving their disclosure/query owners
intact. No canonical format or current execution authority is added.

The native host shares the renderer and CSS. It uses composite origin/Trial keys,
explicit source choice, applied cohort pagination and a100-Trial visual window.
Run/Study/Trial inspection navigates with original identifiers. The host clears
late, mismatched or failed reads, keeps node focus through selection and restores
a useful focus target on refusal. Project/checkpoint/credential/epoch changes
clear its state through the existing native controller. The workspace title now
reflects its supported data/model/history tasks.

All51 actual SQL/contract/SDK/MCP/HTTP tests pass, including credential revocation
during an actual owner read and HTTP401 for revoked inspection. An independent
real Chromium fixture passes nine DOM/CSS/keyboard/lifecycle decisions with
controlled transport. This is not actual imported-project acceptance. The first
SQL run exposes two fixture setup errors, retained and corrected without changing
production guards. Source7f533f0 is frozen while controller-progression-01 opens
another independent copied root on fresh PostgreSQL. Its28-action browser workflow
will inspect both original C106/C107 nodes, unknown ancestry, full exact references,
original C107 Run navigation and Save. Close and final byte equality follow.
[Qualification](reports/native-progression-contract-qualification-01.json).


## Actual imported-history gate complete

The28-action original-history browser passes on7f533f0. Both imported entries,
unknown ancestry and full original references are inspected; the C107 link opens
its original Run. The graph screenshot is visually inspected. Save and Close are
acknowledged. All28661portablefiles,156requiredpayloads, six original history
facts and independent inodes/modes remain exact after Close. Browser, controller
and PostgreSQL stop cleanly, with no protected cold-audit violation. Main local
candidate5636510 has the identical full tracked tree. No live switch or push.
[Actual proof](reports/native-progression-original-history-01.json),
[integration](reports/native-progression-integration-01.json).

This qualifies selected original Trial history, not the complete model portfolio.
Curated model aliases/cards and Trial-free checkpoint/variant provenance remain
required. Native scalar sources are the next slice in native-measurements-design.md.
