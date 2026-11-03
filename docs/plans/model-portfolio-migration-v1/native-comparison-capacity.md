# Complete full-state comparisons

Owner: model portfolio migration; native measurement/API owners. Accepted design;
implementation qualified and fast-forward integrated locally as `65f6caf`.
See [integration](reports/comparison-capacity-integration-01.json).

The complete diversity comparison has 2,120 cases, nine execution subjects and
nine metric states: 171,720 observations. Its five packets fit the existing
128 MiB combined byte budget but exceed the old 100,000-row combined budget.
Truncating cases or silently omitting unavailable states would change the
scientific population. The prepared temporal comparison is a separate scope.

The native measurement owner now distinguishes the 100,000-row per-packet limit
from a 250,000-row combined comparison limit. All byte, source-count, subject,
page-size, case-binding, unit and metric compatibility constraints remain.
No pagination, scalar reduction or persisted packet format changes are needed.

The HTTP operation is public and versioned. Version 1 retains its original
closed response bounds and exact hash; version 2 admits larger complete views.
The workspace rejects a view above the selected version's bound with typed 413,
before returning an incompatible success. The API owns the version-specific
limit, so the workspace does not inspect schema internals. Existing clients
negotiate advertised versions. All 281 previously installed contract hashes are
unchanged; only measurement_compare version 2 is added.

Write set: retained_measurements.py, agent_api.py, project_workspace.py and the
three focused comparison/HTTP/client test files. The private proposal branch is
codex/measurement-comparison-capacity-20261024, based on41974aea. Integration
owner is /root; target is the task's codex/composition-source-artifacts-20261024
line. After review, fast-forward the complete accepted proposal; do not rewrite
shared history. Retain its worktree protected until the qualified source is in
use for cold/UI acceptance, then retire the redundant clean worktree while
preserving accepted commits through the integration branch.

Evidence: the original engine refuses the actual 171,720-row population despite
valid packets; the candidate reduces every retained scalar against the exact
original replays. Twenty-four affected native tests pass, including real HTTP
and SQL plus version-specific workspace refusal. The actual large header is
accepted by the v2 output schema and rejected by unchanged v1. These are native
engine/contract proofs; final project ingestion and rendered UI remain separate.

- [Native SQL tests](reports/full-state-comparison-capacity-SQL-01.json)
- [Contract continuity](reports/comparison-capacity-contract-continuity-01.json)
- [Actual large wire value](reports/full-state-large-comparison-wire-01.json)
- [Original scalar bindings and reductions](reports/full-state-native-packet-qualification-01.json)

Composed-design check: one measurement owner retains table admission and
reduction, one API owner retains versioned wire bounds, and the workspace
coordinates the selected public contract. Consumers send ordinary source
locators and scientific queries; they do not split populations to evade a cap.
An additional packet format, lab-owned reducer or frontend-specific bypass
would duplicate these owners and is not introduced.

The broader client gate found the browser lacked comparison-v2 installation.
Commit `dd1feca` regenerates the full local version matrix and completes the
previous composition SDK/browser tables. See
[client qualification](reports/large-project-client-qualification-01.json).
Backend capacity acceptance alone did not establish rendered browser support.
