# Issues

- Resolved: refreshed the historical handoff's stale closing native status while
  retaining its original inventory snapshot and measured results.
- Resolved: provisioned isolated scientific/application CPU dependencies and a
  private task-owned PostgreSQL 16 cluster; existing catalogs and credentials stay
  intact. Actual Python 3.12.3 differs from historical 3.12.14 and is measured anew.
- Resolved: network and Unix-socket sandbox restrictions required authorized
  provisioning outside the sandbox. The failed first PostgreSQL initialization
  was preserved/stopped; a fresh private second cluster initialized successfully.
- Resolved: Ubuntu stdlib links sitecustomize.py outside /usr and ships python.o.
  Materialize the installed runtime source; omit compiler-only objects/archives
  rather than weakening the captured runtime's contained-source/ELF checks.
- Resolved: historical launcher intentionally rejects residual requests. The
  new local adapter adds that bounded mode and retains its lifecycle fixtures.
- External Rheon merge conflicts and Tuldok app.py edits are outside the write
  set. Fetched main and used clean task checkouts; no resolution or stash applied.
- Resolved: real runtime creation rejected intermediate snapshot directories
  under the host's inherited0002umask. Bootstrap now establishes private0077
  permissions before catalog/source/runtime creation. Snapshot verification is
  unchanged. First preparation remains failed with zero Runs and no optimizer.
- Resolved: first managed CLI open rejected the prepared project's root key.
  Preparation used local-pressure; the pinned CLI driver uses qualification.
  Snapshot identity includes the key. One bootstrap-owned constant now selects
  the driver's root for preparation too. The rejected attempt created no Run,
  approval or optimizer; preserve its failure and reprepare a fresh workspace.
- Resolved: actual managed child input admission used wrong evidence field names.
  Tuldok supplies the closed project-data schema with measured_sha256, not
  captured_sha256/captured_bytes. Scientific source-witness bytes/hash binding
  remains separate. Added producer-schema admission and corruption regression.
  Runaf85e241 failed/drained before data/model/optimizer initialization with no
  artifacts; retain its source, logs, metadata and nonzero execution costs.
- Resolved: default preparation's oracle call lacked a supported mode, then
  inherited a 60 s timeout above the oracle's 10 s admission. Added the bounded
  no-request mode, explicit caller timeout and actual wrapper/drain regression.
  Fresh bootstrap 2 now passes preparation and managed train/restore/evaluation.
- Observed limitation: both fresh GN lineages qualify native cases through
  fallback only (raw 0/32 each). The implementation establishes CPU portability,
  not acceleration, rollout or final generalization.
- Observed workflow boundary: the owner's open UI serves a different normal
  Tuldok catalog. These Project references/Run artifacts do not appear there or
  stream into its Data collection. Existing-catalog integration is separate scope;
  the runbook documents inspection of the isolated qualification catalog.
