# Issues

- The historical handoff's closing native status predates the completed corrective
  pass. Refresh current status while retaining the original inventory snapshot.
- System Python lacks scientific dependencies and PostgreSQL is not running.
  Provision isolated CPU dependencies and a private task-owned local cluster;
  preserve existing catalogs and credentials.
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
