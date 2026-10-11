# Issues and dispositions

- PUB-1 — Blocked: authenticated GitHub integration cannot create the requested
  public repository. Exact error: GraphQL: Resource not accessible by integration
  (createRepository). Owner: repository account/integration administrator. Resolve
  by creating the exact requested public repository with authorized access; no
  different account/name or permission workaround. Local implementation continues.
- API-1 — Local patch/probe accepted; upstream integration deferred. Pinned
  Rheon still zero-starts. patches/ contains the independently reviewed sealed-only
  initial-pressure API and tests; corrected solves qualify. Owner: Rheon maintainers.
- LIC-1 — Open owner decision: new project license unspecified. No license grant
  is invented. Dependencies' independent terms are documented in THIRD_PARTY.md.

- MODEL-1 — Measured limitation, deferred improvement: shallow local CNN fails
  physical qualification for every predicted case; held-out test relative RMSE
  7.664 and max corrected divergence 17.458 s^-1. Owner: experiment maintainer.
  Revisit with broader context/physics-aware training using train/validation only;
  do not claim a useful solver replacement or acceleration from these runs.

- MODEL-2 — Diagnosed: finite receptive field has a remote-input counterexample;
  zero RHS produces nonconstant pressure; pressure-only loss does not control
  residuals. See qualification-diagnosis.md. No implementation bug established.
  One global modal candidate now completes the bounded development comparison
  and succeeds as an initial guess under patched PCG. Its fixed-operator spectral
  prior is a limitation; no general model/simulator claim follows.
- TEST-1 — Protocol updated: original v1 test has been exposed. New seeds
  9000–9015 reserved but not generated; materialize only after freezing choices.
  Existing data integrity checks may read old test references but development
  emits no fresh model test metrics.
