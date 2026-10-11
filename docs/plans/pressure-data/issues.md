# Issues and dispositions

- PUB-1 — Resolved: the owner created public MrScripty/rheon-surrogate-lab.
  Verified repository identity/visibility and normal-pushed the complete history,
  preserving owner initialization8cb010d and experiment closuref80a7e8. Initial
  published head02748ac matches the remote. No repository creation retry,
  alternate destination or force push. See reports/publication.json.
- API-1 — Local patch/probe accepted; upstream integration deferred. Pinned
  Rheon still zero-starts. patches/ contains the independently reviewed sealed-only
  initial-pressure API and tests; corrected solves qualify. Owner: Rheon maintainers.
- LIC-1 — Resolved: owner initialization8cb010d supplied Apache-2.0 LICENSE,
  preserved unchanged. Dependencies' independent terms remain in THIRD_PARTY.md.

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
- TEST-1 — Closed: original v1 test and final seeds9000–9015 are now exposed.
  Final seeds were materialized once after committed freeze af6d2a1, with no
  subsequent tuning. The reservation JSON remains its freeze-time declaration;
  experiments/final/outcome.json records consumption and closure.
  Existing data integrity checks may read old test references but development
  emits no fresh model test metrics.
- FINAL-1 — Measured negative findings retained: raw learned fails the linear
  gate on seed9005 by3.70%; both corrected paths qualify all16. Fresh learned
  paths are slower than zero-PCG; no established advantage over the analytic
  baseline. No tolerance adjustment or follow-on optimization is authorized by
  experiment closure. See reports/first-experiment-closure.md.
