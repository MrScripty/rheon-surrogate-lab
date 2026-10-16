# Findings

- The first fixture exposed relative-path normalization in the new capture
  wrapper. Fixed before broad execution; original failed source/receipt retained.
- Eighteen concurrent CPU trajectories exhausted the unchanged 120-second bound
  for two long strict-8 cases. Recovery owns fresh attempts at lower concurrency;
  failed partial trajectories are not counted as completed or hidden.
- Tuldok limits evidence groups to 64 and rejects changed snapshots under an
  existing ID. Keep a linked historical index and register a new immutable
  project snapshot in the same catalog; preserve historical snapshots.
- The prior 127×5×5 failure and application adapter migration remain out of scope;
  this study makes no arbitrary-grid or deployment-support claim.
- Initial broad replay-file discovery included command timing receipts. Fixed
  by enumerating the selected config/capture paths, with all36 results accepted.
  Failed source retained; no numerical jobs repeated.
- Initial scheduler could reuse a still-busy core in the next task wave and
  did not forward parent cancellation to its command owners. Final scheduler
  leases cores and propagates cancellation, with real active/pending-child
  lifecycle tests. Original executed source/timings remain distinctly archived.
- Pressure error is not uniformly reduced scene by scene by stricter stopping;
  it also costs additional calls. Decision: no promotion. Future late-direction
  training is prepared, not implemented or running; see next-training-protocol.md.
