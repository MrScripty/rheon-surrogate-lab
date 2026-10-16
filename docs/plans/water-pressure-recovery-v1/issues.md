# Findings

- E1 (scientific interpretation): local same-RHS pressure and independently advanced
  trajectory pressure are different metrics. Owner measurement; fix now by reporting
  both, and do not infer autonomous accuracy from fixed-input replay.
- E2 (resource scope): prior24-thread runs and excessive CPU concurrency were slower
  or hit native deadlines. Owner experiment coordinator; reuse two one-thread workers,
  memory reserve and bounded raw retention. Revisit on measured resource breach.
