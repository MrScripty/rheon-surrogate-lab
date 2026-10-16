# Findings

- E1 (scientific interpretation): local same-RHS pressure and independently advanced
  trajectory pressure are different metrics. Owner measurement; fix now by reporting
  both, and do not infer autonomous accuracy from fixed-input replay.
- E2 (resource scope): prior24-thread runs and excessive CPU concurrency were slower
  or hit native deadlines. Owner experiment coordinator; reuse two one-thread workers,
  memory reserve and bounded raw retention. Revisit on measured resource breach.
- E3 (workflow performance): weight0.1 managed train/restore1256.73s versus
  optimizer562.05s. Owner Tuldok managed runtime; defer to a separate runtime
  performance change, preserving custody/attestation. The remainder includes
  CPU restore, preparation and artifact handling, not just GPU idle time.
  Revisit when planning another sweep; use phase clocks before changing owners.

- E1 resolved: local oracle pressure, autonomous paired pressure, spatial modes and temporal bins are separately labelled in results and scalar reports.
- E2 resolved for this sequence: all444 full episodes and fixtures/timings completed without OOM or native deadline failure; bounded-memory oracle and measured CPU concurrency retained.
- E4 (accuracy/work tradeoff): all candidates fail the frozen comparative gates despite physical qualification. Owner next training protocol; defer further fitting, initialization-seed replication and independent8. Revisit with predeclared data-exposure/duration or feedback comparisons after reviewing these records; current data do not establish the cause of the remaining original-group accuracy gap.
