# Issues and dispositions

- I1, scientific: one-pass10,000 failed75 steps on16x23x25. This run's consumed
  development regressions directly test that gap. Owner: training experiment;
  fix-now through controlled evidence, no relaxed thresholds.
- I2, performance: managed pipeline preparation had substantial cost outside
  optimizer intervals. Owner: Tuldok workflow; deferred implementation. Record
  optimizer and complete workflow separately; revisit after this control.
- I3, scientific:127x5x5 is a known convergence failure in the retained control.
  Owner: learning coverage/objective; exploratory regression here, no universal
  resolution claim. Revisit with mixed-odd/elongated training-only feedback.
