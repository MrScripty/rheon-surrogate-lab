# Findings and disposition

- Resolved qualification finding: C63's first source comparison rejected
  `config.json` because its per-Project ID differs from C60. The preceding complete
  configuration comparison had already established equality except that ID.
  Compare the configuration and objective structurally at their declared metadata
  seams, while retaining byte-exact comparisons for unchanged numerical sources.
  Failed attempt 01 and its base receipt remain in the local run directory;
  [failure receipt](qualification-attempt-01.json). No managed admission occurred.
- Resolved admission hardening: a generic qualifier wrote its base-only passed
  result before the composite checks finished. New composite qualifiers publish
  base evidence separately and publish their final receipt only after all checks.
  The controller also requires the arm-specific result before starting any child
  or remote admission. A focused regression check rejects the original base-only
  receipt. C62's previously completed receipt contains its full specialized checks;
  its successful evidence is unchanged.
- Scientific limitation: C62 fits one manufactured fundamental mode but regresses
  on broader requests. It demonstrates the tested fitting capacity, not a lower
  bound on attainable error or evidence of generalization. C53 remains baseline.
- Resolved report finding: the first browser check found horizontal overflow on
  narrow screens from unbroken artifact hashes. Paragraphs now wrap long tokens;
  regenerated report passed the actual browser controls, numeric binding and
  narrow-screen checks. Failed and passed receipts remain under the run directory.
- Measured pipeline cost: C62's managed workflow took 990.457 s, while its trainer
  took 76.528 s. Restore preparation/admission and verified artifact transfer are
  recorded separately in `results.json`. This measures API call costs; it does not
  establish the server's internal root cause or justify changing numerical work.
- Scientific limitation: C63 improves the targeted modes against matched C60,
  while the higher 4,4,4 mode regresses and the worst C53 guards remain unmet.
  A single matched seed establishes this observed comparison, not generalization.
  Keep the same-stage C53/C63 timings together; C60/C62 timings are earlier phases.
