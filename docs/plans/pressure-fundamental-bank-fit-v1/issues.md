# Issues and limitations

- No model or physical contract change is being tested. This is a bounded shared
  optimization screen, with one seed and one train-calibrated amplitude. Failure
  does not establish an architectural impossibility; passing does not establish
  arbitrary-grid or fluid generalization.
- Smaller conditional arms match exposure and learning-rate allocation per case,
  not total updates or interactions between examples. Historical C62 had 2,000
  exposures on one case. C64 has 200 per case and 4,800 total updates, so a direct
  comparison changes both population and budget. Report those differences.
- CUDA learning diagnostics and restored CPU measurements can differ numerically.
  Exact artifact parity establishes the same restored state, not identical
  floating-point arithmetic on the two backends. CPU measurements own the gate.
- Retained workflow timing includes slow typed admission, preflight, approval and
  artifact operations. Current profiles observe API durations, not their internal
  cause. No timeout means a mutation is safe to retry; the durable journal is the
  authority. Controller termination does not prove remote cessation.
- The first controller guard receipt used a genuine base-only qualification. The
  additional complete-fixture guard removes only the bank specialization field
  from a valid complete receipt, supplies all path arguments and fresh directories,
  traps child startup, and verifies the exact ValueError. Both are retained; the
  latter isolates the intended contract failure more precisely.
- The reused generic qualifier emits an existing Torch warning when serializing
  a differentiable loss scalar. The bank qualifier detaches its own recorded
  losses. Qualification performed zero optimizer updates and did not initialize CUDA.
- Original untouched scenes and long simulation rollouts are deliberately excluded
  from this fitting diagnostic. No claim about deployment drift follows from it.

- One CPU evaluation invocation was premature: the controller completion record
  did not yet exist. It failed before output creation or worker startup; no managed
  mutation or optimizer ran. `early-cpu-attempt.json` retains the diagnostic.

- The first report invocation found its output subdirectory absent. It failed
  before publishing evidence. The report now creates its fresh destination after
  validation; no model execution was repeated.
