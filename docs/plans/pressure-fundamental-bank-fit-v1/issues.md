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

- C66 restore submission timed out at the client after 300 seconds. The original
  controller failed and reaped its child; its unresolved journal is retained.
  Read-only exact request-key lookup found restore
  `0b8f06c8e62d4dd481c6b5c2183c2b5b` completed and drained. The existing recovery
  owner verifies the current API request/graph/material/approval/actor/catalog
  before reconciling locally and collecting existing artifacts. No resubmission,
  new restore or retraining is authorized by that timeout. The initial API profile
  excludes recovery calls; completed managed elapsed time includes the whole gap.

- C65's first report correctly passed its numerical gate, but copied the combined
  arm's stop-action wording. Its original report is retained in
  `axis-isolation-report-01`. The corrected report chooses stop actions by arm
  before gate result; only C64 controls whether isolation runs start. Historical
  report/verification source bytes are retained by hash in `source-archive`, with
  an explicit `source-history.json` mapping. Scientific values and model weights
  are unchanged. Verification checks the exact retained source version when the
  current script has evolved; data/config/weight identity checks remain exact.
