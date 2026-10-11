# ADR 0002: Development diagnostics and a reserved final test

Status: accepted. Author: Puma. Date: 2026-10-11.

The first two smoke runs exposed the v1 test results. Treat those as historical
observations, not an untouched final evaluation for future choices. Development
now consumes and evaluates training/validation predictions only by default.
`--include-test` explicitly opts into test evaluation; routine diagnostics do not
use it. Dataset integrity verification still checks the full persisted dataset,
but reports no test prediction metrics or optimization signals.

Reserve the 16 unseen seeds in `docs/final-test-reservation.json` for a future
frozen-model evaluation. They use the existing held-out oscillatory family and
fixed geometry, so this remains a same-family new-seed test, not evidence of a
new unseen physics family. No fields, targets or model outputs for those seeds
have been generated or inspected. A deliberate next schema/data-plan revision
must materialize that reservation after training/validation choices are frozen;
schema v1 will not silently accept different seeds. Do not expand training or
validation into this reserved seed range. The reservation is a declared protocol,
not a claim that a new test dataset already exists or is verified.

Current investigation is fixed before running: zero pressure, the mean training
pressure, one diagonal inverse step, and pinned Rheon Jacobi/SGS reference solves.
The diagonal is extracted via public operator basis products. No iterative solver
is copied. The mean uses training targets only. Read unchanged existing CNN
checkpoints and probe zero response, sign symmetry, double-amplitude scaling,
centered pressure errors, and all numerical gate ratios. No model parameters,
learning rates, stopping steps or normalization values are optimized in this
slice. A single remote-face fixture tests the known finite receptive-field
limitation independently of held-out data.

Baselines and diagnostics are bounded to 12 development cases plus one structural
witness on the current smoke dataset. Record total compute and its included costs.
Comparing errors and accepted results is permitted; do not compare these timings
as acceleration or matched-quality warm-start evidence. The existing upstream
initial-guess limitation remains unchanged. Prepare an upstream patch only when
an admitted integration experiment needs it; this diagnosis does not.
