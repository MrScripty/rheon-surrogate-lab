# Results ledger

C64 complete; C65/C66 running. C53 remains baseline. All residuals below are
relative pressure-equation residuals. Target <=1e-6. No model promoted.

| Model | Fitted cases | Four-pass cases passing | Four-pass worst | Eight-pass cases passing | Eight-pass worst | CUDA trainer seconds | Managed workflow seconds | CPU phase seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C53 on combined bank | 24 | 0 | 1.6063694e-3 | 22 | 2.5973784e-6 | — | — | — |
| C64 combined | 24 | 15 | 3.2972074e-5 | 24 | 2.6550862e-9 | 207.2733 | 1228.3785 | 37.0466 |

C64 reduces the fitted four-pass worst residual about 48.7× and eight-pass worst
about 978×. Three fitted cases regress at one pass and two at two passes; the
predeclared all-case capacity gate fails. This authorizes only the two smaller
isolation screens, each with 200 exposures per case from C53.

C64 passes 240/240 manufactured regression cases at eight passes (C53 230/240),
with worst 1.5049574e-7 (C53 2.5975074e-6). At four passes it passes 75/240
(C53 0/240), worst 2.4178760e-4 (C53 1.6065176e-3).

On 128 consumed real requests C64 passes 1/128 at four passes (C53 7/128), worst
2.6926827e-4 (C53 9.7855327e-5). Both pass 128/128 at eight, but C64 worst is
7.9572892e-7 versus C53 1.4522194e-7. Better manufactured-mode fitting has not
preserved real-request worst-case accuracy. No untouched scenes or new rollouts.

Actual C64 optimizer work: 4,800 updates, 19,200 network forwards, 22,272,000
forward cells. Every one of 24 cases received exactly 200 exposures; teacher zero.
Peak allocated CUDA 20,347,904 bytes; peak trainer RSS 2,414,972,928 bytes.
Weights SHA-256: `36a03244adbe8758b4a864f41a09ccac0ca6fa4979a9bfc392c91f4e73ad4413`.
Training Run `2097763eed3249c186119e920eeee788`; CPU restore
`622caf928da24d3c8b0f2d83433b953d`, both completed/drained; published artifact
parity exact. Controller, monitor and CPU workers completed and reaped.

Per-case measurements, learning traces, source identities, readback timings and
capacity decision are retained in `combined-fundamentals/results.json`. The
aggregate report is a current projection while the conditional screens finish;
per-arm completed results are immutable.
