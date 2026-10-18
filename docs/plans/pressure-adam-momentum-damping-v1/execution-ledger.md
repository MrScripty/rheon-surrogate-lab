# Results and execution record

Completed 2026-10-18. C53 remains deployed. No CUDA training, new simulation,
holdout use, rollout, pressure-network optimizer update or model promotion.

The final CPU measurement pool took **11.5345 seconds**, using three concurrent
one-thread workers on separate CPU cores. It evaluated **30 distinct float32
parameter states**, 35 policy entries, all eight fitting cases, unchanged full
loss and pressure residual at 1/2/4/8 network passes. There are 25 damping entries including five preserved controls;
20 have factors below 1. At update 1 all five factors coincide because no
optimizer history exists. These aliases are not independent experiments.

No damping entry passed every predeclared guard. No damping entry lost an
existing four- or eight-pass qualification, but all had at least one strict
one/two-pass regression. Several gains on the weak case are real relative to
the empirical CPU/GPU value gaps; that does not make them safe overall.

| Saved update | Before weak residual · 4 | Retain 25% first moment | Improvement vs before | One/two-pass regressions | Lost fits · 4/8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.52243114e-4 | 3.93870359e-4 | −158.7114% | 3 | 0 |
| 10 | 1.86398131e-3 | 1.93975486e-3 | −4.0651% | 3 | 0 |
| 100 | 2.46952447e-3 | 2.43906508e-3 | +1.2334% | 7 | 0 |
| 150 | 2.06140567e-3 | 2.05614676e-3 | +0.2551% | 6 | 0 |
| 200 | 2.02616665e-3 | 2.02615305e-3 | +0.000671% | 7 | 0 |

Retaining 25% means damping away 75% of the saved first moment *before* adding
the current gradient. At update 100 its new step length is 40.65% of the
preserved step and direction cosine 0.6567; at update 150, 41.95% and 0.6188.
Both direction and length change. This test does not prove direction alone is
responsible, nor does it simulate subsequent accumulated moment histories.

Removing historical first moment entirely (factor 0) improved weak four-pass
residual by 0.8402%, 2.8376% and 0.5394% at updates 10, 100 and 150. These gains
and their full-loss gains exceeded the predeclared twice-empirical-gap test.
They still caused 3, 6 and 6 short-pass regressions. At update 100, unlike the
previous full reset's 19.43% weak-case gain, factor 0 did not lose cases 1 and 4
four-pass qualification. It offers a smaller but still unqualified tradeoff.
Zero first moment here retains the original second moment and step counter;
it is not the earlier full Adam reset.

The best weak-case entry at update 200 retained 75%, improving by 0.007763%
(absolute 1.57299008e-7). The observed pressure value gap at that snapshot was
2.29375064e-7, so the improvement fails even the single-gap comparison, let alone
the required twice-gap test. Do not interpret this tiny effect as a reliable
training improvement. Target remains <=1e-6; the weak residual remains about
2e-3, far from qualification.

All before-state, observed endpoint and fully preserved first-moment parameter
hashes and CPU metrics match the historical step screen **exactly** at all five
snapshots. CPU Adam reconstruction differed from the observed CUDA endpoint by
at most 7.45058060e-9 per parameter (allowed 2e-7). All saved state vectors were
unchanged. Initial zero-history factors were byte-identical. Independent scalar
recurrences checked every factor, untouched inputs, exact preserved control,
zero-first-moment versus full-reset distinction and typed invalid-input errors.

Peak self-reported worker RSS was **764.4 MiB**; controller `/proc` sampling
observed up to **864.3 MiB** for one process. Host available memory stayed above
**47.42 GiB**. Conservative parameter-vector workspace is **47,362,048 bytes**
(128 bytes per parameter), below 64 MiB. Network/data/runtime allocations are
covered separately by the 2 GiB RSS cap per worker and 12 GiB host reserve.
Qualifier and five workers exited 0 and were reaped. Cancellation receipts
cover SIGTERM during qualifier startup and SIGINT after registration of all
three initial workers, including returncodes and complete owned-child reaping.

Initial measurement pool (`workers-01`) took 11.9834s. Before acceptance, review
increased the conservative vector workspace bound from 80 to 128 bytes per
parameter; final `workers-02` repeated the measurements with the stronger
accounting. All candidate hashes and metrics were exact between runs. The
initial summary and raw records remain in ignored `accounting-preview-01`.
This rerun verifies accounting/reproducibility, not an independent seed.

The interactive report exposes all five snapshots, eight cases, full loss or
four-pass pressure error, percentage change or absolute logarithmic view,
observed/before controls, step lengths/direction and guard counts. Browser and
verification receipts retain the checks and source identities. Methods,
commands, scalar results and small receipts are committed; vectors, weights and
datasets remain in their existing managed/ignored locations.

Browser attempt 01 could not start Chromium under the filesystem sandbox
(`sandbox_host_linux.cc`, shutdown operation not permitted); no report check ran
in that attempt. Authorized local browser verification outside that sandbox
(attempt 02) passed all five snapshots × eight cases × two metrics, retained
values/guards, percentage/absolute target views, keyboard labels and narrow
layout, with no page exceptions. Chromium was reaped and its screenshot was
visually checked. This was an environment startup failure, not model evidence.
Final scalar/guard/source/historical-control/cancellation verification passed.

Staged whitespace review removed an extra final blank line in the report
producer; the summary/page source identity was refreshed without rerunning
numerical workers. Final browser attempt 03 and scalar-record verification
passed against the refreshed page; attempt 02 remains retained as a preview.
