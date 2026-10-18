# Next proposal

Keep C53 deployed. C65 proves joint axis fitting on the first grid; C66 still
misses four-pass targets across grid shapes, including that same first-grid X
case. This points toward shared optimization across geometries as the next
bounded diagnosis. It is not proof of an architectural limit or a noise floor.

First use CPU-only autograd to measure per-shape gradient norms and alignment
at C53 and C66, without optimizer updates or new simulations. Compare the first
13×15×16 X case against the other seven and retain the pairwise matrix. This
can distinguish observed gradient conflict from an unsupported explanation.

Then propose one matched CUDA schedule test on C66's same eight cases, order,
seed, initialization, loss and 1,600 updates. Delay the first LR reduction from
100 to 150 exposures per case, then use 25 exposures at 1e-5, 15 at 1e-6 and
10 at 1e-7. C66's retained run is the control. Its worst four-pass CUDA residual
continued improving from 1.69e-4 at 100 exposures to 5.52e-5 at 150 and 2.29e-5
at 200, so testing decay timing is justified; improvement is not guaranteed.
Keep the same all-case four/eight-pass and early-pass guards, fresh CPU restore,
128 real and 240 manufactured regressions, memory caps and timing records.

Only after that shape gate passes should the changed schedule be tested on all
24 cases at 4,800 updates. A shared four-pass fit would then justify a separate
real-frame retention experiment; C64's real regression shows why that stage
cannot be skipped. Do not enlarge CFD datasets or change the architecture merely
because the present bounded fits missed their gates.

These are proposals, not newly started jobs. This study's authorized sequence is
complete. Separately, profile Tuldok restore/admission latency before a larger
experiment batch: measured managed time greatly exceeds actual CUDA training.
Any infrastructure change should remain separate from model/training changes.
