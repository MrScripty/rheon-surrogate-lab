# Full fit completed; CPU pressure qualification failed

The candidate completed all 5,000 CUDA updates and 5,000 finite-state checks. The
fresh CPU restore reproduced all five required artifacts byte for byte. The model
is not qualified: three scenes rejected their first AI pressure request after
32 calls, and another failed the required pressure-error certificate. Preserve the
retained control; no candidate was promoted or unused holdout consumed.

## Training and restoration

| Measurement | Result |
|---|---:|
| Configured/completed updates | 5,000/5,000 |
| Successful CUDA checks | 5,000 |
| Overflow-only clipping repairs | 2, at updates1,398 and 1,399 |
| Predecessor logged losses through 1,500 | All 15 exactly reproduced |
| Instrumented training-loop wall time | 586.818s (9m46.8s) |
| Trainer wall time | 591.034s |
| Managed training Run through drainage | 638.428s |
| Managed CPU restore Run through drainage | 50.620s |
| Full fit/restore/API/download command | 1,523.979s (25m24.0s) |
| Peak trainer CUDA allocation | 870,590,464 bytes |
| Final float32 parameter count / weights archive | 374,112 /1,502,110 bytes |

Final weights SHA256:
`c38091565d58c2a897c2dd6fa853d223262dcc64a2118ee428ea58e17a3b8ade`.
Original Tuldok catalog `a7da4a8de53541a2a0e2aad08005bb71`, Project
`de4e5d3a600943cdb8b6aa4225d360ce`, training Run
`c01e1741f48a474d9a2a3252b3971827`, CPU restore Run
`707c16c97bd54738b038e6c7b7828689`. The Study is closed. Both managed
Runs exited0 with drained/ECHILD/spawn-closed receipts; all 37 actual downloaded
artifacts and 37 staged sources per Run were verified against their owners.

The retained control's loop was498.238s. Both runs used exactly50,000 optimizer-loss
network forwards and 425,703,026 forward cells. The observed loop ratio is1.1778;
architecture, clipping and instrumentation differences are explicit, so this is
not an isolated architecture-cost estimate or an inference-speed comparison.
Final sampled loss was0.00158726 versus the control's0.000994973; sampled training
loss is not pressure qualification.

## CPU qualification and its limit

The first-frame fixture passed in9.525s:15 native warm-up steps then one AI request,
7 network calls, exact first-frame input closure and independent oracle acceptance.
The same successful numerical outputs were reproduced when qualifying the local
current-schema rejection classifier.

The fresh development campaign is **incomplete**, with these74 scheduled scenes:

| Disposition | Scenes |
|---|---:|
| Complete windows passed | 9 |
| Explicit native AI pressure rejection | 3 |
| Independent pressure-certificate failure | 1 |
| Cancelled after the pool stopped | 15 |
| Not started | 46 |

The nine passing complete windows have22 independent checks;233 were scheduled for
the complete campaign. This is not a9/74pass rate. Cancellation and unstarted
scenes are not model failures. No population gate, global accuracy improvement,
complete call-count comparison or unused-scene eligibility is asserted. The
retained control had74 complete passing scenes and 233 checks; it was reused, not
rerun, after exact comparability/provenance qualification.

All three native rejections occurred at step 16 after 15 native warm-up steps, with
previous-pressure history present. The exact native-control physical frames
reproduce the retained control, so missing warm-up context does not explain them.

| Scene | Calls | Relative equation residual | Same-request native pressure relative L2 |
|---|---:|---:|---:|
| 1 | 32 | 0.0265474 | 0.0112607 |
| 5 | 32 | 2.96207 | 1.16798 |
| 6 | 32 | 0.0163656 | 0.00618757 |

The pressure column is Rheon's same-request native diagnostic, distinct from the
independent spectral-oracle measurements. Scene5's value exceeds1; this candidate
can diverge under repeated correction despite finite training state.

Scene7 completed all 81 AI steps under native physical gates. Its independent
operator defects were3.7e-15–8.9e-15, and every reported pressure bound covered the
oracle error. The required conservative certificate still failed:

| Step | Calls | Measured mean-free pressure relative error | Certificate | Gate |
|---|---:|---:|---:|---|
| 16 | 31 | 4.71836e-8 | 9.76853e-7 | pass |
| 64 | 32 | 5.50476e-8 | 1.13969e-6 | fail |
| 96 | 32 | 7.77920e-8 | 1.61063e-6 | fail |

The certificate limit remains1e-6. The measured errors at these captures are below
that target; do not describe this as measured pressure error exceeding1e-6.
Callback quality and the saved-field replay match exactly. The raw scene 7
infrastructure classification and pool cancellation receipts remain unchanged,
with this gate diagnosis retained separately. Stop rather than alter the gates.

## Fixed-input diagnostics

The144 shared training RHS pairs and 252manufactured frequency/amplitude cases
completed before the broad campaign. These diagnostics advance no simulations
and execute no optimizer updates. Two-call mean-free pressure errors on the
shared pairs were mixed:

| Stage | Control median/max | Candidate median/max |
|---|---|---|
| Early | 0.0107091 /0.196360 | 0.0159305 /0.184245 |
| Late | 0.0204805 /0.169402 | 0.0236160 /0.142037 |

Medians worsened while these maxima improved; this does not establish a rollout
improvement. On pure-low manufactured1Pa modes after 12 calls, candidate median/max
relative pressure error was7.26e-9 /6.92e-5, versus control2.18e-11 /5.59e-10.
The three amplitude scales have similar relative-error behavior; these stationary
spatial tests do not establish a temporal sampling or causal noise-floor effect.
Full per-amplitude tables and CSV are retained.

## Failures, timing and custody

The first CPU campaign stopped on the obsolete rejection-record classifier:
27.455s inside the benchmark,27.694s for the outer phase. The local repair preserves
all canonical numerical sources and accepts only explicit current nested native
rejection evidence. Eleven infrastructure/malformed/timeout/memory/fallback
negatives remained infrastructure. Qualification03 took9.856s; three prefix
admission negatives also passed. An earlier qualifier compared a fresh process
receipt hash as a physical metric; its failure and corrected physical comparison
are retained. Historical and failed receipts were not replaced.

The fresh campaign stopped at the independent gate after 98.441s inside the
benchmark,98.601s for the phase and 98.603s for its CPU owner (outer command98.728s).
Available memory remained at least51,491,577,856 bytes; the monitored conservative
worker RSS sum peaked at 7,342,710,784 bytes with 16 workers. These are CPU campaign
monitoring scopes, distinct from trainer CUDA allocation and sparse machine-wide
samples. No OOM occurred. Saved-field replay02 took8.925s; partial analysis took
0.071s. Serial paired/raw timing was not admitted after the campaign failure;
there is no new qualified inference-speed result.

All CPU/native children drained, controllers/custodians ceased, and the original
Tuldok and Rheon app processes remained open with unchanged start identities.
No background observation service remains. Small receipts and methods are in
this directory; weights, captures, private records, original logs, full commands
and archived command sources remain in workspace-root
`runs/pressure-coarse-full-fit-v1`.

The next numerical work should first reproduce the failing RHS correction curves
and frequency growth, then test a bounded contraction/stability change to the
coarse correction under matched training. More updates alone are not supported
by this result. Keep the retained control and qualification thresholds fixed.
