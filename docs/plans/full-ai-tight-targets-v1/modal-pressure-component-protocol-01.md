# Learned modal pressure component: bounded first fit

Status: component experiment specified before any new optimizer updates. This is an accuracy/capacity probe for the pressure branch, **not** a complete K=1/2/3/4 simulation generator or a new target pass. Full-generator training remains subject to its own complete compute and evaluation contract.

The frozen pressure branch failed the new accuracy objective even with a correct native RHS. A full double-precision transform/multiplier path has now passed execution and precision controls, but those identity/random-multiplier tests did not learn a pressure response. This experiment tests learned coefficients without importing an analytic pressure inverse.

## Model and data

Use `LearnedModalPressure` from `experiments/full-ai-tight-targets-v1/learned_modal_pressure.py`: one free double log-gain per spatial mode for each of the four existing registered geometries. Initialize log-gains uniformly to zero, so non-DC gains start at one. Fix DC output to zero for gauge freedom. Decode with the full cell-centred orthonormal DCT-II or its qualified FFTW equivalent. No physical eigenvalues, reciprocal eigenvalues, numerical pressure solver, corrective projection or teacher field may construct inference weights or enter candidate inference.

These finite geometry tables do not establish unseen-shape generalization. A later geometry-conditioned architecture or additional registered tables would require its own evidence, especially for the higher-density phase. This component does not predict viscosity, advection, velocity, tracer or any future timestep.

The [fixed training view](../../../runs/full-ai-tight-targets-v1/spectral-training-view-01/manifest.json) contains 144 original native double pressure/RHS pairs: all 16 training scenes, states 1/17/33/49/65/81/97/113/128, densities 8/16 cells per metre, four existing shapes. Whole development/reserved scenes are excluded. Original native pressure and RHS remain double; the older full-state training view's float32 pressure is unsuitable for this test. Fixed preprocessing computes only unit conversion and basis projections, with source hashes and independent native forward-operator checks. It computes no ratios, fitted gains or inverse.

## Finite learning contract

Run one managed Tuldok CUDA fit, seed 2026102049, double parameters and arithmetic, no AMP or TF32, deterministic algorithms. Use all four geometry batches per update, equally weighted, for exactly **800 Adam updates**, initial learning rate 0.1 with cosine decay to 0.001. Register the exact source/config/inputs and actual CPU/CUDA constructor qualifications before admission. Wall allowance is 600 seconds; failure or timeout is retained rather than silently retried.

Inside that CUDA learning run only, construct supervision from the native coefficient pairs. For each non-DC sample/mode, accept a logarithmic gain label when pressure and RHS coefficients have the same sign and the RHS magnitude exceeds 1e-10 times that sample's RMS RHS coefficient. The label is `log(abs(native_pressure_coefficient)) - log(abs(native_q_coefficient))`. It is a supervised data label, not an analytically generated operator coefficient. Average squared log-gain error first over valid samples per mode, then equally over observed modes and geometries. Unobserved modes keep their declared initialization. Report coverage and final training errors; do not fill missing gains using an analytic inverse.

No local CPU/GPU fitting, closed-form parameter estimation or alternate credential workflow substitutes for this managed fit. CPU/CUDA forward/backward qualifications make **zero** optimizer updates and preserve parameter bytes.

## Exact restore and evaluation

Publish initial/final typed double parameter inventories, weights, normalization, training history and an actual CPU physical-pressure probe. Restore in a fresh managed CPU run using typed artifacts from the same training run and independently verify byte-identical inventories/artifacts and the complete CPU probe. Existing float32 inventory helpers must not downcast these double weights. Preserve source, input, run and artifact custody bindings.

After restore, evaluate the learned branch on native RHS from consumed development scenes, including all registered shapes and both densities through the 240-step bank. Check mean-free pressure error and the independent native equation residual at every scored state, with the separate double physical output. Compare Torch and FFTW deployment execution. These teacher-input component results remain offline diagnostics; passing them does not prove causal full-state or own-history accuracy.

If either pressure or residual remains above 1e-5, retain the failed component evidence and diagnose coverage/precision/capacity before expanding learning. If it passes, it becomes a candidate learned pressure branch for a separately registered causal architecture. That architecture must still generate all fields/intermediates, meet maximum complete-block CPU time, and pass own-history and fresh whole-scene confirmation. Higher-density training and 600-step campaigns remain gated by the original ordered program.
