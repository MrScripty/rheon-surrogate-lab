# Complete the learned recurrent-sampler prototype

The user explicitly directed completion and testing of this prototype before
moving to another architecture. This overrides the earlier specification's
instruction to stop before training after a failed 1.2 ms cost gate. It does not
change the staged program's accuracy, latency or independent confirmation gates.
No different coordinate architecture will be opened during this completion.

The first complete untrained implementation costs approximately 16–18 ms per
32-cubed state on core 23. Its 48 new double parameters (12 intermediate-query
gates and 36 velocity-mixing coefficients) have not been fitted. Existing point,
viscosity, flux and modal-pressure parameters retain managed-learning provenance.

1. Finish qualification and timing of the versioned integer-address compiler
   optimization. It changes source addressing only; preserve learned coefficients,
   floating-point operations, full input checks, complete outputs and old evidence.
2. Prepare train-only local field samples and intermediate/final transport labels
   from the original 16 native training scenes at densities 8/16 and all four
   existing shapes. The native teacher remains offline. Predict query locations
   with the learned point projection and train all 48 new coefficients through
   managed CUDA. No analytical coefficients or local parameter fit may substitute.
3. Qualify the actual differentiable objective on CPU/CUDA without optimizer
   updates. Freeze the finite fit, sources and new whole-scene reservations before
   managed admission. Restore exact typed double parameters through a separate
   managed CPU Run and verify all artifacts and CPU probes.
4. Score the trained complete generator on consumed development scenes first,
   including every intermediate pressure, velocity and tracer field and native
   equation residual for K=1/2/3/4. Test 240 actual own-history steps, prescribed
   forcing changes and small/odd-domain regressions. Retain failures.
5. Freeze selected weights and deployment sources before independent reserved
   whole-scene confirmation. Rebenchmark complete K blocks on core 23 and measure
   matched native PCG and spectral-reference application costs. Report actual
   accuracy, drift and complete clock scopes even if the target remains unmet.

The 1.2 ms/1e-5 targets remain unpassed. Higher densities and the 600-step
optimization phase remain gated. Completing this prototype is a research
deliverable, and does not establish completion of the wider active objective.
