# Primary sources and practical choices

Retrieved 2026-10-14. These publications motivate architecture and evaluation;
their benchmark gains are **not estimates for Rheon or Tuldok**. No pretrained
weights, paper code or external datasets are required or copied.

| Source | Supported observation | Consequence for this design |
| --- | --- | --- |
| [Tompson et al., Accelerating Eulerian Fluid Simulation With Convolutional Networks](https://arxiv.org/html/1607.03597v7), sections 2/4/5 | Multiresolution context and physics losses are used for pressure projection; discusses supervised-input/rollout mismatch | Use box-spanning context, full operator loss and independent autonomous rollout evaluation |
| [Ronneberger et al., U-Net](https://arxiv.org/abs/1505.04597), architecture | Contracting/expanding paths combine context and spatial localization | A small encoder/decoder is a practical implementation template; this is an architectural analogy, not pressure-quality evidence |
| [Li et al., Fourier Neural Operator](https://arxiv.org/abs/2010.08895v3) | Learns function-space mappings using Fourier-parameterized kernels, with PDE experiments | A plausible global alternative, but rectangular Neumann boundaries require explicit treatment; not an unqualified periodic FFT swap |
| [Kochkov et al., Machine learning accelerated computational fluid dynamics](https://arxiv.org/abs/2102.01010) | Learned components inside a conventional solver are evaluated for long-term stability and forcing/regime generalization | Retain numerical structure; measure long rollouts and distinct regimes, not snapshot loss alone |
| [PETSc KSPSetTolerances](https://petsc.org/release/manualpages/KSP/KSPSetTolerances/) | Residual tolerance differs from solution error and depends on norm conventions | Name full unpreconditioned residual, original RHS scale, absolute gate, maximum iterations and divergence independently |

Our inference from these sources and local evidence is that the smallest useful
next candidate is a multiscale **correction warm start**, rather than unconstrained
pressure replacement. Residual tests decide eligibility and the native solver
controls final quality. This is an experiment decision, not a theorem from a paper.

## Bounded comparison

**Conditioned modal model:** cosine transforms are naturally compatible with the
sealed box. Geometry/density can condition a small coefficient model; few parameters
and float64 CPU are attractive. But the exact eigenvalue response is available for
these uniform boxes. The existing learned fixed-operator result already explored
this prior. A second modal fit would add less evidence than a general convolutional
correction and would still need the analytic baseline.

**Selected multiscale model:** four downsampling stages, skip connections and a
global pooled bottleneck give every output access to the full box, unlike the old
radius-one differences. Native operator loss, fixed gauge and rejection/fallback
keep physics validation explicit. Uses eager CPU Torch convolutions/pooling only;
no compiler, FFT plugins or distributed training. CPU inference may erase any
iteration savings. Record that outcome.

**Deferred FNO:** global Fourier mixing is appealing for larger grids. Nonperiodic
Neumann geometry, truncation of high modes, complex parameter serialization and
CPU transform cost would require more independent qualification. It is not part of
a smoke architecture sweep. If this proposal fails, any FNO comparison is a new
reviewed experiment with new holdouts and compute bounds.

The mandatory analytic baseline must use separable cosine transforms and the
finite-volume operator scaling actually verified against Rheon. A dense
Kronecker matrix would handicap it. An optional optimized DCT implementation
may only enter if its dependencies/runtime are qualified before the freeze;
otherwise use separable dense 1D contractions, label that limitation and avoid
claims over the best available numerical method.
