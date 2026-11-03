# Pressure hybrid options: evidence and hypotheses

Research date: 2026-10-18. External links checked on that date. This note proposes experiments; it reports no new solver measurements, training, or promotion.

## What would actually change

The captured current [physical residual implementation](../../../runs/pressure-detached-correction-v1/c53-u-project-02/physical_residual.py) solves corrections for the uniform sealed cubic finite-volume operator `A=(h/rho)L`. It feeds the network `q=(rho/h)r`, then recomputes the physical residual between corrections. Thus cell spacing already affects assembly; a spacing input channel would be additional conditioning. Algebraically, for this exact operator, `L delta_p=q` removes the scalar spacing/density dependence. Such conditioning should have a specific hypothesis rather than assume missing physics. Obstacles, free surfaces, and variable coefficients require a broader operator contract.

The earlier [spatial coarse branch](../../../runs/pressure-spatial-coarse-v1/candidate-project-01/coarse.py) restricted to four bins per axis, applied a **learned 64-by-64 matrix**, then interpolated. Its [architecture declaration](../../../runs/pressure-spatial-coarse-v1/candidate-project-01/model.py) explicitly excludes an inverse solver. A numerical coarse residual solve is therefore a different experiment. Fully learned coarse/fine mixing has precedent here; calling another learned coarse branch “new” requires identifying its changed information or operator.

The subsequent [full coarse fit](../../../runs/dependency-migration-v1/lab/docs/plans/pressure-coarse-full-fit-v1/results.md) completed all 5,000 updates and reproduced its restored artifacts. CPU qualification nevertheless stopped with three native pressure rejections and one conservative pressure-certificate failure. The certificate failure occurred despite measured pressure errors below its threshold; the rejected scenes are separate failures. No qualified inference-speed improvement was established. Thus the learned coarse approach has a completed negative qualification result, rather than merely an interrupted training attempt.

## Relevant primary evidence

- **Tompson et al., ICML 2017:** learned pressure projection within a conventional fluid pipeline, with a divergence-based objective and multiple-frame training. Their experiments compare both fixed compute and fixed divergence, and distinguish projection timing from total simulation time. This supports physical residual and rollout evaluation, not transferring their speedups to Rheon. [Paper and publication details](https://proceedings.mlr.press/v70/tompson17a.html), [full paper](https://proceedings.mlr.press/v70/tompson17a/tompson17a.pdf).
- **Zhang et al., HINTS, submitted 2022:** alternates neural residual corrections with numerical relaxation to address complementary error modes; stopping uses the PDE residual. Its examples support testing mixed numerical/neural iterations, not assuming that any pairing converges. The authors also report an example dominated by interpolation cost. [Paper](https://arxiv.org/html/2208.13273v2).
- **Um et al., NeurIPS 2020:** solver-in-the-loop training exposes the network to states affected by previous corrections and improves their tested rollouts. This concerns correction under recurrent physical evolution; detached pressure-correction training alone does not establish equivalent trajectory behavior. [Paper](https://arxiv.org/abs/2007.00016).
- **PETSc:** multigrid requires coarse operators, transfer operators, smoothers, and residual evaluation. A Galerkin coarse operator is `Ac=R A P`; rediscretization is a separate construction. Neumann Laplacians have a constant nullspace and require a compatible right-hand side. [Multigrid and singular-system manual](https://petsc.org/main/manual/ksp/), [Galerkin API](https://petsc.org/main/manualpages/PC/PCMGSetGalerkin/).

## First diagnostic: numerical low-mode correction without training

**Hypothesis:** supplying a cheap numerical solution of the poorly corrected large-scale modes can improve retained C53's worst physical error per total CPU time.

Use the *same fine-grid* operator and request residual `r=b-Ap`. Construct orthonormal cell-centred cosine columns `P`, with per-axis mode indices `0..min(3,n-1)`, excluding the all-zero mode. Apply

`p <- p + P (P^T A P)^(-1) P^T r`,

then recompute the full fine residual before any neural call. This is a small Galerkin subspace correction, **not a lower-resolution CFD simulation**. For this sealed constant-coefficient stencil the cosine eigenvalues can evaluate the small inverse directly. The repository already has [analytical modes](../../../runs/pressure-detached-correction-v1/c53-u-project-02/analytical.py) and [orthonormal low-mode coefficients](../../../runs/pressure-detached-correction-v1/c53-u-project-02/band_feedback.py); independently check eigenpair action against the actual physical operator before using them.

Clip counts independently for odd and thin grids; a one-cell axis contributes only its constant factor. The all-one-cell case has no nonconstant correction. Exclude DC rather than invert its zero eigenvalue. Keep the existing compatibility rejection and pressure gauge; do not silently subtract an incompatible RHS and then claim to solve the original request. Pure in-span modes should be solved almost exactly by construction: these are implementation checks, not generalization evidence.

Compare fixed C53 calls `K=1,2,4` against the same calls plus one predeclared numerical correction, coarse-only, and numerical-only controls. Include held-out mixed modes and real request residuals across shapes. Report worst fine-grid equation residual, gradient/velocity error, amplification, and every existing acceptance gate. Measure transfers, residual assembly, conversion, checks, reconstruction, and neural work; disclose cold setup and cached reuse separately. A Python diagnostic can establish a numerical signal but cannot establish native deployment speed. Only a favorable signal justifies architecture training.

## Sparse anchors and literal coarse grids

Sparse pressure values are globally coupled unknowns, not independent voxel calculations. Obtaining exact anchors by a full solve cannot be counted as cheap inference. Eliminating remaining unknowns produces a Schur-complement problem; approximate local solves need specified interface conditions. An oracle-anchor study could test information sufficiency, clearly excluding deployable speed claims. This is a derived design constraint; PETSc's [block solver discussion](https://petsc.org/main/manual/ksp/) supplies the numerical framework.

A literal coarse finite-volume solve needs conservative restriction and geometry-aware prolongation. On odd grids, unequal edge aggregates have different volumes; on thin grids, avoid redundant coarse unknowns. Preserve constants and boundary fluxes, and either form a consistent `RAP` operator or specify the rediscretized geometry, weights, and nullspace. Reusing average restriction and linear interpolation does not itself establish consistency. A highly accurate coarse solve still leaves fine-space error; numerical smoothing or learned correction must be assessed against the fine equation.

## Grid and temporal data changes

Separate counts from cells/metre: increasing counts at fixed spacing enlarges physical extent; increasing counts at fixed extent refines spatial sampling. `128^3` already exceeds the captured [65,536-cell cap](../../../runs/pressure-detached-correction-v1/c53-u-project-02/policy.py). Larger datasets are therefore a resource/contract change, not an inexpensive accuracy knob.

Two `dt/2` teacher steps provide an alternative endpoint to one `dt` step, not automatically the pressure label for that coarse request. **Derived from operator splitting:** advection and forcing also change, and their endpoint differences need not be representable by a pressure gradient. For pressure-only training, collect fine-step states but generate labels for the learner's actual coarse-step pressure equations. Endpoint imitation is a broader update-learning experiment.

First compare `dt`, `dt/2`, and `dt/4` at matching physical times, fixed mesh, and sufficiently converged pressure solves. NASA's 2011 study explicitly separates temporal convergence from spatial convergence; timestep sensitivity is not exact-solution error. [Primary study](https://www.nas.nasa.gov/assets/nas/pdf/papers/housman_j_space_time_accuracy_2011.pdf). Measure pressure residual per request separately from velocity/tracer drift over a declared physical horizon. A later accurate projection does not undo prior trajectory error; no universal safe step count follows from a residual threshold.
