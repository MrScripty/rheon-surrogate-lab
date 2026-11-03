# Diffusion for learned substep corrections

Date: 2026-10-21. Research only; no model, training, or qualification changes.

**Recommendation:** continue directly supervised, deterministic missing-correction training first. Diffusion is worth retaining as an accuracy research option, especially through denoising training or a small learned refinement pass. A conventional many-pass diffusion sampler is a poor first choice for the current latency objective.

## What diffusion would mean here

A conditional diffusion model learns to remove noise from a candidate future field while conditioning on the current state, grid properties, and prescribed controls. Its denoising iterations are additional neural evaluations of the same future physical state; they are not simulation substeps and do not automatically replace them cheaply. This is also distinct from physical diffusion or numerical smoothing.

The project objective supplied for this assessment is worst equation/pressure/velocity/tracer error at most `1e-5`, with complete GPU generation time at most `1.2 ms` per physical step, for blocks K=1–4. The question is therefore whether an extra learned evaluation removes more internal simulation work than it costs, while retaining the required trajectory accuracy. That is an experimental question, not a consequence of choosing diffusion.

## What the primary literature supports

- **PDE-Refiner is the closest precedent.** It first predicts the next state, then performs noise-conditioned refinements at decreasing amplitudes. Its motivation is recovering low-amplitude spatial information that ordinary MSE training can neglect. It demonstrates longer accurate rollouts on 1D Kuramoto–Sivashinsky and 2D Kolmogorov flow. In the reported A100 Kolmogorov test, four network calls cost 16.53 seconds versus 4.04 seconds for the one-call MSE model. The paper explicitly identifies inference cost as a limitation, and reports limited extrapolation assessment. Its own residual-error refinement comparison overfit on that setup; this is a useful warning about testing generalization, not evidence that our different residual task cannot work. [Lippe et al., sections 3–5](https://arxiv.org/html/2308.05732v2)

- **Autoregressive conditional diffusion is useful when multiple plausible futures matter.** A fluid benchmark evaluates incompressible wakes, transonic flow, and isotropic turbulence. It finds favorable stability and posterior samples, but substantially slower inference: roughly 20–100 times the single-pass baseline for the studied ACDMs. It also finds that adding training noise can stabilize direct predictors, without requiring a diffusion sampler at deployment. The benchmark uses rollout MSE, similarity and statistics, which do not establish our worst-field thresholds. [Kohl et al., sections VI and appendix C](https://arxiv.org/html/2309.01745v2)

- **Sparse observation is a stronger use case than fully known short-step prediction.** DiffusionPDE learns a joint prior over PDE coefficients and solutions, then combines partial observations with PDE guidance to infer missing information. This supports future infilling/inverse-problem research; it does not directly establish an efficient replacement for our fully observed, deterministic short-horizon evolution. [Huang et al., sections 1 and 3](https://arxiv.org/html/2406.17763v2)

- **A slower diffusion teacher can be compressed.** Progressive distillation halves a deterministic sampler's evaluation count repeatedly, demonstrating four-step image generation. The newer Phys-Instruct preprint applies physics-guided diffusion distillation to PDE fields with one-to-four generator evaluations. These establish plausible mechanisms, not a guarantee of our trajectory error or GPU latency. A teacher would need to improve paired correction accuracy before adding this training complexity is justified. [Salimans and Ho](https://arxiv.org/abs/2202.00512), [Kong et al.](https://arxiv.org/html/2602.03627v1)

## Practical experiment order

1. Establish the deterministic direct-correction baseline on the existing broad scene collection. Preserve scene-level separation and report results by cell density, grid dimensions/aspect ratio, forcing family and internal-substep regime. Include quiescent and no-motion controls. Reference future states may supply labels, never inference features.
2. If that baseline learns useful corrections but is fragile under rollout, compare small, explicitly scaled noise augmentation during training against identical clean training. Keep deployed inference deterministic and one-pass. This is a proposed ablation inspired by the literature, not a claim that full PDE-Refiner benefits survive removal of its refinement process.
3. If spectral or residual diagnostics identify persistent small-amplitude errors, try one compact conditioned refinement pass and measure its full GPU cost. Compare error reduction per millisecond against spending the same budget on more internal advances or a wider direct correction network.
4. Consider a diffusion/refinement teacher and deterministic student only after the teacher produces demonstrably better paired targets. Continue using reference labels and physical output losses so the student does not merely inherit teacher bias.

For any stochastic candidate, predeclare seeds and score individual trajectories and seed sensitivity. An ensemble average, a statistically plausible vortex pattern, or a small PDE residual does not establish accurate pressure, velocity and tracer along the particular reference trajectory. Ensemble evaluation can diagnose uncertainty offline, but its compute must be counted if required during deployment. Fixing a random seed makes a sampler reproducible; it does not make its prediction correct.

No reviewed source verifies our exact equations, boundary handling, forcing variety, grid densities, worst-case metric, or hardware budget. The strongest immediate opportunity remains teaching the missing corrections directly; diffusion-inspired training is a controlled follow-up rather than a presumed accuracy or speed improvement.
