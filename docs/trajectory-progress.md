# Time-evolving 3D surrogate: evidence and remaining work

The first pressure-only experiment is frozen and unchanged. It established no
simulator acceleration or advantage over the analytic spectral baseline. The
active user goal requires a meaningful learned solve inside a complete 3D
simulation, timed viewport/timeline proof, numeric accuracy, a reproducible book
chapter, PyTorch 2.14 through Pumas Library, and training/inference below 20/8 GB.

The current goal continuation inspected the checkout and found no running
training job. It made progress by adding the separate trajectory probe/model,
generating actual Rheon training trajectories, fitting on CUDA and evaluating
held-out rollouts. The earlier session's completion of the pressure experiment
is progress toward measurement infrastructure, but does not satisfy this goal.

## Experiment two

- `probes/trajectory` uses the sibling Rheon crate without editing it. Both
  methods call identical public 3D advection, projection correction and tracer
  kernels. A learned step receives only its own current preprojection RHS.
- The sealed unit cube uses 32³ cells, density 1.25 kg/m³, dt 1/60 s. Prescribed
  localized 3D forcing and small broadband stirring vary by trajectory seed.
  This is a passive-tracer incompressible flow, not a free-surface liquid.
- Training fits 32,767 modal coefficients from 12 × 8 reference frames, using
  supervised least squares in PyTorch 2.14.0+cu132. The cosine basis supplies
  geometry but no analytic inverse coefficients. All modes are trained; there
  is no PCG or analytic-inverse fallback at inference.
- Whole-trajectory seed splits: train 100–111, validation 500–501, test 900–903.
  Validation spans 60 frames; frozen test spans 120 frames per seed. Test data
  never enter fitting. Coefficients and conditions are frozen before testing.
- `runs/trajectory-v1` is an incomplete development attempt: its original forcing
  exceeded the declared Courant gate in the reference validation rollout. It is
  not successful evidence. Experiment v2 lowers forcing before accessing test.
- v2 validation velocity error ≤7.67e-8. CUDA training tensor allocation peak
  261,103,616 bytes; Python RSS peak 1,495,904,256 bytes. Untrained unit gains fail
  the stored-divergence gate, showing learned coefficients are essential.
- GPU access requires approved host execution in this Codex session; the default
  sandbox has no `/dev/nvidia*`. Host and Pumas both identify an RTX 5090 Laptop
  GPU with 24,463 MiB. No GPU installation or driver change was necessary.

## Completion evidence

| Requirement | Authoritative evidence |
| --- | --- |
| Meaningful AI work | Trained 32,767-parameter checkpoint; zero AI PCG iterations; untrained ablation rejected; pressure was 77% of Jacobi frame computation |
| Faster complete simulation | Parallel medians 16.35 s Jacobi / 9.67 s SGS / 1.76 s AI for 120 frames; 9.30× / 5.50× speedups, with validation-selected 4/8/8 CPU threads |
| Accuracy | Independent full-field replay at selected thread counts across four seeds against both references; worst velocity relative L2 error 9.00e-8 |
| Viewport and timeline | Real-field synchronized voxel viewer; load/play/scrub/orbit/mobile checks and preserved screenshot; live localhost server |
| Book coverage | Chapter 41 installed in requested repository; current Markdown/web, chapter PDF and eight-script physics companion; 50 HTML pages and 5,479 internal links/assets validated |
| Pumas/PyTorch 2.14 | Launcher resolves selected Pumas environment; actual PyTorch 2.14.0+cu132 and CUDA device verified; fit, rollouts and fresh process runs executed there |
| Memory budgets | Training 264 MB CUDA reserved / 1.50 GB Python RSS; evaluation 37.7 MB CUDA reserved / 2.00 GB RSS; fresh inference 942 MB RSS |

The first experiment's 7,200 timed frames and the parallel comparison's 4,320
timed frames all passed physical gates and observed strict relative residual
≤1e-8. Test trajectories were independent rollouts rather than teacher-forced
predictions. The model's fixed-operator scope remains explicit.

The final rejection-path test found a buffered-pipe cleanup race. The current
driver catches close-time BrokenPipeError and always reaps its child. Four
integration tests and a 20-rejection check pass. Exact original timed-driver
bytes are preserved and hash-verified under the frozen experiment; checkpoint
and numerical kernels were not changed. Current-driver fresh-process inference
also qualified all frames, with 3.20 s median including imports/CUDA/model
initialization and Python teardown.

All preserved evidence is in `experiments/trajectory-v2`. Bulk reference and AI
fields remain in `runs/trajectory-v2`. The installed-book receipt verifies all 63
changed files against the staged diff and protects concurrent edits. Original
PDF/companion archives retain their bytes and are labelled historical. No GitHub
publication was requested or performed.

The frozen checkpoint is geometry-specific and much smaller than the user's
suggested 500M starting size. More parameters do not expand its fixed-operator
contract. Any claim about variable geometry, free surfaces, viscosity or general
3D physics would need a different model/data experiment. The actual instruction
suggested 500M; it did not make that count a hard requirement.

## CPU-only control after the hardware-comparison question

The user questioned comparing GPU model inference against CPU Rheon. The
unchanged checkpoint was subsequently run entirely on CPU, with CUDA hidden
before imports, all model tensors asserted on CPU, and identical eight-thread
limits for native kernels and PyTorch inference. Three balanced repetitions over
four 120-frame trajectories measured 15.21 s Jacobi, 9.39 s SGS and 1.86 s learned:
8.20× / 5.06× speedups on CPU alone. All 4,320 timed frames met observed strict
relative residual ≤1e-8 and stored divergence ≤1e-5 s⁻¹. Full-field independent
replay against both references found maximum velocity relative L2 error 9.00e-8.
Python peak RSS including accuracy replay was 972,546,048 bytes.

`experiments/trajectory-cpu-v1` preserves the new CPU freeze, results, verification
and labelled viewer screenshot. The default viewport now replays CPU-only fields;
the Comparison selector explicitly labels and switches to the earlier GPU
recording. These exposed seeds compare backends, without retraining or choosing
model parameters from test results. The book's chapter and companion are updated
on its separate branch. Superiority over an analytic spectral solver remains
unestablished.
