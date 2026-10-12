<a id="project-train-a-3d-fluid-pressure-solver"></a>
# 41  Project train a 3D fluid pressure solver

<a id="physics-start-with-the-job"></a>
## Start with the job the model will replace

A fluid simulator stores a state, advances that state through a small time interval, checks the candidate result, and publishes the accepted new state. For an incompressible flow, pressure helps prevent fluid from accumulating or disappearing inside a cell. A useful AI acceleration project begins by naming one expensive stage and preserving the surrounding numerical contracts.

This project trains a pressure prediction model and inserts it into a real three-dimensional simulation. The numerical reference is [Rheon](https://github.com/MrScripty/Rheon), a Rust fluid toolkit. The executable companion is the separate [Rheon Surrogate Lab](https://github.com/MrScripty/rheon-surrogate-lab). Its `probes/trajectory` and `scripts/trajectory_model.py` are the authoritative implementation for this chapter. Earlier pressure-only experiments in that repository are historical; their small isolated fields did not establish acceleration of a complete simulation.

The current executable files are included in the book's [3D physics companion](../downloads/rheon-physics-companion.zip). Use that download if the lab's public checkout still contains only its earlier pressure experiment. The book update does not imply that new source was pushed to GitHub.

The supported scene is deliberately precise: a sealed one-metre cube, a uniform 32 × 32 × 32 grid, density 1.25 kg/m³, and a time interval of 1/60 second. Prescribed acceleration injects momentum in all three directions. A passive concentration field makes the flow visible. It is a marker carried by fluid that fills the entire box. It is not a liquid surface or a changing liquid volume. This example does not implement obstacles, viscosity, variable density, moving walls, gas compression, splashes or surface tension.

The measured candidate has 32,767 trainable parameters. A suggested starting size such as 500 million is a capacity hypothesis, not a physical requirement. Here the fixed pressure operator has a structure that makes one coefficient per nonconstant global mode sufficient. Increasing this count without changing the task supplies no missing geometry or physics. This is a narrowly scoped learned linear operator, rather than a large neural simulator that learns all dynamics.

Learning pressure inside a conventional simulation is an established research direction. Tompson and colleagues used a convolutional network for pressure projection in two and three dimensions. Their architecture and experimental results differ from this example. Fourier neural operators are another operator-learning family; the modal regression below is simpler than a multilayer FNO and makes no claim to its resolution or parameter generalization. [Tompson et al., 2017](https://proceedings.mlr.press/v70/tompson17a.html), [Li et al., 2021](https://arxiv.org/abs/2010.08895v3).

<a id="physics-understand-the-fields"></a>
## Understand what a three-dimensional field contains

A cell-centred scalar assigns one number to each of 32³ = 32,768 cells. Pressure is stored there. Concentration is another cell-centred scalar. Velocity uses a staggered MAC grid: its X component is stored on X-normal faces, its Y component on Y-normal faces, and its Z component on Z-normal faces. Each component therefore has 33 × 32 × 32 = 33,792 values, with the longer axis depending on the component.

The simulation stores velocities and concentration as float32, pressure and the pressure right-hand side as float64. The array order has X varying fastest: `i + nx * (j + ny * k)`. A flattened cell field reshapes to `[z, y, x]` in NumPy or PyTorch. Do not reshape a face component as if it had the cell shape. Equal element counts also do not make differently placed fields interchangeable.

At a sealed wall, the normal velocity is zero. Let `u*` be the candidate velocity after advection and forcing. In this Rheon operator, the pressure system is `K p = b`. The operator sums differences in pressure across neighbouring cells, with a coefficient `face_area / (density * distance)`. The right-hand side sums inward minus outward velocity flux, divided by the time interval. It has units m³/s². Pressure has units pascals.

After predicting pressure, velocity is corrected by the pressure gradient:

```text
u_new = u* - (dt / density) * gradient(p)
```

The sign and geometry must match the implementation that created the labels. A pressure model trained with the opposite sign can minimize its own incorrectly defined loss and still push fluid in the wrong direction.

Pressure in this closed box is determined only up to an additive constant. Adding the same number to every cell leaves its gradient unchanged. The reference sets cell zero to zero. The model subtracts its predicted cell-zero value. For pressure accuracy, compare mean-centred fields so the metric does not mistake a harmless gauge difference for a physics error.

<a id="physics-generate-real-trajectories"></a>
## Generate the data from real simulation trajectories

The native probe calls Rheon's public velocity advection, pressure solve, velocity correction and tracer advection functions. Each frame begins from the previous accepted state. Training data are saved at the point immediately before projection: the real preprojection right-hand side is the input, and Rheon's pressure is the target. Do not infer the input from the corrected velocity. Projection has already removed information from that field.

The fixed source follows a seed-dependent moving centre. Its smooth acceleration is accompanied by small deterministic broadband stirring. This is part of the declared scene for every split and every solver. The stirring excites high-frequency pressure modes, so a mode is not left untrained merely because a smooth plume barely uses it.

The split is by whole trajectory: seeds 100–111 supply 12 training trajectories with eight frames each; seeds 500 and 501 supply validation trajectories with 60 frames each; seeds 900–903 supply test trajectories with 120 frames each. All frames from a trajectory stay together. Adjacent frames are strongly related; randomly splitting neighbouring frames would make a generalization claim misleading.

These test seeds check new forcing realizations within the declared scene family. They do not test new boundary types, grids or geometries. The saved manifest records grid, units, seeds, timing reports, runtime version and source/binary hashes. The checkpoint and test conditions are frozen before any test trajectory is generated.

<a id="physics-learn-global-pressure"></a>
## Learn the global pressure response

Pressure depends on the entire connected fluid domain. A tiny local convolution cannot generally infer the effect of distant flux from an insufficient receptive field. The first lab's local CNN failed its numerical checks. This example uses a global cosine basis tailored to the uniform sealed box.

In one dimension, the orthonormal basis contains cosine values at cell centres:

```text
Q[k, i] = sqrt(2 / n) * cos(pi * k * (i + 0.5) / n)
Q[0, i] is additionally divided by sqrt(2)
```

Apply this basis along X, Y and Z. A basis coefficient describes the contribution of one global spatial pattern. Transforming the field does not supply a pressure solution: the model still has to learn how strongly each input mode should affect pressure.

The model has a trainable gain for every nonconstant 3D mode. Transform the right-hand side, multiply each modal coefficient by its learned gain, transform back, and apply the gauge. The constant mode is omitted because it is the pressure nullspace. There are `n³ - 1` trainable gains.

This is supervised linear regression. For training examples indexed by `t`, let `x[t,k]` be input mode `k` and `y[t,k]` its target pressure mode. The least-squares fit is:

```text
gain[k] = sum_t(x[t,k] * y[t,k]) / sum_t(x[t,k] * x[t,k])
```

The denominator measures training excitation. A zero denominator means that mode was not learned; reject the training run. Nonfinite or nonpositive gains are also rejected in this fixed positive pressure operator. The calculation uses only training trajectories. It contains no analytic inverse eigenvalues, validation labels or test labels.

Training does not require gradient descent when a closed-form optimum is available. `ModalPressure` is a PyTorch `nn.Module` with a real `nn.Parameter`, but its fit uses supervised sufficient statistics rather than Adam. There are zero optimizer steps and no optimizer-state memory. This is still a fitted model: without the learned coefficients, its predictions fail the physical acceptance gate. Report the method plainly rather than describing a least-squares fit as a deep-network training run.

At inference, there is no Rheon pressure solve, analytic inverse, access to teacher pressure, or PCG correction of the learned result. Rheon still performs velocity advection, forcing, gradient correction, tracer transport and acceptance checks. The model therefore replaces a meaningful numerical stage while retaining the simulator's other operations.

An analytic spectral inverse is available for this particular box. Learning it is useful as a controlled end-to-end example; it does not establish an advantage over that analytic method. If an analytic solver already meets a production task, use it. To extend the learned contract to obstacles or variable coefficients, the current fixed diagonal gains are insufficient: collect those conditions as inputs and labels, use an architecture with enough global context, and repeat the entire validation and rollout benchmark.

<a id="physics-run-through-pumas"></a>
## Run through Pumas Library's managed PyTorch

Select the installed PyTorch **v2.14.0** runtime in Pumas Library. This project uses the Python environment managed by Pumas; it does not install packages into your system Python or treat the weights as a language model. In the measured environment, the actual package is `2.14.0+cu132`, Python is 3.14.7, and the device is an RTX 5090 Laptop GPU with 24,463 MiB VRAM.

The launcher reads Pumas' `launcher-root.json`, verifies `.active-version-torch`, and probes the managed interpreter's actual PyTorch version. An explicit `--pumas-root` supports another installation location. It does not change the active runtime. This is an executable adapter to the managed interpreter, not a claim that Pumas' image-generation or chat interface natively serves fluid fields.

Keep sibling checkouts named `Rheon` and `rheon-surrogate-lab`. The trajectory probe's path dependency refers to the sibling Rheon checkout. The v2 experiment used Rheon commit `2209012bc5f902693d0f695ec1c2d854126af317`; exact numerical-source hashes are recorded because unrelated local documentation changes existed in that checkout. Verify your current source against the receipt or make a clearly labelled replication with a new provenance record.

Extract the physics companion into a new working directory. It creates `rheon-surrogate-lab`. In that same working directory, obtain the reference checkout:

```sh
git clone https://github.com/MrScripty/Rheon.git Rheon
git -C Rheon checkout 2209012bc5f902693d0f695ec1c2d854126af317
cd rheon-surrogate-lab
```

The Rust prerequisite is the pinned 1.92.0 toolchain, which `rust-toolchain.toml` selects through rustup. Pumas supplies PyTorch 2.14 and NumPy. Keep the unpacked `scripts`, `probes`, `viewer` and toolchain file together; this is a complete small project, not a request to reconstruct source from printed fragments.

The original timed driver is preserved with the historical receipts. The current companion fixes a process-cleanup race after rejected predictions; weights and numerical kernels are unchanged. Historical hashes identify the original bytes, while new runs record their current driver. A source hash mismatch must be explained and preserved, never silently rewritten to make a receipt appear current.

From the lab root:

```sh
export CARGO_TARGET_DIR="$PWD/target"
cargo build --release --locked --manifest-path probes/trajectory/Cargo.toml
python3 scripts/pumas_trajectory.py
python3 scripts/pumas_trajectory.py train runs/my-fluid --device cuda
python3 scripts/pumas_trajectory.py evaluate runs/my-fluid
python3 scripts/package_trajectory_viewer.py runs/my-fluid/evaluation
python3 -m http.server 8098 --bind 127.0.0.1
```

Open `http://127.0.0.1:8098/viewer/?run=my-fluid`. Output directories must be new; the scripts refuse to replace an experiment. The first Cargo build needs the pinned small Rust dependencies. The training and inference scripts use PyTorch and NumPy already available in the managed environment.

If `torch.cuda.is_available()` is false, inspect device access as well as the installed build. The authoring Codex sandbox did not expose `/dev/nvidia*`; an approved host execution could access the same installed GPU and runtime. That is different from having no GPU or needing a driver reinstall. On your own machine, run the launcher from a terminal with normal GPU access.

Use `model.eval()` and `torch.inference_mode()` for predictions. CUDA work is asynchronous, so a CPU clock around a launch alone is not a completed-inference measurement. This driver copies the prediction back to CPU before handing it to Rust; that transfer waits for the result. It explicitly synchronizes CUDA around standalone training and warmup measurements. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html).

<a id="physics-check-accuracy-over-time"></a>
## Check accuracy over time, not just one predicted field

Each AI trajectory starts from the same zero state and receives the same prescribed forcing as its numerical reference. Every AI frame then uses its own previous accepted velocity and concentration. No teacher state is supplied between steps. This is an independent rollout, which exposes accumulated error that a sequence of teacher-forced single-step predictions would hide.

Use several metrics with explicit definitions:

- Relative velocity L2 error: `||u_AI - u_reference||₂ / ||u_reference||₂`, concatenating all three full MAC face arrays.
- Relative tracer L2 error and tracer RMSE, measured on every cell.
- Mean-centred relative pressure L2 error.
- Full pressure residual, including the gauge cell, relative to the right-hand side.
- Maximum divergence computed from the corrected stored float32 faces, rather than inferred only from a float64 pressure residual.

Both paths must pass a maximum stored-divergence gate of `1e-5 s⁻¹` and full relative-residual gate of `1e-5`. Native PCG is configured more strictly, with relative residual `1e-8`, absolute residual `1e-12`, and predicted divergence `1e-7 s⁻¹`. This extra native work must be stated in interpreting the timing. The model does not silently repair a failed prediction: it rejects the step. Concentration must remain finite and in [0,1], and the declared Courant measure must stay at most one.

In the completed first benchmark, the AI's largest observed relative residual was `6.73e-9`, so every measured AI frame also met the reference's stricter relative-residual tolerance. Across all four independent test trajectories, maximum corrected AI divergence was `1.43e-6 s⁻¹`, and maximum relative tracer error was `5.89e-8`. The independent artifact audit recomputes errors using float64 norms; the original receipt used a float32 reference norm, causing only a small reduction-precision difference in the last reported digits.

Do not convert an error to a generic “physics accuracy percentage.” Relative L2 agreement measures this discretized numerical reference, not experimental truth, turbulence fidelity or conservation of liquid mass. The passive tracer advection here is semi-Lagrangian and is not a conservative liquid-volume method.

The 120-frame test contains two seconds of simulated time per seed. The initial test rollouts produced maximum relative velocity errors from `8.23e-8` to `8.89e-8` across the four seeds. These small errors are plausible because the fixed-box inverse is diagonal in the chosen basis; they do not imply the same accuracy on unmodelled physics. Full arrays and per-frame errors are saved so the comparison can be recomputed.

<a id="physics-benchmark-the-whole-simulation"></a>
## Benchmark the whole simulation and show its timeline

The relevant time begins before requesting a frame and ends after receiving its accepted result. Counting only matrix multiplications would omit advection, Python/Rust transfer, GPU transfer, pressure-gradient correction and error checks. This experiment times complete 120-frame rollouts with a resident model. Each rollout includes native process startup, all physics stages, inference, transfers, qualification, result parsing and native teardown. It excludes Python imports, model construction, explicit warmup, recording the full visualization arrays and rendering. Those are separate costs, not zero costs.

Compare both Rheon Jacobi PCG and symmetric Gauss–Seidel PCG. Rotate method order across five repetitions and four seeds. Report medians and preserve every repetition, rather than selecting a favourable frame. The first native probe is single-threaded. The separate `trajectory-parallel` probe enables Rheon's real parallel kernels, tunes thread counts using validation seed 500 only, and records a separate frozen CPU-execution comparison. A single-thread timing should never be advertised as a comparison against every possible CPU configuration.

The completed single-thread native benchmark recorded median complete 120-frame times of 22.01 seconds for Jacobi, 19.08 seconds for SGS and 5.18 seconds for the learned path. The learned path was therefore 4.25× faster than Jacobi and 3.69× faster than SGS in that declared setting. This is time to compute two seconds of simulated flow, not a claim to real-time 60 fps. The learned path performed zero PCG iterations; pressure accounted for about 77% of the Jacobi reference's native frame calculation time. Timing varies with hardware, OS scheduling and background activity. Full repetitions remain in `evaluation/result.json`.

For the separate parallel comparison, build the second native probe and run the driver with the managed interpreter printed by the launcher:

```sh
cargo build --release --locked --manifest-path probes/trajectory-parallel/Cargo.toml
PATH_TO_PUMAS_PYTHON scripts/trajectory_parallel.py runs/my-fluid
```

It tunes thread counts 1, 2, 4, 8 and 16 on validation data, freezes those choices, then records three complete repetitions on the four declared test seeds. Its results belong to `parallel/result.json`, with a separate timing scope. They do not overwrite the first benchmark or change its checkpoint.

On the measured machine, validation selected four native CPU threads for Jacobi and eight for SGS and the learned path's remaining Rheon stages. Median complete 120-frame wall times were 16.35 seconds, 9.67 seconds and 1.76 seconds respectively. That is 9.30× faster than Jacobi and 5.50× faster than SGS. The comparison includes each solver's selected thread count; it does not force the numerical reference to remain single-threaded. There is no universal speed guarantee for other processors, scenes, resolutions or memory layouts.

An independent full-field replay at those selected thread counts compared all four test seeds against both CPU implementations. Its worst relative velocity L2 error across both references was `9.00e-8`. All 4,320 timed parallel frames passed the stored-divergence gate and the strict observed `1e-8` relative-residual tolerance. This extends the accuracy evidence to the execution settings used in the parallel speed comparison.

The comparisons above run model inference on the RTX GPU while Rheon runs on the CPU. Their speedups combine an algorithm change with hardware acceleration. To separate those effects, the unchanged checkpoint was also evaluated entirely on CPU, with every method using eight native CPU threads and the model using eight PyTorch CPU threads. CUDA devices were hidden before importing PyTorch, and every model parameter and buffer was checked to reside on CPU.

Across three balanced repetitions and the same four 120-frame trajectories, median complete times were **15.21 seconds for Jacobi, 9.39 seconds for SGS and 1.86 seconds for CPU model inference plus Rheon**. The learned path was **8.20× / 5.06× faster** in this CPU-only comparison. All 4,320 timed frames met the observed strict relative-residual and stored-divergence gates. Separate full-field replay against both references found worst relative velocity L2 error `9.00e-8`. The timing includes CPU inference and process transfer; imports, model setup, warmup and rendering remain separate. These are previously exposed seeds used to compare execution backends; no weights were refitted.

The CPU control supports acceleration over Rheon's iterative pressure solvers on the same hardware. The sealed box also admits an analytic spectral inverse. A comparison against that method would be needed to claim a benefit from learning over the best applicable nonlearned algorithm.

Run the CPU-only comparison through the selected Pumas interpreter:

```sh
python3 scripts/pumas_trajectory.py cpu-compare runs/my-fluid runs/my-fluid/cpu-only-8threads --threads 8
PATH_TO_PUMAS_PYTHON scripts/trajectory_cpu_viewer.py runs/my-fluid runs/my-fluid/cpu-only-8threads
```

The comparison destination must be new. Its freeze and verified result record the CPU device, matching thread limits, hashes, per-frame gates, accuracy and timings. The viewer defaults to the CPU-only comparison; its Comparison selector also exposes the earlier GPU recording with an explicit device label.

The viewport replays recorded accepted states; it does not perform fluid calculations in JavaScript. Orange and teal show the same passive concentration quantity from the CPU and learned rollouts. Orbit either camera to view the data from another angle; scrub or play the timeline to move through accepted frames. Calculation time, pressure-stage time and corrected divergence accompany each selected frame.

The compact display export keeps voxel positions whose concentration exceeds 0.025 and rounds display opacity to five decimal places. It preserves the original full numeric artifacts, which are used for accuracy. Thresholded display pixels must never become training labels or numerical validation data.

<a id="physics-memory-and-quantization"></a>
## Measure memory and keep numerical precision deliberate

The measured CUDA fit allocated a peak of 261,103,616 bytes in PyTorch tensors and reserved 264,241,152 bytes in its CUDA allocator. The Python training process peaked at 1,495,904,256 bytes RSS. This is far below the requested 20 GB training budget, with a small 266 KB checkpoint. The receipt separately records inference peaks and the complete-rollout timing. Native buffers and browser display memory are separate from model tensors.

The full inference/evaluation process, including saved-trajectory comparisons and visualization export, peaked at 1,995,051,008 bytes RSS. Its CUDA allocator peaked at 35,930,112 allocated bytes and 37,748,736 reserved bytes, below the requested 8 GB inference budget. An observed GPU-process sample was 410 MiB including CUDA overhead; that sample is not claimed as a continuous peak. These measurements describe this fit and evaluation, not arbitrary larger grids or differently implemented models.

PyTorch's allocated and reserved peaks do not include every CUDA context, driver or third-party allocation. Use a GPU process monitor alongside allocator measurements, and state the measurement scope. An observed host GPU-memory sample is a sample, not a proven continuous peak. [PyTorch peak allocation API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html), [PyTorch memory profiler scope](https://docs.pytorch.org/docs/stable/torch_cuda_memory).

This model uses float64 deliberately and already fits well below an 8 GB inference budget. Four-bit quantization is therefore unnecessary. `Q4_K_M` is not a general PyTorch tensor dtype or an automatic export format for arbitrary physics networks. Do not promise that a GGUF language-model quantization recipe can quantize this module. If a different physics model needs reduced precision, export it through a supported backend and repeat residual, stored-divergence, rollout-accuracy and completed-inference timing checks. A smaller weight file is not sufficient evidence.

<a id="physics-reproduction-acceptance"></a>
## Know when your reproduction has succeeded

Run the independent eight-cell-per-axis integration checks before the full experiment. They fit on actual Rheon labels, replay an unseen trajectory, verify zero response, linearity and pressure gauge, check a nonlocal response, and confirm that untrained coefficients are rejected:

```sh
PATH_TO_PUMAS_PYTHON -m unittest discover -s scripts -p 'test_trajectory_model.py' -v
PATH_TO_PUMAS_PYTHON scripts/verify_trajectory_results.py runs/my-fluid
PATH_TO_PUMAS_PYTHON scripts/verify_parallel_results.py runs/my-fluid
cargo fmt --manifest-path probes/trajectory/Cargo.toml --check
cargo clippy --locked --manifest-path probes/trajectory/Cargo.toml -- -D warnings
```

Replace `PATH_TO_PUMAS_PYTHON` with the interpreter path printed by the launcher. Do not copy that author's machine-specific path into your own setup.

The parallel verifier requires the completed optional parallel benchmark. It independently replays full fields at each selected thread count and compares both CPU pressure implementations against the AI trajectory; it does not refit the model.

A completed run has a training dataset, a checkpoint, a freeze record, full test fields, a timing receipt, and a working 3D timeline. Check the hashes, inspect every rejected or incomplete run, confirm test seeds never entered fitting, and compute speedup from the declared complete timing scope. If the learned path is slower, the model failed the acceleration objective even if its pressure error is small. If it is fast but fails divergence or rollout gates, it failed the solver objective. Both findings are useful evidence for deciding what to change next.

The broader lesson is to learn an explicitly defined physical operation, preserve units and field placement, and test the operation inside its real repeated use. To build a more general solver, widen the physical contract and collect corresponding data before widening the claims.
