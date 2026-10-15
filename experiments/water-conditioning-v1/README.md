# Water conditioning diagnostics

The [plan](../../docs/plans/water-conditioning-v1/plan.md) governs the fitting,
amplitude and iterative-correction experiments. Historical experiment code and
results remain untouched. Configuration files use the existing measured water
pipeline and declare schedule changes explicitly; no historical hashes change.

Use the installed qualified Torch Python (`$TORCH_PYTHON`) for these commands:

```bash
"$TORCH_PYTHON" experiments/water-conditioning-v1/cpu.py fit-check \
  --packet runs/water-learning-sequence-v1/views/fit-one \
  --artifacts runs/water-learning-sequence-v1/fit-one/managed/stage-01/artifacts
```

Exit 2 means that the restored model fails the 1e-6 equation gate; it is an
expected diagnosis result for the historical fit. `native-controls` takes the
same inputs plus `--binary` (the qualified native water evaluator) and a fresh
`--output` file. It checks actual physical acceptance of the native reference
and float32-rounded ideal correction through the independent native oracle.

`benchmark` measures one restored inference at 1, 2, 4, 8, 12 and available-core
thread counts, with warm-up and retained timing samples. This is a diagnostic
microbenchmark, not full-simulation speedup evidence. Thread changes may change
float32 arithmetic; they do not alter the historical managed CPU witness.

```bash
"$TORCH_PYTHON" experiments/water-conditioning-v1/cpu.py audit \
  --packet runs/water-learning-sequence-v1/views/validation \
  --artifacts runs/water-learning-sequence-v1/scale-1024/managed/stage-01/artifacts \
  --output runs/water-conditioning-v1/amplitude-iteration
```

The pool uses at most available cores minus one, one Torch thread per process,
and distinct worker affinities. It preserves each case and reports worker
identity and used cores. Positive and negative residual scaling keep the
operator and preceding pressure fixed. Repeated-correction experiments update
the base estimate and residual; they are algorithmic experiments outside the
single-proposal training distribution. The scalar minimum-residual step uses
the actual operator and is explicitly separate from a pure AI prediction.
There is no optimizer or CUDA context in this diagnostic pool.

Assemble each schedule through `experiments/water-learning-sequence-v1/pipeline.py
assemble`, using the existing fit-one, validation and reserved view paths and
the chosen new configuration. Source must be committed before assembly. Execute
the normal managed train/restore stages using `workflow.execute`; Tuldok owns
the catalog, grants, Run lifecycle and checkpoint custody. Reserved inference
is not part of this development experiment. Evaluate fitted and validation
proposals through the existing native `pipeline.py evaluate` command. Retain
all completed and failed evidence; never treat fallback as model success.


`training/` is the separately versioned model-change slice. Its CUDA/CPU pipeline
has the same commands and custody path as the historical pipeline, with explicit
`conditioning` configurations. `scalar-units.json` changes only density/timestep
feature units relative to the original one-frame fit. `amplitude-smoke.json`
qualifies two CUDA updates and fresh CPU restore with per-frame residual RMS
normalization and output amplitude reconstruction. Amplitude restoration enforces
positive scaling; it does not enforce oddness or superposition. Network parameter
count and float32 arithmetic stay unchanged. Neither variant is compatible with
the old deployment preprocessing contract merely because weight shapes match.
The new trainer records sampled post-update physical residual, learning rate,
clipping and optimization timestamps. Training loss retains its pre-update
meaning. Run its focused checks separately to avoid flat-module name collisions:

```bash
"$TORCH_PYTHON" -m unittest discover \
  -s experiments/water-conditioning-v1/training -p 'test_conditioning.py'
```


`native_parallel.py` checks independent teacher-state cases on separate CPU
processes. It consumes the existing managed predictions and original captures,
uses the unchanged native evaluator and independent oracle, and preserves each
native child's termination/reaping receipt. `--workers` controls outer
parallelism; each worker has one native/Torch thread. The source is qualified
against the historical one-frame fixture and all 128 validation results;
scientific fields match exactly, excluding three explicitly named timing fields.
This does not change inference arithmetic, thresholds or model behavior.

```bash
"$TORCH_PYTHON" experiments/water-conditioning-v1/native_parallel.py \
  --packet runs/water-learning-sequence-v1/views/validation \
  --inference runs/water-learning-sequence-v1/scale-1024/managed/stage-02/artifacts \
  --binary runs/sealed-water-v1-3/evaluation-build/rheon-sealed-water-evaluation \
  --output runs/water-conditioning-v1/my-native-check
```


`cpu_throughput.py` compares disjoint persistent worker/core partitions, retaining
startup separately from synchronized repeated passes. It replays actual recorded
pressure inputs, without advancing native simulations or training. The default
layouts include three eight-threaded workers and 23 single-threaded workers.

Version-2 GPU configurations `gpu-baseline`, `gpu-resident` and `gpu-batch4` add
execution policy separately from model conditioning. Each runs 500 CUDA updates,
records a short CPU/CUDA profiler summary after warm-up, and restores on CPU.
Resident inputs and combined finite checks must preserve the batch-1 checkpoint
inventory exactly. The batch-4 path stacks only identical grid shapes and changes
sample exposure; it is a throughput probe rather than a controlled accuracy gain.
Profiler summaries, transfer/scalar extraction counts, update rates and peak GPU
memory are retained as managed artifacts. The current Tuldok supervisor executes
one Run at a time, so concurrent GPU training is a distinct future scheduler task.
