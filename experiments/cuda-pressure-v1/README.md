# CUDA training, CPU deployment

This version adds device portability to the unchanged seven-channel GroupNorm
pressure model: 371,729 float32 parameters, the same initialization, normalization,
physical operator, supervised/equation loss and cubic scalar spacing. Historical
CPU experiments stay immutable. Grid axes may differ; cell edges share one h.
Existing 65,536-cell and batch-size bounds remain. Network float32 is unchanged;
physical operator/loss accumulation and gauge-fixed pressure reconstruction retain
float64 as before. There is no mixed precision or TF32 training.

Training requires CUDA, including parameters, conditioning/fields, gradients and
Adam moment buffers. No automatic CPU fallback. The CPU path uses native operations.
CUDA preserves average/nearest forward operators, with mathematically equivalent
explicit backward reductions for deterministic gradients; replicated convolution
padding uses concatenation on CUDA. These remove unsupported/nondeterministic Torch
CUDA backward operations without changing parameter names or the forward model.

Checkpoints are bounded NPZ float32 weights, portable to CPU, with per-tensor byte
inventories. They do not contain resumable optimizer state. The training Run checks
weight bits survive CPU transfer, publishes CPU deployment predictions, and records
CUDA/CPU forward differences. A distinct managed CPU restore consumes typed retained
Run artifacts, performs zero optimizer updates, checks every checkpoint and seeded
initial state, and repeats all twelve CPU predictions exactly. Cross-device bit
identity is neither promised nor required.

`configs/qualification.json` is a 20-update infrastructure qualification, not a
scientific accuracy experiment. Configurations separate training from the existing
qualified packet, allow independent initialization/sampling seeds, 1–10,000 updates,
2–11 explicit checkpoint states, and bounded optional learning-rate changes. The
600-second cap is elapsed time, irrespective of device. No model/loss ablations or
long training runs are started by assembly.

Register a measured installed Torch runtime using Tuldok's `torch_runtimes.py` owner
CLI. The existing catalog and normal approved project Runs own both stages. Then
use that Torch release's Python (with Tuldok's normal CLI dependencies available):

```
python experiments/cuda-pressure-v1/pipeline.py assemble \
  --packet QUALIFIED_V2_PACKET --runtime-manifest REGISTERED_MANIFEST_JSON \
  --tuldok TULDOK_CHECKOUT --config experiments/cuda-pressure-v1/configs/qualification.json \
  --output runs/cuda-pressure-v1-workspace-N/project
```

Assembly requires committed clean lab sources. It verifies existing packet lineage,
checks every grid and exact zero-residual bypass without constructing an optimizer,
and freezes source/runtime/configuration/normalization witnesses. The selected
manifest's measured identity is used in both runtime-device recipes. Execute the
stages through the existing `workflow.execute(client, project, output, observer)`
HTTP adapter with an authenticated Tuldok client; it grants only exact preflighted
material, observes completion/draining, downloads and re-downloads all artifacts.

```
python experiments/cuda-pressure-v1/pipeline.py evaluate \
  --packet QUALIFIED_V2_PACKET --binary QUALIFIED_V2_NATIVE_BINARY \
  --training TRAIN_RUN_ARTIFACTS --restored CPU_RESTORE_ARTIFACTS \
  --output runs/cuda-pressure-v1-workspace-N/evaluation
```

Native diagnostics reuse the qualified unchanged Rheon operator/binary and retain
its original results. The current float32 output acceptance profile is reported
separately, preserving every physical gate. The live observer publishes retained
capture/proposal previews for Rheon egui and Tuldok. GPU execution qualification
does not establish learned pressure accuracy or acceleration.
