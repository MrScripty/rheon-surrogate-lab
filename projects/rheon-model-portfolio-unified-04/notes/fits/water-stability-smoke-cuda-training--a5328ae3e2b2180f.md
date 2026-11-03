# water-stability-smoke · CUDA training

Retrospective source review · fit-a5328ae3e2b2180f

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.815072 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.158924 | 8.95657 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.200816 | 8.53092 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.114839 | 8.51240 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.18176 | 8.67684 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.0217883 | 8.53115 |
| 8 | 7x16x5 | 0.016666667 | 2 | 0.0184824 | 9.21775 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.128264 | 8.63608 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.0449681 | 8.93509 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.0331398 | 8.54041 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.114025 | 8.63772 |
| 8 | 9x16x9 | 0.016666667 | 2 | 0.0133849 | 8.60917 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.0083515 | 8.64898 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.0813828 | 8.59289 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.034005 | 8.50688 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.141871 | 8.84025 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.162622 | 8.59290 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.0117488 | 8.59041 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.14503 | 9.29385 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.0937811 | 9.14901 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.134595 | 9.07239 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.136985 | 8.67823 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.052551 | 8.84083 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.0339656 | 9.07158 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.197971 | 9.07992 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.11612 | 8.58967 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.0388973 | 9.04115 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.182593 | 8.69035 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.0700492 | 8.67636 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.156241 | 8.62011 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.145601 | 8.54067 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.0491507 | 8.56035 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.126764 | 9.17035 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.0334078 | 9.02247 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.0845965 | 8.52699 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.191376 | 8.74471 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.0685282 | 8.58585 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.0642901 | 8.55106 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.0359352 | 8.63606 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.0272703 | 8.54452 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.0568588 | 9.17624 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.0933349 | 8.59360 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.0241802 | 8.90022 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.116231 | 9.00867 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.111486 | 9.21486 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.0665964 | 8.54398 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.215887 | 8.83675 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.038864 | 9.21919 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.29154 | 8.57347 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.0412893 | 8.61889 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.118291 | 8.76682 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.11445 | 8.67661 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.0308432 | 8.58939 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.135943 | 8.62688 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.0279959 | 8.68736 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.0982255 | 8.58507 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.0687254 | 8.56982 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.131668 | 9.19716 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.126877 | 8.88339 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.289629 | 8.63941 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.120648 | 8.57420 |
| 16 | 15x5x5 | 0.016666667 | 2 | 0.0150618 | 9.15908 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.125066 | 8.64867 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.148341 | 9.04687 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.148775 | 8.85087 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: stability-smoke. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 370,016-parameter F32 one-channel multiscale pressure CNN uses physical q=(rho/h)r and 4 internal correction(s), recomputing the F64 equation residual between corrections. GroupNorm is removed; SiLU and all convolution biases are also removed. The captured output scale is correction-rms; cell-zero-gauged corrections accumulate in F64. The objective is canonical. dt is validated metadata, not an independent neural channel. It emits pressure only and advances no physical simulation time.

**Purpose.** This unnamed fit is retained to qualify the captured physical residual/objective/auxiliary-stream path before its full-budget study. It is distinct from any later named final-budget candidate.

**Experiment.** 2 CUDA Adam update(s), batch 1, rate 0.001, seed 20261015, private physical sampling seed 2026102501. The actual optimizer population has 1 physical parent(s)/1 frame(s); validation has 64 parents/128 frames. Training metadata contains grid resolutions [8.0] cells per metre and fixed 1/60 s steps. Execution strategy is {'batch_mode': 'serial', 'finite_checks': 'combined', 'prepared_fields': False, 'profile_updates': 0, 'resident_inputs': True}. Active auxiliary streams: {'analytical_feedback': {'fields': 128, 'proposals': 2, 'weight': 0.1, 'sampling_order_rederived': True, 'grid_count': 1}}. Manufactured fields and frozen captured correction directions are additional operator examples, not independent simulation scenes. Original typed input/output closures, every retained checkpoint and sample orders verify without importing model code.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.29154 and worst equation residual relative L2 is 0.486778. Median case GPU latency is 8.65121 ms per pressure request; the largest case median is 9.33539 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** A one-, two- or ten-update qualification run proves execution/custody coverage at that budget, not a full scientific training result. Validation population size does not enlarge the optimizer population. Folder names such as early/late do not establish direction stage: preserve the actual recorded auxiliary manifest and stage. The common benchmark now quantifies this retained default policy; its training objective and smoke/publication duration are not pressure accuracy or deployment latency. Use the recorded same-request errors and complete pressure-call cost when deciding whether to retain or extend this design.

**Reproduction.** Use the exact captured typed train/restore recipe, source witness, runtime, original input blobs, whole-parent selection, train-only scales, private sample order and all checkpoint inventories. Preserve execution strategy and separately counted auxiliary streams. The historical trainer performs CPU publication/restore neural forwards and needs a GPU-only route before replay; the separate original-checkpoint GPU evaluation and timing are now qualified. Clean retraining and new-Rheon regeneration remain unqualified; the current common GPU pressure accuracy and latency are qualified. Original receipts/configuration and historical Trial ancestry remain unchanged. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-stability-learning-v1/stability-smoke/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "a5328ae3e2b2180f571928abefec1c6f1e168bb376018ed0f1943c27bf6a83d4"
  },
  "history": {
    "path": "runs/water-stability-learning-v1/stability-smoke/managed-smoke/stage-00/artifacts/history.json",
    "sha256": "5047038926cbf656f9632457a747af0776e3c340e490971958349f570cf05dc6"
  },
  "sources": [
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/analytical.py",
      "sha256": "8c2f9e91988c7e3558926830f3758d2752f5efed953a6853a12cea793b96a726"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/config.json",
      "sha256": "a698d5e34a31aa8ad328f83ff616f683c65fb819f0ac522db969e7102e2c6ed2"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/model.py",
      "sha256": "c771465b97827513d6baba8911552bdef90325626693abb0c26d92c8f5a8195e"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/policy.py",
      "sha256": "bcaafde7eee4dfb728800bcd5241004532a46c11f562d884760317bed7014921"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/qualification.json",
      "sha256": "20b0d4d652453f38dd1f687e520e63879deb3516da82c8711d74ef1c96d82139"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-stability-learning-v1/stability-smoke/project/train.py",
      "sha256": "baaacbebd73377c51299adde2b872aa61dd153a96b4d20b9c569229b044b5b99"
    }
  ],
  "source_witness": {
    "path": "runs/water-stability-learning-v1/stability-smoke/project/source-witness.json",
    "sha256": "6a1a9392cdbfe79906a9240d81b2b270e1611783f28a4181ede440a2adc88418"
  },
  "training_population": {
    "path": "runs/water-stability-learning-v1/stability-smoke/managed-smoke/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "bytes": 2275660,
      "path": "runs/water-stability-learning-v1/stability-smoke/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset"
    },
    {
      "bytes": 2278650,
      "path": "runs/water-stability-learning-v1/stability-smoke/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-stability-learning-v1/stability-smoke/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-stability-learning-v1/stability-smoke/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-stability-learning-v1/stability-smoke/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 374928,
      "path": "runs/water-stability-learning-v1/stability-smoke/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-stability-learning-v1/stability-smoke/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-stability-learning-v1/stability-smoke/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-stability-learning-v1/stability-smoke/project/inputs/validation_splits.json",
      "sha256": "9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5",
      "slot": "validation_splits"
    }
  ]
}
```

## Previous source-review statements

These paragraphs describe the earlier source-review stage; the common GPU measurements above update evaluation status.

```json
{
  "lesson": "A one-, two- or ten-update qualification run proves execution/custody coverage at that budget, not a full scientific training result. Validation population size does not enlarge the optimizer population. Folder names such as early/late do not establish direction stage: preserve the actual recorded auxiliary manifest and stage.",
  "outcome": "Original final sampled loss 0.14979111885506965; recorded validation objective 0.025690434932161665 → 0.020552286083366643. Final-update weights remain selected; no optimizer-resume state is retained. These are historical objective values, not worst pressure relative error. A matched current GPU benchmark for this unnamed fit, full-state accuracy and own-history drift remain pending.",
  "reproduction": "Use the exact captured typed train/restore recipe, source witness, runtime, original input blobs, whole-parent selection, train-only scales, private sample order and all checkpoint inventories. Preserve execution strategy and separately counted auxiliary streams. The historical trainer performs CPU publication/restore neural forwards; qualify a GPU-only route before rerunning under current policy. Clean retraining, new-Rheon regeneration and a current common benchmark for this unnamed fit remain unqualified. Original receipts/configuration and historical Trial ancestry remain unchanged."
}
```

## Superseded benchmark-stage reproduction statement

The current paragraph above corrects its qualification status. This original statement is retained for history.

```json
"Use the exact captured typed train/restore recipe, source witness, runtime, original input blobs, whole-parent selection, train-only scales, private sample order and all checkpoint inventories. Preserve execution strategy and separately counted auxiliary streams. The historical trainer performs CPU publication/restore neural forwards; qualify a GPU-only route before rerunning under current policy. Clean retraining, new-Rheon regeneration and a current common benchmark for this unnamed fit remain unqualified. Original receipts/configuration and historical Trial ancestry remain unchanged. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift."
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-a5328ae3e2b2180f",
    "kind": "training_attempt",
    "name": "water-stability-smoke · CUDA training",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-physical-residual-pressure-v1",
      "input_channels": 1,
      "encoder_widths": [
        8,
        16,
        24,
        32
      ],
      "bottleneck_width": 48,
      "decoder_widths": [
        32,
        24,
        16,
        8
      ],
      "kernel": [
        3,
        3,
        3
      ],
      "activation": "silu",
      "dtype": "float32",
      "pressure_dtype": "float64",
      "correction": {
        "activation": "linear",
        "steps": 4
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 65536
    },
    "receipt_sha256": "a5328ae3e2b2180f571928abefec1c6f1e168bb376018ed0f1943c27bf6a83d4",
    "receipt_paths": [
      "runs/water-stability-learning-v1/stability-smoke/managed-smoke/stage-00/artifacts/receipt.json",
      "runs/water-stability-learning-v1/stability-smoke/managed-smoke/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/97b2b4faccc14336a44af87c27bfe8d4/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/97b2b4faccc14336a44af87c27bfe8d4/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/97b2b4faccc14336a44af87c27bfe8d4/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/97b2b4faccc14336a44af87c27bfe8d4/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/97b2b4faccc14336a44af87c27bfe8d4/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-97b2b4faccc14336a44af87c27bfe8d4/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-97b2b4faccc14336a44af87c27bfe8d4/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-97b2b4faccc14336a44af87c27bfe8d4/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-97b2b4faccc14336a44af87c27bfe8d4/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/97b2b4faccc14336a44af87c27bfe8d4/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/97b2b4faccc14336a44af87c27bfe8d4/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/97b2b4faccc14336a44af87c27bfe8d4/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/97b2b4faccc14336a44af87c27bfe8d4/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-97b2b4faccc14336a44af87c27bfe8d4/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-97b2b4faccc14336a44af87c27bfe8d4/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-97b2b4faccc14336a44af87c27bfe8d4/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-97b2b4faccc14336a44af87c27bfe8d4/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-97b2b4faccc14336a44af87c27bfe8d4/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "97b2b4faccc14336a44af87c27bfe8d4",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-stability-smoke · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "a698d5e34a31aa8ad328f83ff616f683c65fb819f0ac522db969e7102e2c6ed2",
      "inputs": {
        "dataset": {
          "bytes": 2275660,
          "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"
        },
        "dataset_extra": {
          "bytes": 2278650,
          "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"
        },
        "reserved_test_dataset": {
          "bytes": 2561142,
          "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"
        },
        "reserved_test_dataset_extra": {
          "bytes": 2564086,
          "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"
        },
        "reserved_test_splits": {
          "bytes": 339677,
          "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"
        },
        "splits": {
          "bytes": 374928,
          "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
        },
        "validation_dataset": {
          "bytes": 2209622,
          "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"
        },
        "validation_dataset_extra": {
          "bytes": 2212566,
          "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"
        },
        "validation_splits": {
          "bytes": 372865,
          "sha256": "9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"
        }
      },
      "normalization_sha256": "9b4abbc5b02e75c134fe7012dc599d721886a7f2ae53fbebc17ca9357220139e",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925",
      "source_witness_sha256": "6a1a9392cdbfe79906a9240d81b2b270e1611783f28a4181ede440a2adc88418"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-002.npz",
        "sha256": "eb60c960329d21aba62ccf89305b501c6457bb24d6ff33ce27b8c6d9e41fd7fa",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "eb60c960329d21aba62ccf89305b501c6457bb24d6ff33ce27b8c6d9e41fd7fa",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2323853312,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [],
    "purpose": {
      "status": "source_review_required",
      "evidence": "Captured architecture, training configuration and linked named entries."
    },
    "outcome": {
      "status": "retained_training_evidence_only",
      "fresh_common_benchmark": "pending",
      "scientific_promotion": null
    },
    "lesson": {
      "status": "review_required_no_automatic_causal_claim"
    },
    "reproduction": {
      "receipt": "runs/water-stability-learning-v1/stability-smoke/managed-smoke/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-stability-learning-v1/stability-smoke/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "a5328ae3e2b2180f571928abefec1c6f1e168bb376018ed0f1943c27bf6a83d4",
    "bytes": 8823
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {},
    "qualified_latency": false
  },
  "interpretation": {
    "card_status": "retrospective_evidence_draft",
    "architecture_description": "Recorded fields; source semantics still require review.",
    "purpose_and_lesson": "No causal lesson or completion claim is inferred from a training receipt.",
    "reproduction": "Source/data/checkpoint identities retained. Clean reproduction remains unverified.",
    "historical_original_Run_observations": "References only; no native Run, Trial, ancestry or current execution authority is created."
  }
}
```

## Exact original receipt bytes

```json
{"architecture":{"activation":"silu","analytical_training_feedback":{"patterns":32,"seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":4},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"correction-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"a698d5e34a31aa8ad328f83ff616f683c65fb819f0ac522db969e7102e2c6ed2","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"9b4abbc5b02e75c134fe7012dc599d721886a7f2ae53fbebc17ca9357220139e","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925","source_witness_sha256":"6a1a9392cdbfe79906a9240d81b2b270e1611783f28a4181ede440a2adc88418"},"elapsed_s":6.369717352994485,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":3.0363514156306337e-06,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":31177216,"peak_reserved_bytes":54525952,"tf32":false},"mode":"train","optimizer_updates":2,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-002.npz":{"bytes":1485452,"role":"checkpoint","sha256":"eb60c960329d21aba62ccf89305b501c6457bb24d6ff33ce27b8c6d9e41fd7fa"},"history.json":{"bytes":20485,"role":"training_diagnostics","sha256":"5047038926cbf656f9632457a747af0776e3c340e490971958349f570cf05dc6"},"normalization.json":{"bytes":1563,"role":"preprocessing","sha256":"9b4abbc5b02e75c134fe7012dc599d721886a7f2ae53fbebc17ca9357220139e"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"1efa94f1bd5a1237aa311b3668813db5f63215c30aec257cbd54e2ad0dfeab2f"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":831,"role":"result_summary","sha256":"33a14204f15eda0575cd2a29ed123b05ddc3922587964db9fd16f398fd2884cf"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"eb60c960329d21aba62ccf89305b501c6457bb24d6ff33ce27b8c6d9e41fd7fa"}},"per_case_inference_s":[0.02904110701638274,0.02283313500811346,0.008905109018087387,0.012880689988378435,0.007648441998753697,0.0051175949920434505,0.010719799000071362,0.008263382012955844,0.00891811601468362,0.00940196801093407,0.010233248001895845,0.017845102003775537,0.006757235998520628],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"64ebce7ac9baf9793fdb5822e109368b014d0c606de76d3c872bfaba6aa5a69e","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"3de6f30f628230dfcfbda73bfb1aea9ac4c9498259aee6ad16137cbf2fea57db","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"2ea1062eb95037730a66c422f5a6e5adda5b8b906f6f3f697646d730dfd7228a","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"25009ceae693258f6912d0086b7084c95965029a4609e5e01bc86db6310f10f4","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"164a3f6c91d744cfa4c608c4599fb5c3e5975f054416ee6778f7deb91b3dcae0","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"ac4e5adbe19cbbbc04529c2ee36cdadd15f7d9bd31ab6f40e724317c49e026fe","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"d2dcd0643f8e464ef2edfaebca3900af8f7435e1158f62c430e4193371ec1e5f","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"afaf657d63dc34335974a0a98266a92e0700dcc5ba320ea7bdaf9f1221192357","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"112fb58ae3ec7f4d8ef46c74ea5a7776453d0c7f0ddcfc26e3cbf28354790f72","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"19fa1811ba2aa93c1eff237498fa023e0365fcf308dc1955aba926a9d8d4c9ac","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"869b6178da5d8fc0b868ec26f3f29175c4400bd7c4f3fdd10a9ed7a8aab09a54","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"918c7964a8773043b5136ad670bab98e8696481e89b8f807e5af14a64ceff844","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"a247593ac99ce8b9dce595836989f10de110832527293e1b74ad878287f92df4","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2323853312,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
