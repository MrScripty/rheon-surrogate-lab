# water-sequence-pipeline-smoke · CUDA training

Retrospective source review · fit-6d1558cced039d83

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.479436 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.192718 | 2.71602 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.200046 | 2.58164 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.103393 | 2.60801 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.182187 | 2.55226 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.0203518 | 2.58511 |
| 8 | 7x16x5 | 0.016666667 | 2 | 0.0158925 | 2.65803 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.130363 | 2.60679 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.0459407 | 2.78793 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.0345113 | 2.64385 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.111102 | 2.70989 |
| 8 | 9x16x9 | 0.016666667 | 2 | 0.0130346 | 2.66619 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.00866885 | 2.68575 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.0723719 | 2.76271 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.0349177 | 2.47364 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.142872 | 2.80647 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.157108 | 2.67046 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.0121654 | 2.60502 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.149819 | 2.79990 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.0889804 | 2.64792 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.13372 | 2.60283 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.135404 | 2.63124 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.0527768 | 2.64946 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.0339198 | 2.62049 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.198918 | 2.60932 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.116839 | 2.77464 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.0379469 | 2.58054 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.185513 | 2.68858 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.0711882 | 2.65991 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.154786 | 2.52753 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.150542 | 2.58889 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.0559637 | 2.62419 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.128131 | 2.80964 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.0301687 | 2.86979 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.0804694 | 2.59787 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.197011 | 2.76228 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.0628496 | 2.69240 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.0644105 | 2.65808 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.0367842 | 2.63252 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.0281097 | 2.63571 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.0568765 | 2.63072 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.0919306 | 2.64894 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.0227816 | 2.65944 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.117481 | 2.83153 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.108848 | 2.65123 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.0675377 | 2.64511 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.220005 | 2.64234 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.0378979 | 2.68717 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.263251 | 2.51608 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.0414361 | 2.60165 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.1199 | 2.79805 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.112327 | 2.47759 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.0319748 | 2.61120 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.151027 | 2.60696 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.0262714 | 2.67552 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.0979957 | 2.62468 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.0751999 | 2.60511 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.131738 | 2.64293 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.128713 | 2.68475 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.29114 | 2.69273 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.128459 | 2.55390 |
| 16 | 15x5x5 | 0.016666667 | 2 | 0.0141786 | 2.63457 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.131734 | 2.72510 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.149131 | 2.74332 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.148734 | 2.61926 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: sequence-pipeline. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 371,729-parameter seven-channel F32 multiscale CNN predicts an additive pressure correction from native residual, previous pressure, physical XYZ coordinates, fluid density and dt. Four encoder levels, global-mean bottleneck and skip decoder retain non-affine GroupNorm and SiLU. Correction scaling/gauge and equation checks use F64. It emits pressure only and advances no physical simulation time.

**Purpose.** This unnamed fit is retained to CUDA managed training/restore qualification. It is distinct from any later named final-budget candidate.

**Experiment.** 2 CUDA Adam update(s), batch 1, rate 0.001, seed 20261015, private physical sampling seed 2026102501. The actual optimizer population has 1 physical parent(s)/1 frame(s); validation has 64 parents/128 frames. Training metadata contains grid resolutions [8.0] cells per metre and fixed 1/60 s steps. Execution strategy is {}. Active auxiliary streams: {}. Manufactured fields and frozen captured correction directions are additional operator examples, not independent simulation scenes. Original typed input/output closures, every retained checkpoint and sample orders verify without importing model code.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.29114 and worst equation residual relative L2 is 0.473326. Median case GPU latency is 2.64927 ms per pressure request; the largest case median is 2.96821 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** A one-, two- or ten-update qualification run proves execution/custody coverage at that budget, not a full scientific training result. Validation population size does not enlarge the optimizer population. Folder names such as early/late do not establish direction stage: preserve the actual recorded auxiliary manifest and stage. The common benchmark now quantifies this retained default policy; its training objective and smoke/publication duration are not pressure accuracy or deployment latency. Use the recorded same-request errors and complete pressure-call cost when deciding whether to retain or extend this design.

**Reproduction.** Use the exact captured typed train/restore recipe, source witness, runtime, original input blobs, whole-parent selection, train-only scales, private sample order and all checkpoint inventories. Preserve execution strategy and separately counted auxiliary streams. The historical trainer performs CPU publication/restore neural forwards and needs a GPU-only route before replay; the separate original-checkpoint GPU evaluation and timing are now qualified. Clean retraining and new-Rheon regeneration remain unqualified; the current common GPU pressure accuracy and latency are qualified. Original receipts/configuration and historical Trial ancestry remain unchanged. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-learning-sequence-v1/pipeline-smoke/managed/stage-00/artifacts/receipt.json",
    "sha256": "6d1558cced039d832fa07ec6ae1efe3f5bb0d27d909e3f5adc15217ad93a76b5"
  },
  "history": {
    "path": "runs/water-learning-sequence-v1/pipeline-smoke/managed/stage-00/artifacts/history.json",
    "sha256": "aebb81db1c9782de2804e6cd1910c35c0bbcd525971f1024c7b4b8ed8c08ffe9"
  },
  "sources": [
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/config.json",
      "sha256": "1abaadef3edf678b0d8f813ce5ba3cb0d656041e3401b8575100e6e00f109aa9"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/model.py",
      "sha256": "d42453ac9060275c7fc6676eab0e23eee7188cf2604a82714bfd4570104d7c0c"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/policy.py",
      "sha256": "5e6fb00db31f558df0bf3665422c7afd4a6ed14e8608e9514576fb86632858cd"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/qualification.json",
      "sha256": "149cf659de3d7d2c2d4a2b5b529faa6f195499b545da9cbe963e304e4257c553"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/test_inference.py",
      "sha256": "43bc71643e362670c960c184b6b1a378f93aba7a6408a2763a960059592a1b56"
    },
    {
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/train.py",
      "sha256": "75a6e4281f70646ef0e66fc3b93163b573df05b66e54006ca6630c0051338be5"
    }
  ],
  "source_witness": {
    "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/source-witness.json",
    "sha256": "a77386f095a691b58e91849e96d4a47b86c3a1ae6e5a23dca604ac973613caee"
  },
  "training_population": {
    "path": "runs/water-learning-sequence-v1/pipeline-smoke/managed/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "bytes": 2275660,
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset"
    },
    {
      "bytes": 2278650,
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 374928,
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-learning-sequence-v1/pipeline-smoke/project/inputs/validation_splits.json",
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
  "outcome": "Original final sampled loss 0.027872713033076466; recorded validation objective 17.70540289055894 → 0.019036769352204139. Final-update weights remain selected; no optimizer-resume state is retained. These are historical objective values, not worst pressure relative error. A matched current GPU benchmark for this unnamed fit, full-state accuracy and own-history drift remain pending.",
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
    "item_id": "fit-6d1558cced039d83",
    "kind": "training_attempt",
    "name": "water-sequence-pipeline-smoke · CUDA training",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-cubic-groupnorm-model-v3",
      "input_channels": 7,
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
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 65536
    },
    "receipt_sha256": "6d1558cced039d832fa07ec6ae1efe3f5bb0d27d909e3f5adc15217ad93a76b5",
    "receipt_paths": [
      "runs/water-learning-sequence-v1/pipeline-smoke/managed/stage-00/artifacts/receipt.json",
      "runs/water-learning-sequence-v1/pipeline-smoke/managed/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/2fd5aeceb09649d69421030f805e416d/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/2fd5aeceb09649d69421030f805e416d/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/2fd5aeceb09649d69421030f805e416d/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/2fd5aeceb09649d69421030f805e416d/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/2fd5aeceb09649d69421030f805e416d/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-2fd5aeceb09649d69421030f805e416d/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-2fd5aeceb09649d69421030f805e416d/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-2fd5aeceb09649d69421030f805e416d/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-2fd5aeceb09649d69421030f805e416d/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/2fd5aeceb09649d69421030f805e416d/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/2fd5aeceb09649d69421030f805e416d/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/2fd5aeceb09649d69421030f805e416d/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/2fd5aeceb09649d69421030f805e416d/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-2fd5aeceb09649d69421030f805e416d/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-2fd5aeceb09649d69421030f805e416d/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-2fd5aeceb09649d69421030f805e416d/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-2fd5aeceb09649d69421030f805e416d/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-2fd5aeceb09649d69421030f805e416d/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "2fd5aeceb09649d69421030f805e416d",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-sequence-pipeline-smoke · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "1abaadef3edf678b0d8f813ce5ba3cb0d656041e3401b8575100e6e00f109aa9",
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
      "normalization_sha256": "37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925",
      "source_witness_sha256": "a77386f095a691b58e91849e96d4a47b86c3a1ae6e5a23dca604ac973613caee"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c",
        "bytes": 1497594,
        "stage": "initial"
      },
      {
        "name": "checkpoint-002.npz",
        "sha256": "38a36f05ca0fcf6d4fa706b578e9ae88631f894840612e7101f09b05f7a4e71b",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "38a36f05ca0fcf6d4fa706b578e9ae88631f894840612e7101f09b05f7a4e71b",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2267480064,
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
      "receipt": "runs/water-learning-sequence-v1/pipeline-smoke/managed/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-learning-sequence-v1/pipeline-smoke/managed/stage-00/artifacts/receipt.json",
    "sha256": "6d1558cced039d832fa07ec6ae1efe3f5bb0d27d909e3f5adc15217ad93a76b5",
    "bytes": 8260
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
{"architecture":{"activation":"silu","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","schema":"rheon-cubic-groupnorm-model-v3","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"1abaadef3edf678b0d8f813ce5ba3cb0d656041e3401b8575100e6e00f109aa9","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925","source_witness_sha256":"a77386f095a691b58e91849e96d4a47b86c3a1ae6e5a23dca604ac973613caee"},"elapsed_s":4.517019270017045,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":3.719549072300765e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":8774656,"peak_reserved_bytes":10485760,"tf32":false},"mode":"train","optimizer_updates":2,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-002.npz":{"bytes":1497594,"role":"checkpoint","sha256":"38a36f05ca0fcf6d4fa706b578e9ae88631f894840612e7101f09b05f7a4e71b"},"history.json":{"bytes":11852,"role":"training_diagnostics","sha256":"aebb81db1c9782de2804e6cd1910c35c0bbcd525971f1024c7b4b8ed8c08ffe9"},"normalization.json":{"bytes":1173,"role":"preprocessing","sha256":"37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"d89aeb31c4b2dbee5ed0631d5fbea62e5ee800f055cc007e2635cd6cc58066f3"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":827,"role":"result_summary","sha256":"3942ca7a57481cbf33e4200ba3ac1515da899cf2c89c7cdb179506ebe450bdc8"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"38a36f05ca0fcf6d4fa706b578e9ae88631f894840612e7101f09b05f7a4e71b"}},"per_case_inference_s":[0.01439208400552161,0.009966954006813467,0.002758824994089082,0.00399227999150753,0.0027437160024419427,0.0017890330054797232,0.003656977991340682,0.0027735430048778653,0.0030025280138943344,0.0032529429881833494,0.0038207359903026372,0.00788622599793598,0.0030344319820869714],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"dcfa4efefd9c6ac1b4ae1400b8901a36add0204ab86c0afdfe5bf3107400b953","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"d9d08e3abfa3e6c632f4aa652907e76a045d80e1dffb3a823efdce84a55009bf","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"735b3d9a0e35bb1b4e99ef1c544a4a33812b66c20027ec7b6ae0ab898192a637","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"bba4d3fa38542825930295ebf30fe35864dc57d4e0cfc7f3cca6172ccfb1af23","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"2d2524f0f0d4dd902bac3b603dc31141685af7b83d1618d70fd72f9b38c526ae","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"7ac964fec72ce294cae50bbc962ca546d6a0a5e03df2641349fc77af37d7928d","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"4020834e814472dd1ff466586d35c2b21eebe754840eb628048e77bf8c012584","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"4efd531c2537d3a87203fe562cfbfc6eb4080c7c421c8f843f93adaa164b9d12","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"811404bf4bf73521f0e3a4be3215f3c5fa66ccc17bb988ecb47e67b99f27e208","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"8e6a6b10f2ed1b20424fd5f056ed690f5f89bb0161be53a3d5fe8078a69866d9","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"21690ed1008ad3d1d3de27816b434ade911f53aadf0a4115c703b094b63a0e12","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"099f56bd4d35cb8dc561eaf7c0fed362b1280d2302d3a7b5b3fd457c11d7f8d7","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"645bc43dc56bf160d26639d557720bb822c10fa53221df5416bb31d2cadf2c95","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2267480064,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
