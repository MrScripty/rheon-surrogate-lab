# water-multigrid-smoke · CUDA training

Retrospective source review · fit-135202a799d78633

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.218624 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.115365 | 2.15093 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.140049 | 2.10448 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.0698276 | 2.17784 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.0891127 | 2.08061 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.0125458 | 2.05322 |
| 8 | 7x16x5 | 0.016666667 | 2 | 0.0143356 | 2.08746 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.0878113 | 2.10363 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.0291086 | 2.07403 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.0135539 | 2.15759 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.0704791 | 2.14792 |
| 8 | 9x16x9 | 0.016666667 | 2 | 0.00890315 | 2.10611 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.00519452 | 2.10880 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.0491425 | 2.06762 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.0228215 | 2.09777 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.0963277 | 2.11055 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.071428 | 2.10708 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.00725012 | 2.12426 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.109474 | 2.10587 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.0576631 | 2.08966 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.0854518 | 2.06078 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.0813731 | 2.09928 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.0318489 | 2.07936 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.0243362 | 2.17556 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.149028 | 2.11166 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.072105 | 2.10569 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.0250225 | 2.08762 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.119351 | 2.11231 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.0484674 | 2.13389 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.101311 | 2.10011 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.128704 | 2.07901 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.0413744 | 2.06719 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.0997897 | 2.21392 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.0213386 | 2.07150 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.0597517 | 2.15435 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.154771 | 2.05054 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.0693693 | 2.11699 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.046253 | 2.09470 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.0244894 | 2.16114 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.0167806 | 2.07811 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.0407708 | 2.12932 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.0671797 | 2.08438 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.0114819 | 2.10792 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.0865445 | 2.10204 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.0560315 | 2.10204 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.0378615 | 2.18288 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.136174 | 2.06508 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.0172485 | 2.10333 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.161037 | 2.06224 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.023277 | 2.10752 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.0801736 | 2.08595 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.0752964 | 2.07858 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.0181711 | 2.09136 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.101463 | 2.09160 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.0185083 | 2.12044 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.0722125 | 2.05883 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.0498911 | 2.09849 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.0952066 | 2.10452 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.0766724 | 2.09621 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.203797 | 2.12556 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.0665384 | 2.12517 |
| 16 | 15x5x5 | 0.016666667 | 2 | 0.0079226 | 2.07532 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.0791241 | 2.07600 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.100352 | 2.17871 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.113487 | 2.10816 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: multigrid-smoke. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 370,016-parameter F32 one-channel multiscale pressure CNN uses physical q=(rho/h)r and 1 internal correction(s), recomputing the F64 equation residual between corrections. GroupNorm is removed; SiLU and all convolution biases are also removed. The captured output scale is previous-pressure-rms; cell-zero-gauged corrections accumulate in F64. The objective is canonical. dt is validated metadata, not an independent neural channel. It emits pressure only and advances no physical simulation time.

**Purpose.** This unnamed fit is retained to qualify manufactured multi-grid feedback before a longer scientific fit. It is distinct from any later named final-budget candidate.

**Experiment.** 10 CUDA Adam update(s), batch 1, rate 0.001, seed 20261015, private physical sampling seed 2026102501. The actual optimizer population has 1 physical parent(s)/1 frame(s); validation has 64 parents/128 frames. Training metadata contains grid resolutions [8.0] cells per metre and fixed 1/60 s steps. Execution strategy is {'batch_mode': 'serial', 'finite_checks': 'combined', 'prepared_fields': False, 'profile_updates': 0, 'resident_inputs': True}. Active auxiliary streams: {'analytical_feedback': {'fields': 640, 'proposals': 10, 'weight': 0.1, 'sampling_order_rederived': True, 'grid_count': 5}}. Manufactured fields and frozen captured correction directions are additional operator examples, not independent simulation scenes. Original typed input/output closures, every retained checkpoint and sample orders verify without importing model code.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.203797 and worst equation residual relative L2 is 0.365132. Median case GPU latency is 2.09857 ms per pressure request; the largest case median is 2.24738 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** A one-, two- or ten-update qualification run proves execution/custody coverage at that budget, not a full scientific training result. Validation population size does not enlarge the optimizer population. Folder names such as early/late do not establish direction stage: preserve the actual recorded auxiliary manifest and stage. The common benchmark now quantifies this retained default policy; its training objective and smoke/publication duration are not pressure accuracy or deployment latency. Use the recorded same-request errors and complete pressure-call cost when deciding whether to retain or extend this design.

**Reproduction.** Use the exact captured typed train/restore recipe, source witness, runtime, original input blobs, whole-parent selection, train-only scales, private sample order and all checkpoint inventories. Preserve execution strategy and separately counted auxiliary streams. The historical trainer performs CPU publication/restore neural forwards and needs a GPU-only route before replay; the separate original-checkpoint GPU evaluation and timing are now qualified. Clean retraining and new-Rheon regeneration remain unqualified; the current common GPU pressure accuracy and latency are qualified. Original receipts/configuration and historical Trial ancestry remain unchanged. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-multigrid-learning-v1/multigrid-smoke/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "135202a799d786331964a8aecba8ea885cc53aab187dbe2db6bf2bb9e1d29865"
  },
  "history": {
    "path": "runs/water-multigrid-learning-v1/multigrid-smoke/managed-smoke/stage-00/artifacts/history.json",
    "sha256": "299fa9e1f9211b52e25eabe872504194db975f409bcdc8f8a5087006d858f607"
  },
  "sources": [
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/analytical.py",
      "sha256": "8443a6234a5b9ea1217f7fb07cbc4bdc7afb36d158f0193c988a36cf1e39a883"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/config.json",
      "sha256": "89fa2bb65c3777939ba4fff74e1cdc71fb39f0c35b34ba314fecbf5c3e02f03f"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/model.py",
      "sha256": "c35b1bfa491fb966ba062243c25659c3cd02ea75031a50ac87afdb5290258be2"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/policy.py",
      "sha256": "faccd15f908436b77e8d8ee07dcc260d59f31bc7e52dde57c03fa820c5a89618"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/qualification.json",
      "sha256": "c2ffaa7d93d9805de250ae128b4d602054d433805005455ef1340f39b16f434a"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/train.py",
      "sha256": "baaacbebd73377c51299adde2b872aa61dd153a96b4d20b9c569229b044b5b99"
    }
  ],
  "source_witness": {
    "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/source-witness.json",
    "sha256": "4e508bf1cbe96949ea00dc684069611b4b4ba2758d235bea517213980f5d1806"
  },
  "training_population": {
    "path": "runs/water-multigrid-learning-v1/multigrid-smoke/managed-smoke/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "bytes": 2275660,
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset"
    },
    {
      "bytes": 2278650,
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 374928,
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-multigrid-learning-v1/multigrid-smoke/project/inputs/validation_splits.json",
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
  "outcome": "Original final sampled loss 0.088264017834858405; recorded validation objective 0.050698485330109955 → 0.012374413892466506. Final-update weights remain selected; no optimizer-resume state is retained. These are historical objective values, not worst pressure relative error. A matched current GPU benchmark for this unnamed fit, full-state accuracy and own-history drift remain pending.",
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
    "item_id": "fit-135202a799d78633",
    "kind": "training_attempt",
    "name": "water-multigrid-smoke · CUDA training",
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
        "steps": 1
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 65536
    },
    "receipt_sha256": "135202a799d786331964a8aecba8ea885cc53aab187dbe2db6bf2bb9e1d29865",
    "receipt_paths": [
      "runs/water-multigrid-learning-v1/multigrid-smoke/managed-smoke/stage-00/artifacts/receipt.json",
      "runs/water-multigrid-learning-v1/multigrid-smoke/managed-smoke/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/6327bdba13f3465ab82474241f23d52f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/6327bdba13f3465ab82474241f23d52f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/6327bdba13f3465ab82474241f23d52f/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/6327bdba13f3465ab82474241f23d52f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/6327bdba13f3465ab82474241f23d52f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-6327bdba13f3465ab82474241f23d52f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-6327bdba13f3465ab82474241f23d52f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-6327bdba13f3465ab82474241f23d52f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-6327bdba13f3465ab82474241f23d52f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/6327bdba13f3465ab82474241f23d52f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/6327bdba13f3465ab82474241f23d52f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/6327bdba13f3465ab82474241f23d52f/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/6327bdba13f3465ab82474241f23d52f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-6327bdba13f3465ab82474241f23d52f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-6327bdba13f3465ab82474241f23d52f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-6327bdba13f3465ab82474241f23d52f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-6327bdba13f3465ab82474241f23d52f/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-6327bdba13f3465ab82474241f23d52f/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "6327bdba13f3465ab82474241f23d52f",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-multigrid-smoke · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "89fa2bb65c3777939ba4fff74e1cdc71fb39f0c35b34ba314fecbf5c3e02f03f",
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
      "normalization_sha256": "73d1cb4a0d9e37c5dc10b40328d10e93d881151310c6c330978eece155c38dd0",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "b393978842a0fa3d3e1470196f098f473f9678e72463cb65ec4ab5581856c2e4",
      "source_witness_sha256": "4e508bf1cbe96949ea00dc684069611b4b4ba2758d235bea517213980f5d1806"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-005.npz",
        "sha256": "a268f7fee9e7941d99c20a8d747ea1604b3ef1b8cc2e5b04b52e84403fb8fceb",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-010.npz",
        "sha256": "9c4499d2c22e8ea7804a0ee7f81ebaad109db8415504d2a3cc02372b17c48fe0",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "9c4499d2c22e8ea7804a0ee7f81ebaad109db8415504d2a3cc02372b17c48fe0",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2327146496,
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
      "receipt": "runs/water-multigrid-learning-v1/multigrid-smoke/managed-smoke/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-multigrid-learning-v1/multigrid-smoke/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "135202a799d786331964a8aecba8ea885cc53aab187dbe2db6bf2bb9e1d29865",
    "bytes": 9285
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":1},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"89fa2bb65c3777939ba4fff74e1cdc71fb39f0c35b34ba314fecbf5c3e02f03f","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"73d1cb4a0d9e37c5dc10b40328d10e93d881151310c6c330978eece155c38dd0","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"b393978842a0fa3d3e1470196f098f473f9678e72463cb65ec4ab5581856c2e4","source_witness_sha256":"4e508bf1cbe96949ea00dc684069611b4b4ba2758d235bea517213980f5d1806"},"elapsed_s":4.49333768201177,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.36161845919014e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":531972096,"peak_reserved_bytes":551550976,"tf32":false},"mode":"train","optimizer_updates":10,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-005.npz":{"bytes":1485452,"role":"checkpoint","sha256":"a268f7fee9e7941d99c20a8d747ea1604b3ef1b8cc2e5b04b52e84403fb8fceb"},"checkpoint-010.npz":{"bytes":1485452,"role":"checkpoint","sha256":"9c4499d2c22e8ea7804a0ee7f81ebaad109db8415504d2a3cc02372b17c48fe0"},"history.json":{"bytes":80112,"role":"training_diagnostics","sha256":"299fa9e1f9211b52e25eabe872504194db975f409bcdc8f8a5087006d858f607"},"normalization.json":{"bytes":1384,"role":"preprocessing","sha256":"73d1cb4a0d9e37c5dc10b40328d10e93d881151310c6c330978eece155c38dd0"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"e21a8aa3e26c006ec3a1df1a901858cb0683d2b243ccaad5423c01dd72f32770"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":833,"role":"result_summary","sha256":"40d906506c21bf45c9f81acd03562f48e22a4b7100c9b20876cab44d3626f696"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"9c4499d2c22e8ea7804a0ee7f81ebaad109db8415504d2a3cc02372b17c48fe0"}},"per_case_inference_s":[0.012600866000866517,0.008467667998047546,0.0021607439848594368,0.0033603440097067505,0.0021529450023081154,0.0013681020063813776,0.0027912490186281502,0.0019900939951185137,0.002149512991309166,0.0022531989961862564,0.0023408859851770103,0.005021477001719177,0.0018280439835507423],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"689a585d4b5ca9f186d98e56524cb9e074019e17013f60cdd8a5b39a1b83893c","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"c972fb13eeb9e2492b16279c4e9420c6f02eca30195c725134b2c9e3739c043f","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"0912dc7e44d1b3e9129ccb69de56bdc1271543c54e327795df52981a3eba3fee","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"3d4aec830ca2e918f480f25ad4854565dd55e38a37527361c23a908c5499dc31","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"3b164e9797c6fa7dc6b12693150d6b2de3d73eb125197695b9cc3ce70ada266a","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"334f3c2c589758faa821814e3b0c39d91cbaf3aaf8d19f2fcd59bf352734b5eb","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"8d4d71769d9338943e6bd94a33e9275769de3af5ba3ee3bb60a797cf694d89e2","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"5dd2d7f860bd4ad357fd9dff69155d81d4fd457c6bb2f8230e1f1f8c39f5a09b","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"ba0b4491365ce29bf25df3a696f8173ed57100c8a01f589bc3e66a763d831600","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"94fe8bdfebea2609ec3713a8ea08ffe9b0aedaa12cb4d0a9b79f560017e3f8fc","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"ba3c591e64217af14bd98805c231461094e0f9effc65be59f4b2ef233d447774","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"e5021034c901f01d8010abbe1c1a4fdc6d8299f072f91609a0333686837d995e","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"d710f6ae25356ee50fc4c2b39fabe326a4efe1a4f18859dc0e0afe469f0277c0","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2327146496,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
