# water-conditioning-gpu-prepared · CUDA training

Retrospective source review · fit-9fa632f2b8082525

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.427094 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.574298 | 2.63421 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.382771 | 2.61083 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.161546 | 2.81999 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.162976 | 2.79124 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.0247925 | 2.66836 |
| 8 | 7x16x5 | 0.016666667 | 2 | 0.0366463 | 2.82261 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.234987 | 2.54651 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.0248512 | 2.56243 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.026028 | 2.74272 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.185553 | 2.63873 |
| 8 | 9x16x9 | 0.016666667 | 2 | 0.0156592 | 2.57023 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.0119935 | 2.55236 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.147976 | 2.57166 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.0441285 | 2.54038 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.0541553 | 2.91723 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.197604 | 2.51527 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.0152737 | 2.59984 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.21819 | 2.59096 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.103262 | 2.57762 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.255734 | 2.85328 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.280919 | 2.86386 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.107034 | 2.56423 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.144621 | 2.54892 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.212438 | 2.56473 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.216311 | 2.73385 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.0815932 | 2.51626 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.196946 | 2.55839 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.11767 | 2.58054 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.145134 | 2.54700 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.379562 | 2.58374 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.114889 | 2.53172 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.174745 | 2.92813 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.0335792 | 2.56677 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.0759425 | 2.70445 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.134879 | 2.60237 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.0814934 | 2.53887 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.0736358 | 3.03165 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.0252227 | 2.71551 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.0246278 | 2.52136 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.0902704 | 2.53671 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.0988169 | 2.56078 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.021241 | 2.55468 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.118897 | 2.52178 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.0741008 | 2.55732 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.0706917 | 2.72421 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.132738 | 2.87247 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.0342959 | 2.69944 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.163585 | 2.50044 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.0324461 | 2.77559 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.163801 | 2.58417 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.0711098 | 2.59437 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.0260868 | 2.56557 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.169097 | 2.56706 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.0302522 | 2.54040 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.123443 | 2.58999 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.0440991 | 2.51953 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.105566 | 2.58002 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.113785 | 2.70104 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.175755 | 2.57863 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.100557 | 2.64650 |
| 16 | 15x5x5 | 0.016666667 | 2 | 0.00554819 | 2.59265 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.125965 | 2.86074 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.142751 | 2.93520 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.193137 | 2.58762 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: conditioning-prepared. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 371,729-parameter seven-channel F32 multiscale CNN predicts an additive pressure correction from native residual, previous pressure, physical XYZ coordinates, fluid density and dt. Four encoder levels, global-mean bottleneck and skip decoder retain non-affine GroupNorm and SiLU. Correction scaling/gauge and equation checks use F64. Per-sample residual RMS is normalized relative to train RMS and restored at output; fixed units rescale rho by 1,000 and dt by 1/60. This changes conditioning without adding parameters. It emits pressure only and advances no physical simulation time.

**Purpose.** This unnamed fit is retained to compare training execution strategies with the same one-frame amplitude-conditioned task. It is distinct from any later named final-budget candidate.

**Experiment.** 500 CUDA Adam update(s), batch 1, rate 0.001, seed 20261015, private physical sampling seed 2026102501. The actual optimizer population has 1 physical parent(s)/1 frame(s); validation has 64 parents/128 frames. Training metadata contains grid resolutions [8.0] cells per metre and fixed 1/60 s steps. Execution strategy is {'batch_mode': 'serial', 'finite_checks': 'combined', 'prepared_fields': True, 'profile_updates': 10, 'resident_inputs': True}. Active auxiliary streams: {}. Manufactured fields and frozen captured correction directions are additional operator examples, not independent simulation scenes. Original typed input/output closures, every retained checkpoint and sample orders verify without importing model code.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.574298 and worst equation residual relative L2 is 0.471107. Median case GPU latency is 2.58568 ms per pressure request; the largest case median is 3.13312 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** The baseline, resident and prepared variants fit one identical physical frame; batch four repeats that same frame four times per update. Those names describe execution, not four-scene coverage. Exact retained tensor comparison can establish value parity, while historical training throughput observations alone cannot establish current inference speed or broad generalization. The common benchmark now quantifies this retained default policy; its training objective and smoke/publication duration are not pressure accuracy or deployment latency. Use the recorded same-request errors and complete pressure-call cost when deciding whether to retain or extend this design.

**Reproduction.** Use the exact captured typed train/restore recipe, source witness, runtime, original input blobs, whole-parent selection, train-only scales, private sample order and all checkpoint inventories. Preserve execution strategy and separately counted auxiliary streams. The historical trainer performs CPU publication/restore neural forwards and needs a GPU-only route before replay; the separate original-checkpoint GPU evaluation and timing are now qualified. Clean retraining and new-Rheon regeneration remain unqualified; the current common GPU pressure accuracy and latency are qualified. Original receipts/configuration and historical Trial ancestry remain unchanged. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-conditioning-v1/gpu-prepared/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "9fa632f2b80825250e74ca0e610863cd2149f0921da22f66dee29a1f43a0b4ae"
  },
  "history": {
    "path": "runs/water-conditioning-v1/gpu-prepared/managed-smoke/stage-00/artifacts/history.json",
    "sha256": "6ccf19f4b6bd529b8c0cf7f773f5fd7d88857a3b414693ba40bf70b09e41aa9b"
  },
  "sources": [
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/config.json",
      "sha256": "0e00707661e4f544e1b228e52101c022eaf41df1919e0af0337dd214ac64d993"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/model.py",
      "sha256": "4fb774b9a0d698e228b7afbe329b9f281cea59a22b4361d78463f57a7055a31e"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/policy.py",
      "sha256": "837e6a740e13809e13195688afafd4a928999138bf200708f72129afc7ab25ff"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/prepared.py",
      "sha256": "e286c00c7bfe60dfe4182a1f95194d876372fdcc500895a3e042389ad4e0adc7"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/qualification.json",
      "sha256": "5a3361c4eedb7811dc1871295eec9da46c16bfc48ecee20bee873e304eec4fe6"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-conditioning-v1/gpu-prepared/project/train.py",
      "sha256": "247b246e19e40f755e70dacbdcd45725c0fdd608baaf65d51211490524cd98ae"
    }
  ],
  "source_witness": {
    "path": "runs/water-conditioning-v1/gpu-prepared/project/source-witness.json",
    "sha256": "a345fadb9750f7552561acd4f49cd0a563605ef31f0ad2ccc668db65070b0e2e"
  },
  "training_population": {
    "path": "runs/water-conditioning-v1/gpu-prepared/managed-smoke/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "bytes": 2275660,
      "path": "runs/water-conditioning-v1/gpu-prepared/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset"
    },
    {
      "bytes": 2278650,
      "path": "runs/water-conditioning-v1/gpu-prepared/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-conditioning-v1/gpu-prepared/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-conditioning-v1/gpu-prepared/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-conditioning-v1/gpu-prepared/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 374928,
      "path": "runs/water-conditioning-v1/gpu-prepared/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-conditioning-v1/gpu-prepared/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-conditioning-v1/gpu-prepared/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-conditioning-v1/gpu-prepared/project/inputs/validation_splits.json",
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
  "lesson": "The baseline, resident and prepared variants fit one identical physical frame; batch four repeats that same frame four times per update. Those names describe execution, not four-scene coverage. Exact retained tensor comparison can establish value parity, while historical training throughput observations alone cannot establish current inference speed or broad generalization.",
  "outcome": "Original final sampled loss 2.8855478862083511e-05; recorded validation objective 9.6544146763660095 → 0.01912558997877711. Final-update weights remain selected; no optimizer-resume state is retained. These are historical objective values, not worst pressure relative error. A matched current GPU benchmark for this unnamed fit, full-state accuracy and own-history drift remain pending.",
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
    "item_id": "fit-9fa632f2b8082525",
    "kind": "training_attempt",
    "name": "water-conditioning-gpu-prepared · CUDA training",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-conditioned-cubic-groupnorm-v1",
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
    "receipt_sha256": "9fa632f2b80825250e74ca0e610863cd2149f0921da22f66dee29a1f43a0b4ae",
    "receipt_paths": [
      "runs/water-conditioning-v1/gpu-prepared/managed-smoke/stage-00/artifacts/receipt.json",
      "runs/water-conditioning-v1/gpu-prepared/managed-smoke/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/d23f2b2b2228412287de872a3569bcb3/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/d23f2b2b2228412287de872a3569bcb3/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/d23f2b2b2228412287de872a3569bcb3/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/d23f2b2b2228412287de872a3569bcb3/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/d23f2b2b2228412287de872a3569bcb3/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-d23f2b2b2228412287de872a3569bcb3/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-d23f2b2b2228412287de872a3569bcb3/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-d23f2b2b2228412287de872a3569bcb3/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-d23f2b2b2228412287de872a3569bcb3/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/d23f2b2b2228412287de872a3569bcb3/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/d23f2b2b2228412287de872a3569bcb3/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/d23f2b2b2228412287de872a3569bcb3/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/d23f2b2b2228412287de872a3569bcb3/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-d23f2b2b2228412287de872a3569bcb3/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-d23f2b2b2228412287de872a3569bcb3/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-d23f2b2b2228412287de872a3569bcb3/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-d23f2b2b2228412287de872a3569bcb3/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-d23f2b2b2228412287de872a3569bcb3/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "d23f2b2b2228412287de872a3569bcb3",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-conditioning-gpu-prepared · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 500,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "0e00707661e4f544e1b228e52101c022eaf41df1919e0af0337dd214ac64d993",
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
      "normalization_sha256": "59b9e7be6968b788fa5ec3fa189745002d10cd7d747f4d4328f25fe0c359044b",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "668946bab9868b28489bb906205ee1026045c8bcd3ca62a1bdf733c65491351b",
      "source_witness_sha256": "a345fadb9750f7552561acd4f49cd0a563605ef31f0ad2ccc668db65070b0e2e"
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
        "sha256": "3308e95fa8ed42b74ba3c65ac2d1c9cdc3ab930d5cdd74d858a8c824d01b8bc8",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "9c2cb3aa659d4ddf23bed0dc295a2f0f5641a3ba94709f63990e9cb7102a5777",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "9c2cb3aa659d4ddf23bed0dc295a2f0f5641a3ba94709f63990e9cb7102a5777",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2793070592,
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
      "receipt": "runs/water-conditioning-v1/gpu-prepared/managed-smoke/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-conditioning-v1/gpu-prepared/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "9fa632f2b80825250e74ca0e610863cd2149f0921da22f66dee29a1f43a0b4ae",
    "bytes": 8742
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
{"architecture":{"activation":"silu","amplitude":"per-sample residual RMS / training residual RMS; restore output amplitude","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"conditioning":"amplitude-and-units","convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","scalar_feature_units":{"dt_s":0.016666666666666666,"rho_kg_m3":1000.0},"schema":"rheon-conditioned-cubic-groupnorm-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"0e00707661e4f544e1b228e52101c022eaf41df1919e0af0337dd214ac64d993","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"59b9e7be6968b788fa5ec3fa189745002d10cd7d747f4d4328f25fe0c359044b","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"668946bab9868b28489bb906205ee1026045c8bcd3ca62a1bdf733c65491351b","source_witness_sha256":"a345fadb9750f7552561acd4f49cd0a563605ef31f0ad2ccc668db65070b0e2e"},"elapsed_s":13.961967832001392,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":0.00015331459428580274,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":14617600,"peak_reserved_bytes":33554432,"tf32":false},"mode":"train","optimizer_updates":500,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-002.npz":{"bytes":1497594,"role":"checkpoint","sha256":"3308e95fa8ed42b74ba3c65ac2d1c9cdc3ab930d5cdd74d858a8c824d01b8bc8"},"checkpoint-500.npz":{"bytes":1497594,"role":"checkpoint","sha256":"9c2cb3aa659d4ddf23bed0dc295a2f0f5641a3ba94709f63990e9cb7102a5777"},"history.json":{"bytes":19071,"role":"training_diagnostics","sha256":"6ccf19f4b6bd529b8c0cf7f773f5fd7d88857a3b414693ba40bf70b09e41aa9b"},"normalization.json":{"bytes":1179,"role":"preprocessing","sha256":"59b9e7be6968b788fa5ec3fa189745002d10cd7d747f4d4328f25fe0c359044b"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"0dc008756f6a9843b52a3a50f9863716707943baaa5b5d6ecec581aadb3ca7e7"},"profile.json":{"bytes":6566,"role":"training_diagnostics","sha256":"110a91d27e2aa233500cf9f325d0d9316319f4ea177110d67efab37f55be712e"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":830,"role":"result_summary","sha256":"ab0c35f1120c040b577cae37d7900484a0688041994543db854a674ae29f99d1"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"9c2cb3aa659d4ddf23bed0dc295a2f0f5641a3ba94709f63990e9cb7102a5777"}},"per_case_inference_s":[0.007715245999861509,0.0067042790178675205,0.002637249999679625,0.004091387992957607,0.0026642989832907915,0.0017227520002052188,0.0034843610192183405,0.0025324790040031075,0.0026266819913871586,0.002598791994387284,0.002885268011596054,0.005831674992805347,0.0021851630008313805],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"138a97c026d1fd72ed89897f1f0d9eee135bfba66bf918df31f72a00001f7cb0","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"93bb22a059d92555be2f4f03977b643b04ea66d7ceae77ebd49a462c5456aeb7","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"f3a0fe072c7edbd5250fecb5f7c7694f696984b960eeb468759dbbc4f0d43c05","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"1185510e1ef3c0a7a2acb2454a70c7ab60c4356d3065e9e35a09d64610a06307","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"57688b85aaa4921624bc84fcc3766108448e36ba05862cde9d0501f68b338aa7","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"0e3d263206ffd5cea22173aea9b1104cb10d31be23c0401d6ab1ed4cfd1d8ef4","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"544428accce946055643a35fc2e49591a7b5b5e2953ae4659d12829fbb7db29d","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"b2a4f0fd7f79575d8b203cc06fc47c7d053a9aad1321a41f317eeb17f4bf9ec9","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"fd43bd62c89ffc9e4cdacfd31cd37b87095a01e0fe311c30c0c7753a22328af1","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"ee0d66a8bf6471a19e4a76dc5541174049c8d9b347854ce88231a8d197229a28","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"579f9fbee86c4662fca2562ab358142729a77c94e0cdbbc933bebf79e9d6dee1","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"5515cc814dd2a7ff7bbe12b127e2e1508c879aad584d06e6ed1abf11fd7f4909","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"9e426378c6650d131aa9a0fc20db9d28a09af581b27364f2f226b3c747c27f3c","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2793070592,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
