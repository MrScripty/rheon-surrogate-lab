# water-residual-smoke · CUDA training

Retrospective source review · fit-733a339f563172ab

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.303001 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.193267 | 2.14743 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.199866 | 2.16303 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.0982281 | 2.14879 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.177328 | 2.10877 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.0200182 | 2.16230 |
| 8 | 7x16x5 | 0.016666667 | 2 | 0.018198 | 2.08935 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.128706 | 2.20517 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.0443824 | 2.17589 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.0340873 | 2.13358 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.108718 | 2.17016 |
| 8 | 9x16x9 | 0.016666667 | 2 | 0.013527 | 2.18461 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.00800361 | 2.24093 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.0696342 | 2.18989 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.0345862 | 2.20691 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.142055 | 2.17316 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.15586 | 2.19174 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.0117888 | 2.18776 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.149635 | 2.25128 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.0876962 | 2.22828 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.131354 | 2.16881 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.132558 | 2.17429 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.0519299 | 2.16436 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.0323489 | 2.18049 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.198041 | 2.17366 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.113471 | 2.18447 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.0366415 | 2.18776 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.184335 | 2.20996 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.0692104 | 2.24414 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.152866 | 2.20186 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.145474 | 2.15800 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.0529471 | 2.12941 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.127057 | 2.17075 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.0293775 | 2.17301 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.0770166 | 2.19138 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.190748 | 2.16577 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.0605428 | 2.23421 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.0626492 | 2.20152 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.0356286 | 2.15754 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.0271588 | 2.17874 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.0558355 | 2.18276 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.0912325 | 2.18544 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.0226069 | 2.17287 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.115991 | 2.21511 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.106808 | 2.18148 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.0660364 | 2.16559 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.217111 | 2.11995 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.0362576 | 2.11573 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.255788 | 2.19382 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.0406263 | 2.14733 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.118195 | 2.28141 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.106723 | 2.22620 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.0305267 | 2.15654 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.141018 | 2.15921 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.0277233 | 2.22630 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.0941923 | 2.16495 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.0700193 | 2.21241 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.130549 | 2.18797 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.126478 | 2.16338 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.2864 | 2.21993 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.122928 | 2.18496 |
| 16 | 15x5x5 | 0.016666667 | 2 | 0.0146328 | 2.20621 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.126939 | 2.16487 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.14318 | 2.16129 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.147708 | 2.21912 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: residual-smoke. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 370,433-parameter F32 one-channel multiscale pressure CNN uses physical q=(rho/h)r and 1 internal correction(s), recomputing the F64 equation residual between corrections. GroupNorm is removed; SiLU and convolution biases remain. The captured output scale is previous-pressure-rms; cell-zero-gauged corrections accumulate in F64. The objective is canonical. dt is validated metadata, not an independent neural channel. It emits pressure only and advances no physical simulation time.

**Purpose.** This unnamed fit is retained to qualify the captured physical residual/objective/auxiliary-stream path before its full-budget study. It is distinct from any later named final-budget candidate.

**Experiment.** 2 CUDA Adam update(s), batch 1, rate 0.001, seed 20261015, private physical sampling seed 2026102501. The actual optimizer population has 1 physical parent(s)/1 frame(s); validation has 64 parents/128 frames. Training metadata contains grid resolutions [8.0] cells per metre and fixed 1/60 s steps. Execution strategy is {'batch_mode': 'serial', 'finite_checks': 'combined', 'prepared_fields': True, 'profile_updates': 0, 'resident_inputs': True}. Active auxiliary streams: {}. Manufactured fields and frozen captured correction directions are additional operator examples, not independent simulation scenes. Original typed input/output closures, every retained checkpoint and sample orders verify without importing model code.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.2864 and worst equation residual relative L2 is 0.450763. Median case GPU latency is 2.17982 ms per pressure request; the largest case median is 2.29112 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** A one-, two- or ten-update qualification run proves execution/custody coverage at that budget, not a full scientific training result. Validation population size does not enlarge the optimizer population. Folder names such as early/late do not establish direction stage: preserve the actual recorded auxiliary manifest and stage. The common benchmark now quantifies this retained default policy; its training objective and smoke/publication duration are not pressure accuracy or deployment latency. Use the recorded same-request errors and complete pressure-call cost when deciding whether to retain or extend this design.

**Reproduction.** Use the exact captured typed train/restore recipe, source witness, runtime, original input blobs, whole-parent selection, train-only scales, private sample order and all checkpoint inventories. Preserve execution strategy and separately counted auxiliary streams. The historical trainer performs CPU publication/restore neural forwards and needs a GPU-only route before replay; the separate original-checkpoint GPU evaluation and timing are now qualified. Clean retraining and new-Rheon regeneration remain unqualified; the current common GPU pressure accuracy and latency are qualified. Original receipts/configuration and historical Trial ancestry remain unchanged. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-residual-learning-v1/residual-smoke/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "733a339f563172abfebe8e9cd354e533c171a55e865bd6017a871d9974055bcd"
  },
  "history": {
    "path": "runs/water-residual-learning-v1/residual-smoke/managed-smoke/stage-00/artifacts/history.json",
    "sha256": "65307b2b008c5a12ac41c23dcad7f4a804a68ff1b235369af17184c74c7658eb"
  },
  "sources": [
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/config.json",
      "sha256": "9c8e82d411069298ebed10d6819e15e812ffcd61d185efe7aeef98fa521b4324"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/model.py",
      "sha256": "c55228fea3716954f97c63cc60a1db06ccfac86ebc339d0c31eb17660d1ec1c1"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/physical_residual.py",
      "sha256": "361e5091a3467cef8e58980cbd740a471dde80c48f2a6466631c7032d626e01a"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/policy.py",
      "sha256": "9b8c8ab145f9167c34018d68482094770535879f32650fe08a8a93b15d2f4d73"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/qualification.json",
      "sha256": "fbc90e275d888167356253293a193617203502f3802717109f595b09e0eaaabc"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-residual-learning-v1/residual-smoke/project/train.py",
      "sha256": "bff0a1b76d5541be2efd06ea682d918103a56137e1496287451334853e0fcd71"
    }
  ],
  "source_witness": {
    "path": "runs/water-residual-learning-v1/residual-smoke/project/source-witness.json",
    "sha256": "ea874ac23d6535bbda4ebaca98b6670d40d73ff18e03b5e8a701f016581331ee"
  },
  "training_population": {
    "path": "runs/water-residual-learning-v1/residual-smoke/managed-smoke/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "bytes": 2275660,
      "path": "runs/water-residual-learning-v1/residual-smoke/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset"
    },
    {
      "bytes": 2278650,
      "path": "runs/water-residual-learning-v1/residual-smoke/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-residual-learning-v1/residual-smoke/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-residual-learning-v1/residual-smoke/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-residual-learning-v1/residual-smoke/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 374928,
      "path": "runs/water-residual-learning-v1/residual-smoke/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-residual-learning-v1/residual-smoke/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-residual-learning-v1/residual-smoke/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-residual-learning-v1/residual-smoke/project/inputs/validation_splits.json",
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
  "outcome": "Original final sampled loss 0.014395458089541721; recorded validation objective 0.020072869909707119 → 0.017884301585607651. Final-update weights remain selected; no optimizer-resume state is retained. These are historical objective values, not worst pressure relative error. A matched current GPU benchmark for this unnamed fit, full-state accuracy and own-history drift remain pending.",
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
    "item_id": "fit-733a339f563172ab",
    "kind": "training_attempt",
    "name": "water-residual-smoke · CUDA training",
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
        "activation": "silu",
        "steps": 1
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 65536
    },
    "receipt_sha256": "733a339f563172abfebe8e9cd354e533c171a55e865bd6017a871d9974055bcd",
    "receipt_paths": [
      "runs/water-residual-learning-v1/residual-smoke/managed-smoke/stage-00/artifacts/receipt.json",
      "runs/water-residual-learning-v1/residual-smoke/managed-smoke/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/455586b9f3c24a06ba0b50df1d932c12/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/455586b9f3c24a06ba0b50df1d932c12/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/455586b9f3c24a06ba0b50df1d932c12/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/455586b9f3c24a06ba0b50df1d932c12/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/455586b9f3c24a06ba0b50df1d932c12/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-455586b9f3c24a06ba0b50df1d932c12/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-455586b9f3c24a06ba0b50df1d932c12/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-455586b9f3c24a06ba0b50df1d932c12/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-455586b9f3c24a06ba0b50df1d932c12/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/455586b9f3c24a06ba0b50df1d932c12/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/455586b9f3c24a06ba0b50df1d932c12/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/455586b9f3c24a06ba0b50df1d932c12/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/455586b9f3c24a06ba0b50df1d932c12/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-455586b9f3c24a06ba0b50df1d932c12/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-455586b9f3c24a06ba0b50df1d932c12/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-455586b9f3c24a06ba0b50df1d932c12/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-455586b9f3c24a06ba0b50df1d932c12/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-455586b9f3c24a06ba0b50df1d932c12/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "455586b9f3c24a06ba0b50df1d932c12",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-residual-smoke · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "9c8e82d411069298ebed10d6819e15e812ffcd61d185efe7aeef98fa521b4324",
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
      "sample_order_sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925",
      "source_witness_sha256": "ea874ac23d6535bbda4ebaca98b6670d40d73ff18e03b5e8a701f016581331ee"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "d94f07e2873160a4aa9f7a56ca5694067d428fb94a89df293b3145a09c1b1b2a",
        "bytes": 1492410,
        "stage": "initial"
      },
      {
        "name": "checkpoint-002.npz",
        "sha256": "9002f34e6b12fadf69e81635c622684d2bcdb553b3bb2623e809262bdc00ffbb",
        "bytes": 1492410,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "9002f34e6b12fadf69e81635c622684d2bcdb553b3bb2623e809262bdc00ffbb",
        "bytes": 1492410,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2227286016,
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
      "receipt": "runs/water-residual-learning-v1/residual-smoke/managed-smoke/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-residual-learning-v1/residual-smoke/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "733a339f563172abfebe8e9cd354e533c171a55e865bd6017a871d9974055bcd",
    "bytes": 8695
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
{"architecture":{"activation":"silu","block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","correction":{"activation":"silu","steps":1},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times train pressure RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"9c8e82d411069298ebed10d6819e15e812ffcd61d185efe7aeef98fa521b4324","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"73d1cb4a0d9e37c5dc10b40328d10e93d881151310c6c330978eece155c38dd0","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925","source_witness_sha256":"ea874ac23d6535bbda4ebaca98b6670d40d73ff18e03b5e8a701f016581331ee"},"elapsed_s":5.582493032998173,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":3.148824617937862e-06,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":14272512,"peak_reserved_bytes":31457280,"tf32":false},"mode":"train","optimizer_updates":2,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1492410,"role":"checkpoint","sha256":"d94f07e2873160a4aa9f7a56ca5694067d428fb94a89df293b3145a09c1b1b2a"},"checkpoint-002.npz":{"bytes":1492410,"role":"checkpoint","sha256":"9002f34e6b12fadf69e81635c622684d2bcdb553b3bb2623e809262bdc00ffbb"},"history.json":{"bytes":12172,"role":"training_diagnostics","sha256":"65307b2b008c5a12ac41c23dcad7f4a804a68ff1b235369af17184c74c7658eb"},"normalization.json":{"bytes":1384,"role":"preprocessing","sha256":"73d1cb4a0d9e37c5dc10b40328d10e93d881151310c6c330978eece155c38dd0"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"a723ff6f9d41101fc3a5dc92e8971b37ebfd0a2cc5dfbcbd42d01e6093f9d5cf"},"profile.json":{"bytes":188,"role":"training_diagnostics","sha256":"d1fc041c4980d802aa10536461bea5aec61340227250c108a7da53154257c07e"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":829,"role":"result_summary","sha256":"5a7a9f772be35856a22a46512bdf4a15fa04839e95d2594d184210f81c107e37"},"weights.npz":{"bytes":1492410,"role":"checkpoint","sha256":"9002f34e6b12fadf69e81635c622684d2bcdb553b3bb2623e809262bdc00ffbb"}},"per_case_inference_s":[0.02930508297868073,0.021636763005517423,0.007153690996346995,0.008190558990463614,0.005539164005313069,0.0037824689934495836,0.007636608992470428,0.00508570100646466,0.005637826980091631,0.005884946993319318,0.006552404985995963,0.011740087997168303,0.00448957399930805],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"0445657139098318ac45bdce92b1eccb02fba25cea2b283fc390184feddc0864","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"d8ddba1ff61a60700b21b84b6fb5c5a2fb211c7ac74526541048b4016c010a5a","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"eeafa33a99380a8d036f67dd5efd6d3c1722e292705679dbe151b04546e1fe57","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"c6a55e3a542bda5998c7aaa63d4cd7b002fa9868e80cf487a1b31f2079385b05","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"8c81d5cf9547b7f03aa503666b943cfd84175460b6264c54aad58bb29ebe0863","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"c0431ec1595c4a6c3d94f753904594249c673d28a27ae8137efc991af77c4ac0","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"b736d67ae0501511e6b110fb040569f68c0844cc4da88b5ffa35c0bbd4d5001c","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"d008dc86975dee37185190d11462cbd0ac778bf0071a76df2a25aa9b4e72a4d1","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"a1857a34737756e9881be3c2e289eb6ffc2a4dfd7991b76b1770b6dd9dfa21fb","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"784e84f4de99ed9fdb3f03af14ba5213e8258e585253eb227a18c2d649a429c7","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"4f91205a16089a14dbce5ce5b53bacee0e1a942b0aa49aa8c21ea24f4603501a","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"3acc598bb5f7e9a6350545ad1163c50764fa2305d05246f1e6c7a62fd8473a4d","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"3b040381f07fc166c8bdd1d10bb87f8468e64c77d31d4e1fba5651473ceeade4","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2227286016,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
