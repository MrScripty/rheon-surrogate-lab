# water-multigrid-cohort16-smoke · CUDA training

Retrospective source review · fit-c8638134ae71a368

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.202158 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.116189 | 2.02282 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.143678 | 2.08837 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.0645599 | 2.04523 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.118339 | 2.04613 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.0141898 | 2.07604 |
| 8 | 7x16x5 | 0.016666667 | 2 | 0.0154478 | 2.06228 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.102257 | 2.06783 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.033535 | 2.09409 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.0184421 | 2.06650 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.0860445 | 2.12911 |
| 8 | 9x16x9 | 0.016666667 | 2 | 0.0107128 | 2.08947 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.00530627 | 2.16693 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.048149 | 2.05646 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.0266502 | 2.06896 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.110099 | 2.07128 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.0998414 | 2.13347 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.0084025 | 2.10571 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.123611 | 2.11071 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.062732 | 2.07028 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.0997679 | 2.12772 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.0984371 | 2.09849 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.0389636 | 2.10946 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.0235404 | 2.05671 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.160701 | 2.12661 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.0812905 | 2.09877 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.0275786 | 2.11132 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.129816 | 2.11123 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.0529841 | 2.15323 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.114064 | 2.08210 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.127719 | 2.07839 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.0426356 | 2.06519 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.107342 | 2.07970 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.0187436 | 2.08661 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.0561254 | 2.04753 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.164926 | 2.06795 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.0600529 | 2.12296 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.0490445 | 2.10684 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.0274752 | 2.06631 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.0174343 | 2.17315 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.044483 | 2.04806 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.0728293 | 2.09013 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.0147087 | 2.11159 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.096556 | 2.14976 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.0730191 | 2.11465 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.0474294 | 2.06365 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.159755 | 2.08093 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.0216998 | 2.08054 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.184488 | 2.06709 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.0285944 | 2.11571 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.0925496 | 2.07526 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.081927 | 2.08423 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.0227548 | 2.09787 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.0965703 | 2.13487 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.0216255 | 2.12444 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.068502 | 2.11666 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.0523915 | 2.06748 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.104804 | 2.11416 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.086065 | 2.11145 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.232645 | 2.18661 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.0771704 | 2.07235 |
| 16 | 15x5x5 | 0.016666667 | 2 | 0.00975004 | 2.04348 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.0874186 | 2.06874 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.109608 | 2.07652 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.124962 | 2.09872 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: multigrid-cohort16. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 370,016-parameter F32 one-channel multiscale pressure CNN uses physical q=(rho/h)r and 1 internal correction(s), recomputing the F64 equation residual between corrections. GroupNorm is removed; SiLU and all convolution biases are also removed. The captured output scale is previous-pressure-rms; cell-zero-gauged corrections accumulate in F64. The objective is canonical. dt is validated metadata, not an independent neural channel. It emits pressure only and advances no physical simulation time.

**Purpose.** This unnamed fit is retained to qualify manufactured multi-grid feedback before a longer scientific fit. It is distinct from any later named final-budget candidate.

**Experiment.** 10 CUDA Adam update(s), batch 1, rate 0.001, seed 20261015, private physical sampling seed 2026102501. The actual optimizer population has 16 physical parent(s)/32 frame(s); validation has 64 parents/128 frames. Training metadata contains grid resolutions [8.0, 16.0] cells per metre and fixed 1/60 s steps. Execution strategy is {'batch_mode': 'serial', 'finite_checks': 'combined', 'prepared_fields': False, 'profile_updates': 0, 'resident_inputs': True}. Active auxiliary streams: {'analytical_feedback': {'fields': 640, 'proposals': 10, 'weight': 0.1, 'sampling_order_rederived': True, 'grid_count': 5}}. Manufactured fields and frozen captured correction directions are additional operator examples, not independent simulation scenes. Original typed input/output closures, every retained checkpoint and sample orders verify without importing model code.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.232645 and worst equation residual relative L2 is 0.340387. Median case GPU latency is 2.08794 ms per pressure request; the largest case median is 2.22206 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** A one-, two- or ten-update qualification run proves execution/custody coverage at that budget, not a full scientific training result. Validation population size does not enlarge the optimizer population. Folder names such as early/late do not establish direction stage: preserve the actual recorded auxiliary manifest and stage. The common benchmark now quantifies this retained default policy; its training objective and smoke/publication duration are not pressure accuracy or deployment latency. Use the recorded same-request errors and complete pressure-call cost when deciding whether to retain or extend this design.

**Reproduction.** Use the exact captured typed train/restore recipe, source witness, runtime, original input blobs, whole-parent selection, train-only scales, private sample order and all checkpoint inventories. Preserve execution strategy and separately counted auxiliary streams. The historical trainer performs CPU publication/restore neural forwards and needs a GPU-only route before replay; the separate original-checkpoint GPU evaluation and timing are now qualified. Clean retraining and new-Rheon regeneration remain unqualified; the current common GPU pressure accuracy and latency are qualified. Original receipts/configuration and historical Trial ancestry remain unchanged. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "c8638134ae71a368d6943dbd8526c78d969cf434398342b9b696909245993f82"
  },
  "history": {
    "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/managed-smoke/stage-00/artifacts/history.json",
    "sha256": "39a68ba88ac7f4f611d7fd8e5ebb69e416ff982b5d63147fe3b24929d568f763"
  },
  "sources": [
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/analytical.py",
      "sha256": "b4b9bb8ad37f51bf0e7190f881588c5503c5d80d85b9c110d06fdaa42068ffcf"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/config.json",
      "sha256": "759702f4e323c8928823139401d27134556431495a520a892314261a1ed84a41"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/model.py",
      "sha256": "0886a7301877b92a0e62c1e6d759193262ec28adf40b0954b182e6f782fdb702"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/policy.py",
      "sha256": "6fac4ba260eb6c627722b68bf9cde7c7b934797cacc244204b6fad6afad347cc"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/qualification.json",
      "sha256": "98b71ef87aec2cab739a136981c678c9b5f63ec3d42ff594b86b457ec28ae2ef"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/train.py",
      "sha256": "5001b5123a6d468be8ae401c0bf54af60dd4289c577c8f930b812b2e978e493b"
    }
  ],
  "source_witness": {
    "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/source-witness.json",
    "sha256": "9bf3d793a1cb593a35bc01de32533d8a4a5ee9ee6f452cf6b643cb6d3d73fa6f"
  },
  "training_population": {
    "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/managed-smoke/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/project/inputs/validation_splits.json",
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
  "outcome": "Original final sampled loss 0.096908049712356795; recorded validation objective 0.035807094125703494 → 0.01082350775699922. Final-update weights remain selected; no optimizer-resume state is retained. These are historical objective values, not worst pressure relative error. A matched current GPU benchmark for this unnamed fit, full-state accuracy and own-history drift remain pending.",
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
    "item_id": "fit-c8638134ae71a368",
    "kind": "training_attempt",
    "name": "water-multigrid-cohort16-smoke · CUDA training",
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
    "receipt_sha256": "c8638134ae71a368d6943dbd8526c78d969cf434398342b9b696909245993f82",
    "receipt_paths": [
      "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/managed-smoke/stage-00/artifacts/receipt.json",
      "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/managed-smoke/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/d25f7512a7294af9bfb0d7db4fc89a1f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/d25f7512a7294af9bfb0d7db4fc89a1f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/d25f7512a7294af9bfb0d7db4fc89a1f/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/d25f7512a7294af9bfb0d7db4fc89a1f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/d25f7512a7294af9bfb0d7db4fc89a1f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-d25f7512a7294af9bfb0d7db4fc89a1f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-d25f7512a7294af9bfb0d7db4fc89a1f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-d25f7512a7294af9bfb0d7db4fc89a1f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-d25f7512a7294af9bfb0d7db4fc89a1f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/d25f7512a7294af9bfb0d7db4fc89a1f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/d25f7512a7294af9bfb0d7db4fc89a1f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/d25f7512a7294af9bfb0d7db4fc89a1f/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/d25f7512a7294af9bfb0d7db4fc89a1f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-d25f7512a7294af9bfb0d7db4fc89a1f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-d25f7512a7294af9bfb0d7db4fc89a1f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-d25f7512a7294af9bfb0d7db4fc89a1f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-d25f7512a7294af9bfb0d7db4fc89a1f/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-d25f7512a7294af9bfb0d7db4fc89a1f/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "d25f7512a7294af9bfb0d7db4fc89a1f",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-multigrid-cohort16-smoke · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "759702f4e323c8928823139401d27134556431495a520a892314261a1ed84a41",
      "inputs": {
        "dataset": {
          "bytes": 2820310,
          "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"
        },
        "dataset_extra": {
          "bytes": 2823990,
          "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"
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
          "bytes": 411192,
          "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
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
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "05facd8f56db80e159fdbc8ffc3612f558140ce5327eae18bd920b206aad41b0",
      "source_witness_sha256": "9bf3d793a1cb593a35bc01de32533d8a4a5ee9ee6f452cf6b643cb6d3d73fa6f"
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
        "sha256": "1b3aa11690401ab7b5a08571513acf0e72b7ddf50f4debb064625c2a44552177",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-010.npz",
        "sha256": "ec6d1c1e7f2dc1bcf97100ba842449bd2d0855a682ee9cebb460c66ad2ea2135",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "ec6d1c1e7f2dc1bcf97100ba842449bd2d0855a682ee9cebb460c66ad2ea2135",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2349309952,
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
      "receipt": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/managed-smoke/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-smoke/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "c8638134ae71a368d6943dbd8526c78d969cf434398342b9b696909245993f82",
    "bytes": 12649
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":1},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"759702f4e323c8928823139401d27134556431495a520a892314261a1ed84a41","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"05facd8f56db80e159fdbc8ffc3612f558140ce5327eae18bd920b206aad41b0","source_witness_sha256":"9bf3d793a1cb593a35bc01de32533d8a4a5ee9ee6f452cf6b643cb6d3d73fa6f"},"elapsed_s":4.753400071989745,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.056024042218496e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":533117440,"peak_reserved_bytes":553648128,"tf32":false},"mode":"train","optimizer_updates":10,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-005.npz":{"bytes":1485452,"role":"checkpoint","sha256":"1b3aa11690401ab7b5a08571513acf0e72b7ddf50f4debb064625c2a44552177"},"checkpoint-010.npz":{"bytes":1485452,"role":"checkpoint","sha256":"ec6d1c1e7f2dc1bcf97100ba842449bd2d0855a682ee9cebb460c66ad2ea2135"},"history.json":{"bytes":80524,"role":"training_diagnostics","sha256":"39a68ba88ac7f4f611d7fd8e5ebb69e416ff982b5d63147fe3b24929d568f763"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"ab56f79b6d87eefb0d9a7c771594d0adf2b7d8b97359a0ed31148726b96556be"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":977,"role":"result_summary","sha256":"1b7b60332019783fe148b7ec27f6c6532a41df4a3fe111e04e5462d005330153"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"ec6d1c1e7f2dc1bcf97100ba842449bd2d0855a682ee9cebb460c66ad2ea2135"}},"per_case_inference_s":[0.012066915980540216,0.001905217010062188,0.0023402369988616556,0.0018118919979315251,0.0016056440072134137,0.0015429740014951676,0.0019099900091532618,0.0019891189876943827,0.004883375018835068,0.0020552319765556604,0.0024371049948967993,0.0021028760238550603,0.006327131006401032,0.00172691501211375,0.002974392002215609,0.0017095229995902628,0.001260113000171259,0.0026605399907566607,0.0018447159964125603,0.0018842499994207174,0.0020006340055260807,0.0022634089982602745,0.00454856498981826,0.0018007030012086034],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"bb61939b8776ecff5d0ec3d7334fb27982c9a9225a5cb734098a0c243b161558","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"4b552b78131fb73208ed83e9a5fa101efecb7f34acb31a3c89c4b04076f5747f","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"627ca08b700c56f6798a40814d299753a1be2e49ffe82489e1aba792824dc8e8","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"ffe356fdce500f1d2e09c734bbf200750e8ef224e07f5a6d41e1f046dd8faaef","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"958ce5bed073adaaa1f8586dbd313736e8947d30e6ef48a7bbc3efa5cf533c0e","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"e5cca2b80ee1cfafbf79534376594476e2b98e74984cbabd2db14999b7840b9f","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"899f818330bec36027920922d8a2795481dc4d0b5fb076dace1e66cc989c9f63","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"12958c89e04721dfeae15622cac367bc60c3afcdca86b3ea6126baa884dad725","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"a7592bade38b23115f1b10a33f72916b89d3e6469793062c0ca47a5524aa21bf","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"68d20dab351356a537f8ab389a49e7b57add769d673270097e0482ed709eeaa5","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"99e50da0a7d4100d6f775d2ea0bbcd7af53f4ad05af71b2605a517cc00122d82","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"c60b4220cb83a79ea02a06948771248bf463e694d017d98d82e029e2975c4de8","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"54e0803cbde49eeefa7d5d04da62e8857f7341aa9dd4b3fcf58adc9563a54770","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"b06bcf6577885fe63ac2efd1ec1dba96386bba43e7efbbe38fa13cb19ec4704c","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"1a8986d8edb4ef108fefb2a1e74677059f4892554070cca2ed391169caf906e4","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"f8cc817499a9aa717c41465fc24b5e3b8c6f9bc205cb910ccc3b53df9dc21028","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"2d503690a3d0f9c13c2f174ab26561847b229adc0b86832725c6ce9eb2e9f5f2","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"a3ae9ad3ccaa7aefd99d7f7c6bd50ec6e25829ca7c8d3599875beae5abbaf2ca","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"ac6c7b0cced394f954c6131914297db8e6d09b162f11c1f0e64ec9228e90c3ef","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"6ad47cad0117734339f70a9b55bde8c60ef666059dc537c58705cd12455a64d2","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"78b000795258ca842ce449a4e3111ee73196232b8a610e49f7244c7dc6374c10","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"365213898591da392b3db18688432e3bfddd2a4cb138ebbf86396c1797ebf1a3","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"3c2501a1198b639ca236c92a15ecb96e4d198fa480de68a33bea0c6193bddabd","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"7a3d6d438385a92a4765a75dc5f374a9ac3af728016a72640e6b38cb1de6a86c","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2349309952,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
