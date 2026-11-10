# water-direction-smoke · CUDA training

Retrospective source review · fit-803a07c660a70a93

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.414864 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.305328 | 4.13168 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.325668 | 4.32070 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.236938 | 4.11548 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.22989 | 4.34429 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.0383024 | 4.13905 |
| 8 | 7x16x5 | 0.016666667 | 2 | 0.0266282 | 4.15479 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.133849 | 4.16953 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.0473967 | 4.44240 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.0408148 | 4.15617 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.168307 | 4.35946 |
| 8 | 9x16x9 | 0.016666667 | 2 | 0.013606 | 4.21297 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.0139842 | 4.16909 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.168549 | 4.39676 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.0335623 | 4.19434 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.158754 | 4.28602 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.234477 | 4.45303 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.0117029 | 4.48119 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.141527 | 4.18833 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.144412 | 4.44076 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.174639 | 4.35109 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.155193 | 4.16979 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.0632329 | 4.50046 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.0584844 | 4.22271 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.210226 | 4.41239 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.152858 | 4.16608 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.0660009 | 4.19204 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.254537 | 4.19035 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.0811716 | 4.15898 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.182633 | 4.14677 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.219115 | 4.40394 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.0697418 | 4.12982 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.126419 | 4.39549 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.0756201 | 4.44009 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.16026 | 4.17929 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.242022 | 4.43646 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.156373 | 4.14805 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.11964 | 4.17575 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.0378335 | 4.29809 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.0330293 | 4.45813 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.0707218 | 4.17083 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.18906 | 4.12249 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.0302058 | 4.15081 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.120873 | 4.14180 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.173732 | 4.24181 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.109906 | 4.17367 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.238632 | 4.16411 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.0621239 | 4.18205 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.641487 | 4.11129 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.045974 | 4.37226 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.119535 | 4.34613 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.224795 | 4.15461 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.0371696 | 4.28783 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.190603 | 4.48442 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.0357737 | 4.16545 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.218435 | 4.36527 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.0881778 | 4.14603 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.136097 | 4.39367 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.171281 | 4.43942 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.358713 | 4.23803 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.128512 | 4.27666 |
| 16 | 15x5x5 | 0.016666667 | 2 | 0.0191224 | 4.45409 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.15449 | 4.17499 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.277173 | 4.42368 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.160956 | 4.42968 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: direction-smoke. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 370,016-parameter F32 one-channel multiscale pressure CNN uses physical q=(rho/h)r and 2 internal correction(s), recomputing the F64 equation residual between corrections. GroupNorm is removed; SiLU and all convolution biases are also removed. The captured output scale is previous-pressure-rms; cell-zero-gauged corrections accumulate in F64. The objective is canonical-intermediate-contraction. dt is validated metadata, not an independent neural channel. It emits pressure only and advances no physical simulation time.

**Purpose.** This unnamed fit is retained to qualify the captured physical residual/objective/auxiliary-stream path before its full-budget study. It is distinct from any later named final-budget candidate.

**Experiment.** 1 CUDA Adam update(s), batch 1, rate 0.001, seed 20261015, private physical sampling seed 2026102501. The actual optimizer population has 16 physical parent(s)/32 frame(s); validation has 64 parents/128 frames. Training metadata contains grid resolutions [8.0, 16.0] cells per metre and fixed 1/60 s steps. Execution strategy is {'batch_mode': 'serial', 'finite_checks': 'combined', 'prepared_fields': False, 'profile_updates': 0, 'resident_inputs': True}. Active auxiliary streams: {'analytical_feedback': {'fields': 1024, 'proposals': 1, 'weight': 0.1, 'sampling_order_rederived': True, 'grid_count': 8}, 'supplemental_feedback': {'fields': 768, 'proposals': 1, 'weight': 0.1, 'sampling_order_rederived': True, 'grid_count': 6}, 'direction_feedback': {'cases': 144, 'stage': 'early', 'groups': 15, 'proposals': 1, 'weight': 0.1, 'sampling_order_rederived': True, 'label_arrays_decoded': False}}. Manufactured fields and frozen captured correction directions are additional operator examples, not independent simulation scenes. Original typed input/output closures, every retained checkpoint and sample orders verify without importing model code.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.641487 and worst equation residual relative L2 is 1.35042. Median case GPU latency is 4.21393 ms per pressure request; the largest case median is 4.56415 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** A one-, two- or ten-update qualification run proves execution/custody coverage at that budget, not a full scientific training result. Validation population size does not enlarge the optimizer population. Folder names such as early/late do not establish direction stage: preserve the actual recorded auxiliary manifest and stage. The common benchmark now quantifies this retained default policy; its training objective and smoke/publication duration are not pressure accuracy or deployment latency. Use the recorded same-request errors and complete pressure-call cost when deciding whether to retain or extend this design.

**Reproduction.** Use the exact captured typed train/restore recipe, source witness, runtime, original input blobs, whole-parent selection, train-only scales, private sample order and all checkpoint inventories. Preserve execution strategy and separately counted auxiliary streams. The historical trainer performs CPU publication/restore neural forwards and needs a GPU-only route before replay; the separate original-checkpoint GPU evaluation and timing are now qualified. Clean retraining and new-Rheon regeneration remain unqualified; the current common GPU pressure accuracy and latency are qualified. Original receipts/configuration and historical Trial ancestry remain unchanged. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-late-residual-v1/smoke/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "803a07c660a70a9345ae4522de601f801e7fa5d37187b76a2a17d52b2c26ae4c"
  },
  "history": {
    "path": "runs/water-late-residual-v1/smoke/managed-smoke/stage-00/artifacts/history.json",
    "sha256": "01e7dc6a00acdfb52f9256e339b2c69c080039e5c16b096966802ab3ba391419"
  },
  "sources": [
    {
      "path": "runs/water-late-residual-v1/smoke/project/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/config.json",
      "sha256": "bbf04e7d25b525d6b3a7b7f4802abc6f776009e788bc2b9b5b3011a17fac9cdc"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/model.py",
      "sha256": "ea450a65f1b5fa75cead93e449b85070a4a24ae342182ac1fc1d0dbe355d6102"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/policy.py",
      "sha256": "8c82ff6067ac4ea097a08f638f376d0ef0d669b8ed35115bddf2e5012ee6753e"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/qualification.json",
      "sha256": "af8f182135bb14ab0647bbccc86bab8e06551e69841955e23fb68ba0f255bc4d"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-late-residual-v1/smoke/project/train.py",
      "sha256": "758c9b54cc028233ebb163b23aa700f8ba93ff9597c8991feeca1a2f5d5f1315"
    }
  ],
  "source_witness": {
    "path": "runs/water-late-residual-v1/smoke/project/source-witness.json",
    "sha256": "d4b75b2ad021f21110a6c1b2fe14590b0a8cd2ea849507b2292f3cee2efcd2bc"
  },
  "training_population": {
    "path": "runs/water-late-residual-v1/smoke/managed-smoke/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-late-residual-v1/smoke/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-late-residual-v1/smoke/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 21428086,
      "path": "runs/water-late-residual-v1/smoke/project/inputs/direction_dataset.npz",
      "sha256": "83f147a056f94366ddedeeaf3505eab51e8efba6bd33e033283c07a0e4b06a08",
      "slot": "direction_dataset"
    },
    {
      "bytes": 383531,
      "path": "runs/water-late-residual-v1/smoke/project/inputs/direction_manifest.json",
      "sha256": "4b0a35e50ad5958bbc69c074f90ddacaa76c10435e1c3b731cf9be19b40f9135",
      "slot": "direction_manifest"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-late-residual-v1/smoke/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-late-residual-v1/smoke/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-late-residual-v1/smoke/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-late-residual-v1/smoke/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-late-residual-v1/smoke/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-late-residual-v1/smoke/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-late-residual-v1/smoke/project/inputs/validation_splits.json",
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
  "outcome": "Original final sampled loss 573090.3487606314; recorded validation objective 0.11340079977840754 → 0.11271450277420847. Final-update weights remain selected; no optimizer-resume state is retained. These are historical objective values, not worst pressure relative error. A matched current GPU benchmark for this unnamed fit, full-state accuracy and own-history drift remain pending.",
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
    "item_id": "fit-803a07c660a70a93",
    "kind": "training_attempt",
    "name": "water-direction-smoke · CUDA training",
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
        "steps": 2
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 65536
    },
    "receipt_sha256": "803a07c660a70a9345ae4522de601f801e7fa5d37187b76a2a17d52b2c26ae4c",
    "receipt_paths": [
      "runs/water-late-residual-v1/smoke/managed-smoke/stage-00/artifacts/receipt.json",
      "runs/water-late-residual-v1/smoke/managed-smoke/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/f5818120176842448f2e0ef520a25c1c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/f5818120176842448f2e0ef520a25c1c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/f5818120176842448f2e0ef520a25c1c/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/f5818120176842448f2e0ef520a25c1c/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/f5818120176842448f2e0ef520a25c1c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-f5818120176842448f2e0ef520a25c1c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-f5818120176842448f2e0ef520a25c1c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-f5818120176842448f2e0ef520a25c1c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-f5818120176842448f2e0ef520a25c1c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/f5818120176842448f2e0ef520a25c1c/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/f5818120176842448f2e0ef520a25c1c/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/f5818120176842448f2e0ef520a25c1c/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/f5818120176842448f2e0ef520a25c1c/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-f5818120176842448f2e0ef520a25c1c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-f5818120176842448f2e0ef520a25c1c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-f5818120176842448f2e0ef520a25c1c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-f5818120176842448f2e0ef520a25c1c/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-f5818120176842448f2e0ef520a25c1c/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "f5818120176842448f2e0ef520a25c1c",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-direction-smoke · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "bbf04e7d25b525d6b3a7b7f4802abc6f776009e788bc2b9b5b3011a17fac9cdc",
      "inputs": {
        "dataset": {
          "bytes": 2820310,
          "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"
        },
        "dataset_extra": {
          "bytes": 2823990,
          "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"
        },
        "direction_dataset": {
          "bytes": 21428086,
          "sha256": "83f147a056f94366ddedeeaf3505eab51e8efba6bd33e033283c07a0e4b06a08"
        },
        "direction_manifest": {
          "bytes": 383531,
          "sha256": "4b0a35e50ad5958bbc69c074f90ddacaa76c10435e1c3b731cf9be19b40f9135"
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
      "sample_order_sha256": "f137c1b9244c69befd532cde2486d4fe8b6f681deda0487b8e5757e0d09619ec",
      "source_witness_sha256": "d4b75b2ad021f21110a6c1b2fe14590b0a8cd2ea849507b2292f3cee2efcd2bc"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-001.npz",
        "sha256": "d091fca502f0acfb7130e4eca82c3c078041164547aebe18ba89ff9e4c1179b8",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "d091fca502f0acfb7130e4eca82c3c078041164547aebe18ba89ff9e4c1179b8",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2338951168,
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
      "receipt": "runs/water-late-residual-v1/smoke/managed-smoke/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-late-residual-v1/smoke/managed-smoke/stage-00/artifacts/receipt.json",
    "sha256": "803a07c660a70a9345ae4522de601f801e7fa5d37187b76a2a17d52b2c26ae4c",
    "bytes": 13755
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"4b0a35e50ad5958bbc69c074f90ddacaa76c10435e1c3b731cf9be19b40f9135","seed":2026102701,"stage":"early","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"bbf04e7d25b525d6b3a7b7f4802abc6f776009e788bc2b9b5b3011a17fac9cdc","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"direction_dataset":{"bytes":21428086,"sha256":"83f147a056f94366ddedeeaf3505eab51e8efba6bd33e033283c07a0e4b06a08"},"direction_manifest":{"bytes":383531,"sha256":"4b0a35e50ad5958bbc69c074f90ddacaa76c10435e1c3b731cf9be19b40f9135"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"f137c1b9244c69befd532cde2486d4fe8b6f681deda0487b8e5757e0d09619ec","source_witness_sha256":"d4b75b2ad021f21110a6c1b2fe14590b0a8cd2ea849507b2292f3cee2efcd2bc"},"elapsed_s":5.996854814000017,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.45396417923871e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":762757632,"peak_reserved_bytes":786432000,"tf32":false},"mode":"train","optimizer_updates":1,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-001.npz":{"bytes":1485452,"role":"checkpoint","sha256":"d091fca502f0acfb7130e4eca82c3c078041164547aebe18ba89ff9e4c1179b8"},"history.json":{"bytes":204170,"role":"training_diagnostics","sha256":"01e7dc6a00acdfb52f9256e339b2c69c080039e5c16b096966802ab3ba391419"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"2e27406566f4067e82b46f04073cc72faf1ab8c401080b654d991c851c2c1ce9"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":974,"role":"result_summary","sha256":"b0d586fbb288731c34a9bcff42fa2a74cc347d56cb70d4f71679394c4d3e0e6b"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"d091fca502f0acfb7130e4eca82c3c078041164547aebe18ba89ff9e4c1179b8"}},"per_case_inference_s":[0.019365673999800492,0.0037023620002401003,0.004474263999782124,0.0032042349998846476,0.002843243000370421,0.0027847119999933057,0.003368981000221538,0.0036561349998009973,0.00742790000003879,0.004395222999846737,0.005693182999948476,0.005279910999888671,0.011237589999836928,0.004233765000208223,0.006543612999848847,0.00431088200002705,0.0027141289997416607,0.005407335000199964,0.0038563759999306058,0.003983599000093818,0.004232032999880175,0.004541808999874775,0.008810536000055436,0.0033813229997576855],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"3759b85f3128234c88c843b3ceba4f146ae1f0e175dd39322cbb01abf2868db3","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"77981df7316a128428e147e0cdafa6dd5b65bc68a327a05854c9bd4f2b1928e4","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"1a13ce9ed5bafc3cdfc0848de3b01816f6d0b0a751748edbd0e20e7dd28bfb9c","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"03b102de7c48c7290ccb6d62f6956832a6b8a811270cfe802b8d2f400fe1fd07","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"6090ca3364331763ff269e98de013114cb1e7ff1c485ad2507f8f0f60702835d","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"5c306e7cbebb1e1676fb478c8d4b7ccb0f753c46f667189e537edace15dfa06e","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"58a6f6ac22c26ecceedc3fea98f4aee857b1e6a37189f3b6e447700658ec97e0","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"3695b982830af1e7b706dc5ef28bbe0ddf702c6c8268db5a46b67c8ec9d4e6f3","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"cb713092a4540383d97afbd48e08d72220fa1e6fc11192e56d69f0da4ccd18a3","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"17c0a2a140e657c4051d3895c987371c6182dd359d6520a8896443e3cd174179","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"f132cc2a88d9f1a6489af2ab195b77aec7a16fc23e1034e27788bf9681a4fb1f","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"714883bb4836d66d6f8f4c97b9002822a8b49246fd163774d8c5c7df9baaa32f","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"e3ec232254b067072cb7ead3e61ee617cc7ff710854a2b817fa5bfeffeae0d19","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"7794eabc51812a247c87aeafbb4965dc13f73d3e2a2cc04983ea869a173799ed","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"c0a8e833064fee8a432f48ca8d7de0271967e4ce0e1f3b391cc98fe05b6cf6d4","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"f51069ccec82b1d881014e42435f237b6d0204936e758ad14c5303d05b0a2027","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"6f95ef26fff1c39f31c1065b35e42e0746511ab2f98c443b2fd57ef19b3ccc62","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"846b2fa43ff337c43bb0573efb4a4084e4a8cdca77964160ed8e6b2a9d657db2","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"0a980becc85504f41f46f83786f4ac3e47adf5429724855e556b32485b772de8","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"d88ddfa48e06ffc94f6e14d781d546234b079035121193be627599da038c9b05","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"f5895374842bd765d9733c873ea219cf0334cf00a72eecf836f7252647b9f1d1","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"ad7098256cf0a48cbde4db931c3008512358119c8f87462f5cfcc04816f5da87","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"9a42edeb0727e0f949439f5eeb0604db0055540587eddfa3c7cff5d869193b27","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"ae110b5eec3182d8094b3ec4f2d0143c25f14e1d833bed3f12247a9bc879c2d1","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2338951168,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
