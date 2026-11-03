# water-pressure-recovery-smoke · CUDA training

Retrospective source review · fit-befe8d913f41d1c8

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.389547 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.305328 | 4.16019 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.325667 | 4.17821 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.236939 | 4.18160 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.22989 | 4.43038 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.0383024 | 4.35184 |
| 8 | 7x16x5 | 0.016666667 | 2 | 0.0266282 | 4.15704 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.133849 | 4.21865 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.0473968 | 4.21540 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.0408149 | 4.16339 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.168306 | 4.32121 |
| 8 | 9x16x9 | 0.016666667 | 2 | 0.013606 | 4.39310 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.0139843 | 4.37629 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.168549 | 4.18892 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.0335623 | 4.15911 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.158754 | 4.41692 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.234477 | 4.27400 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.0117029 | 4.20948 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.141527 | 4.49489 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.144411 | 4.25942 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.17464 | 4.35617 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.155193 | 4.21877 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.063233 | 4.23604 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.0584843 | 4.23959 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.210226 | 4.43410 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.152858 | 4.20350 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.0660009 | 4.45780 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.254537 | 4.16473 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.0811717 | 4.40173 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.182633 | 4.29225 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.219115 | 4.46820 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.0697418 | 4.18360 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.126419 | 4.47462 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.07562 | 4.19880 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.16026 | 4.16399 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.242022 | 4.13379 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.156373 | 4.41304 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.11964 | 4.19330 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.0378335 | 4.17449 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.0330292 | 4.31028 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.0707219 | 4.29789 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.189059 | 4.40189 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.0302059 | 4.41643 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.120873 | 4.38164 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.173732 | 4.23332 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.109906 | 4.15885 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.238632 | 4.32798 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.0621238 | 4.17997 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.641487 | 4.27306 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.0459741 | 4.44221 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.119535 | 4.14798 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.224795 | 4.17243 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.0371697 | 4.23340 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.190603 | 4.15902 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.0357737 | 4.26764 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.218435 | 4.46034 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.0881777 | 4.22079 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.136097 | 4.56475 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.171281 | 4.35487 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.358713 | 4.15935 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.128513 | 4.32833 |
| 16 | 15x5x5 | 0.016666667 | 2 | 0.0191225 | 4.20187 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.154491 | 4.24463 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.277173 | 4.51352 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.160957 | 4.21368 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: pressure-recovery. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 370,016-parameter F32 one-channel multiscale pressure CNN uses physical q=(rho/h)r and 2 internal correction(s), recomputing the F64 equation residual between corrections. GroupNorm is removed; SiLU and all convolution biases are also removed. The captured output scale is previous-pressure-rms; cell-zero-gauged corrections accumulate in F64. The objective is canonical-intermediate-contraction. dt is validated metadata, not an independent neural channel. It emits pressure only and advances no physical simulation time.

**Purpose.** This unnamed fit is retained to qualify the captured physical residual/objective/auxiliary-stream path before its full-budget study. It is distinct from any later named final-budget candidate.

**Experiment.** 1 CUDA Adam update(s), batch 1, rate 0.001, seed 20261015, private physical sampling seed 2026102501. The actual optimizer population has 16 physical parent(s)/32 frame(s); validation has 64 parents/128 frames. Training metadata contains grid resolutions [8.0, 16.0] cells per metre and fixed 1/60 s steps. Execution strategy is {'batch_mode': 'serial', 'finite_checks': 'combined', 'prepared_fields': False, 'profile_updates': 0, 'resident_inputs': True}. Active auxiliary streams: {'analytical_feedback': {'fields': 1024, 'proposals': 1, 'weight': 0.1, 'sampling_order_rederived': True, 'grid_count': 8}, 'supplemental_feedback': {'fields': 768, 'proposals': 1, 'weight': 0.1, 'sampling_order_rederived': True, 'grid_count': 6}, 'direction_feedback': {'cases': 192, 'stage': 'late', 'groups': 21, 'proposals': 1, 'weight': 0.1, 'sampling_order_rederived': True, 'label_arrays_decoded': False}}. Manufactured fields and frozen captured correction directions are additional operator examples, not independent simulation scenes. Original typed input/output closures, every retained checkpoint and sample orders verify without importing model code.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.641487 and worst equation residual relative L2 is 1.35042. Median case GPU latency is 4.22614 ms per pressure request; the largest case median is 4.57802 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** A one-, two- or ten-update qualification run proves execution/custody coverage at that budget, not a full scientific training result. Validation population size does not enlarge the optimizer population. Folder names such as early/late do not establish direction stage: preserve the actual recorded auxiliary manifest and stage. The common benchmark now quantifies this retained default policy; its training objective and smoke/publication duration are not pressure accuracy or deployment latency. Use the recorded same-request errors and complete pressure-call cost when deciding whether to retain or extend this design.

**Reproduction.** Use the exact captured typed train/restore recipe, source witness, runtime, original input blobs, whole-parent selection, train-only scales, private sample order and all checkpoint inventories. Preserve execution strategy and separately counted auxiliary streams. The historical trainer performs CPU publication/restore neural forwards and needs a GPU-only route before replay; the separate original-checkpoint GPU evaluation and timing are now qualified. Clean retraining and new-Rheon regeneration remain unqualified; the current common GPU pressure accuracy and latency are qualified. Original receipts/configuration and historical Trial ancestry remain unchanged. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-pressure-recovery-v1/smoke/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "befe8d913f41d1c876f1466d14aba6d21f5d2411890c6ea0478c428e18db39f8"
  },
  "history": {
    "path": "runs/water-pressure-recovery-v1/smoke/managed-full/stage-00/artifacts/history.json",
    "sha256": "c31627760edfe7376e5409c6265596cff36b8ab23a8e4c87cf7e704c32f0767d"
  },
  "sources": [
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/config.json",
      "sha256": "333c786be3e0e8fe6361a06a44b5eb60eeb670b7894657868f42dde1a336fcd6"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/model.py",
      "sha256": "0bce2799411bb8da0e2dec5d82638f63959d9849924265b4c95fc239fcd57e99"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/policy.py",
      "sha256": "60aafd2c399fecf7bc6852b7e5ec386617741415cdd424e6aaeb29226a4ce7da"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/qualification.json",
      "sha256": "af8f182135bb14ab0647bbccc86bab8e06551e69841955e23fb68ba0f255bc4d"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-pressure-recovery-v1/smoke/project/train.py",
      "sha256": "62c7dd0ad8bcc1b4e707800d6d329c318962d8e4abce973bf722e832f5f791b8"
    }
  ],
  "source_witness": {
    "path": "runs/water-pressure-recovery-v1/smoke/project/source-witness.json",
    "sha256": "66ce771cb19dfe38b2ad366bfedf0b0a202729cff7af3a4f41313f82ad165d9f"
  },
  "training_population": {
    "path": "runs/water-pressure-recovery-v1/smoke/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-pressure-recovery-v1/smoke/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-pressure-recovery-v1/smoke/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 24476086,
      "path": "runs/water-pressure-recovery-v1/smoke/project/inputs/direction_dataset.npz",
      "sha256": "4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6",
      "slot": "direction_dataset"
    },
    {
      "bytes": 513808,
      "path": "runs/water-pressure-recovery-v1/smoke/project/inputs/direction_manifest.json",
      "sha256": "15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06",
      "slot": "direction_manifest"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-pressure-recovery-v1/smoke/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-pressure-recovery-v1/smoke/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-pressure-recovery-v1/smoke/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-pressure-recovery-v1/smoke/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-pressure-recovery-v1/smoke/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-pressure-recovery-v1/smoke/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-pressure-recovery-v1/smoke/project/inputs/validation_splits.json",
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
  "outcome": "Original final sampled loss 573114.40262296004; recorded validation objective 0.11340079977840754 → 0.11271434997911732. Final-update weights remain selected; no optimizer-resume state is retained. These are historical objective values, not worst pressure relative error. A matched current GPU benchmark for this unnamed fit, full-state accuracy and own-history drift remain pending.",
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
    "item_id": "fit-befe8d913f41d1c8",
    "kind": "training_attempt",
    "name": "water-pressure-recovery-smoke · CUDA training",
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
    "receipt_sha256": "befe8d913f41d1c876f1466d14aba6d21f5d2411890c6ea0478c428e18db39f8",
    "receipt_paths": [
      "runs/water-pressure-recovery-v1/smoke/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-pressure-recovery-v1/smoke/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/b19583136d444bd89f1138dea59b94ad/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/b19583136d444bd89f1138dea59b94ad/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/b19583136d444bd89f1138dea59b94ad/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/b19583136d444bd89f1138dea59b94ad/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/b19583136d444bd89f1138dea59b94ad/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-b19583136d444bd89f1138dea59b94ad/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-b19583136d444bd89f1138dea59b94ad/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-b19583136d444bd89f1138dea59b94ad/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-b19583136d444bd89f1138dea59b94ad/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/b19583136d444bd89f1138dea59b94ad/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/b19583136d444bd89f1138dea59b94ad/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/b19583136d444bd89f1138dea59b94ad/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/b19583136d444bd89f1138dea59b94ad/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-b19583136d444bd89f1138dea59b94ad/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-b19583136d444bd89f1138dea59b94ad/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-b19583136d444bd89f1138dea59b94ad/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-b19583136d444bd89f1138dea59b94ad/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-b19583136d444bd89f1138dea59b94ad/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "b19583136d444bd89f1138dea59b94ad",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-pressure-recovery-smoke · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "333c786be3e0e8fe6361a06a44b5eb60eeb670b7894657868f42dde1a336fcd6",
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
          "bytes": 24476086,
          "sha256": "4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"
        },
        "direction_manifest": {
          "bytes": 513808,
          "sha256": "15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06"
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
      "source_witness_sha256": "66ce771cb19dfe38b2ad366bfedf0b0a202729cff7af3a4f41313f82ad165d9f"
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
        "sha256": "c968f7cd99c3e73fab70184ebd7953aa92c43a512f62387718a14329c37a14d4",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "c968f7cd99c3e73fab70184ebd7953aa92c43a512f62387718a14329c37a14d4",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2353602560,
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
      "receipt": "runs/water-pressure-recovery-v1/smoke/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-pressure-recovery-v1/smoke/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "befe8d913f41d1c876f1466d14aba6d21f5d2411890c6ea0478c428e18db39f8",
    "bytes": 13801
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"first_correction_pressure_feedback_weight":0.1,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"333c786be3e0e8fe6361a06a44b5eb60eeb670b7894657868f42dde1a336fcd6","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":513808,"sha256":"15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"f137c1b9244c69befd532cde2486d4fe8b6f681deda0487b8e5757e0d09619ec","source_witness_sha256":"66ce771cb19dfe38b2ad366bfedf0b0a202729cff7af3a4f41313f82ad165d9f"},"elapsed_s":6.695533006000915,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.652934250590988e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":767402496,"peak_reserved_bytes":790626304,"tf32":false},"mode":"train","optimizer_updates":1,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-001.npz":{"bytes":1485452,"role":"checkpoint","sha256":"c968f7cd99c3e73fab70184ebd7953aa92c43a512f62387718a14329c37a14d4"},"history.json":{"bytes":204322,"role":"training_diagnostics","sha256":"c31627760edfe7376e5409c6265596cff36b8ab23a8e4c87cf7e704c32f0767d"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"d72a2a63598b68cdcb11a23245d13665a435ceca87c0fad2b862ca5dc42c8bc7"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":974,"role":"result_summary","sha256":"774206830b709a173fdf02d135363a7ce31d171502b877705b7dfebfc1007c2c"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"c968f7cd99c3e73fab70184ebd7953aa92c43a512f62387718a14329c37a14d4"}},"per_case_inference_s":[0.021180814001127146,0.004344033004599623,0.005502441999851726,0.004417304997332394,0.0038845980016048998,0.0036857980012428015,0.004640503000700846,0.005003703001420945,0.009811178999370895,0.0053999689989723265,0.006227535996004008,0.005523038998944685,0.011911748995771632,0.004400749996420927,0.007013768001343124,0.004878307001490612,0.0031082519999472424,0.006377107994921971,0.004668323002988473,0.004930153998429887,0.00525346799986437,0.006051142998330761,0.013007913999899756,0.004760463998536579],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"a4c44277625c55142d9f50579f15b71860a5cf3e158553690e0a0067a5cd54df","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"05d9e0d174eb14871bd9d3390e741e76cce22bf0eeedeb42b0be8abf1f5f7501","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"bce8f5124bd0c5fc615ab1ee5a11f0f98122f9948d4940d790fad4c42be2a371","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"7924a7dc113383a8233d4ddebe5bea4b2c796f949f4527aa9f7fbce768170968","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"812a0cbe5cd2d5d205606d583c6120093b994550fc04c5d09801ef84012b15bf","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"99a7103ce44be7d3082c3c5aefc4a8321410fa1c016d2c8c4f69511ef97b6b0a","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"bb3049aa6e3130bd4d8ca051944223c98ed98defcc0cdc8c2b786538969e30f0","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"f95396f50f3df1373cb309beab67516b13cc11136d2a3129f4cf7dccd89192dc","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"55c1a3b5903c52dde3df5b490b5347528e3519283b6a934889e9b886b938b987","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"cb8b8c49f2b5e61dd084c787207951f5ae091b60171c62fb2c4453f379242ecd","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"5eabefd7919f10fdf82f0eaff4d92ec90728c7f22457f977ad599ab873cab5ba","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"88dad1595588eee3231b0e6882e712339741595cded0d494c8d1a731b94cf4de","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"6c9fb1be3dce6d39462c3050014f10a8c92ab65790f4af742af6e8d97ae50514","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"33358a2c54394b931fd951cbe5b5e576777d40a2ebd6a9ef7c8cf4ce358bb942","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"4742983915f30d7bbb0d85a4643319524721bd3dbe05bcf69bb01db208976993","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"e47de141eaf73ecb2269434976530583695bfd3fbc761b9c75f8874c3acc0243","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"24b033a34959cbe304bb7a5fa9ddbcbcda170ff5282a9c44cf67c447b2b605a6","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"383e90ba86ef0e3ebae2ef9eff5aba463a21cdc9fcabcf1ceaed8e17f73a4373","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"ba60acbe43ced0b7b83f6aadbe89389ffff8fbeb7726bb0852cf915f7d6f70d8","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"5195ab176dab80aabeab198441c6e3e09a8a40bd1c9d1ddf126b11b33fea6590","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"1059c2bf557655e271c955f18e41a6c9a1b301d91c0ae3dd43785c733a649d13","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"f9e09d52c94ce7237835f7bc5c8432bb79c1fe81e9f7efd8f44f00e52a747230","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"16bf351fb3a3cd77e8bcb2942e0d483a56144f96008fbbb8ba33a9420ac003a1","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"017fe390577c2e24722fbd20e9f7649b7208229916e9a3a0f1a48e01a92599ba","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2353602560,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
