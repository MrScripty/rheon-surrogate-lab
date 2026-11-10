# water-coarse-control CUDA

Retrospective source review · fit-047049129e1e2833

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.351042 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.000943575 | 4.11667 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.00515387 | 4.13671 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.00124977 | 4.13330 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.00237343 | 4.12291 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.000261875 | 4.12376 |
| 8 | 7x16x5 | 0.016666667 | 2 | 9.15537e-05 | 4.10980 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.00271775 | 4.37256 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.000957489 | 4.38922 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.000374268 | 4.24422 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.00137323 | 4.09653 |
| 8 | 9x16x9 | 0.016666667 | 2 | 6.41764e-05 | 4.14260 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.00016374 | 4.13263 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.0012628 | 4.40729 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.000516109 | 4.20072 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.00228939 | 4.15312 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.00207202 | 4.15880 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.000342452 | 4.42295 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.00214157 | 4.35466 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.001702 | 4.32686 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.00162384 | 4.13707 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.00506489 | 4.20414 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.00108084 | 4.15289 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.00097839 | 4.18400 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.00314996 | 4.14829 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.00352534 | 4.30652 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.00021379 | 4.39010 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.00602154 | 4.13744 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.00174138 | 4.20971 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.0014434 | 4.12559 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.00550607 | 4.20833 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.00131009 | 4.20183 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.00336827 | 4.18563 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.000586011 | 4.31232 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.00082191 | 4.32466 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.00442927 | 4.34985 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.00490658 | 4.13409 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.000670917 | 4.16476 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.000634117 | 4.12355 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.000566305 | 4.15292 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.00173921 | 4.14688 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.00206493 | 4.13168 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.000625654 | 4.42921 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.000431156 | 4.35549 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.00126738 | 4.33250 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.000824418 | 4.15387 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.00301048 | 4.13849 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.00068902 | 4.16733 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.00393296 | 4.11458 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.000605029 | 4.15675 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.00212468 | 4.35504 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.00158584 | 4.14710 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.00036592 | 4.14782 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.00373396 | 4.16163 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.000259161 | 4.17232 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.00536618 | 4.16073 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.00230699 | 4.38663 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.00306426 | 4.15963 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.00199928 | 4.10502 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.00350293 | 4.14799 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.00600278 | 4.19342 |
| 16 | 15x5x5 | 0.016666667 | 2 | 0.000173194 | 4.32052 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.00521885 | 4.14876 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.00370903 | 4.14657 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.0020004 | 4.35166 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: coarse-control. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 370,016-parameter linear, bias-free F32 multiscale pressure CNN with one physically scaled residual channel and 19 weight tensors. Two shared neural pressure corrections recompute the F64 sealed cubic-grid equation residual, apply fixed train-only scales and accumulate cell-zero-gauged F64 pressure. Inherited GroupNorm/SiLU labels do not describe its executed linear blocks. It returns pressure, not a physical simulation step; no learned spatial coarse branch is present.

**Purpose.** Establish the base-network control for adding the learned spatial coarse branch; this is a separate completed fit with parameter bytes equal to C53.

**Experiment.** 5,000 CUDA Adam updates, batch one, rate 0.001, model seed 20261015, physical-index seed 2026102501, schedule [{'learning_rate': 0.0001, 'update': 3000}, {'learning_rate': 1e-05, 'update': 4500}]. Sixteen physical fitting trajectories/32 frames at 8/16 cells per metre, dt=1/60 s and fluid density 1000 kg/m3. Both manufactured banks and the fixed captured-direction bank remain active; an additional frozen current-policy bank supplies late directions. First/final low-spatial-mode feedback weight is 0.03. The actual entrypoint is evidence.py; diagnostics use training parents only. The typed evidence wrapper also queries one already-consumed validation frame for publication; it supplies no independent holdout.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.00602154 and worst equation residual relative L2 is 0.00390012. Median case GPU latency is 4.16087 ms per pressure request; the largest case median is 4.50064 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** Keep lifecycle qualification, completed optimizer budgets and scientific accuracy qualification distinct. Repeating an unchanged fit for publication or scheduling can reproduce parameter bytes without creating new scene diversity or an architectural improvement. Byte-equal weights may share retained storage later, but original fit/Run evidence remains separate. A one-update smoke is not a full training result or generalization test. The common population exactly reproduces all 128 pressure fields of C53; retain separate fit histories despite shared final parameter bytes. This establishes only this population, not general execution-policy equivalence.

**Reproduction.** Retain the exact typed entrypoint and its explicit recipe source bindings, original config/runtime/source witness, all physical/manufactured/captured inputs, train-only scales, draw order and both checkpoints. Preserve native profile renaming and any original consumed publication probe. The historical training and publication paths still need GPU-only replay qualification; the separate original-checkpoint GPU evaluation and timing are now qualified. No original receipt alias, Trial ancestry or scientific scalar is rewritten. Clean retraining/new-Rheon generation and long own-history drift remain pending. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-spatial-coarse-v1/control-study-01/train/receipt.json",
    "sha256": "047049129e1e28332143b4e532d3eb3b1868e76a68c54ee375fffef266427011"
  },
  "history": {
    "path": "runs/pressure-spatial-coarse-v1/control-study-01/train/history.json",
    "sha256": "d923701b3b8e0e5f7495a44bc6084c38e2de627c60b4c811187d1c309a991a9e"
  },
  "sources": [
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/config.json",
      "sha256": "1793bcc8bede3b79c29654cb950c70ef2b65cd49958997b8835024bdc58469bb"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/custody-validator.py",
      "sha256": "247c573d6a9bfdc4e189b3a940650c91102cb40273751225f0fef6dd79999e6f"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/train.py",
      "sha256": "152d714442cca48a62afbcd0f2e0c2d43000d2a17710400b50ec1788a5bb6fa7"
    },
    {
      "path": "runs/pressure-spatial-coarse-v1/control-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-spatial-coarse-v1/control-project-01/source-witness.json",
    "sha256": "5fa6dbbb6adc91409b4312b383be707bbcba1dcb97b35bfe53cc4cec324c4e4e"
  },
  "training_population": {
    "path": "runs/pressure-spatial-coarse-v1/control-study-01/train/splits.json",
    "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
  },
  "training_inputs": [
    {
      "bytes": 21428086,
      "path": "runs/pressure-current-policy-v1/bank-02/late.npz",
      "sha256": "8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521",
      "slot": "current_dataset"
    },
    {
      "bytes": 168912,
      "path": "runs/pressure-current-policy-v1/bank-02/manifest.json",
      "sha256": "fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e",
      "slot": "current_manifest"
    },
    {
      "bytes": 610070,
      "path": "runs/dependency-migration-v1/training-project-6/inputs/dataset.npz",
      "sha256": "89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e",
      "slot": "dataset"
    },
    {
      "bytes": 610806,
      "path": "runs/dependency-migration-v1/training-project-6/inputs/dataset_extra.npz",
      "sha256": "c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5",
      "slot": "dataset_extra"
    },
    {
      "bytes": 24476086,
      "path": "runs/dependency-migration-v1/training-project-6/inputs/direction_dataset.npz",
      "sha256": "4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6",
      "slot": "direction_dataset"
    },
    {
      "bytes": 298490,
      "path": "runs/dependency-migration-v1/training-project-6/inputs/direction_manifest.json",
      "sha256": "ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc",
      "slot": "direction_manifest"
    },
    {
      "bytes": 256361,
      "path": "runs/dependency-migration-v1/training-project-6/inputs/splits.json",
      "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e",
      "slot": "splits"
    }
  ]
}
```

## Previous source-review statements

These paragraphs describe the earlier source-review stage; the common GPU measurements above update evaluation status.

```json
{
  "lesson": "Keep lifecycle qualification, completed optimizer budgets and scientific accuracy qualification distinct. Repeating an unchanged fit for publication or scheduling can reproduce parameter bytes without creating new scene diversity or an architectural improvement. Byte-equal weights may share retained storage later, but original fit/Run evidence remains separate. A one-update smoke is not a full training result or generalization test.",
  "outcome": "Retained train-only final objective 3.0391492799270182e-06; last optimizer loss 0.00099497308716583978. Final parameter archive is byte-identical to C53; this does not merge original fit receipts or establish identical execution contracts. The three one-update pipeline/prospective/scheduler fits share exact initial and final parameter values. Four prospective outputs missing from its partial local download are freshly verified in the existing native content store using the original receipt hashes; no replacement or copy is created. No fresh common GPU inference, complete simulation-step timing or drift benchmark is added for these fit identities. Historical scheduler/training wall time and normalized diagnostic loss are not deployed latency or worst pressure relative error.",
  "reproduction": "Retain the exact typed entrypoint and its explicit recipe source bindings, original config/runtime/source witness, all physical/manufactured/captured inputs, train-only scales, draw order and both checkpoints. Preserve native profile renaming and any original consumed publication probe. The historical trainer/restore/publication invokes CPU neural inference; qualify GPU-only equivalents before replay. No original receipt alias, Trial ancestry or scientific scalar is rewritten. Clean retraining/new-Rheon generation and long own-history drift remain pending."
}
```

## Superseded benchmark-stage reproduction statement

The current paragraph above corrects its qualification status. This original statement is retained for history.

```json
"Retain the exact typed entrypoint and its explicit recipe source bindings, original config/runtime/source witness, all physical/manufactured/captured inputs, train-only scales, draw order and both checkpoints. Preserve native profile renaming and any original consumed publication probe. The historical trainer/restore/publication invokes CPU neural inference; qualify GPU-only equivalents before replay. No original receipt alias, Trial ancestry or scientific scalar is rewritten. Clean retraining/new-Rheon generation and long own-history drift remain pending. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift."
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-047049129e1e2833",
    "kind": "training_attempt",
    "name": "water-coarse-control CUDA",
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
    "receipt_sha256": "047049129e1e28332143b4e532d3eb3b1868e76a68c54ee375fffef266427011",
    "receipt_paths": [
      "runs/pressure-spatial-coarse-v1/control-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "ab4a6325b4e94631be46567530bb563c",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "water-coarse-control CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "1793bcc8bede3b79c29654cb950c70ef2b65cd49958997b8835024bdc58469bb",
      "inputs": {
        "current_dataset": {
          "bytes": 21428086,
          "sha256": "8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"
        },
        "current_manifest": {
          "bytes": 168912,
          "sha256": "fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"
        },
        "dataset": {
          "bytes": 610070,
          "sha256": "89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"
        },
        "dataset_extra": {
          "bytes": 610806,
          "sha256": "c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"
        },
        "direction_dataset": {
          "bytes": 24476086,
          "sha256": "4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"
        },
        "direction_manifest": {
          "bytes": 298490,
          "sha256": "ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"
        },
        "splits": {
          "bytes": 256361,
          "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423",
      "source_witness_sha256": "5fa6dbbb6adc91409b4312b383be707bbcba1dcb97b35bfe53cc4cec324c4e4e"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2626793472,
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
      "scientific_promotion": false
    },
    "lesson": {
      "status": "review_required_no_automatic_causal_claim"
    },
    "reproduction": {
      "receipt": "runs/pressure-spatial-coarse-v1/control-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-spatial-coarse-v1/control-study-01/train/receipt.json",
    "sha256": "047049129e1e28332143b4e532d3eb3b1868e76a68c54ee375fffef266427011",
    "bytes": 9967
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"1793bcc8bede3b79c29654cb950c70ef2b65cd49958997b8835024bdc58469bb","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"5fa6dbbb6adc91409b4312b383be707bbcba1dcb97b35bfe53cc4cec324c4e4e"},"diagnostic_population":"train_only","elapsed_s":502.04522853699746,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":3.050874384769031e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":870009344,"peak_reserved_bytes":920649728,"tf32":false},"heldout_evaluations":0,"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"history.json":{"bytes":216262,"role":"training_diagnostics","sha256":"d923701b3b8e0e5f7495a44bc6084c38e2de627c60b4c811187d1c309a991a9e"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"a55fdf3197ac0546cfb3a375edb2b1859c5691d64a00825f7b9ede692f22eda3"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":473,"role":"result_summary","sha256":"3c8bba44056bfa49ca367cabbea613cbe0faca38f92896586970793824247b26"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"}},"per_case_inference_s":[0.02260004599520471,0.004385659005492926,0.005554280010983348,0.004403882994665764,0.003861666002194397,0.003717496001627296,0.004669191999710165,0.004831381011172198,0.00981888099340722,0.005471640994073823,0.006573115009814501,0.005907684986595996],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"7e2579de4103ca626bd53bf4618fd923748dab2f1a71ebca4822c102d9afddad","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"995b42ccf1127f0d7ba80cdbbfac351b963b7fca7ee7f51fceccaadc17f8fb06","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"265a9fe3db154abcf00600e81ee7348a6c5cf1038ad062c589d3daa5cd9e4164","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"303a102a331cac8d1aff9fc04036feb23a6e04f1fc9ab3ca33ef70757c92927b","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"d995ef96214192c6bc82ade8d2776534b6958af6f885a4d385c3567ead9a297d","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"d1e0659bc4f5a17ec720c5bcf10c9c1610ca6ebf4192997e2e10dbff0d33b0b3","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"b5e4ee1f2284eb3a2ef2be866a193bc4a2f5aec70acf56d77574faff452b5bde","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"730ffca8fd5b89a3f35becad556842e9bd73700c4f019b805515b9639f92cd44","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"c46c772f435d5be2ac599f35ef21245fa9f42422f9294863d6ad9c372f43cc74","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"82501ba60b28031fab17c502d108e3aa926acb396707f2610a08f62b706489c2","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"8a857e8be1345b7e6e2cbea3c67deedec7abfc995cbe3517ff4fb2d2e710e38f","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"cf784c34b847cace35831e224e50fda75b91ca90a16c64856994aeffe9556355","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2626793472,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
