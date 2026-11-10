# water-band-control CUDA

Retrospective source review · fit-a928b38bd18e1c6a

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.363821 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.0011571 | 4.28072 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.0097332 | 4.13833 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.000965393 | 4.35661 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.00223956 | 4.11802 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.000328576 | 4.09688 |
| 8 | 7x16x5 | 0.016666667 | 2 | 0.000241986 | 4.13523 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.00359544 | 4.13437 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.000396293 | 4.27792 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.00023366 | 4.12012 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.00241003 | 4.14999 |
| 8 | 9x16x9 | 0.016666667 | 2 | 0.000119704 | 4.18829 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.000279691 | 4.15568 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.00136128 | 4.32706 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.000335074 | 4.28007 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.00271363 | 4.14238 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.00167813 | 4.18500 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.000256238 | 4.37795 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.00193869 | 4.15482 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.00253576 | 4.40626 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.00382525 | 4.11159 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.00821432 | 4.17276 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.00113674 | 4.16193 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.000666113 | 4.13988 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.00226042 | 4.15477 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.00309108 | 4.14888 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.000290728 | 4.13727 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.00730034 | 4.12183 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.00118564 | 4.14940 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.00462921 | 4.26542 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.00293794 | 4.14522 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.00127936 | 4.42127 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.00252695 | 4.16232 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.000794992 | 4.27486 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.00269049 | 4.10792 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.00148079 | 4.33680 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.00233941 | 4.18216 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.00073113 | 4.11912 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.000814804 | 4.31692 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.000517917 | 4.08771 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.00272964 | 4.27743 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.00313356 | 4.14106 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.000511132 | 4.14622 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.00158346 | 4.11961 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.00129237 | 4.35761 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.000662826 | 4.19994 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.00375187 | 4.12162 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.000471788 | 4.17389 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.00638709 | 4.37011 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.000937127 | 4.12973 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.00254935 | 4.27199 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.0028502 | 4.37980 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.000539519 | 4.13195 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.00241098 | 4.12623 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.000354632 | 4.12499 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.00428707 | 4.13719 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.00152446 | 4.21937 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.00288295 | 4.17611 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.00291249 | 4.09741 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.00426612 | 4.15087 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.00512829 | 4.16447 |
| 16 | 15x5x5 | 0.016666667 | 2 | 0.000138036 | 4.40912 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.00238523 | 4.14500 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.00518559 | 4.14512 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.00232824 | 4.40007 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: band-control. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 370,016-parameter linear, bias-free F32 multiscale pressure CNN with one physically scaled residual channel and 19 weight tensors. Two shared neural pressure corrections recompute the F64 sealed cubic-grid equation residual, apply fixed train-only scales and accumulate cell-zero-gauged F64 pressure. Inherited GroupNorm/SiLU labels do not describe its executed linear blocks. It returns pressure, not a physical simulation step; no learned spatial coarse branch is present.

**Purpose.** Establish the full-length zero-low-mode-weight throughput/control arm; its final parameter archive equals named C45.

**Experiment.** 5,000 CUDA Adam updates, batch one, rate 0.001, model seed 20261015, physical-index seed 2026102501, schedule [{'learning_rate': 0.0001, 'update': 3000}, {'learning_rate': 1e-05, 'update': 4500}]. Sixteen physical fitting trajectories/32 frames at 8/16 cells per metre, dt=1/60 s and fluid density 1000 kg/m3. Both manufactured banks and the fixed captured-direction bank remain active; First/final low-spatial-mode feedback weight is 0. The actual entrypoint is evidence.py; diagnostics use training parents only. The typed evidence wrapper also queries one already-consumed validation frame for publication; it supplies no independent holdout.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.0097332 and worst equation residual relative L2 is 0.0056079. Median case GPU latency is 4.15383 ms per pressure request; the largest case median is 4.46220 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** Keep lifecycle qualification, completed optimizer budgets and scientific accuracy qualification distinct. Repeating an unchanged fit for publication or scheduling can reproduce parameter bytes without creating new scene diversity or an architectural improvement. Byte-equal weights may share retained storage later, but original fit/Run evidence remains separate. A one-update smoke is not a full training result or generalization test. The common population exactly reproduces all 128 pressure fields of C45; retain separate fit histories despite shared final parameter bytes. This establishes only this population, not general execution-policy equivalence.

**Reproduction.** Retain the exact typed entrypoint and its explicit recipe source bindings, original config/runtime/source witness, all physical/manufactured/captured inputs, train-only scales, draw order and both checkpoints. Preserve native profile renaming and any original consumed publication probe. The historical training and publication paths still need GPU-only replay qualification; the separate original-checkpoint GPU evaluation and timing are now qualified. No original receipt alias, Trial ancestry or scientific scalar is rewritten. Clean retraining/new-Rheon generation and long own-history drift remain pending. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-throughput-feedback-v1/control-study-02/train/receipt.json",
    "sha256": "a928b38bd18e1c6af9c6a14eab09672d45f35c621253a348003a1ad74b899110"
  },
  "history": {
    "path": "runs/pressure-throughput-feedback-v1/control-study-02/train/history.json",
    "sha256": "388db8ebf7fa6dee7c93055faea3c2ef00481ad8cabfc0f7d1c63e652e880718"
  },
  "sources": [
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/config.json",
      "sha256": "44e999522818da362424759dbca3d23a79468aaeebe6092fbfa4373e65eee7ad"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/model.py",
      "sha256": "8d7bca5b7296396ffb5316df78c58fb1b3fc5a9776be08409bf6bd09b435bd01"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/policy.py",
      "sha256": "2b4d0fc25fb75a7a2563b042b13d435076391d8f06adb29ad18c220567797eaa"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/train.py",
      "sha256": "d450b3230a379b5a4440a12eb8d0cd133359bd27dbe9765837ef4908fd8f68c4"
    },
    {
      "path": "runs/pressure-throughput-feedback-v1/control-project-04/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-throughput-feedback-v1/control-project-04/source-witness.json",
    "sha256": "ee850a004366b176bf39fe473914e6849156abc3c36b54c0ed6c158e4997fb26"
  },
  "training_population": {
    "path": "runs/pressure-throughput-feedback-v1/control-study-02/train/splits.json",
    "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
  },
  "training_inputs": [
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
  "outcome": "Retained train-only final objective 3.2031162777493653e-06; last optimizer loss 0.00059696448348051795. Final parameter archive is byte-identical to C45; this does not merge original fit receipts or establish identical execution contracts. The three one-update pipeline/prospective/scheduler fits share exact initial and final parameter values. Four prospective outputs missing from its partial local download are freshly verified in the existing native content store using the original receipt hashes; no replacement or copy is created. No fresh common GPU inference, complete simulation-step timing or drift benchmark is added for these fit identities. Historical scheduler/training wall time and normalized diagnostic loss are not deployed latency or worst pressure relative error.",
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
    "item_id": "fit-a928b38bd18e1c6a",
    "kind": "training_attempt",
    "name": "water-band-control CUDA",
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
    "receipt_sha256": "a928b38bd18e1c6af9c6a14eab09672d45f35c621253a348003a1ad74b899110",
    "receipt_paths": [
      "runs/pressure-throughput-feedback-v1/control-study-02/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "9c9777fcb4b541e98dc03e66c1df4f73",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "water-band-control CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "44e999522818da362424759dbca3d23a79468aaeebe6092fbfa4373e65eee7ad",
      "inputs": {
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
      "source_witness_sha256": "ee850a004366b176bf39fe473914e6849156abc3c36b54c0ed6c158e4997fb26"
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
        "sha256": "26b66232928ce81d428dbdb1c8cc750a0cf5c0bc5f678977efc111a0a56d2d9a",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "26b66232928ce81d428dbdb1c8cc750a0cf5c0bc5f678977efc111a0a56d2d9a",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2424713216,
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
      "receipt": "runs/pressure-throughput-feedback-v1/control-study-02/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-throughput-feedback-v1/control-study-02/train/receipt.json",
    "sha256": "a928b38bd18e1c6af9c6a14eab09672d45f35c621253a348003a1ad74b899110",
    "bytes": 9573
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.0},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"44e999522818da362424759dbca3d23a79468aaeebe6092fbfa4373e65eee7ad","inputs":{"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"ee850a004366b176bf39fe473914e6849156abc3c36b54c0ed6c158e4997fb26"},"diagnostic_population":"train_only","elapsed_s":365.77180759799376,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.2601437683557037e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":770565632,"peak_reserved_bytes":819986432,"tf32":false},"heldout_evaluations":0,"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"26b66232928ce81d428dbdb1c8cc750a0cf5c0bc5f678977efc111a0a56d2d9a"},"history.json":{"bytes":214480,"role":"training_diagnostics","sha256":"388db8ebf7fa6dee7c93055faea3c2ef00481ad8cabfc0f7d1c63e652e880718"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"07e9417ebf5f07c1619600e1f66aa3876e585151cfe06891bc8eee45bf87f284"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":473,"role":"result_summary","sha256":"bc3b36f62a669661abb81a8120a3b143fed456cded027cb28cbf60e1b2f1c609"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"26b66232928ce81d428dbdb1c8cc750a0cf5c0bc5f678977efc111a0a56d2d9a"}},"per_case_inference_s":[0.0211585049983114,0.004010592005215585,0.005176559992833063,0.004091102993697859,0.004285514005459845,0.003339967006468214,0.00415702500322368,0.004327793008997105,0.008914076999644749,0.004981199002941139,0.005893249996006489,0.005300429998897016],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"51f48b0d5b9ccff71bef2f7a6d35b7adee37010eccb55c72b2a8aeb7c2ec62e7","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"bdb70ef53531cf143c04f11e292a284d02dd756db8b8decbb5b9e80366ea076a","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"3802be5e1955e2685ad920ed53a028fc03b115cf26bacf927a739ce19dcca004","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"5444de39457a81e6b890052a2e41fd3da8303db10e13daee9f75de996b83bfe1","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"c8d4c482c629336e9c86d874e2508d8af756380c4b62720d4f9e08bf87e60834","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"88d604e26c9e4f71f9dcaffcd1462292b6355ba43a534ccd9fefaa7c6739b32b","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"630d3803638e5381ed479197ac435eac1be09f7acadba66e99a3c63cb90e63ec","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"ce8532e047b1621e51a595e58e0b0d4432218e88551b6f6135058c292950b661","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"4080003938bfaadac1e8f0ecb3983d259ab153958fbec1dafbe286d360a7342d","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"5e34e18259a61328d4c646d02957354494529e7312180cadc964cd8d60bc9b97","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"b12b401e2d311ea26d8be602706c930c7df211eef186c26ee9bad3faf8c9f2f5","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"96869199052fbd87ceb2564d63fe8c4f371dd9706cbac4b399917a0df422569c","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2424713216,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
