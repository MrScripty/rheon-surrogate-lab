# Pressure methods · v2-small16 · C53 CUDA

Retrospective source review · fit-80e3e9ce8eb8ab94

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.377207 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.00098116 | 4.17817 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.00521396 | 4.29976 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.000783736 | 4.37037 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.00180095 | 4.11316 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.000279532 | 4.35134 |
| 8 | 7x16x5 | 0.016666667 | 2 | 0.000212161 | 4.11781 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.00274567 | 4.13890 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.000444787 | 4.13116 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.00099626 | 4.36266 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.00130713 | 4.21915 |
| 8 | 9x16x9 | 0.016666667 | 2 | 0.000225814 | 4.19311 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.00011867 | 4.38749 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.00158829 | 4.16164 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.00038921 | 4.33082 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.00110705 | 4.58504 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.00122871 | 4.31019 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.000429846 | 4.19157 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.0016619 | 4.17495 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.00149242 | 4.12609 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.0028609 | 4.15969 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.00308424 | 4.44338 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.000895482 | 4.12923 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.000764052 | 4.15562 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.00199013 | 4.16414 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.00266744 | 4.13820 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.000401224 | 4.19064 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.00685662 | 4.33196 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.00157191 | 4.13843 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.00348963 | 4.41179 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.00598623 | 4.17003 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.00105648 | 4.28656 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.00480581 | 4.43678 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.00031104 | 4.09692 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.000819769 | 4.17296 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.000861236 | 4.11030 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.00483654 | 4.21827 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.000568592 | 4.30518 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.000539727 | 4.41788 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.00115761 | 4.35578 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.00236644 | 4.12887 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.00186444 | 4.20084 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.000408497 | 4.15234 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.00168038 | 4.16055 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.00080648 | 4.17222 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.000829547 | 4.28003 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.00549071 | 4.13027 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.00109694 | 4.16256 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.00198619 | 4.39835 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.000439024 | 4.11813 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.002187 | 4.15467 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.00181841 | 4.37265 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.000142387 | 4.29520 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.00527893 | 4.29699 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.000444585 | 4.36412 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.00623751 | 4.14997 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.00171275 | 4.22810 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.00322533 | 4.17476 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.00112713 | 4.16186 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.00168408 | 4.41006 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.00306271 | 4.15724 |
| 16 | 15x5x5 | 0.016666667 | 2 | 0.000377017 | 4.12136 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.00512825 | 4.57059 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.00222298 | 4.40705 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.00173516 | 4.17278 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: v2-small16. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 370,016-parameter linear bias-free one-channel F32 multiscale pressure CNN initialized from exact C53 values. Physical residual q=(rho/h)r is normalized using the fixed original scales; F64 equation assembly, residual recomputation, gauging and pressure accumulation surround shared neural corrections. Default deployment executes two pressure corrections and advances no physical simulation state.

**Purpose.** Establish the 1,600-update small-only replay control for the v2 expansion and retained-exposure comparisons.

**Experiment.** Configured 1,600 CUDA Adam updates, initial rate 1e-05, schedule [{'update': 1201, 'learning_rate': 1e-06}, {'update': 1501, 'learning_rate': 1e-07}], model seed 2026101820, index seed 2026101804. 16 training trajectories/32 physical requests; actual tier exposures {'16': 1600}, sampled voxel count 1,823,416. Inherited manufactured/direction/current-policy banks are disabled. Only recorded physical requests enter fitting.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.00685662 and worst equation residual relative L2 is 0.0049531. Median case GPU latency is 4.18558 ms per pressure request; the largest case median is 4.68221 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** An equal-update comparison does not match sampled voxel work or small-scene exposure. Report the actual tier counts, original versus added parents and fixed scales. The v2 private small stream preserves the relevant sample subsequence when expansion is enabled. The common population exactly reproduces all 128 pressure fields of C78; retain separate fit histories despite shared final parameter bytes. This establishes only this population, not general execution-policy equivalence.

**Reproduction.** Retain the exact typed evidence recipe, all source/config/runtime and input/output bindings, C53 warm tensors, fixed normalization and original private sampling. Native-view manifests retain generation receipts, time mappings and target-conversion policy; recreate these using the preserved old reference before separately qualifying new Rheon. The Adam arm also needs its aggregate gradient order and observational archives. Historical training and publication replay still needs a GPU-only route; the separate original-checkpoint GPU evaluation and timing are now qualified. Clean retraining, full-state drift and new-Rheon regeneration remain unqualified; preserve original receipts and ancestry. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v2/v2-small16-study-02/train/receipt.json",
    "sha256": "80e3e9ce8eb8ab94c8233a2b66242c0139ecfce9315185007aa3895b2468fc37"
  },
  "history": {
    "path": "runs/pressure-methods-v2/v2-small16-study-02/train/history.json",
    "sha256": "ff1dc1166e14cf0b7afd28bf3549f18ec2b92e6fe06c391fdc57fce9f3f6751b"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/config.json",
      "sha256": "653625bd38c24ea233f59295d9d27e4dbde127308506cfb0005a0832f2c2e0c0"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/data.py",
      "sha256": "cf01d003cdf525e78cb3ec2b5e004186fa28c65ecadd63ffea969d85009674f9"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/fine-tuning.json",
      "sha256": "7055a8835e19d1947f96b7418a224e778c6ea6e5bee6f4e96ee7db1824af582f"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/methods_view.py",
      "sha256": "3418389fb087ae3036549b49338bc7377291d497fa8636c0978f2f382131294f"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/mixed_order.py",
      "sha256": "fdbbfa8d48001aa8fb6e8d530c0bceca3de6d99caa44b945af77a81ba5dd6de7"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/policy.py",
      "sha256": "83d2f46900b6a8018a02006520c33c6b2b8e2ed5e46dd2fc92337a570b6e36b4"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/portable_policy.py",
      "sha256": "b890bf302e911c024ce54082c06043700f940d2fd812ffdcdc0fbb3d785a2d39"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/train.py",
      "sha256": "5e02e78d3eca1cb7f9eeb04ed0372dc71c8f11da5f26b7cadf629510a929b81b"
    },
    {
      "path": "runs/pressure-methods-v2/v2-small16-project-02/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v2/v2-small16-project-02/source-witness.json",
    "sha256": "559e0c3fc104c12b84b5a45744081533e3ce9fdcc1c45eb678f98e0ec8eaf5ed"
  },
  "training_population": {
    "path": "runs/pressure-methods-v2/v2-small16-study-02/train/splits.json",
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
      "bytes": 1485452,
      "path": "runs/pressure-current-policy-v1/current-late-study-03/restore/weights.npz",
      "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
      "slot": "initial_weights"
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
  "lesson": "An equal-update comparison does not match sampled voxel work or small-scene exposure. Report the actual tier counts, original versus added parents and fixed scales. The v2 private small stream preserves the relevant sample subsequence when expansion is enabled.",
  "outcome": "Retained final fitting loss 14.949307171011972. This unnamed fit has no fresh common GPU inference benchmark under its own retained identity. Original training, field-probe and observer diagnostics do not establish worst-case physical accuracy, deployment speed or drift.",
  "reproduction": "Retain the exact typed evidence recipe, all source/config/runtime and input/output bindings, C53 warm tensors, fixed normalization and original private sampling. Native-view manifests retain generation receipts, time mappings and target-conversion policy; recreate these using the preserved old reference before separately qualifying new Rheon. The Adam arm also needs its aggregate gradient order and observational archives. Historical publication/restore contains CPU neural queries; qualify a GPU-only equivalent before replay. Clean retraining, full-state drift and new-Rheon regeneration remain unqualified; preserve original receipts and ancestry."
}
```

## Superseded benchmark-stage reproduction statement

The current paragraph above corrects its qualification status. This original statement is retained for history.

```json
"Retain the exact typed evidence recipe, all source/config/runtime and input/output bindings, C53 warm tensors, fixed normalization and original private sampling. Native-view manifests retain generation receipts, time mappings and target-conversion policy; recreate these using the preserved old reference before separately qualifying new Rheon. The Adam arm also needs its aggregate gradient order and observational archives. Historical publication/restore contains CPU neural queries; qualify a GPU-only equivalent before replay. Clean retraining, full-state drift and new-Rheon regeneration remain unqualified; preserve original receipts and ancestry. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift."
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-80e3e9ce8eb8ab94",
    "kind": "training_attempt",
    "name": "Pressure methods · v2-small16 · C53 CUDA",
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
      "max_cells": 262144
    },
    "receipt_sha256": "80e3e9ce8eb8ab94c8233a2b66242c0139ecfce9315185007aa3895b2468fc37",
    "receipt_paths": [
      "runs/pressure-methods-v2/v2-small16-study-02/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "d697cd8f35424acd9d29ee1450870d86",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods · v2-small16 · C53 CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1600,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "653625bd38c24ea233f59295d9d27e4dbde127308506cfb0005a0832f2c2e0c0",
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
        "initial_weights": {
          "bytes": 1485452,
          "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"
        },
        "splits": {
          "bytes": 256361,
          "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "9d6532d2873d341b93ee2f41369f6c8ad91dc31bd5a3bc68fd0499104960054a",
      "source_witness_sha256": "559e0c3fc104c12b84b5a45744081533e3ce9fdcc1c45eb678f98e0ec8eaf5ed"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-1600.npz",
        "sha256": "e1b54a52bcd9e894f58540b70a744253af2475eb40ae1b990c22e22c46403a73",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "e1b54a52bcd9e894f58540b70a744253af2475eb40ae1b990c22e22c46403a73",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2395459584,
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
      "receipt": "runs/pressure-methods-v2/v2-small16-study-02/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v2/v2-small16-study-02/train/receipt.json",
    "sha256": "80e3e9ce8eb8ab94c8233a2b66242c0139ecfce9315185007aa3895b2468fc37",
    "bytes": 10764
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"v2-small16","early_weight":0.05,"initial_weights_sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":262144,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Balanced spatial tiers; private RNG; fixed update budget.","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"653625bd38c24ea233f59295d9d27e4dbde127308506cfb0005a0832f2c2e0c0","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"9d6532d2873d341b93ee2f41369f6c8ad91dc31bd5a3bc68fd0499104960054a","source_witness_sha256":"559e0c3fc104c12b84b5a45744081533e3ce9fdcc1c45eb678f98e0ec8eaf5ed"},"diagnostic_population":"train_only","elapsed_s":38.773311088007176,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.404221662288819e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":17328128,"peak_reserved_bytes":35651584,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":1600,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"checkpoint-1600.npz":{"bytes":1485452,"role":"checkpoint","sha256":"e1b54a52bcd9e894f58540b70a744253af2475eb40ae1b990c22e22c46403a73"},"history.json":{"bytes":10654,"role":"training_diagnostics","sha256":"ff1dc1166e14cf0b7afd28bf3549f18ec2b92e6fe06c391fdc57fce9f3f6751b"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"9b0dbc29a6f74320c2f7ab7a4acef02dc1c5b4311c4e685199651b3081a5edfa"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":478,"role":"result_summary","sha256":"4b1046bf841ece3fb66114ce2d35bd1d5411d4e68da4f3642d45223ae6277540"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"e1b54a52bcd9e894f58540b70a744253af2475eb40ae1b990c22e22c46403a73"}},"per_case_inference_s":[0.023837985994759947,0.004534515988780186,0.005732296995120123,0.004571063996991143,0.003948518977267668,0.003739450010471046,0.004779862996656448,0.004956629010848701,0.010289985017152503,0.005666657001711428,0.0067784849961753935,0.006141015008324757],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"c47727e7f9571e58a62d6c20c51d242f7533520f0f9b28d7d2a9c9e986daccdd","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"a87c46f9cf2d15bb0f09eb451a23946010cf37cecf2d0ffa2824a492bf2abd2f","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"f0b02b5ad2af68a6520425932646b1f3c560640a4e0230d133efd2bf58d79bd3","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"0c7db59a3ce1c3f9d9d7db6b8588d8a8531c2bb42fec4e1f45e576358172014c","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"57706ef59bdf78496a8c9d6378bd563a8bba5350cc4421ee7a204be78a091591","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"e240a9da6ce800661d7535acd6287f998eac446c2e7b1bc826a0c170197effbb","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"22512443ef21e73e13110871dc9d59636a13dce0f81737885c5476f01bf1b4de","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"fedc1a0f993d218013810936109c7791919f68ff17b840a4f9cd8e192ea921a1","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"bc3a1a91eccbc7edcd912fa0584d9542ac1d1a84734cb099274f53a3104991fc","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"8e47c84da396b6e412e837f77223a86bfdb88604af52db6da037613d30f072fb","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"ad266decb325c3d390796dd41d48be94457bf2733709024d4e77158ccbd97a62","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"d743d3fc828235fac4aa266aca109c84248fbbabec94b6d063fc40dbd888b7d7","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2395459584,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
