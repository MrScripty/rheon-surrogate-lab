# Pressure four-pass · c53-u · C53 fine-tune CUDA

Retrospective source review · fit-8992e19d389ea016

Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.

## Common GPU pressure benchmark

NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.

Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; this fit has median paired overhead 0.390495 ms/request. No overhead is subtracted or negative differences clipped.

| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |
| --- | --- | --- | --- | --- | --- |
| 8 | 5x6x5 | 0.016666667 | 2 | 0.00134026 | 4.11641 |
| 8 | 5x6x15 | 0.016666667 | 2 | 0.0113305 | 4.12081 |
| 8 | 6x7x8 | 0.016666667 | 2 | 0.00298755 | 4.11508 |
| 8 | 6x12x7 | 0.016666667 | 2 | 0.00257243 | 4.20526 |
| 8 | 7x8x6 | 0.016666667 | 2 | 0.000402218 | 4.11918 |
| 8 | 7x16x5 | 0.016666667 | 2 | 0.000161826 | 4.13452 |
| 8 | 8x12x10 | 0.016666667 | 2 | 0.00622608 | 4.40572 |
| 8 | 8x15x10 | 0.016666667 | 2 | 0.00109714 | 4.40511 |
| 8 | 9x6x8 | 0.016666667 | 2 | 0.000501125 | 4.15597 |
| 8 | 9x6x11 | 0.016666667 | 2 | 0.0027973 | 4.14550 |
| 8 | 9x16x9 | 0.016666667 | 2 | 0.000302249 | 4.18289 |
| 8 | 10x5x13 | 0.016666667 | 2 | 0.000283131 | 4.46262 |
| 8 | 10x8x12 | 0.016666667 | 2 | 0.00196305 | 4.35550 |
| 8 | 10x11x12 | 0.016666667 | 2 | 0.00114559 | 4.29203 |
| 8 | 10x15x10 | 0.016666667 | 2 | 0.00638133 | 4.12794 |
| 8 | 11x8x9 | 0.016666667 | 2 | 0.00232624 | 4.14511 |
| 8 | 11x12x16 | 0.016666667 | 2 | 0.000231678 | 4.36154 |
| 8 | 11x15x14 | 0.016666667 | 2 | 0.00262558 | 4.37919 |
| 8 | 12x10x6 | 0.016666667 | 2 | 0.00306155 | 4.33621 |
| 8 | 13x10x11 | 0.016666667 | 2 | 0.00230812 | 4.19531 |
| 8 | 13x12x8 | 0.016666667 | 2 | 0.0024558 | 4.13771 |
| 8 | 14x5x12 | 0.016666667 | 2 | 0.00246375 | 4.18412 |
| 8 | 14x7x11 | 0.016666667 | 2 | 0.00153913 | 4.15655 |
| 8 | 14x7x14 | 0.016666667 | 2 | 0.00839808 | 4.16889 |
| 8 | 14x13x9 | 0.016666667 | 2 | 0.00537001 | 4.37395 |
| 8 | 14x14x7 | 0.016666667 | 2 | 0.00213089 | 4.29412 |
| 8 | 15x6x8 | 0.016666667 | 2 | 0.00995581 | 4.45325 |
| 8 | 15x6x14 | 0.016666667 | 2 | 0.00294512 | 4.30782 |
| 8 | 15x9x7 | 0.016666667 | 2 | 0.00191994 | 4.15890 |
| 8 | 15x12x5 | 0.016666667 | 2 | 0.0083027 | 4.19184 |
| 8 | 16x6x5 | 0.016666667 | 2 | 0.00226941 | 4.17313 |
| 8 | 16x16x16 | 0.016666667 | 2 | 0.00727906 | 4.19949 |
| 16 | 5x7x11 | 0.016666667 | 2 | 0.000963801 | 4.30585 |
| 16 | 5x8x16 | 0.016666667 | 2 | 0.00236135 | 4.38055 |
| 16 | 5x16x5 | 0.016666667 | 2 | 0.00151698 | 4.44477 |
| 16 | 5x16x6 | 0.016666667 | 2 | 0.00887967 | 4.48497 |
| 16 | 5x16x9 | 0.016666667 | 2 | 0.0012617 | 4.12550 |
| 16 | 6x10x16 | 0.016666667 | 2 | 0.00160798 | 4.27537 |
| 16 | 7x8x9 | 0.016666667 | 2 | 0.000204907 | 4.13837 |
| 16 | 7x9x13 | 0.016666667 | 2 | 0.000872081 | 4.17420 |
| 16 | 7x12x14 | 0.016666667 | 2 | 0.00569401 | 4.16310 |
| 16 | 8x15x11 | 0.016666667 | 2 | 0.000995458 | 4.33914 |
| 16 | 8x16x8 | 0.016666667 | 2 | 0.00251652 | 4.28746 |
| 16 | 8x16x13 | 0.016666667 | 2 | 0.00202107 | 4.26534 |
| 16 | 9x7x8 | 0.016666667 | 2 | 0.00165544 | 4.17660 |
| 16 | 9x11x11 | 0.016666667 | 2 | 0.00366001 | 4.17380 |
| 16 | 9x15x10 | 0.016666667 | 2 | 0.000317422 | 4.21163 |
| 16 | 10x5x5 | 0.016666667 | 2 | 0.0124987 | 4.12578 |
| 16 | 10x7x15 | 0.016666667 | 2 | 0.000692277 | 4.17769 |
| 16 | 10x12x12 | 0.016666667 | 2 | 0.00569193 | 4.34401 |
| 16 | 11x5x15 | 0.016666667 | 2 | 0.00130163 | 4.15801 |
| 16 | 11x6x8 | 0.016666667 | 2 | 0.000551656 | 4.15906 |
| 16 | 11x12x7 | 0.016666667 | 2 | 0.00941107 | 4.11851 |
| 16 | 11x14x14 | 0.016666667 | 2 | 0.000256361 | 4.46481 |
| 16 | 11x16x5 | 0.016666667 | 2 | 0.00713327 | 4.15134 |
| 16 | 12x13x7 | 0.016666667 | 2 | 0.00246721 | 4.44422 |
| 16 | 13x15x16 | 0.016666667 | 2 | 0.00186201 | 4.18461 |
| 16 | 14x6x6 | 0.016666667 | 2 | 0.00282528 | 4.13531 |
| 16 | 14x6x16 | 0.016666667 | 2 | 0.0143179 | 4.15778 |
| 16 | 14x9x11 | 0.016666667 | 2 | 0.0179659 | 4.17221 |
| 16 | 15x5x5 | 0.016666667 | 2 | 5.06226e-05 | 4.41933 |
| 16 | 16x6x12 | 0.016666667 | 2 | 0.0109222 | 4.13887 |
| 16 | 16x10x6 | 0.016666667 | 2 | 0.00587607 | 4.15562 |
| 16 | 16x14x11 | 0.016666667 | 2 | 0.00405395 | 4.39355 |

Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. Clear compute-PID snapshots do not prove continuous hardware exclusivity.

Unified pressure table: cef4fe48e1105104e437dde1b18893adacfe76962d3d966d45970025232c7fb7; subject: c53-u. The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.

**Method.** A 370,016-parameter float32 3D pressure U-Net with encoder widths8/16/24/32, bottleneck48 and reversed decoder; physical residual reconstruction is float64. It starts from the exact C53 checkpoint. Training unrolls four pressure corrections on one equation. These corrections are not future physical simulation steps.

**Purpose.** Compare unchanged four-pass fine tuning with attached and detached auxiliary gradient paths under the same initialization, pattern order, depth schedule and optimizer budget. Retain the four-pass control without the auxiliary loss.

**Experiment.** The optimizer uses eight synthetic analytical patterns: one [0,0,1] mode on each of eight rectangular grids, at8or16cells/m. Each gets200exposures, with50at each of four auxiliary depths. The original16train trajectories calibrate amplitude/normalization; they are not the optimizer sample stream. No companion physical, direction or current-policy loss runs in the captured fitting-bank updater. Teacher weight is zero and counted teacher forwards are zero. All arms run1,600updates; U counts6,400network forwards and A/D8,000. The loss combines analytical pressure loss, log-scaled final equation residual and 0.05mean early residual feedback; A/D add0.05auxiliary loss.

**Observed outcome.** On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is 0.0179659 and worst equation residual relative L2 is 0.00588282. Median case GPU latency is 4.18235 ms per pressure request; the largest case median is 4.48820 ms/request. Two fresh process phases preserve every input, reference and output hash, the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.

**Lesson.** The eight-pattern objective fails to improve this stronger parent on the consumed native pressure bank, including at its trained four-correction depth. Extra corrections reduce absolute errors but do not reverse the parent ranking. The earlier C37 default comparison also changed one to two corrections; the new matched matrix confirms that C37 benefits only conditionally on budget and metric. Preserve distinct fit identities even when checkpoint bytes and tested outputs coincide. No sole causal explanation or unseen-scene claim follows. The subsequent matched-work CUDA matrix confirms these parent regressions at every tested one/two/four/eight correction budget. C37 benefits at two/four/eight corrections, while its worst pressure error worsens at one correction; its saved parent default uses one and its fine-tunes use two. Keep metric and actual work explicit. The common population exactly reproduces all 128 pressure fields of C66; retain separate fit histories despite shared final parameter bytes. This establishes only this population, not general execution-policy equivalence.

**Reproduction.** Use the exact captured project.json/train-recipe.json and evidence.py entrypoint with the original input bindings, source witness, C53 checkpoint, fitting bank and fine-tuning.json. Clean retraining and new-Rheon data equivalence remain unqualified; the current common GPU pressure accuracy and latency are qualified. These results alone do not authorize checkpoint retirement. The original CUDA training receipt is sufficient for the declared training-checkpoint replay; do not invent a CPU-restore receipt or a cohort alias for C53-U. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-detached-correction-v1/c53-u-study-01/train/receipt.json",
    "sha256": "8992e19d389ea0164a88beace7cfe013ea8ec35e857a11683ad33438f2ed4dcd"
  },
  "history": {
    "path": "runs/pressure-detached-correction-v1/c53-u-study-01/train/history.json",
    "sha256": "8e3f11a0b1e99ed615ee7766620801ac3fb7e7b9c8644f32a8b1e1ac87725fe9"
  },
  "sources": [
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/config.json",
      "sha256": "3ec43d3218df78b967137fdebd6156a30ce19a1f4ba53cd9e17f7b565aae2629"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/fine-tuning.json",
      "sha256": "6c959f156a0b0d6d6d057e9ba4e062ab8af0ca739004fa3f4b87b02e92162693"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/fine_tuning.py",
      "sha256": "57658249095fd62c38f8d3bcbbe201d5ebe288009f0042001941bca82110795e"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/fit_bank.py",
      "sha256": "bbdbabfc89805410df196dd0724b24347954548dd83790456994ca46a6f5e347"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/model.py",
      "sha256": "a2ebefe6a8ce12c7e6e1e12fa82b835029f364b50e91284f33b60bfe31b0eaee"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/original_fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/original_fit_bank.py",
      "sha256": "e68c1fa6723331af3448edec8294ca0dc67a5b97576a1cd251c5a789e9bf0668"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/train.py",
      "sha256": "b9c98c38dfd22ec77ed1bc6dcbad8e692b276dc96fe9cb5cf928ec2b6be0774b"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c53-u-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-detached-correction-v1/c53-u-project-01/source-witness.json",
    "sha256": "9cce3cbd2a21736de9b0308acaf9609adce27d02a9b2acbfff19744616afd3dd"
  },
  "training_population": {
    "path": "runs/pressure-detached-correction-v1/c53-u-study-01/train/splits.json",
    "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
  },
  "training_inputs": [
    {
      "path": "runs/pressure-current-policy-v1/bank-02/late.npz",
      "sha256": "8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521",
      "slot": "current_dataset",
      "bytes": 21428086
    },
    {
      "path": "runs/pressure-current-policy-v1/bank-02/manifest.json",
      "sha256": "fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e",
      "slot": "current_manifest",
      "bytes": 168912
    },
    {
      "path": "runs/dependency-migration-v1/training-project-6/inputs/dataset.npz",
      "sha256": "89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e",
      "slot": "dataset",
      "bytes": 610070
    },
    {
      "path": "runs/dependency-migration-v1/training-project-6/inputs/dataset_extra.npz",
      "sha256": "c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5",
      "slot": "dataset_extra",
      "bytes": 610806
    },
    {
      "path": "runs/dependency-migration-v1/training-project-6/inputs/direction_dataset.npz",
      "sha256": "4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6",
      "slot": "direction_dataset",
      "bytes": 24476086
    },
    {
      "path": "runs/dependency-migration-v1/training-project-6/inputs/direction_manifest.json",
      "sha256": "ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc",
      "slot": "direction_manifest",
      "bytes": 298490
    },
    {
      "path": "runs/pressure-current-policy-v1/current-late-study-03/restore/weights.npz",
      "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
      "slot": "initial_weights",
      "bytes": 1485452
    },
    {
      "path": "runs/dependency-migration-v1/training-project-6/inputs/splits.json",
      "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e",
      "slot": "splits",
      "bytes": 256361
    }
  ]
}
```

## Previous source-review statements

These paragraphs describe the earlier source-review stage; the common GPU measurements above update evaluation status.

```json
{
  "lesson": "The eight-pattern objective fails to improve this stronger parent on the consumed native pressure bank, including at its trained four-correction depth. Extra corrections reduce absolute errors but do not reverse the parent ranking. The earlier C37 default comparison also changed one to two corrections; the new matched matrix confirms that C37 benefits only conditionally on budget and metric. Preserve distinct fit identities even when checkpoint bytes and tested outputs coincide. No sole causal explanation or unseen-scene claim follows. The subsequent matched-work CUDA matrix confirms these parent regressions at every tested one/two/four/eight correction budget. C37 benefits at two/four/eight corrections, while its worst pressure error worsens at one correction; its saved parent default uses one and its fine-tunes use two. Keep metric and actual work explicit.",
  "outcome": "C53-U (unnamed completed fit): worst mean-free pressure relative L2 0.017965911104999222 and equation relative L2 0.0058828190343905911 at the captured two-correction default. All 128 consumed old-solver equations were also replayed at one, two, four and eight corrections; all three C53 fine-tunes are worse than C53 on both worst metrics at every matched budget. Its default outputs match C66 byte-for-byte on all 128 equations; this does not equate their fit histories. No timing or own-history result is implied.",
  "reproduction": "Use the exact captured project.json/train-recipe.json and evidence.py entrypoint with the original input bindings, source witness, C53 checkpoint, fitting bank and fine-tuning.json. Clean retraining, fresh qualified latency and new-Rheon data equivalence remain unqualified; no checkpoint can be retired on this review alone. The original CUDA training receipt is sufficient for the declared training-checkpoint replay; do not invent a CPU-restore receipt or a cohort alias for C53-U."
}
```

## Superseded benchmark-stage reproduction statement

The current paragraph above corrects its qualification status. This original statement is retained for history.

```json
"Use the exact captured project.json/train-recipe.json and evidence.py entrypoint with the original input bindings, source witness, C53 checkpoint, fitting bank and fine-tuning.json. Clean retraining, fresh qualified latency and new-Rheon data equivalence remain unqualified; no checkpoint can be retired on this review alone. The original CUDA training receipt is sufficient for the declared training-checkpoint replay; do not invent a CPU-restore receipt or a cohort alias for C53-U. The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift."
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-8992e19d389ea016",
    "kind": "training_attempt",
    "name": "Pressure four-pass · c53-u · C53 fine-tune CUDA",
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
    "receipt_sha256": "8992e19d389ea0164a88beace7cfe013ea8ec35e857a11683ad33438f2ed4dcd",
    "receipt_paths": [
      "runs/pressure-detached-correction-v1/c53-u-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "d4211bfd98de484d941682eb93a8212c",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure four-pass · c53-u · C53 fine-tune CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1600,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "3ec43d3218df78b967137fdebd6156a30ce19a1f4ba53cd9e17f7b565aae2629",
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
      "sample_order_sha256": "21ed766152674818e22d0abb5ed5540690ab31aa14507d78fcc73a381e5d058a",
      "source_witness_sha256": "9cce3cbd2a21736de9b0308acaf9609adce27d02a9b2acbfff19744616afd3dd"
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
        "sha256": "69b49daa62ea6945726571dcf01eecc6d0084a89c40acd5531c5bd6607260655",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "69b49daa62ea6945726571dcf01eecc6d0084a89c40acd5531c5bd6607260655",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2413813760,
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
      "receipt": "runs/pressure-detached-correction-v1/c53-u-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-detached-correction-v1/c53-u-study-01/train/receipt.json",
    "sha256": "8992e19d389ea0164a88beace7cfe013ea8ec35e857a11683ad33438f2ed4dcd",
    "bytes": 59975
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"c53-u","early_weight":0.05,"fitting_bank":{"amplitude_reference":"original-train-correction-rms","cache_max_bytes":8388608,"grids":[{"cells_per_metre":16,"counts_xyz":[13,15,16]},{"cells_per_metre":16,"counts_xyz":[9,11,11]},{"cells_per_metre":8,"counts_xyz":[15,9,7]},{"cells_per_metre":16,"counts_xyz":[11,5,15]},{"cells_per_metre":8,"counts_xyz":[7,16,5]},{"cells_per_metre":16,"counts_xyz":[11,12,7]},{"cells_per_metre":8,"counts_xyz":[5,13,9]},{"cells_per_metre":16,"counts_xyz":[16,7,11]}],"modes_zyx":[[0,0,1]],"sampling_seed":2026102101},"initial_weights_sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954","relative_equation_target":1e-06,"rollout_feedback":{"depth_seed":2026101807,"mode":"U","weight":0.05},"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":4},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"3ec43d3218df78b967137fdebd6156a30ce19a1f4ba53cd9e17f7b565aae2629","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"21ed766152674818e22d0abb5ed5540690ab31aa14507d78fcc73a381e5d058a","source_witness_sha256":"9cce3cbd2a21736de9b0308acaf9609adce27d02a9b2acbfff19744616afd3dd"},"diagnostic_population":"train_only","elapsed_s":72.506504451012,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.5254371930950583e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":19713024,"peak_reserved_bytes":37748736,"tf32":false},"fine_tuning_statistics":{"fundamental_bank":{"amplitude_rms_pa":2.8535734640782584,"cases":[{"cells_per_metre":16,"counts_xyz":[13,15,16],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[9,11,11],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[15,9,7],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[11,5,15],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[7,16,5],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[11,12,7],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[5,13,9],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[16,7,11],"mode_zyx":[0,0,1]}],"description_sha256":"30f7092460b6a649b28e4bf380a5769c2955d1e29461e0f64a6e1d474dab960f","expected_exposures":[200,200,200,200,200,200,200,200],"optimizer_examples":1600,"optimizer_exposures":[200,200,200,200,200,200,200,200],"resident_tensor_bytes":297152,"sampling_order_sha256":"d0ea221a7e03a83673ece7fc9cdf4f5212bb54f3023511f93d95c84e45647899","trace":[{"exposures_per_case":10,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.15784835736723601},{"case":0,"passes":2,"relative_equation_residual":0.021452784432344826},{"case":0,"passes":4,"relative_equation_residual":0.0004125454222632154},{"case":0,"passes":8,"relative_equation_residual":1.542688023556251e-07},{"case":1,"passes":1,"relative_equation_residual":0.08057004468536987},{"case":1,"passes":2,"relative_equation_residual":0.004859584804531118},{"case":1,"passes":4,"relative_equation_residual":2.277255172579694e-05},{"case":1,"passes":8,"relative_equation_residual":4.296367418778232e-10},{"case":2,"passes":1,"relative_equation_residual":0.1407147710249634},{"case":2,"passes":2,"relative_equation_residual":0.007357598322205652},{"case":2,"passes":4,"relative_equation_residual":1.0348535276890004e-05},{"case":2,"passes":8,"relative_equation_residual":6.07546868126484e-11},{"case":3,"passes":1,"relative_equation_residual":0.0821411126891672},{"case":3,"passes":2,"relative_equation_residual":0.0043489873163382285},{"case":3,"passes":4,"relative_equation_residual":7.6183536505070915e-06},{"case":3,"passes":8,"relative_equation_residual":3.5714508256808926e-11},{"case":4,"passes":1,"relative_equation_residual":0.06544883540456518},{"case":4,"passes":2,"relative_equation_residual":0.004603964587550759},{"case":4,"passes":4,"relative_equation_residual":1.886686940760885e-05},{"case":4,"passes":8,"relative_equation_residual":1.2719402845860506e-10},{"case":5,"passes":1,"relative_equation_residual":0.0821396350510775},{"case":5,"passes":2,"relative_equation_residual":0.004348693874046727},{"case":5,"passes":4,"relative_equation_residual":7.618178386063889e-06},{"case":5,"passes":8,"relative_equation_residual":3.5796686256731997e-11},{"case":6,"passes":1,"relative_equation_residual":0.029812664074410432},{"case":6,"passes":2,"relative_equation_residual":0.0009208512050393766},{"case":6,"passes":4,"relative_equation_residual":3.4423767225137867e-07},{"case":6,"passes":8,"relative_equation_residual":7.344260957640186e-13},{"case":7,"passes":1,"relative_equation_residual":0.1713229121606898},{"case":7,"passes":2,"relative_equation_residual":0.031166381123513942},{"case":7,"passes":4,"relative_equation_residual":0.0015799835036444743},{"case":7,"passes":8,"relative_equation_residual":4.6357817584227455e-06}],"update":80},{"exposures_per_case":20,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.13698179776487807},{"case":0,"passes":2,"relative_equation_residual":0.01174115699840943},{"case":0,"passes":4,"relative_equation_residual":6.0068877197883573e-05},{"case":0,"passes":8,"relative_equation_residual":3.4201867161475744e-09},{"case":1,"passes":1,"relative_equation_residual":0.07134697760937395},{"case":1,"passes":2,"relative_equation_residual":0.002521746272669943},{"case":1,"passes":4,"relative_equation_residual":2.0846144747459164e-06},{"case":1,"passes":8,"relative_equation_residual":3.01670460380165e-12},{"case":2,"passes":1,"relative_equation_residual":0.22015621651339617},{"case":2,"passes":2,"relative_equation_residual":0.012933248365811214},{"case":2,"passes":4,"relative_equation_residual":1.0165577394089904e-05},{"case":2,"passes":8,"relative_equation_residual":5.896659837087968e-11},{"case":3,"passes":1,"relative_equation_residual":0.09656468303374738},{"case":3,"passes":2,"relative_equation_residual":0.0036521435704908135},{"case":3,"passes":4,"relative_equation_residual":4.891330071145361e-06},{"case":3,"passes":8,"relative_equation_residual":9.499410259856631e-12},{"case":4,"passes":1,"relative_equation_residual":0.04124117766600117},{"case":4,"passes":2,"relative_equation_residual":0.0015760130014089019},{"case":4,"passes":4,"relative_equation_residual":1.656967404808876e-06},{"case":4,"passes":8,"relative_equation_residual":4.08555852193308e-12},{"case":5,"passes":1,"relative_equation_residual":0.09656775824325611},{"case":5,"passes":2,"relative_equation_residual":0.0036523503399774815},{"case":5,"passes":4,"relative_equation_residual":4.891340700910644e-06},{"case":5,"passes":8,"relative_equation_residual":1.0766311644025094e-11},{"case":6,"passes":1,"relative_equation_residual":0.02815354617767298},{"case":6,"passes":2,"relative_equation_residual":0.0007505469714708086},{"case":6,"passes":4,"relative_equation_residual":1.0486410875480467e-06},{"case":6,"passes":8,"relative_equation_residual":9.133086348311602e-13},{"case":7,"passes":1,"relative_equation_residual":0.20665056187128175},{"case":7,"passes":2,"relative_equation_residual":0.028364432593754867},{"case":7,"passes":4,"relative_equation_residual":0.0002936499504002973},{"case":7,"passes":8,"relative_equation_residual":4.4644841739466175e-08}],"update":160},{"exposures_per_case":30,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.21806122816734302},{"case":0,"passes":2,"relative_equation_residual":0.043058054786577524},{"case":0,"passes":4,"relative_equation_residual":0.0019054757081728654},{"case":0,"passes":8,"relative_equation_residual":3.7333819664578807e-06},{"case":1,"passes":1,"relative_equation_residual":0.11440505042049874},{"case":1,"passes":2,"relative_equation_residual":0.011233158246186075},{"case":1,"passes":4,"relative_equation_residual":0.00011891559642224195},{"case":1,"passes":8,"relative_equation_residual":1.3156762812481425e-08},{"case":2,"passes":1,"relative_equation_residual":0.21163447072550876},{"case":2,"passes":2,"relative_equation_residual":0.01129222122556913},{"case":2,"passes":4,"relative_equation_residual":4.5761676846470186e-05},{"case":2,"passes":8,"relative_equation_residual":2.3452319229425137e-10},{"case":3,"passes":1,"relative_equation_residual":0.09367374961494006},{"case":3,"passes":2,"relative_equation_residual":0.004449386789359839},{"case":3,"passes":4,"relative_equation_residual":8.927193072050128e-06},{"case":3,"passes":8,"relative_equation_residual":8.128301364158176e-11},{"case":4,"passes":1,"relative_equation_residual":0.06861031377310192},{"case":4,"passes":2,"relative_equation_residual":0.0034701318671028197},{"case":4,"passes":4,"relative_equation_residual":7.400097595233373e-06},{"case":4,"passes":8,"relative_equation_residual":6.243039612784178e-11},{"case":5,"passes":1,"relative_equation_residual":0.09366938361271399},{"case":5,"passes":2,"relative_equation_residual":0.004449935581659129},{"case":5,"passes":4,"relative_equation_residual":8.928266406241033e-06},{"case":5,"passes":8,"relative_equation_residual":8.144212800526709e-11},{"case":6,"passes":1,"relative_equation_residual":0.03323043451004816},{"case":6,"passes":2,"relative_equation_residual":0.0009972455926713817},{"case":6,"passes":4,"relative_equation_residual":8.116895962819617e-07},{"case":6,"passes":8,"relative_equation_residual":2.191840038854352e-12},{"case":7,"passes":1,"relative_equation_residual":0.12609987974279746},{"case":7,"passes":2,"relative_equation_residual":0.010817313797939414},{"case":7,"passes":4,"relative_equation_residual":0.00025258607015095884},{"case":7,"passes":8,"relative_equation_residual":6.054348635283862e-08}],"update":240},{"exposures_per_case":40,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.1450273388646824},{"case":0,"passes":2,"relative_equation_residual":0.019040168167704867},{"case":0,"passes":4,"relative_equation_residual":0.0003814617264575381},{"case":0,"passes":8,"relative_equation_residual":1.4803019813577063e-07},{"case":1,"passes":1,"relative_equation_residual":0.08399736115212873},{"case":1,"passes":2,"relative_equation_residual":0.005812803179562194},{"case":1,"passes":4,"relative_equation_residual":2.212750187125205e-05},{"case":1,"passes":8,"relative_equation_residual":1.9535781424896728e-10},{"case":2,"passes":1,"relative_equation_residual":0.15142794689430858},{"case":2,"passes":2,"relative_equation_residual":0.003753720374905611},{"case":2,"passes":4,"relative_equation_residual":5.695370494725789e-06},{"case":2,"passes":8,"relative_equation_residual":5.2143643367136305e-11},{"case":3,"passes":1,"relative_equation_residual":0.07133363271648602},{"case":3,"passes":2,"relative_equation_residual":0.0014414912470226361},{"case":3,"passes":4,"relative_equation_residual":1.988600398594813e-06},{"case":3,"passes":8,"relative_equation_residual":7.340197488912713e-12},{"case":4,"passes":1,"relative_equation_residual":0.050134526982222295},{"case":4,"passes":2,"relative_equation_residual":0.0024694428560295585},{"case":4,"passes":4,"relative_equation_residual":4.272111499950944e-06},{"case":4,"passes":8,"relative_equation_residual":1.3825209601346822e-11},{"case":5,"passes":1,"relative_equation_residual":0.07134211449988437},{"case":5,"passes":2,"relative_equation_residual":0.0014418405139530114},{"case":5,"passes":4,"relative_equation_residual":1.9891097199374088e-06},{"case":5,"passes":8,"relative_equation_residual":1.4610299691907528e-11},{"case":6,"passes":1,"relative_equation_residual":0.023646336664461566},{"case":6,"passes":2,"relative_equation_residual":0.0006890093348279064},{"case":6,"passes":4,"relative_equation_residual":5.831975245287507e-07},{"case":6,"passes":8,"relative_equation_residual":9.688740601904195e-13},{"case":7,"passes":1,"relative_equation_residual":0.17798874363459635},{"case":7,"passes":2,"relative_equation_residual":0.024155442125151878},{"case":7,"passes":4,"relative_equation_residual":0.0007212028156620575},{"case":7,"passes":8,"relative_equation_residual":1.173510913719458e-06}],"update":320},{"exposures_per_case":50,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.15027161909314968},{"case":0,"passes":2,"relative_equation_residual":0.014922637948856104},{"case":0,"passes":4,"relative_equation_residual":0.00019778256624955693},{"case":0,"passes":8,"relative_equation_residual":3.423688662787851e-08},{"case":1,"passes":1,"relative_equation_residual":0.0672327351030092},{"case":1,"passes":2,"relative_equation_residual":0.00230590377081164},{"case":1,"passes":4,"relative_equation_residual":4.1895856937800455e-06},{"case":1,"passes":8,"relative_equation_residual":2.1352572581081185e-11},{"case":2,"passes":1,"relative_equation_residual":0.14203667167092565},{"case":2,"passes":2,"relative_equation_residual":0.008911765047466389},{"case":2,"passes":4,"relative_equation_residual":3.2967716259343955e-05},{"case":2,"passes":8,"relative_equation_residual":4.888361872715871e-10},{"case":3,"passes":1,"relative_equation_residual":0.05049656276478319},{"case":3,"passes":2,"relative_equation_residual":0.0018197359469323887},{"case":3,"passes":4,"relative_equation_residual":6.359021730866205e-06},{"case":3,"passes":8,"relative_equation_residual":4.860006332558155e-11},{"case":4,"passes":1,"relative_equation_residual":0.03195286523026984},{"case":4,"passes":2,"relative_equation_residual":0.0008314567878468408},{"case":4,"passes":4,"relative_equation_residual":8.673552201272692e-07},{"case":4,"passes":8,"relative_equation_residual":1.5630630508098077e-12},{"case":5,"passes":1,"relative_equation_residual":0.05049560115697069},{"case":5,"passes":2,"relative_equation_residual":0.001819439590374939},{"case":5,"passes":4,"relative_equation_residual":6.357099224398094e-06},{"case":5,"passes":8,"relative_equation_residual":5.438335320107502e-11},{"case":6,"passes":1,"relative_equation_residual":0.04045301727737821},{"case":6,"passes":2,"relative_equation_residual":0.0014273794169327265},{"case":6,"passes":4,"relative_equation_residual":1.774380588699443e-06},{"case":6,"passes":8,"relative_equation_residual":9.456734647387955e-12},{"case":7,"passes":1,"relative_equation_residual":0.1688407424361907},{"case":7,"passes":2,"relative_equation_residual":0.024542182827325332},{"case":7,"passes":4,"relative_equation_residual":0.0007918360700840129},{"case":7,"passes":8,"relative_equation_residual":1.9096368435580713e-06}],"update":400},{"exposures_per_case":60,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.19884347466804536},{"case":0,"passes":2,"relative_equation_residual":0.020094654435498634},{"case":0,"passes":4,"relative_equation_residual":6.30618587406555e-05},{"case":0,"passes":8,"relative_equation_residual":1.4394124173807284e-09},{"case":1,"passes":1,"relative_equation_residual":0.07589912857164756},{"case":1,"passes":2,"relative_equation_residual":0.004587407124036452},{"case":1,"passes":4,"relative_equation_residual":1.4316879561305528e-05},{"case":1,"passes":8,"relative_equation_residual":1.1839371148334393e-10},{"case":2,"passes":1,"relative_equation_residual":0.17502616553655723},{"case":2,"passes":2,"relative_equation_residual":0.00795806063583392},{"case":2,"passes":4,"relative_equation_residual":2.1005083503089805e-05},{"case":2,"passes":8,"relative_equation_residual":1.7198068714225893e-10},{"case":3,"passes":1,"relative_equation_residual":0.07032552596471098},{"case":3,"passes":2,"relative_equation_residual":0.001912493517927299},{"case":3,"passes":4,"relative_equation_residual":3.5917404934902636e-06},{"case":3,"passes":8,"relative_equation_residual":5.641434624027e-11},{"case":4,"passes":1,"relative_equation_residual":0.037456766298442946},{"case":4,"passes":2,"relative_equation_residual":0.0021825011461465123},{"case":4,"passes":4,"relative_equation_residual":1.1320669651739332e-05},{"case":4,"passes":8,"relative_equation_residual":3.024741488631126e-10},{"case":5,"passes":1,"relative_equation_residual":0.0703284511245146},{"case":5,"passes":2,"relative_equation_residual":0.001912585731567015},{"case":5,"passes":4,"relative_equation_residual":3.592764126266207e-06},{"case":5,"passes":8,"relative_equation_residual":5.800458987369675e-11},{"case":6,"passes":1,"relative_equation_residual":0.022636650567076637},{"case":6,"passes":2,"relative_equation_residual":0.0008782547950601771},{"case":6,"passes":4,"relative_equation_residual":7.951188343360555e-07},{"case":6,"passes":8,"relative_equation_residual":4.872025407404756e-12},{"case":7,"passes":1,"relative_equation_residual":0.17463762455837092},{"case":7,"passes":2,"relative_equation_residual":0.023450001854754546},{"case":7,"passes":4,"relative_equation_residual":0.0005982405606817479},{"case":7,"passes":8,"relative_equation_residual":5.756281970838669e-07}],"update":480},{"exposures_per_case":70,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.14514982676597946},{"case":0,"passes":2,"relative_equation_residual":0.010816508647933728},{"case":0,"passes":4,"relative_equation_residual":5.164458268127416e-05},{"case":0,"passes":8,"relative_equation_residual":2.2190305917448394e-09},{"case":1,"passes":1,"relative_equation_residual":0.06043007175306412},{"case":1,"passes":2,"relative_equation_residual":0.0022959373179152953},{"case":1,"passes":4,"relative_equation_residual":2.5726985097873183e-06},{"case":1,"passes":8,"relative_equation_residual":8.801329009854286e-12},{"case":2,"passes":1,"relative_equation_residual":0.15609882859040852},{"case":2,"passes":2,"relative_equation_residual":0.0034928834037593537},{"case":2,"passes":4,"relative_equation_residual":3.989436828515531e-05},{"case":2,"passes":8,"relative_equation_residual":4.647126712862691e-10},{"case":3,"passes":1,"relative_equation_residual":0.06858870173825818},{"case":3,"passes":2,"relative_equation_residual":0.0034253668459929463},{"case":3,"passes":4,"relative_equation_residual":1.0395790903221588e-05},{"case":3,"passes":8,"relative_equation_residual":1.0477808703993247e-10},{"case":4,"passes":1,"relative_equation_residual":0.05769902862674901},{"case":4,"passes":2,"relative_equation_residual":0.00200977072407706},{"case":4,"passes":4,"relative_equation_residual":3.104230637049004e-06},{"case":4,"passes":8,"relative_equation_residual":9.129214509175453e-12},{"case":5,"passes":1,"relative_equation_residual":0.06858881777799332},{"case":5,"passes":2,"relative_equation_residual":0.0034255867160833623},{"case":5,"passes":4,"relative_equation_residual":1.0394804042692275e-05},{"case":5,"passes":8,"relative_equation_residual":1.0510363563315748e-10},{"case":6,"passes":1,"relative_equation_residual":0.01538813929238632},{"case":6,"passes":2,"relative_equation_residual":0.0003409217623803326},{"case":6,"passes":4,"relative_equation_residual":9.546202522388428e-08},{"case":6,"passes":8,"relative_equation_residual":7.814773084954291e-13},{"case":7,"passes":1,"relative_equation_residual":0.17578865242006242},{"case":7,"passes":2,"relative_equation_residual":0.017260500537268773},{"case":7,"passes":4,"relative_equation_residual":0.00016924728152365194},{"case":7,"passes":8,"relative_equation_residual":7.309896306494592e-08}],"update":560},{"exposures_per_case":80,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.24519887876852103},{"case":0,"passes":2,"relative_equation_residual":0.02550874672423578},{"case":0,"passes":4,"relative_equation_residual":5.887044945913065e-05},{"case":0,"passes":8,"relative_equation_residual":4.751094917695571e-09},{"case":1,"passes":1,"relative_equation_residual":0.06647545913393155},{"case":1,"passes":2,"relative_equation_residual":0.0017293979594443694},{"case":1,"passes":4,"relative_equation_residual":1.3625307713614914e-06},{"case":1,"passes":8,"relative_equation_residual":8.086569732519488e-12},{"case":2,"passes":1,"relative_equation_residual":0.18185161964437324},{"case":2,"passes":2,"relative_equation_residual":0.004823737041069923},{"case":2,"passes":4,"relative_equation_residual":9.978472662240905e-06},{"case":2,"passes":8,"relative_equation_residual":7.933799706505827e-11},{"case":3,"passes":1,"relative_equation_residual":0.06407426449826525},{"case":3,"passes":2,"relative_equation_residual":0.0023912573018586188},{"case":3,"passes":4,"relative_equation_residual":3.0669348132805252e-06},{"case":3,"passes":8,"relative_equation_residual":1.0357464660354599e-11},{"case":4,"passes":1,"relative_equation_residual":0.032955932810789554},{"case":4,"passes":2,"relative_equation_residual":0.0006809728314985391},{"case":4,"passes":4,"relative_equation_residual":1.2658747803909557e-06},{"case":4,"passes":8,"relative_equation_residual":1.5755879574179249e-12},{"case":5,"passes":1,"relative_equation_residual":0.06407321607056948},{"case":5,"passes":2,"relative_equation_residual":0.00239111053401032},{"case":5,"passes":4,"relative_equation_residual":3.0660604705194016e-06},{"case":5,"passes":8,"relative_equation_residual":2.8198586185712928e-11},{"case":6,"passes":1,"relative_equation_residual":0.0063612039583002735},{"case":6,"passes":2,"relative_equation_residual":0.00028523387763240194},{"case":6,"passes":4,"relative_equation_residual":1.1851792124813299e-07},{"case":6,"passes":8,"relative_equation_residual":5.0906038312952095e-12},{"case":7,"passes":1,"relative_equation_residual":0.13812796971744984},{"case":7,"passes":2,"relative_equation_residual":0.012624988137801427},{"case":7,"passes":4,"relative_equation_residual":0.00023748156398659595},{"case":7,"passes":8,"relative_equation_residual":9.495895292318057e-08}],"update":640},{"exposures_per_case":90,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.15453866041044625},{"case":0,"passes":2,"relative_equation_residual":0.012084809208062593},{"case":0,"passes":4,"relative_equation_residual":4.935040518046107e-05},{"case":0,"passes":8,"relative_equation_residual":1.266948477958243e-09},{"case":1,"passes":1,"relative_equation_residual":0.04089137949485456},{"case":1,"passes":2,"relative_equation_residual":0.001392975690353332},{"case":1,"passes":4,"relative_equation_residual":2.721347048480589e-06},{"case":1,"passes":8,"relative_equation_residual":1.0125878735032723e-11},{"case":2,"passes":1,"relative_equation_residual":0.14723614320304035},{"case":2,"passes":2,"relative_equation_residual":0.008711728158070425},{"case":2,"passes":4,"relative_equation_residual":2.4877289466691115e-05},{"case":2,"passes":8,"relative_equation_residual":1.4891730436058998e-10},{"case":3,"passes":1,"relative_equation_residual":0.07412629005297841},{"case":3,"passes":2,"relative_equation_residual":0.003069411281905517},{"case":3,"passes":4,"relative_equation_residual":3.4863915479274144e-06},{"case":3,"passes":8,"relative_equation_residual":1.1976030622196034e-11},{"case":4,"passes":1,"relative_equation_residual":0.027080282437637906},{"case":4,"passes":2,"relative_equation_residual":0.0007293255884543537},{"case":4,"passes":4,"relative_equation_residual":1.2190563892154369e-06},{"case":4,"passes":8,"relative_equation_residual":3.2808824878472348e-12},{"case":5,"passes":1,"relative_equation_residual":0.07412376468429553},{"case":5,"passes":2,"relative_equation_residual":0.0030698234242727486},{"case":5,"passes":4,"relative_equation_residual":3.4872001106144424e-06},{"case":5,"passes":8,"relative_equation_residual":3.006212900764851e-11},{"case":6,"passes":1,"relative_equation_residual":0.02314335422054106},{"case":6,"passes":2,"relative_equation_residual":0.0006798046195985322},{"case":6,"passes":4,"relative_equation_residual":5.947693718883068e-07},{"case":6,"passes":8,"relative_equation_residual":1.2427209765031346e-11},{"case":7,"passes":1,"relative_equation_residual":0.1766908620852341},{"case":7,"passes":2,"relative_equation_residual":0.023192694464134436},{"case":7,"passes":4,"relative_equation_residual":0.00023149761807931662},{"case":7,"passes":8,"relative_equation_residual":3.7973023606021915e-07}],"update":720},{"exposures_per_case":100,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.27309939685484713},{"case":0,"passes":2,"relative_equation_residual":0.030336498939413883},{"case":0,"passes":4,"relative_equation_residual":0.00014514841385376042},{"case":0,"passes":8,"relative_equation_residual":5.010137737133314e-09},{"case":1,"passes":1,"relative_equation_residual":0.11335870523715237},{"case":1,"passes":2,"relative_equation_residual":0.005334310235243186},{"case":1,"passes":4,"relative_equation_residual":1.4766091883595065e-05},{"case":1,"passes":8,"relative_equation_residual":6.278441205185665e-11},{"case":2,"passes":1,"relative_equation_residual":0.20568588388824766},{"case":2,"passes":2,"relative_equation_residual":0.009183836124698615},{"case":2,"passes":4,"relative_equation_residual":3.4387869303319644e-05},{"case":2,"passes":8,"relative_equation_residual":1.0944994402255257e-09},{"case":3,"passes":1,"relative_equation_residual":0.0967642558463696},{"case":3,"passes":2,"relative_equation_residual":0.005203550263767863},{"case":3,"passes":4,"relative_equation_residual":9.973879422799432e-06},{"case":3,"passes":8,"relative_equation_residual":1.0308691010725175e-10},{"case":4,"passes":1,"relative_equation_residual":0.058545330167150765},{"case":4,"passes":2,"relative_equation_residual":0.002205103173894834},{"case":4,"passes":4,"relative_equation_residual":1.0145211046265982e-05},{"case":4,"passes":8,"relative_equation_residual":2.011496519522709e-10},{"case":5,"passes":1,"relative_equation_residual":0.09676080061279482},{"case":5,"passes":2,"relative_equation_residual":0.00520286758291376},{"case":5,"passes":4,"relative_equation_residual":9.971180343619272e-06},{"case":5,"passes":8,"relative_equation_residual":1.0941423790890251e-10},{"case":6,"passes":1,"relative_equation_residual":0.003989968916822674},{"case":6,"passes":2,"relative_equation_residual":0.0001387162324598856},{"case":6,"passes":4,"relative_equation_residual":3.2046481535261806e-07},{"case":6,"passes":8,"relative_equation_residual":6.322336617551939e-12},{"case":7,"passes":1,"relative_equation_residual":0.19602843044948692},{"case":7,"passes":2,"relative_equation_residual":0.022167011717892333},{"case":7,"passes":4,"relative_equation_residual":0.00016928819540043772},{"case":7,"passes":8,"relative_equation_residual":1.2125634033352293e-07}],"update":800},{"exposures_per_case":110,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.21846131583796888},{"case":0,"passes":2,"relative_equation_residual":0.017925162765373166},{"case":0,"passes":4,"relative_equation_residual":2.908905977914298e-05},{"case":0,"passes":8,"relative_equation_residual":7.170920749228369e-10},{"case":1,"passes":1,"relative_equation_residual":0.06804571084595337},{"case":1,"passes":2,"relative_equation_residual":0.001499769670896302},{"case":1,"passes":4,"relative_equation_residual":4.820159831053242e-07},{"case":1,"passes":8,"relative_equation_residual":8.822080507986536e-12},{"case":2,"passes":1,"relative_equation_residual":0.1602421080860631},{"case":2,"passes":2,"relative_equation_residual":0.005276181374248987},{"case":2,"passes":4,"relative_equation_residual":3.813935268549722e-06},{"case":2,"passes":8,"relative_equation_residual":4.903026565576697e-11},{"case":3,"passes":1,"relative_equation_residual":0.05459909381540989},{"case":3,"passes":2,"relative_equation_residual":0.001410251841097633},{"case":3,"passes":4,"relative_equation_residual":6.314894252118933e-07},{"case":3,"passes":8,"relative_equation_residual":7.332037255133808e-12},{"case":4,"passes":1,"relative_equation_residual":0.026626184856251206},{"case":4,"passes":2,"relative_equation_residual":0.0010443683276704391},{"case":4,"passes":4,"relative_equation_residual":1.3523526588936161e-06},{"case":4,"passes":8,"relative_equation_residual":1.5436214769201468e-12},{"case":5,"passes":1,"relative_equation_residual":0.054598258112513336},{"case":5,"passes":2,"relative_equation_residual":0.0014102467867083371},{"case":5,"passes":4,"relative_equation_residual":6.308792001690283e-07},{"case":5,"passes":8,"relative_equation_residual":4.044843504729232e-11},{"case":6,"passes":1,"relative_equation_residual":0.01963181886763094},{"case":6,"passes":2,"relative_equation_residual":0.0003428037388218421},{"case":6,"passes":4,"relative_equation_residual":1.3630782392977443e-07},{"case":6,"passes":8,"relative_equation_residual":9.049918416408632e-12},{"case":7,"passes":1,"relative_equation_residual":0.1562215921172045},{"case":7,"passes":2,"relative_equation_residual":0.016691456843295822},{"case":7,"passes":4,"relative_equation_residual":9.870798671654634e-05},{"case":7,"passes":8,"relative_equation_residual":1.104771191436377e-07}],"update":880},{"exposures_per_case":120,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.22110178856320906},{"case":0,"passes":2,"relative_equation_residual":0.01872463692665355},{"case":0,"passes":4,"relative_equation_residual":3.755427900361414e-05},{"case":0,"passes":8,"relative_equation_residual":1.1332781434792174e-09},{"case":1,"passes":1,"relative_equation_residual":0.059241491896284436},{"case":1,"passes":2,"relative_equation_residual":0.0010902175150257472},{"case":1,"passes":4,"relative_equation_residual":6.927039220166282e-07},{"case":1,"passes":8,"relative_equation_residual":7.504483480707476e-12},{"case":2,"passes":1,"relative_equation_residual":0.15766269175142048},{"case":2,"passes":2,"relative_equation_residual":0.004563372653988339},{"case":2,"passes":4,"relative_equation_residual":2.0639108469373923e-06},{"case":2,"passes":8,"relative_equation_residual":2.5037656104281412e-11},{"case":3,"passes":1,"relative_equation_residual":0.04671220473705708},{"case":3,"passes":2,"relative_equation_residual":0.0012671706833900407},{"case":3,"passes":4,"relative_equation_residual":3.357962809302679e-07},{"case":3,"passes":8,"relative_equation_residual":5.209187649860203e-12},{"case":4,"passes":1,"relative_equation_residual":0.015439560667916111},{"case":4,"passes":2,"relative_equation_residual":0.000577573710625175},{"case":4,"passes":4,"relative_equation_residual":6.95054710610812e-07},{"case":4,"passes":8,"relative_equation_residual":9.128583236313045e-13},{"case":5,"passes":1,"relative_equation_residual":0.0467116960566797},{"case":5,"passes":2,"relative_equation_residual":0.00126750382861263},{"case":5,"passes":4,"relative_equation_residual":3.3591762258102507e-07},{"case":5,"passes":8,"relative_equation_residual":2.0289893280812946e-11},{"case":6,"passes":1,"relative_equation_residual":0.018429783161769103},{"case":6,"passes":2,"relative_equation_residual":0.0002570485956838282},{"case":6,"passes":4,"relative_equation_residual":4.393936634485214e-08},{"case":6,"passes":8,"relative_equation_residual":5.759818933488689e-12},{"case":7,"passes":1,"relative_equation_residual":0.1503484862448845},{"case":7,"passes":2,"relative_equation_residual":0.014231856390272676},{"case":7,"passes":4,"relative_equation_residual":5.800463777560048e-05},{"case":7,"passes":8,"relative_equation_residual":5.976400586910816e-08}],"update":960},{"exposures_per_case":130,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2282348913755673},{"case":0,"passes":2,"relative_equation_residual":0.022175134663742695},{"case":0,"passes":4,"relative_equation_residual":9.772409633139853e-05},{"case":0,"passes":8,"relative_equation_residual":2.720518436455343e-09},{"case":1,"passes":1,"relative_equation_residual":0.06155564817902378},{"case":1,"passes":2,"relative_equation_residual":0.0007359589834255686},{"case":1,"passes":4,"relative_equation_residual":4.808038889717112e-07},{"case":1,"passes":8,"relative_equation_residual":1.2157364830267373e-11},{"case":2,"passes":1,"relative_equation_residual":0.15595264873344},{"case":2,"passes":2,"relative_equation_residual":0.004283091199692565},{"case":2,"passes":4,"relative_equation_residual":5.467016511075971e-06},{"case":2,"passes":8,"relative_equation_residual":1.3727596670846767e-11},{"case":3,"passes":1,"relative_equation_residual":0.03922338355407412},{"case":3,"passes":2,"relative_equation_residual":0.0011330901442411197},{"case":3,"passes":4,"relative_equation_residual":3.0922502942241844e-07},{"case":3,"passes":8,"relative_equation_residual":5.700840254608199e-12},{"case":4,"passes":1,"relative_equation_residual":0.009502341936691132},{"case":4,"passes":2,"relative_equation_residual":9.676316968247144e-05},{"case":4,"passes":4,"relative_equation_residual":1.50066117114268e-07},{"case":4,"passes":8,"relative_equation_residual":7.192418691221466e-13},{"case":5,"passes":1,"relative_equation_residual":0.03922423633623295},{"case":5,"passes":2,"relative_equation_residual":0.0011330097301599112},{"case":5,"passes":4,"relative_equation_residual":3.1037339140429434e-07},{"case":5,"passes":8,"relative_equation_residual":4.151863355084963e-11},{"case":6,"passes":1,"relative_equation_residual":0.016205597600569797},{"case":6,"passes":2,"relative_equation_residual":5.0075757349461096e-05},{"case":6,"passes":4,"relative_equation_residual":3.073381406555882e-08},{"case":6,"passes":8,"relative_equation_residual":5.6462775814917234e-12},{"case":7,"passes":1,"relative_equation_residual":0.14734184624711363},{"case":7,"passes":2,"relative_equation_residual":0.012152051892775408},{"case":7,"passes":4,"relative_equation_residual":2.3085841546764322e-05},{"case":7,"passes":8,"relative_equation_residual":3.6214794732832107e-09}],"update":1040},{"exposures_per_case":140,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2232391235846681},{"case":0,"passes":2,"relative_equation_residual":0.01813740457284024},{"case":0,"passes":4,"relative_equation_residual":2.860548776839654e-05},{"case":0,"passes":8,"relative_equation_residual":8.023033887032483e-10},{"case":1,"passes":1,"relative_equation_residual":0.0586240686212192},{"case":1,"passes":2,"relative_equation_residual":0.00041823611369461524},{"case":1,"passes":4,"relative_equation_residual":1.965543290185819e-07},{"case":1,"passes":8,"relative_equation_residual":6.4576099837507594e-12},{"case":2,"passes":1,"relative_equation_residual":0.1554277865450247},{"case":2,"passes":2,"relative_equation_residual":0.004293247886073158},{"case":2,"passes":4,"relative_equation_residual":2.4085860546979722e-06},{"case":2,"passes":8,"relative_equation_residual":1.8878935134863423e-11},{"case":3,"passes":1,"relative_equation_residual":0.03845186547764437},{"case":3,"passes":2,"relative_equation_residual":0.001078795388245931},{"case":3,"passes":4,"relative_equation_residual":2.0290153977431093e-07},{"case":3,"passes":8,"relative_equation_residual":8.298227549754999e-12},{"case":4,"passes":1,"relative_equation_residual":0.007451929233039551},{"case":4,"passes":2,"relative_equation_residual":9.574578282285142e-05},{"case":4,"passes":4,"relative_equation_residual":1.2587766848124007e-07},{"case":4,"passes":8,"relative_equation_residual":3.97426814469595e-13},{"case":5,"passes":1,"relative_equation_residual":0.0384601100552054},{"case":5,"passes":2,"relative_equation_residual":0.0010791734563313127},{"case":5,"passes":4,"relative_equation_residual":2.0609601892233744e-07},{"case":5,"passes":8,"relative_equation_residual":5.939669998399736e-11},{"case":6,"passes":1,"relative_equation_residual":0.016490670392104258},{"case":6,"passes":2,"relative_equation_residual":0.00011716537789566339},{"case":6,"passes":4,"relative_equation_residual":2.9631278727844078e-08},{"case":6,"passes":8,"relative_equation_residual":8.465438265482263e-12},{"case":7,"passes":1,"relative_equation_residual":0.14993187554616538},{"case":7,"passes":2,"relative_equation_residual":0.013244771476962647},{"case":7,"passes":4,"relative_equation_residual":4.3706497767266646e-05},{"case":7,"passes":8,"relative_equation_residual":1.9920485668939514e-08}],"update":1120},{"exposures_per_case":150,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.23080509027296367},{"case":0,"passes":2,"relative_equation_residual":0.020661125301705534},{"case":0,"passes":4,"relative_equation_residual":5.522447289420199e-05},{"case":0,"passes":8,"relative_equation_residual":1.3553373348913343e-09},{"case":1,"passes":1,"relative_equation_residual":0.06052609420678913},{"case":1,"passes":2,"relative_equation_residual":0.0003858925033839682},{"case":1,"passes":4,"relative_equation_residual":2.3633369562852127e-07},{"case":1,"passes":8,"relative_equation_residual":9.077217116394176e-12},{"case":2,"passes":1,"relative_equation_residual":0.16134812668421428},{"case":2,"passes":2,"relative_equation_residual":0.004440506379091602},{"case":2,"passes":4,"relative_equation_residual":1.9777588025477205e-06},{"case":2,"passes":8,"relative_equation_residual":3.7352507218466257e-11},{"case":3,"passes":1,"relative_equation_residual":0.037951081009954334},{"case":3,"passes":2,"relative_equation_residual":0.0011105005554550467},{"case":3,"passes":4,"relative_equation_residual":1.2435592766104778e-07},{"case":3,"passes":8,"relative_equation_residual":7.742852892095513e-12},{"case":4,"passes":1,"relative_equation_residual":0.007578954635167368},{"case":4,"passes":2,"relative_equation_residual":0.00015363460737012182},{"case":4,"passes":4,"relative_equation_residual":2.067677021180388e-07},{"case":4,"passes":8,"relative_equation_residual":4.106391800257e-13},{"case":5,"passes":1,"relative_equation_residual":0.037951955223783126},{"case":5,"passes":2,"relative_equation_residual":0.001110897156636307},{"case":5,"passes":4,"relative_equation_residual":1.2662886444663864e-07},{"case":5,"passes":8,"relative_equation_residual":2.962478005828502e-11},{"case":6,"passes":1,"relative_equation_residual":0.015468946656653685},{"case":6,"passes":2,"relative_equation_residual":3.4327046654723154e-05},{"case":6,"passes":4,"relative_equation_residual":2.855250022008596e-08},{"case":6,"passes":8,"relative_equation_residual":5.094251734811443e-12},{"case":7,"passes":1,"relative_equation_residual":0.14644357338653588},{"case":7,"passes":2,"relative_equation_residual":0.011720730855907715},{"case":7,"passes":4,"relative_equation_residual":5.414933501556971e-05},{"case":7,"passes":8,"relative_equation_residual":2.8112556715510504e-08}],"update":1200},{"exposures_per_case":160,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.22591941869058227},{"case":0,"passes":2,"relative_equation_residual":0.01866397473011903},{"case":0,"passes":4,"relative_equation_residual":2.4079006417809207e-05},{"case":0,"passes":8,"relative_equation_residual":5.597748293957559e-10},{"case":1,"passes":1,"relative_equation_residual":0.057028066754939453},{"case":1,"passes":2,"relative_equation_residual":0.0001936241660713595},{"case":1,"passes":4,"relative_equation_residual":9.718104696550719e-08},{"case":1,"passes":8,"relative_equation_residual":5.919097272491313e-12},{"case":2,"passes":1,"relative_equation_residual":0.1582278710253509},{"case":2,"passes":2,"relative_equation_residual":0.004338183876001547},{"case":2,"passes":4,"relative_equation_residual":1.4159624631059826e-06},{"case":2,"passes":8,"relative_equation_residual":2.0679242923550775e-11},{"case":3,"passes":1,"relative_equation_residual":0.03842717235756075},{"case":3,"passes":2,"relative_equation_residual":0.0010508042601034275},{"case":3,"passes":4,"relative_equation_residual":3.911414482466426e-08},{"case":3,"passes":8,"relative_equation_residual":7.560666591158083e-12},{"case":4,"passes":1,"relative_equation_residual":0.005510030699733465},{"case":4,"passes":2,"relative_equation_residual":2.1486355405653306e-05},{"case":4,"passes":4,"relative_equation_residual":1.4920571717470606e-08},{"case":4,"passes":8,"relative_equation_residual":2.4168080297061484e-13},{"case":5,"passes":1,"relative_equation_residual":0.03842073996250419},{"case":5,"passes":2,"relative_equation_residual":0.0010507107967656535},{"case":5,"passes":4,"relative_equation_residual":4.173867738257715e-08},{"case":5,"passes":8,"relative_equation_residual":2.2795188144214506e-11},{"case":6,"passes":1,"relative_equation_residual":0.015803546189247005},{"case":6,"passes":2,"relative_equation_residual":3.119148274422217e-05},{"case":6,"passes":4,"relative_equation_residual":2.4731404269773235e-08},{"case":6,"passes":8,"relative_equation_residual":1.1791108245831862e-11},{"case":7,"passes":1,"relative_equation_residual":0.14878513763459678},{"case":7,"passes":2,"relative_equation_residual":0.01230711734191774},{"case":7,"passes":4,"relative_equation_residual":2.141396653940946e-05},{"case":7,"passes":8,"relative_equation_residual":4.709775371300152e-09}],"update":1280},{"exposures_per_case":170,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2264839801954962},{"case":0,"passes":2,"relative_equation_residual":0.018632570144441023},{"case":0,"passes":4,"relative_equation_residual":2.332265911483911e-05},{"case":0,"passes":8,"relative_equation_residual":5.128470210630623e-10},{"case":1,"passes":1,"relative_equation_residual":0.05622430559410974},{"case":1,"passes":2,"relative_equation_residual":0.0001676222674176783},{"case":1,"passes":4,"relative_equation_residual":7.641246458226039e-08},{"case":1,"passes":8,"relative_equation_residual":1.0558134046268421e-11},{"case":2,"passes":1,"relative_equation_residual":0.15888131093873145},{"case":2,"passes":2,"relative_equation_residual":0.004463833144164893},{"case":2,"passes":4,"relative_equation_residual":1.4360795060513654e-06},{"case":2,"passes":8,"relative_equation_residual":1.7027010485246356e-11},{"case":3,"passes":1,"relative_equation_residual":0.03855366362526828},{"case":3,"passes":2,"relative_equation_residual":0.0010566857698716668},{"case":3,"passes":4,"relative_equation_residual":2.6612500358603668e-08},{"case":3,"passes":8,"relative_equation_residual":7.58045371458871e-12},{"case":4,"passes":1,"relative_equation_residual":0.004670902149759425},{"case":4,"passes":2,"relative_equation_residual":1.1717835814750547e-05},{"case":4,"passes":4,"relative_equation_residual":9.506277063669392e-09},{"case":4,"passes":8,"relative_equation_residual":3.3912007948931064e-13},{"case":5,"passes":1,"relative_equation_residual":0.038557393801033964},{"case":5,"passes":2,"relative_equation_residual":0.001056864779764436},{"case":5,"passes":4,"relative_equation_residual":3.542099899399028e-08},{"case":5,"passes":8,"relative_equation_residual":3.236827935118426e-11},{"case":6,"passes":1,"relative_equation_residual":0.016054662000321084},{"case":6,"passes":2,"relative_equation_residual":3.4737330819844795e-05},{"case":6,"passes":4,"relative_equation_residual":2.713190867338551e-08},{"case":6,"passes":8,"relative_equation_residual":1.028781720732472e-11},{"case":7,"passes":1,"relative_equation_residual":0.14820933326781247},{"case":7,"passes":2,"relative_equation_residual":0.012154338919840059},{"case":7,"passes":4,"relative_equation_residual":1.963417314960855e-05},{"case":7,"passes":8,"relative_equation_residual":7.734624971690207e-09}],"update":1360},{"exposures_per_case":180,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.22651384668046792},{"case":0,"passes":2,"relative_equation_residual":0.01855498383822884},{"case":0,"passes":4,"relative_equation_residual":2.300786422226735e-05},{"case":0,"passes":8,"relative_equation_residual":4.786245369332811e-10},{"case":1,"passes":1,"relative_equation_residual":0.05577373669866897},{"case":1,"passes":2,"relative_equation_residual":0.00012108315350381604},{"case":1,"passes":4,"relative_equation_residual":6.713467889165006e-08},{"case":1,"passes":8,"relative_equation_residual":1.0250187457839961e-11},{"case":2,"passes":1,"relative_equation_residual":0.1592215437689119},{"case":2,"passes":2,"relative_equation_residual":0.004479320778578681},{"case":2,"passes":4,"relative_equation_residual":1.4008323761289344e-06},{"case":2,"passes":8,"relative_equation_residual":6.430448353258702e-11},{"case":3,"passes":1,"relative_equation_residual":0.038917229886085765},{"case":3,"passes":2,"relative_equation_residual":0.0010655167408357045},{"case":3,"passes":4,"relative_equation_residual":2.6240104288519385e-08},{"case":3,"passes":8,"relative_equation_residual":7.0284376667901025e-12},{"case":4,"passes":1,"relative_equation_residual":0.004505060434715678},{"case":4,"passes":2,"relative_equation_residual":1.029826823429122e-05},{"case":4,"passes":4,"relative_equation_residual":2.103943160792295e-08},{"case":4,"passes":8,"relative_equation_residual":5.067277535525751e-13},{"case":5,"passes":1,"relative_equation_residual":0.038917623238248475},{"case":5,"passes":2,"relative_equation_residual":0.0010658972052422634},{"case":5,"passes":4,"relative_equation_residual":3.4531119961388445e-08},{"case":5,"passes":8,"relative_equation_residual":3.1153662927203717e-11},{"case":6,"passes":1,"relative_equation_residual":0.016088069241558988},{"case":6,"passes":2,"relative_equation_residual":3.365808371088405e-05},{"case":6,"passes":4,"relative_equation_residual":2.2247602049593575e-08},{"case":6,"passes":8,"relative_equation_residual":9.654242354578727e-12},{"case":7,"passes":1,"relative_equation_residual":0.148503015246012},{"case":7,"passes":2,"relative_equation_residual":0.012147929095609161},{"case":7,"passes":4,"relative_equation_residual":1.914549205706029e-05},{"case":7,"passes":8,"relative_equation_residual":7.583990912192894e-09}],"update":1440},{"exposures_per_case":190,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2266741780608643},{"case":0,"passes":2,"relative_equation_residual":0.01859156216909767},{"case":0,"passes":4,"relative_equation_residual":2.2959581211645166e-05},{"case":0,"passes":8,"relative_equation_residual":4.796915695398109e-10},{"case":1,"passes":1,"relative_equation_residual":0.05585152874850129},{"case":1,"passes":2,"relative_equation_residual":0.00013353584923316418},{"case":1,"passes":4,"relative_equation_residual":6.580775314669455e-08},{"case":1,"passes":8,"relative_equation_residual":1.6239169812602707e-11},{"case":2,"passes":1,"relative_equation_residual":0.15928865966340033},{"case":2,"passes":2,"relative_equation_residual":0.004505125287763706},{"case":2,"passes":4,"relative_equation_residual":1.4206457183257958e-06},{"case":2,"passes":8,"relative_equation_residual":4.5656477322442204e-11},{"case":3,"passes":1,"relative_equation_residual":0.03882736231560392},{"case":3,"passes":2,"relative_equation_residual":0.0010673625278887341},{"case":3,"passes":4,"relative_equation_residual":2.3147285026052815e-08},{"case":3,"passes":8,"relative_equation_residual":5.8263926360715484e-12},{"case":4,"passes":1,"relative_equation_residual":0.004447369526418747},{"case":4,"passes":2,"relative_equation_residual":8.082932581435681e-06},{"case":4,"passes":4,"relative_equation_residual":1.820261965894203e-08},{"case":4,"passes":8,"relative_equation_residual":2.4426658174235757e-13},{"case":5,"passes":1,"relative_equation_residual":0.038825229197476725},{"case":5,"passes":2,"relative_equation_residual":0.0010671715545806028},{"case":5,"passes":4,"relative_equation_residual":3.1024477706255804e-08},{"case":5,"passes":8,"relative_equation_residual":2.7945108552919673e-11},{"case":6,"passes":1,"relative_equation_residual":0.016086314170141145},{"case":6,"passes":2,"relative_equation_residual":3.395827246976974e-05},{"case":6,"passes":4,"relative_equation_residual":2.184839357971249e-08},{"case":6,"passes":8,"relative_equation_residual":7.485987201364047e-12},{"case":7,"passes":1,"relative_equation_residual":0.14834324054294154},{"case":7,"passes":2,"relative_equation_residual":0.0121155176936671},{"case":7,"passes":4,"relative_equation_residual":1.8864002058115286e-05},{"case":7,"passes":8,"relative_equation_residual":8.276499173578175e-09}],"update":1520},{"exposures_per_case":200,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2266594465438059},{"case":0,"passes":2,"relative_equation_residual":0.018586505225131233},{"case":0,"passes":4,"relative_equation_residual":2.2941952813000607e-05},{"case":0,"passes":8,"relative_equation_residual":4.871498130969093e-10},{"case":1,"passes":1,"relative_equation_residual":0.05582941321702509},{"case":1,"passes":2,"relative_equation_residual":0.00013381885369150067},{"case":1,"passes":4,"relative_equation_residual":6.525212188897881e-08},{"case":1,"passes":8,"relative_equation_residual":1.639783242991297e-11},{"case":2,"passes":1,"relative_equation_residual":0.15928367562377765},{"case":2,"passes":2,"relative_equation_residual":0.004505626599399743},{"case":2,"passes":4,"relative_equation_residual":1.4194111651375633e-06},{"case":2,"passes":8,"relative_equation_residual":3.2077439627141265e-11},{"case":3,"passes":1,"relative_equation_residual":0.038807714366743906},{"case":3,"passes":2,"relative_equation_residual":0.0010653816631056275},{"case":3,"passes":4,"relative_equation_residual":2.3910606284254108e-08},{"case":3,"passes":8,"relative_equation_residual":7.801768046423325e-12},{"case":4,"passes":1,"relative_equation_residual":0.004413013267759279},{"case":4,"passes":2,"relative_equation_residual":8.219389237200617e-06},{"case":4,"passes":4,"relative_equation_residual":1.7701568837364112e-08},{"case":4,"passes":8,"relative_equation_residual":4.224819585528187e-13},{"case":5,"passes":1,"relative_equation_residual":0.03880578856268417},{"case":5,"passes":2,"relative_equation_residual":0.0010654170110784348},{"case":5,"passes":4,"relative_equation_residual":2.711790940906923e-08},{"case":5,"passes":8,"relative_equation_residual":1.5074530626787666e-11},{"case":6,"passes":1,"relative_equation_residual":0.016103932429406637},{"case":6,"passes":2,"relative_equation_residual":3.410812779589425e-05},{"case":6,"passes":4,"relative_equation_residual":2.279756328103873e-08},{"case":6,"passes":8,"relative_equation_residual":1.0458283908567727e-11},{"case":7,"passes":1,"relative_equation_residual":0.14830364112196967},{"case":7,"passes":2,"relative_equation_residual":0.01211266620672953},{"case":7,"passes":4,"relative_equation_residual":1.886816957477154e-05},{"case":7,"passes":8,"relative_equation_residual":8.3757728599066e-09}],"update":1600}]},"rollout_feedback":{"auxiliary_losses":0,"depth_exposures":[[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50]],"depth_order_sha256":"3adbd1b18830b55ac918ce265241ded42841f53a56341f43b1e333ebd0615257","mode":"U","original_sampling_order_sha256":"d0ea221a7e03a83673ece7fc9cdf4f5212bb54f3023511f93d95c84e45647899"},"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":1600,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"checkpoint-1600.npz":{"bytes":1485452,"role":"checkpoint","sha256":"69b49daa62ea6945726571dcf01eecc6d0084a89c40acd5531c5bd6607260655"},"history.json":{"bytes":10204,"role":"training_diagnostics","sha256":"8e3f11a0b1e99ed615ee7766620801ac3fb7e7b9c8644f32a8b1e1ac87725fe9"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"ad53923127cb58929e915a05f6a871dad3f25c7eff44e32db230dcdb0b3480b5"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":478,"role":"result_summary","sha256":"edf6fa553221cb639896142dd4b945432369e0ba0f09484daec261b8d8effaf2"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"69b49daa62ea6945726571dcf01eecc6d0084a89c40acd5531c5bd6607260655"}},"per_case_inference_s":[0.023504262004280463,0.004541875998256728,0.005697590997442603,0.004512754007009789,0.003968228003941476,0.003749499999685213,0.0047596769873052835,0.005006651015719399,0.010302980983396992,0.005644615972414613,0.006737427989719436,0.0060749350232072175],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"ba2dce51dd2b168576782c5ff36a74d7cbe4834e6f25c8bd54201560b50726b8","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"d2f3a79fa528407e82f82dec003f305a458af91de33e270b90ca68e42bb31636","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"1ff8a95102024b00127ed210c3079191d71e92590a15c0a705d03b877e664622","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"a7f55d6f2c2398ae073d6295b417ee77e8c7033ae31a72ad9eac2873225f5981","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"f25dbc9bf20729ed78fb4475a08268ca5c0e12cd484aeb70b2ac03964b4c90e0","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"689a75d6348942cd86916d5bddee0e6dde30e47e72119b81384fcc805b022e51","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"f90da57c07d7a87ee1ba55ef386bb0922d527a46a17be6e026bd0f53160c6798","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"0a0b5861d57e13ceed509385689a329333ecde4894982747a21f06da38e1289f","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"70b5c26f3d6c97bdae07ac0cbdab3b3c8ba3f535cade07628abce40e199b514b","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"43198c528358599ea6aad0c15f1ccedba4b47794bff302961305f322cbc928ef","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"3f8ecea647578b9cc038766d3a934221fc0294f8a21af5c6297ccb7068b10beb","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"8cfe4e7289f34e1a859535c7df44d657c981debb9f217751f11e7650a09ef4b2","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2413813760,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
