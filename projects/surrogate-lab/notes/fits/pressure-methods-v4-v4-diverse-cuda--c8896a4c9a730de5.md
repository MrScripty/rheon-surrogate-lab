# Pressure methods v4 · v4-diverse CUDA

Retrospective source review · fit-c8896a4c9a730de5

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 371,214-parameter F32 linear multiscale pressure CNN plus learned global branch: 370,016 fine coefficients, 63 coarse logits and a 1,135-parameter grid-shape head. Two ordinary internal pressure corrections each apply a residual-gated low-mode cosine proposal and fine neural correction accumulated in F64. Physical q=(rho/h)r is the single neural field input, using fixed C53 scales; dt is scalar validation metadata. This is a pressure-only request, not two physical simulation outputs.

**Purpose.** Add eight native-generated diverse pressure-request parents to the 34-parent physical replay bank while retaining C101 warm parameters and the fixed fitting budget.

**Experiment.** 2,400 CUDA Adam updates, batch one, clip norm one; fine rate 1e-05 with captured late drops. There are 42 parent groups / 84 saved examples, all train-only. The sampler preserves 1,600 small draws and interleaves 800 larger-tier draws; exact parent/frame exposures and voxel work verify. Grids extend through 128³ at 8/16/32 cells per metre, with fixed dt=1/60 s and fluid density 1,000 kg/m³. Both start from exact completed C101 weights. The original 34-parent bank is retained. Eight additional physical captures cover cubes and rectangles at 8/16 cells per metre, with two snapshots each. The unchanged final/intermediate physical equation-feedback objective uses two neural proposals, early weight 0.05, target 1e-6 and no eight-pass teacher. Captured learning work is 4,800 root neural forwards / 1,209,357,700 input voxels; inherited auxiliary banks and the old teacher are inactive.

**Observed outcome.** Retained final logged training loss 14.794723954859286. The existing common 128-request result has worst mean-free pressure relative L2 0.0052055710127473552, equation relative L2 0.00079813341693507156 and GPU median of case-median pressure calls 6.23391875 ms. Scope: Complete captured two-correction pressure proposal; no velocity/tracer advance. Timings include host-owned input/output, validation/packing, transfers and synchronization for the captured readout, with setup/scoring separate. These are retained consumed small-grid cases, not independent large-grid or trajectory results. No new model execution occurs here.

**Lesson.** The eight added trajectories alter the larger-parent sample distribution under a fixed 800-draw budget; they do not add 800 more updates. The common small-grid pressure tail is slightly higher than C108 while equation error is slightly lower. New native training scenes are not independent common-test scenes.

**Reproduction.** Retain the exact typed native recipe, all witnessed source/config/runtime, fixed normalization, complete original input packet, source population and initial/final tensors. Preserve balanced private RNG streams, physical parent/frame counts and final-update selection. Retain native augmentation receipts or manufactured-pattern generator as applicable; post-coarse and differentiable hierarchy training require their exact numerical arithmetic and gauge/adjoint. The historical wrapper/restore runs CPU neural inference and needs a qualified GPU-only route before replay. Clean retraining, new-Rheon regeneration, complete-composition benchmarking and drift remain unqualified; no historical Trial ancestry is invented.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v4/v4-diverse-study-01/train/receipt.json",
    "sha256": "c8896a4c9a730de5fdc72cc7eff50b9234646d70cdafd7e99d177106f6d58f4f"
  },
  "history": {
    "path": "runs/pressure-methods-v4/v4-diverse-study-01/train/history.json",
    "sha256": "d1dece1cd01c5b118377b4a34b81b1aecd5c1669df5f2682ff02748665eae762"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/coarse.py",
      "sha256": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/config.json",
      "sha256": "92da9085afd5168afb8d3500f346f5305e07224d6ae582a0bdcefdaf8772f132"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/data.py",
      "sha256": "e97b5215fa7207f2ced9c6aa7571aca532fee047a1ec4c21026a5b98662f8e2d"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/fine-tuning.json",
      "sha256": "1d5bb6dfef83d96f6dcf673149491c50db87c4951fa1b4a10c221f4e602a9bbd"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/frozen_gn.py",
      "sha256": "ce280f6ce86aeccc31521b13e4df039780fda8c112c63616457771377903e08d"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/head_optimizer.py",
      "sha256": "b96a4d5196e514ef8a13561415184df8f7ac1540474782b933a5928b5b8f98a6"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/hybrid-method.json",
      "sha256": "7286631ecc547c35be01b10413acc6e3f85649357bac8c308e25019cffccfbe7"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/learned_coarse_model.py",
      "sha256": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/methods_view.py",
      "sha256": "9fab1a3a29a50efa16c82dd1c67e3c8e9a8a7f0a21b6f04e1291ff1a49e79f88"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/mixed_order.py",
      "sha256": "93d987b0f10101fb5b380e7310aa77eb6f5c7557198f6d43d90f102393e47b04"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/model.py",
      "sha256": "63425e0a06f96ccd35476c94fabccb9bdfbd3691c706d98f2fa362ea00b3b4d9"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/native_reconstruct.py",
      "sha256": "332e4d197ce9f2fd1b1ddaf644e8cf89ffac44ab7bad488e7e6b3ec0045bc2f0"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/original_train.py",
      "sha256": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/physical_residual.py",
      "sha256": "6e1392c26880c034d09c72437fefd4c1865d6540f40e245825a6ad6a4e7f4204"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/policy.py",
      "sha256": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/portable_policy.py",
      "sha256": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/reference_spectral.py",
      "sha256": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/train.py",
      "sha256": "128f32615c708983e2302a760324c9d014e90eae01d495aad69861c192d9b880"
    },
    {
      "path": "runs/pressure-methods-v4/v4-diverse-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v4/v4-diverse-project-01/source-witness.json",
    "sha256": "a790c82583a674983e4fa558bc168589084761ba971a3cf066d979aeada4e52d"
  },
  "training_population": {
    "path": "runs/pressure-methods-v4/v4-diverse-study-01/train/splits.json",
    "sha256": "c9625fa89cab4eedd5cbdbcbc4ef7f895e4e7c37e381a2fe049576da494503b0"
  },
  "training_inputs": [
    {
      "bytes": 21725324,
      "path": "runs/pressure-methods-v4/diverse-training-view-01/dataset-00.npz",
      "sha256": "67550093feb355665cb2a393b3c07358a11ddcbe18e8369227a4b337a4a769a1",
      "slot": "dataset"
    },
    {
      "bytes": 25596700,
      "path": "runs/pressure-methods-v4/diverse-training-view-01/dataset-01.npz",
      "sha256": "40a32d21dabb649282007dda29c99fb0ba3006450fd1c28e6f28b77caef130ee",
      "slot": "dataset_extra"
    },
    {
      "bytes": 1491574,
      "path": "runs/pressure-methods-v3/v3-native-study-07/restore/weights.npz",
      "sha256": "42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4",
      "slot": "initial_weights"
    },
    {
      "bytes": 72752,
      "path": "runs/pressure-methods-v4/diverse-training-view-01/manifest.json",
      "sha256": "c9625fa89cab4eedd5cbdbcbc4ef7f895e4e7c37e381a2fe049576da494503b0",
      "slot": "splits"
    },
    {
      "bytes": 22266499,
      "path": "runs/pressure-methods-v4/diverse-training-view-01/dataset-02.npz",
      "sha256": "e45cedbf79c56be57746693e44c9574f26d4b650fc597d3754aa9b7bd2a5367d",
      "slot": "view_02"
    },
    {
      "bytes": 23241588,
      "path": "runs/pressure-methods-v4/diverse-training-view-01/dataset-03.npz",
      "sha256": "7d787d92f596e703bcc054a37a03502dfce7673fd2fb74986cb80633cd39cf26",
      "slot": "view_03"
    },
    {
      "bytes": 22110479,
      "path": "runs/pressure-methods-v4/diverse-training-view-01/dataset-04.npz",
      "sha256": "e24733c8d071ce59e3c3fa00ccdd689116be581fc35c617327c2332d002215b1",
      "slot": "view_04"
    },
    {
      "bytes": 25958557,
      "path": "runs/pressure-methods-v4/diverse-training-view-01/dataset-05.npz",
      "sha256": "bde0a43b3eea813399d832d9978e110ab0f320df41deb179bc4bb4ebfd9b56a0",
      "slot": "view_05"
    },
    {
      "bytes": 21506413,
      "path": "runs/pressure-methods-v4/diverse-training-view-01/dataset-06.npz",
      "sha256": "fe1d383e77df98af6af8b8eb955f39fc5ea1dd116526de5c00f823a5924df6ae",
      "slot": "view_06"
    },
    {
      "bytes": 22257574,
      "path": "runs/pressure-methods-v4/diverse-training-view-01/dataset-07.npz",
      "sha256": "4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9",
      "slot": "view_07"
    },
    {
      "bytes": 22215890,
      "path": "runs/pressure-methods-v4/diverse-training-view-01/dataset-08.npz",
      "sha256": "47a46a0851690135e3f985a4ce5062dde945d7a05ebbe4cf7f0cd0e86cfff708",
      "slot": "view_08"
    },
    {
      "bytes": 7281689,
      "path": "runs/pressure-methods-v4/diverse-training-view-01/dataset-09.npz",
      "sha256": "8fff684d534c8d4846478ca3760f21745bfcae6e3e285d7d62a917345e38af53",
      "slot": "view_09"
    }
  ],
  "data_generation_source": {
    "manifest_slot": "splits",
    "path": "experiments/pressure-methods-v4/pack_expansion.py",
    "sha256": "3ef028b5e9dd45659e459f3429cf630aae2f4ea512dcbcba24a7f34e4e5c815b"
  }
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-c8896a4c9a730de5",
    "kind": "training_attempt",
    "name": "Pressure methods v4 · v4-diverse CUDA",
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
      "learned_coarse_revision": {
        "basis": "Orthonormal separable Neumann cosine modes 0..3, excluding DC",
        "bound": "max_axis_cells^2/2",
        "coefficients": "Positive learned sigmoid gains; no analytical inverse eigenvalues",
        "gate": "Accept learned coarse proposal only when its actual fine-equation residual norm does not increase",
        "initialization": "Gain 1 on 16^3; all original neural tensors preserved",
        "limitation": "Gate controls coarse proposal only; final neural pass and trajectory still require independent evaluation",
        "placement": "Learned coarse proposal before every fine neural pass",
        "trainable_parameters": 63
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 2097152
    },
    "receipt_sha256": "c8896a4c9a730de5fdc72cc7eff50b9234646d70cdafd7e99d177106f6d58f4f",
    "receipt_paths": [
      "runs/pressure-methods-v4/v4-diverse-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "d01b20aeb1784d51aeeb60051fe78cc9",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods v4 · v4-diverse CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2400,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "92da9085afd5168afb8d3500f346f5305e07224d6ae582a0bdcefdaf8772f132",
      "inputs": {
        "dataset": {
          "bytes": 21725324,
          "sha256": "67550093feb355665cb2a393b3c07358a11ddcbe18e8369227a4b337a4a769a1"
        },
        "dataset_extra": {
          "bytes": 25596700,
          "sha256": "40a32d21dabb649282007dda29c99fb0ba3006450fd1c28e6f28b77caef130ee"
        },
        "initial_weights": {
          "bytes": 1491574,
          "sha256": "42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4"
        },
        "splits": {
          "bytes": 72752,
          "sha256": "c9625fa89cab4eedd5cbdbcbc4ef7f895e4e7c37e381a2fe049576da494503b0"
        },
        "view_02": {
          "bytes": 22266499,
          "sha256": "e45cedbf79c56be57746693e44c9574f26d4b650fc597d3754aa9b7bd2a5367d"
        },
        "view_03": {
          "bytes": 23241588,
          "sha256": "7d787d92f596e703bcc054a37a03502dfce7673fd2fb74986cb80633cd39cf26"
        },
        "view_04": {
          "bytes": 22110479,
          "sha256": "e24733c8d071ce59e3c3fa00ccdd689116be581fc35c617327c2332d002215b1"
        },
        "view_05": {
          "bytes": 25958557,
          "sha256": "bde0a43b3eea813399d832d9978e110ab0f320df41deb179bc4bb4ebfd9b56a0"
        },
        "view_06": {
          "bytes": 21506413,
          "sha256": "fe1d383e77df98af6af8b8eb955f39fc5ea1dd116526de5c00f823a5924df6ae"
        },
        "view_07": {
          "bytes": 22257574,
          "sha256": "4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9"
        },
        "view_08": {
          "bytes": 22215890,
          "sha256": "47a46a0851690135e3f985a4ce5062dde945d7a05ebbe4cf7f0cd0e86cfff708"
        },
        "view_09": {
          "bytes": 7281689,
          "sha256": "8fff684d534c8d4846478ca3760f21745bfcae6e3e285d7d62a917345e38af53"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "51c444f634e10540abe8b6493b16ffadbb2c38b32d417b654ae16641db62d124",
      "source_witness_sha256": "a790c82583a674983e4fa558bc168589084761ba971a3cf066d979aeada4e52d"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4",
        "bytes": 1491574,
        "stage": "initial"
      },
      {
        "name": "checkpoint-2400.npz",
        "sha256": "20d4fe4562d06db4f7126844723a4a8e6c0d88078059ddf3cceeec025d6339a0",
        "bytes": 1491574,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "20d4fe4562d06db4f7126844723a4a8e6c0d88078059ddf3cceeec025d6339a0",
        "bytes": 1491574,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 4794449920,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/108"
    ],
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
      "receipt": "runs/pressure-methods-v4/v4-diverse-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v4/v4-diverse-study-01/train/receipt.json",
    "sha256": "c8896a4c9a730de5fdc72cc7eff50b9234646d70cdafd7e99d177106f6d58f4f",
    "bytes": 13947
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C110": {
        "name": "v4-diverse",
        "weights_sha256": "20d4fe4562d06db4f7126844723a4a8e6c0d88078059ddf3cceeec025d6339a0",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.0007981334169350716,
          "pressure_mean_free_relative_l2": 0.005205571012747355,
          "pressure_absolute_rms": 0.04629560973975524,
          "pressure_absolute_max": 0.09443127844786647
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0025",
            "native_frame": 63,
            "counts_xyz": [
              9,
              11,
              11
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0060",
            "native_frame": 63,
            "counts_xyz": [
              5,
              6,
              15
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.00041476544791668735,
              "pressure_mean_free_relative_l2": 0.005205571012747355,
              "pressure_absolute_rms": 0.04629560973975524,
              "pressure_absolute_max": 0.09443127844786647
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.0007981334169350716,
              "pressure_mean_free_relative_l2": 0.002879119139864127,
              "pressure_absolute_rms": 0.019449240932253323,
              "pressure_absolute_max": 0.05225442580288822
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C110/results.json",
          "sha256": "26d1caca057567d1c04e437c4ae77b98a5838133949bc1a860b1b5fea0b854b6"
        },
        "CNN_operation_AST_preserved": null,
        "CPU_neural_forward": false
      }
    },
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"v4-diverse","early_weight":0.05,"initial_weights_sha256":"42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"learned_coarse_revision":{"basis":"Orthonormal separable Neumann cosine modes 0..3, excluding DC","bound":"max_axis_cells^2/2","coefficients":"Positive learned sigmoid gains; no analytical inverse eigenvalues","gate":"Accept learned coarse proposal only when its actual fine-equation residual norm does not increase","initialization":"Gain 1 on 16^3; all original neural tensors preserved","limitation":"Gate controls coarse proposal only; final neural pass and trajectory still require independent evaluation","placement":"Learned coarse proposal before every fine neural pass","trainable_parameters":63},"max_cells":2097152,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Balanced spatial tiers; private RNG; fixed update budget.","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","shape_informed_coarse":{"added_parameters":1135,"features":"log(Nz/16), log(Ny/16), log(Nx/16)","fine_initialization":"Exact completed v2-headshape CPU restore; matched continuation control","head_learning_rate_multiplier":1000,"network":"3 -> 16 tanh -> 63 additive coarse logits; final layer initially zero","operator":"Existing sealed constant-density equation"},"supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[12,6,11],"frame":1,"native_frame":63,"parent":3,"split":"train","trajectory_id":"baseline--water-train-0003"},{"cell_size_m":0.0625,"counts_xyz":[13,11,8],"frame":0,"native_frame":15,"parent":7,"split":"train","trajectory_id":"baseline--water-train-0007"},{"cell_size_m":0.0625,"counts_xyz":[10,5,7],"frame":1,"native_frame":63,"parent":11,"split":"train","trajectory_id":"baseline--water-train-0011"},{"cell_size_m":0.125,"counts_xyz":[6,16,8],"frame":0,"native_frame":15,"parent":14,"split":"train","trajectory_id":"baseline--water-train-0014"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":1,"native_frame":63,"parent":18,"split":"train","trajectory_id":"large-train-64-cpm8"},{"cell_size_m":0.125,"counts_xyz":[61,39,27],"frame":0,"native_frame":15,"parent":22,"split":"train","trajectory_id":"v2-train-01-cpm8"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":1,"native_frame":15,"parent":26,"split":"train","trajectory_id":"v3-train-00-n64-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[128,128,128],"frame":0,"native_frame":0,"parent":29,"split":"train","trajectory_id":"v3-train-00-n128-cpm32"},{"cell_size_m":0.03125,"counts_xyz":[128,128,128],"frame":1,"native_frame":15,"parent":33,"split":"train","trajectory_id":"v3-train-01-n128-cpm32"},{"cell_size_m":0.125,"counts_xyz":[61,45,29],"frame":0,"native_frame":0,"parent":37,"split":"train","trajectory_id":"v4-diverse-train-03"},{"cell_size_m":0.125,"counts_xyz":[61,45,29],"frame":1,"native_frame":15,"parent":41,"split":"train","trajectory_id":"v4-diverse-train-07"}],"closure":{"experiment_sha256":"92da9085afd5168afb8d3500f346f5305e07224d6ae582a0bdcefdaf8772f132","inputs":{"dataset":{"bytes":21725324,"sha256":"67550093feb355665cb2a393b3c07358a11ddcbe18e8369227a4b337a4a769a1"},"dataset_extra":{"bytes":25596700,"sha256":"40a32d21dabb649282007dda29c99fb0ba3006450fd1c28e6f28b77caef130ee"},"initial_weights":{"bytes":1491574,"sha256":"42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4"},"splits":{"bytes":72752,"sha256":"c9625fa89cab4eedd5cbdbcbc4ef7f895e4e7c37e381a2fe049576da494503b0"},"view_02":{"bytes":22266499,"sha256":"e45cedbf79c56be57746693e44c9574f26d4b650fc597d3754aa9b7bd2a5367d"},"view_03":{"bytes":23241588,"sha256":"7d787d92f596e703bcc054a37a03502dfce7673fd2fb74986cb80633cd39cf26"},"view_04":{"bytes":22110479,"sha256":"e24733c8d071ce59e3c3fa00ccdd689116be581fc35c617327c2332d002215b1"},"view_05":{"bytes":25958557,"sha256":"bde0a43b3eea813399d832d9978e110ab0f320df41deb179bc4bb4ebfd9b56a0"},"view_06":{"bytes":21506413,"sha256":"fe1d383e77df98af6af8b8eb955f39fc5ea1dd116526de5c00f823a5924df6ae"},"view_07":{"bytes":22257574,"sha256":"4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9"},"view_08":{"bytes":22215890,"sha256":"47a46a0851690135e3f985a4ce5062dde945d7a05ebbe4cf7f0cd0e86cfff708"},"view_09":{"bytes":7281689,"sha256":"8fff684d534c8d4846478ca3760f21745bfcae6e3e285d7d62a917345e38af53"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"51c444f634e10540abe8b6493b16ffadbb2c38b32d417b654ae16641db62d124","source_witness_sha256":"a790c82583a674983e4fa558bc168589084761ba971a3cf066d979aeada4e52d"},"diagnostic_population":"train_only","elapsed_s":178.44668869499583,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.046274601890218e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":2596036608,"peak_reserved_bytes":3298820096,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":2400,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1491574,"role":"checkpoint","sha256":"42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4"},"checkpoint-2400.npz":{"bytes":1491574,"role":"checkpoint","sha256":"20d4fe4562d06db4f7126844723a4a8e6c0d88078059ddf3cceeec025d6339a0"},"history.json":{"bytes":16311,"role":"training_diagnostics","sha256":"d1dece1cd01c5b118377b4a34b81b1aecd5c1669df5f2682ff02748665eae762"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":1573526,"role":"predictions","sha256":"52ef4e0fa4ab5fa70f3c5c82d822eca2bc176502d0b990730d9858ab55f0af15"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":72752,"role":"training_splits","sha256":"c9625fa89cab4eedd5cbdbcbc4ef7f895e4e7c37e381a2fe049576da494503b0"},"summary.json":{"bytes":773,"role":"result_summary","sha256":"0840b926db56b38cb28b924c192976120fe5c18ca19e31554b4e371aa62b8d7a"},"weights.npz":{"bytes":1491574,"role":"checkpoint","sha256":"20d4fe4562d06db4f7126844723a4a8e6c0d88078059ddf3cceeec025d6339a0"}},"per_case_inference_s":[0.01662920499802567,0.006230315018910915,0.0065866179938893765,0.004496587993344292,0.005688769975677133,0.673526374011999,0.3336051480146125,0.7524217299942393,4.71456669899635,4.571286786987912,0.37127469800179824,0.3712377760093659],"prediction_export_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"ad120bdb6f16425cde8aecae7bf95a350797a2c55857cf5fbac70d35eee68f71","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"00e8a3a24ea58f84af19ddb5905a273f4ab2a7bfa83918d149a371ac47acb9ec","shape":[11,6,12]},"case_002":{"bytes":9152,"dtype":"<f8","sha256":"f02311a9c8729072a9d87b5c06756eede787b58542178120d18834bcd101c630","shape":[8,11,13]},"case_003":{"bytes":2800,"dtype":"<f8","sha256":"693a97035b76d100bf1878aa50a904973021e1e5f5c79404dcd6f087178c30f2","shape":[7,5,10]},"case_004":{"bytes":6144,"dtype":"<f8","sha256":"34115888b974bba6538e9ed6618a05e332a68f5e3c665b9fdcd2a4351735da42","shape":[8,16,6]},"case_005":{"bytes":262144,"dtype":"<f8","sha256":"57794eba9cb1f6ea238e09daf0d5a9cfd390c351863245cf510919a9bfd16e82","shape":[32,32,32]},"case_006":{"bytes":133920,"dtype":"<f8","sha256":"344224bca67f112f9c68e608e08e0a96fcdfb0cddf1a227e0d1a3614015d9141","shape":[27,20,31]},"case_007":{"bytes":262144,"dtype":"<f8","sha256":"ad6a46f78128cc4ea21fe69cb8dbe9e01965e005bc5eeca342b65957943ae739","shape":[32,32,32]},"case_008":{"bytes":262144,"dtype":"<f8","sha256":"65d9b8fa700769d8e36d8ae134b63a455aba887c15a5a79fb580eddc763e615a","shape":[32,32,32]},"case_009":{"bytes":262144,"dtype":"<f8","sha256":"e04443aa7c1b07125ede77769ba7cdc0169a7415183686cae4aa6955cea5ee1d","shape":[32,32,32]},"case_010":{"bytes":165416,"dtype":"<f8","sha256":"470216f1104be46a4109c7d6d9e1409ca10e8ad5515122b8bf4e993cdccab370","shape":[29,23,31]},"case_011":{"bytes":165416,"dtype":"<f8","sha256":"5ab43a049655664d38979aa35985db32aeb8ee1a70588fefd2f9815e4788eef9","shape":[29,23,31]}},"prediction_export_policy":"NPZ samples at most32cells/axis; full prediction_inventory hashes every full-resolution value and is checked exactly on fresh CPU restore","prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"ad120bdb6f16425cde8aecae7bf95a350797a2c55857cf5fbac70d35eee68f71","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"00e8a3a24ea58f84af19ddb5905a273f4ab2a7bfa83918d149a371ac47acb9ec","shape":[11,6,12]},"case_002":{"bytes":9152,"dtype":"<f8","sha256":"f02311a9c8729072a9d87b5c06756eede787b58542178120d18834bcd101c630","shape":[8,11,13]},"case_003":{"bytes":2800,"dtype":"<f8","sha256":"693a97035b76d100bf1878aa50a904973021e1e5f5c79404dcd6f087178c30f2","shape":[7,5,10]},"case_004":{"bytes":6144,"dtype":"<f8","sha256":"34115888b974bba6538e9ed6618a05e332a68f5e3c665b9fdcd2a4351735da42","shape":[8,16,6]},"case_005":{"bytes":2097152,"dtype":"<f8","sha256":"7bb4c85a09e45ce172dd2800a0262fedf27f89aeaee37373e7fd80baafd6f794","shape":[64,64,64]},"case_006":{"bytes":513864,"dtype":"<f8","sha256":"e253092496acb99bcdce5248e1f37a63a2b8e608b0ae25033c6b8682e2dc0af3","shape":[27,39,61]},"case_007":{"bytes":2097152,"dtype":"<f8","sha256":"3a2b6795b9eec88b6e6e341d6a72f565564e57cc8921c11a82a4f71c09723d1b","shape":[64,64,64]},"case_008":{"bytes":16777216,"dtype":"<f8","sha256":"a7b0f0fc0fa811847cbda2ced07d80be3f45b4a83544dcfdb1dfecb1449c936e","shape":[128,128,128]},"case_009":{"bytes":16777216,"dtype":"<f8","sha256":"fac450b73621c37b85a79a877c4519aa678b34d39882125ebc24a561d619bb9a","shape":[128,128,128]},"case_010":{"bytes":636840,"dtype":"<f8","sha256":"f346dc5e07ef174b374284911253c61bfaf7d69f072cdbab3a92d98d0186ba73","shape":[29,45,61]},"case_011":{"bytes":636840,"dtype":"<f8","sha256":"c188bdaa3f34b0aeef7365903e5427f4ac5e72dc3dd73232f435a7fb3ed04a8d","shape":[29,45,61]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":4794449920,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
