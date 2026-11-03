# Pressure methods v3 · v3-c37half CUDA

Retrospective source review · fit-12380a9f8e9b0ffe

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 92,560-parameter F32 linear bias-free multiscale fine CNN, widths 4/8/12/16 and bottleneck 24, follows a numerical Galerkin coarse solve at each of two internal pressure corrections. Clamped trilinear interpolation at up to six aggregate centers per axis defines P; the F64 coarse system is PᵀLP with a fixed gauge and Cholesky solve. Restriction, solve, prolongation, residual action and neural inference all belong to the pressure call. This is a hybrid pressure request, not a full fluid simulation step.

**Purpose.** Train a genuinely half-width fine CNN sliced from C37, coupled to a numerical six-node-per-axis Galerkin stage, on the expanded physical replay packet.

**Experiment.** 2,400 CUDA Adam updates, batch one, clip norm one. Fine rate 1e-05; executed optimizer multipliers [1.0]. Every update uses one recorded physical request: 1,600 preserved small-grid samples plus 800 round-robin size-tier samples. There are 34 train-only parents and 68 selected frames, fixed dt=1/60 s and fluid density 1,000 kg/m³. Actual parent grid resolutions: {'8.0': 15, '16.0': 17, '32.0': 2}; sample tier exposures: {'16': 1600, '32': 267, '64': 267, '128': 266}. Actual fine-network work is 4,800 root forwards / 1,224,395,120 input voxels. Inherited manufactured, direction and current-policy banks are disabled despite retained config labels. Final/intermediate physical equation feedback remains, with early log-residual weight 0.05 and target 1e-6. The old eight-pass teacher has weight zero. 

**Observed outcome.** Final retained weighted training loss 25.367542666494106. On the common 128 original consumed requests, worst mean-free pressure relative L2 is 0.013365002281655054, equation relative L2 0.07103538667860862. Retained complete GPU pressure-call median of case medians is 6.1863037500000004 ms, with CPU input/output, validation/packing, transfers, all solver work and synchronization included. Setup and scoring are separate. The native training diagnostic named validation is train-only, and the wrapper probe is already consumed. The common test covers small 8/16-cells-per-metre grids, not the larger training domains. No fresh forward, latency, trajectory or held-out result is added by this review.

**Lesson.** A smaller neural parameter count does not imply lower complete GPU latency: numerical coarse solves and field movement remain. The archived common pressure/equation errors are larger than those of the full-width learned-global replay models.

**Reproduction.** Use the exact captured native evidence recipe, all witnessed sources/configuration, runtime, fixed normalization, typed input packet, initial parameter inventory and final-update checkpoint. Preserve the private small/large RNG streams and physical exposure budget, disabled auxiliary banks and executed optimizer groups. Retain channel-slice mappings or exact warm-parent bytes, and distinguish teacher work from student work. Original CPU publication/restore performs neural inference and must be replaced by a qualified GPU-only route before replay. Native recipe rebuilding, clean retraining, new-Rheon regeneration and own-history drift remain unqualified; historical Trial ancestry is unchanged.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v3/v3-c37half-study-07/train/receipt.json",
    "sha256": "12380a9f8e9b0ffe07586e603febc5763c9ae5ab5fd0b1f82794a1c92ef951ea"
  },
  "history": {
    "path": "runs/pressure-methods-v3/v3-c37half-study-07/train/history.json",
    "sha256": "74e7f5e04e65bc8040b5b18e5fc5bf1cb8d6605f2db3afef4fb80d308f9be1f9"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/channel_slice.py",
      "sha256": "0e4369f63c476a35ead196f7f5633a04a4d8ca4d7f417464b8d8de55f7a693c3"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/coarse.py",
      "sha256": "e133b2ced49768da22b86a39dd36239d7cb812a170b865d47dffd4ec7b90fd91"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/config.json",
      "sha256": "cba2804ad71da96eb1aa5f2d1160d19f85ad567d3d6b763242b289ae0891fe23"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/data.py",
      "sha256": "09231fe0119461205b74578dab9ce77cbcbea7c537900941432e97f52e51f16d"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/fine-tuning.json",
      "sha256": "187359af63a4f259c5a6983b6d0851e96d0d44fe259d8396bbbe9c3a1da0ff38"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/frozen_gn.py",
      "sha256": "ce280f6ce86aeccc31521b13e4df039780fda8c112c63616457771377903e08d"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/hybrid-method.json",
      "sha256": "ab70994aa4938351d480f0d11e6d3734cf16b9e4f75ccbcfc30d9c56abe91f89"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/methods_view.py",
      "sha256": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/mixed_order.py",
      "sha256": "93d987b0f10101fb5b380e7310aa77eb6f5c7557198f6d43d90f102393e47b04"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/model.py",
      "sha256": "2bfc5ab54020b4b06b5fc9d5305b1c94486b3c2d2cd164bca49ae0e9882f2d8e"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/native_reconstruct.py",
      "sha256": "d2db7ae0a145e82560ede85518a542a2fc83f7e7c6de33697823457b96d74445"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/original_train.py",
      "sha256": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/physical_residual.py",
      "sha256": "8327721a87e6bbfc1cf0e5b92d253f4c6044a82d11adcaa19b0df0b12ff3e139"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/policy.py",
      "sha256": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/portable_policy.py",
      "sha256": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/reference_spectral.py",
      "sha256": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/train.py",
      "sha256": "138f79a1275392dfc305eeae2cb536b613c7cd32a89fdcc8bcc3b6534532df40"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c37half-project-07/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v3/v3-c37half-project-07/source-witness.json",
    "sha256": "35687b637ad0e129da89cefb40eb4cecb5b1d9468550f8c9a960cfb2fa2a2177"
  },
  "training_population": {
    "path": "runs/pressure-methods-v3/v3-c37half-study-07/train/splits.json",
    "sha256": "8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"
  },
  "training_inputs": [
    {
      "bytes": 22711895,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-00.npz",
      "sha256": "20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5",
      "slot": "dataset"
    },
    {
      "bytes": 25135473,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-01.npz",
      "sha256": "2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368",
      "slot": "dataset_extra"
    },
    {
      "bytes": 375628,
      "path": "runs/pressure-methods-v3/C37-half-initialization-01/weights.npz",
      "sha256": "42670d7e427d6b958be4bcf888b6e09003135331f0d3051c77b7b334a05b4f76",
      "slot": "initial_weights"
    },
    {
      "bytes": 57361,
      "path": "runs/pressure-methods-v3/derived-training-view-01/manifest.json",
      "sha256": "8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169",
      "slot": "splits"
    },
    {
      "bytes": 22277708,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-02.npz",
      "sha256": "85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb",
      "slot": "view_02"
    },
    {
      "bytes": 23650806,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-03.npz",
      "sha256": "98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8",
      "slot": "view_03"
    },
    {
      "bytes": 22239100,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-04.npz",
      "sha256": "1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc",
      "slot": "view_04"
    },
    {
      "bytes": 24113006,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-05.npz",
      "sha256": "6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda",
      "slot": "view_05"
    },
    {
      "bytes": 19897861,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-06.npz",
      "sha256": "8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080",
      "slot": "view_06"
    },
    {
      "bytes": 22257574,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-07.npz",
      "sha256": "4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9",
      "slot": "view_07"
    },
    {
      "bytes": 18319876,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-08.npz",
      "sha256": "2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30",
      "slot": "view_08"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-12380a9f8e9b0ffe",
    "kind": "training_attempt",
    "name": "Pressure methods v3 · v3-c37half CUDA",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-physical-residual-pressure-v1",
      "input_channels": 1,
      "encoder_widths": [
        4,
        8,
        12,
        16
      ],
      "bottleneck_width": 24,
      "decoder_widths": [
        16,
        12,
        8,
        4
      ],
      "kernel": [
        3,
        3,
        3
      ],
      "activation": "linear",
      "dtype": "float32",
      "pressure_dtype": "float64",
      "correction": {
        "activation": "linear",
        "steps": 2
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 2097152
    },
    "receipt_sha256": "12380a9f8e9b0ffe07586e603febc5763c9ae5ab5fd0b1f82794a1c92ef951ea",
    "receipt_paths": [
      "runs/pressure-methods-v3/v3-c37half-study-07/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "ac109922546e472484b1ec862121cfa7",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods v3 · v3-c37half CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2400,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "cba2804ad71da96eb1aa5f2d1160d19f85ad567d3d6b763242b289ae0891fe23",
      "inputs": {
        "dataset": {
          "bytes": 22711895,
          "sha256": "20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5"
        },
        "dataset_extra": {
          "bytes": 25135473,
          "sha256": "2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368"
        },
        "initial_weights": {
          "bytes": 375628,
          "sha256": "42670d7e427d6b958be4bcf888b6e09003135331f0d3051c77b7b334a05b4f76"
        },
        "splits": {
          "bytes": 57361,
          "sha256": "8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"
        },
        "view_02": {
          "bytes": 22277708,
          "sha256": "85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb"
        },
        "view_03": {
          "bytes": 23650806,
          "sha256": "98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8"
        },
        "view_04": {
          "bytes": 22239100,
          "sha256": "1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc"
        },
        "view_05": {
          "bytes": 24113006,
          "sha256": "6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda"
        },
        "view_06": {
          "bytes": 19897861,
          "sha256": "8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080"
        },
        "view_07": {
          "bytes": 22257574,
          "sha256": "4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9"
        },
        "view_08": {
          "bytes": 18319876,
          "sha256": "2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "157178a36ed5bebdd0971c3643a6720aa377962175614cdde7115ddef5cfbf56",
      "source_witness_sha256": "35687b637ad0e129da89cefb40eb4cecb5b1d9468550f8c9a960cfb2fa2a2177"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "42670d7e427d6b958be4bcf888b6e09003135331f0d3051c77b7b334a05b4f76",
        "bytes": 375628,
        "stage": "initial"
      },
      {
        "name": "checkpoint-2400.npz",
        "sha256": "27af00158cc8c8be28bef22666fb4ee00e1c789ebda4e64bf6b03c504625593c",
        "bytes": 375628,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "27af00158cc8c8be28bef22666fb4ee00e1c789ebda4e64bf6b03c504625593c",
        "bytes": 375628,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 4156370944,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/100"
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
      "receipt": "runs/pressure-methods-v3/v3-c37half-study-07/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v3/v3-c37half-study-07/train/receipt.json",
    "sha256": "12380a9f8e9b0ffe07586e603febc5763c9ae5ab5fd0b1f82794a1c92ef951ea",
    "bytes": 13371
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C102": {
        "name": "v3-c37half",
        "weights_sha256": "27af00158cc8c8be28bef22666fb4ee00e1c789ebda4e64bf6b03c504625593c",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.07103538667860862,
          "pressure_mean_free_relative_l2": 0.013365002281655054,
          "pressure_absolute_rms": 0.29430564573762685,
          "pressure_absolute_max": 2.8072680649006623
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0031",
            "native_frame": 15,
            "counts_xyz": [
              14,
              6,
              16
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0055",
            "native_frame": 15,
            "counts_xyz": [
              11,
              16,
              5
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.06070074224322485,
              "pressure_mean_free_relative_l2": 0.013080189068292863,
              "pressure_absolute_rms": 0.29430564573762685,
              "pressure_absolute_max": 2.8072680649006623
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.07103538667860862,
              "pressure_mean_free_relative_l2": 0.013365002281655054,
              "pressure_absolute_rms": 0.04904147870510078,
              "pressure_absolute_max": 0.38781958109708325
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C102/results.json",
          "sha256": "069555e108a3bc663430aec929533bcd44f19aafe9f33b8c9d5176216bcf8256"
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
{"architecture":{"activation":"linear","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":24,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[16,12,8,4],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[4,8,12,16],"fine_tuning":{"arm":"v3-c37half","early_weight":0.05,"initial_weights_sha256":"42670d7e427d6b958be4bcf888b6e09003135331f0d3051c77b7b334a05b4f76","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"Exact deterministic channel-sliced named checkpoint; all retained channels trainable","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":2097152,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Balanced spatial tiers; private RNG; fixed update budget.","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"numerical_training_method":{"coarse_nodes_per_axis":6,"cost":"Count geometry setup, restriction/transform, solve/selected inverse, prolongation, residual and neural inference","kind":"numerical-coarse","numerical_operator":"Same sealed fine equation; no reference labels","operator":"Galerkin Ac=P^T L P, float64 Cholesky with fixed coarse gauge","placement":"Numerical stage before each neural pass","transfer":"Clamped trilinear interpolation at coarse aggregate centers; conservative transpose restriction"},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[12,6,11],"frame":1,"native_frame":63,"parent":3,"split":"train","trajectory_id":"baseline--water-train-0003"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":0,"native_frame":15,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":0,"native_frame":15,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":15,"parent":18,"split":"train","trajectory_id":"large-train-64-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[31,23,17],"frame":1,"native_frame":63,"parent":21,"split":"train","trajectory_id":"v2-train-00-cpm16"},{"cell_size_m":0.125,"counts_xyz":[48,13,9],"frame":0,"native_frame":15,"parent":24,"split":"train","trajectory_id":"v2-train-02-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[64,64,64],"frame":1,"native_frame":15,"parent":27,"split":"train","trajectory_id":"v3-train-00-n64-cpm16"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":0,"parent":30,"split":"train","trajectory_id":"v3-train-01-n64-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[128,128,128],"frame":1,"native_frame":15,"parent":33,"split":"train","trajectory_id":"v3-train-01-n128-cpm32"}],"closure":{"experiment_sha256":"cba2804ad71da96eb1aa5f2d1160d19f85ad567d3d6b763242b289ae0891fe23","inputs":{"dataset":{"bytes":22711895,"sha256":"20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5"},"dataset_extra":{"bytes":25135473,"sha256":"2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368"},"initial_weights":{"bytes":375628,"sha256":"42670d7e427d6b958be4bcf888b6e09003135331f0d3051c77b7b334a05b4f76"},"splits":{"bytes":57361,"sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"view_02":{"bytes":22277708,"sha256":"85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb"},"view_03":{"bytes":23650806,"sha256":"98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8"},"view_04":{"bytes":22239100,"sha256":"1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc"},"view_05":{"bytes":24113006,"sha256":"6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda"},"view_06":{"bytes":19897861,"sha256":"8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080"},"view_07":{"bytes":22257574,"sha256":"4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9"},"view_08":{"bytes":18319876,"sha256":"2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"157178a36ed5bebdd0971c3643a6720aa377962175614cdde7115ddef5cfbf56","source_witness_sha256":"35687b637ad0e129da89cefb40eb4cecb5b1d9468550f8c9a960cfb2fa2a2177"},"diagnostic_population":"train_only","elapsed_s":143.78916081701755,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.1077372832346555e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":1766802944,"peak_reserved_bytes":2013265920,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":2400,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":375628,"role":"checkpoint","sha256":"42670d7e427d6b958be4bcf888b6e09003135331f0d3051c77b7b334a05b4f76"},"checkpoint-2400.npz":{"bytes":375628,"role":"checkpoint","sha256":"27af00158cc8c8be28bef22666fb4ee00e1c789ebda4e64bf6b03c504625593c"},"history.json":{"bytes":14401,"role":"training_diagnostics","sha256":"74e7f5e04e65bc8040b5b18e5fc5bf1cb8d6605f2db3afef4fb80d308f9be1f9"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":1238286,"role":"predictions","sha256":"badd8b4bad4efa788b7f2f690bbbc42ab6778818c4d023e3405443158cc0720a"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":57361,"role":"training_splits","sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"summary.json":{"bytes":681,"role":"result_summary","sha256":"306f98e2234ab7a34c89beedafe8d32fbe53227829eaf41521396003fa9fda3f"},"weights.npz":{"bytes":375628,"role":"checkpoint","sha256":"27af00158cc8c8be28bef22666fb4ee00e1c789ebda4e64bf6b03c504625593c"}},"per_case_inference_s":[0.009004166000522673,0.0044962510000914335,0.0038449510175269097,0.004500462993746623,0.004770035011461005,0.004952563991537318,0.5817049180041067,0.019368921988643706,0.00927340000635013,0.558925346005708,0.5506741589924786,2.8983822200098075],"prediction_export_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"f2a1aa8c52c6a1d62fd1040fe0f24951092f88fb64faa10cc52616ed44d86277","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"d01884605ce4ea5768f09f46d3b05ced191d75e36621ff416f2cfd0520bcf1d2","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"2da59e14c43baa69d218ebff888d0309247960062fe420dabd10003c282871c0","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"4031f71f7408c6f967cd199e68dcddf7301fdf54358bd0f68019835f3e4ec51e","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"d829711b3e88efc339482042d830469682e25320dff537f22388c287fba1d723","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"d92a95b5ee3548d03b7340e0983d97eef6eedc1221fe5f185494c09bf79e721a","shape":[7,15,14]},"case_006":{"bytes":262144,"dtype":"<f8","sha256":"d4e0948a6e75dc4cf6933ed80b34648b05d8aa99a951deed2e713f4a7b9d76db","shape":[32,32,32]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"dbb784de1429ee27025855de160b69acbc8f043d862a624c146e8af007eb6a8c","shape":[17,23,31]},"case_008":{"bytes":22464,"dtype":"<f8","sha256":"d1f6a8cfe6c82b427f0707c71d68ae85ec256da832b9d295427b54a8a9e5a4af","shape":[9,13,24]},"case_009":{"bytes":262144,"dtype":"<f8","sha256":"c8db920d3a2375250e57377a5a869e58edc250244d588ffef901c509c715f1ac","shape":[32,32,32]},"case_010":{"bytes":262144,"dtype":"<f8","sha256":"0bc70237a0eaf38864d0c2036b3230fcda61ecab1d46cc6442334f500a1b401a","shape":[32,32,32]},"case_011":{"bytes":262144,"dtype":"<f8","sha256":"30f0f6ca63e57df9656430f0e96084ee679e10c476a22120aadeb36f86ded09f","shape":[32,32,32]}},"prediction_export_policy":"NPZ samples at most32cells/axis; full prediction_inventory hashes every full-resolution value and is checked exactly on fresh CPU restore","prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"f2a1aa8c52c6a1d62fd1040fe0f24951092f88fb64faa10cc52616ed44d86277","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"d01884605ce4ea5768f09f46d3b05ced191d75e36621ff416f2cfd0520bcf1d2","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"2da59e14c43baa69d218ebff888d0309247960062fe420dabd10003c282871c0","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"4031f71f7408c6f967cd199e68dcddf7301fdf54358bd0f68019835f3e4ec51e","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"d829711b3e88efc339482042d830469682e25320dff537f22388c287fba1d723","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"d92a95b5ee3548d03b7340e0983d97eef6eedc1221fe5f185494c09bf79e721a","shape":[7,15,14]},"case_006":{"bytes":2097152,"dtype":"<f8","sha256":"c64591770c87d48e1b2d771bdf085c40452988909679fc7df3519f99714bdcd5","shape":[64,64,64]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"dbb784de1429ee27025855de160b69acbc8f043d862a624c146e8af007eb6a8c","shape":[17,23,31]},"case_008":{"bytes":44928,"dtype":"<f8","sha256":"7905ccf9cf83b5bb67da7b3001e9fb83700a65027c942d8f6bbaa432730db935","shape":[9,13,48]},"case_009":{"bytes":2097152,"dtype":"<f8","sha256":"9253ad8915fbb6b90951214f109209aa1e29a7b00efc24bb01fc70423f0a2a3f","shape":[64,64,64]},"case_010":{"bytes":2097152,"dtype":"<f8","sha256":"3c22bb6302c89d3dbeda106061deb02793f1cf5a75ec6fe063f8c12498785944","shape":[64,64,64]},"case_011":{"bytes":16777216,"dtype":"<f8","sha256":"39c7fe60d083dc3f5f60314b888307b3996ca65aa016dcc74590a6c730398034","shape":[128,128,128]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":4156370944,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
