# Pressure methods v3 · v3-c44half CUDA

Retrospective source review · fit-047a8434eca89fea

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 92,560-parameter F32 linear bias-free multiscale fine CNN, widths 4/8/12/16 and bottleneck 24, follows a numerical Galerkin coarse solve at each of two internal pressure corrections. Clamped trilinear interpolation at up to six aggregate centers per axis defines P; the F64 coarse system is PᵀLP with a fixed gauge and Cholesky solve. Restriction, solve, prolongation, residual action and neural inference all belong to the pressure call. This is a hybrid pressure request, not a full fluid simulation step.

**Purpose.** Change the half-width numerical-coarse warm start from C37 to C44 while retaining C102 geometry, data, sampling and optimization budget.

**Experiment.** 2,400 CUDA Adam updates, batch one, clip norm one. Fine rate 1e-05; executed optimizer multipliers [1.0]. Every update uses one recorded physical request: 1,600 preserved small-grid samples plus 800 round-robin size-tier samples. There are 34 train-only parents and 68 selected frames, fixed dt=1/60 s and fluid density 1,000 kg/m³. Actual parent grid resolutions: {'8.0': 15, '16.0': 17, '32.0': 2}; sample tier exposures: {'16': 1600, '32': 267, '64': 267, '128': 266}. Actual fine-network work is 4,800 root forwards / 1,224,395,120 input voxels. Inherited manufactured, direction and current-policy banks are disabled despite retained config labels. Final/intermediate physical equation feedback remains, with early log-residual weight 0.05 and target 1e-6. The old eight-pass teacher has weight zero. 

**Observed outcome.** Final retained weighted training loss 24.7279623636636. On the common 128 original consumed requests, worst mean-free pressure relative L2 is 0.012007447374872218, equation relative L2 0.061543352130061915. Retained complete GPU pressure-call median of case medians is 6.1375039999999998 ms, with CPU input/output, validation/packing, transfers, all solver work and synchronization included. Setup and scoring are separate. The native training diagnostic named validation is train-only, and the wrapper probe is already consumed. The common test covers small 8/16-cells-per-metre grids, not the larger training domains. No fresh forward, latency, trajectory or held-out result is added by this review.

**Lesson.** This is the same reduced architecture and numerical stage as C102 with a different named-checkpoint initialization. Common tails improve modestly over C102, but no seed replication or independent large-grid qualification follows.

**Reproduction.** Use the exact captured native evidence recipe, all witnessed sources/configuration, runtime, fixed normalization, typed input packet, initial parameter inventory and final-update checkpoint. Preserve the private small/large RNG streams and physical exposure budget, disabled auxiliary banks and executed optimizer groups. Retain channel-slice mappings or exact warm-parent bytes, and distinguish teacher work from student work. Original CPU publication/restore performs neural inference and must be replaced by a qualified GPU-only route before replay. Native recipe rebuilding, clean retraining, new-Rheon regeneration and own-history drift remain unqualified; historical Trial ancestry is unchanged.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v3/v3-c44half-study-07/train/receipt.json",
    "sha256": "047a8434eca89fea925edefb2729add16cfc26b5e68fd45667c19a969d1bac2b"
  },
  "history": {
    "path": "runs/pressure-methods-v3/v3-c44half-study-07/train/history.json",
    "sha256": "9cb74516daa616d25090150594659fc6dafb91fd8532a8ed5646335858559711"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/channel_slice.py",
      "sha256": "0e4369f63c476a35ead196f7f5633a04a4d8ca4d7f417464b8d8de55f7a693c3"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/coarse.py",
      "sha256": "e133b2ced49768da22b86a39dd36239d7cb812a170b865d47dffd4ec7b90fd91"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/config.json",
      "sha256": "035f5649b12097da8edfb312113d6c4f3cebaff533c95d1662854629f1711bb2"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/data.py",
      "sha256": "09231fe0119461205b74578dab9ce77cbcbea7c537900941432e97f52e51f16d"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/fine-tuning.json",
      "sha256": "0de9b630aa8eb523d0b3c5458919b06765ad3cc9e5239ede202fbaaa69c7ff0e"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/frozen_gn.py",
      "sha256": "ce280f6ce86aeccc31521b13e4df039780fda8c112c63616457771377903e08d"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/hybrid-method.json",
      "sha256": "ab70994aa4938351d480f0d11e6d3734cf16b9e4f75ccbcfc30d9c56abe91f89"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/methods_view.py",
      "sha256": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/mixed_order.py",
      "sha256": "93d987b0f10101fb5b380e7310aa77eb6f5c7557198f6d43d90f102393e47b04"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/model.py",
      "sha256": "2bfc5ab54020b4b06b5fc9d5305b1c94486b3c2d2cd164bca49ae0e9882f2d8e"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/native_reconstruct.py",
      "sha256": "d2db7ae0a145e82560ede85518a542a2fc83f7e7c6de33697823457b96d74445"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/original_train.py",
      "sha256": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/physical_residual.py",
      "sha256": "8327721a87e6bbfc1cf0e5b92d253f4c6044a82d11adcaa19b0df0b12ff3e139"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/policy.py",
      "sha256": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/portable_policy.py",
      "sha256": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/reference_spectral.py",
      "sha256": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/train.py",
      "sha256": "138f79a1275392dfc305eeae2cb536b613c7cd32a89fdcc8bcc3b6534532df40"
    },
    {
      "path": "runs/pressure-methods-v3/v3-c44half-project-07/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v3/v3-c44half-project-07/source-witness.json",
    "sha256": "cf4a2b5e64791dc9e425486917538a699f546306d6a8158217c49f4b6ffdd2b8"
  },
  "training_population": {
    "path": "runs/pressure-methods-v3/v3-c44half-study-07/train/splits.json",
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
      "path": "runs/pressure-methods-v3/C44-half-initialization-01/weights.npz",
      "sha256": "f024065e99b1fa7cb98e44e947c77381d44fdd3506a75d01b15b51d6cea8d70b",
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
    "item_id": "fit-047a8434eca89fea",
    "kind": "training_attempt",
    "name": "Pressure methods v3 · v3-c44half CUDA",
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
    "receipt_sha256": "047a8434eca89fea925edefb2729add16cfc26b5e68fd45667c19a969d1bac2b",
    "receipt_paths": [
      "runs/pressure-methods-v3/v3-c44half-study-07/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "0c034d9a8e2e421fbb15309e809c41ff",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods v3 · v3-c44half CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2400,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "035f5649b12097da8edfb312113d6c4f3cebaff533c95d1662854629f1711bb2",
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
          "sha256": "f024065e99b1fa7cb98e44e947c77381d44fdd3506a75d01b15b51d6cea8d70b"
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
      "source_witness_sha256": "cf4a2b5e64791dc9e425486917538a699f546306d6a8158217c49f4b6ffdd2b8"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "f024065e99b1fa7cb98e44e947c77381d44fdd3506a75d01b15b51d6cea8d70b",
        "bytes": 375628,
        "stage": "initial"
      },
      {
        "name": "checkpoint-2400.npz",
        "sha256": "dd4f4a3e36c0131ea0408d56a964c3108c76a2bd3623032fcb0eb0f051445029",
        "bytes": 375628,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "dd4f4a3e36c0131ea0408d56a964c3108c76a2bd3623032fcb0eb0f051445029",
        "bytes": 375628,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 4185882624,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/101"
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
      "receipt": "runs/pressure-methods-v3/v3-c44half-study-07/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v3/v3-c44half-study-07/train/receipt.json",
    "sha256": "047a8434eca89fea925edefb2729add16cfc26b5e68fd45667c19a969d1bac2b",
    "bytes": 13369
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C103": {
        "name": "v3-c44half",
        "weights_sha256": "dd4f4a3e36c0131ea0408d56a964c3108c76a2bd3623032fcb0eb0f051445029",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.061543352130061915,
          "pressure_mean_free_relative_l2": 0.012007447374872218,
          "pressure_absolute_rms": 0.25828889289569296,
          "pressure_absolute_max": 1.4107489149125978
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
              "common_equation_relative_l2": 0.05889865679474413,
              "pressure_mean_free_relative_l2": 0.01152685276674378,
              "pressure_absolute_rms": 0.25828889289569296,
              "pressure_absolute_max": 1.4107489149125978
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.061543352130061915,
              "pressure_mean_free_relative_l2": 0.012007447374872218,
              "pressure_absolute_rms": 0.07894474679922225,
              "pressure_absolute_max": 0.37753723836843184
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C103/results.json",
          "sha256": "4ecc420d425398dc7dd530ef217982d15b0d2fa7705ebb872439f1b538166f17"
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
{"architecture":{"activation":"linear","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":24,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[16,12,8,4],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[4,8,12,16],"fine_tuning":{"arm":"v3-c44half","early_weight":0.05,"initial_weights_sha256":"f024065e99b1fa7cb98e44e947c77381d44fdd3506a75d01b15b51d6cea8d70b","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"Exact deterministic channel-sliced named checkpoint; all retained channels trainable","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":2097152,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Balanced spatial tiers; private RNG; fixed update budget.","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"numerical_training_method":{"coarse_nodes_per_axis":6,"cost":"Count geometry setup, restriction/transform, solve/selected inverse, prolongation, residual and neural inference","kind":"numerical-coarse","numerical_operator":"Same sealed fine equation; no reference labels","operator":"Galerkin Ac=P^T L P, float64 Cholesky with fixed coarse gauge","placement":"Numerical stage before each neural pass","transfer":"Clamped trilinear interpolation at coarse aggregate centers; conservative transpose restriction"},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[12,6,11],"frame":1,"native_frame":63,"parent":3,"split":"train","trajectory_id":"baseline--water-train-0003"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":0,"native_frame":15,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":0,"native_frame":15,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":15,"parent":18,"split":"train","trajectory_id":"large-train-64-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[31,23,17],"frame":1,"native_frame":63,"parent":21,"split":"train","trajectory_id":"v2-train-00-cpm16"},{"cell_size_m":0.125,"counts_xyz":[48,13,9],"frame":0,"native_frame":15,"parent":24,"split":"train","trajectory_id":"v2-train-02-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[64,64,64],"frame":1,"native_frame":15,"parent":27,"split":"train","trajectory_id":"v3-train-00-n64-cpm16"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":0,"parent":30,"split":"train","trajectory_id":"v3-train-01-n64-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[128,128,128],"frame":1,"native_frame":15,"parent":33,"split":"train","trajectory_id":"v3-train-01-n128-cpm32"}],"closure":{"experiment_sha256":"035f5649b12097da8edfb312113d6c4f3cebaff533c95d1662854629f1711bb2","inputs":{"dataset":{"bytes":22711895,"sha256":"20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5"},"dataset_extra":{"bytes":25135473,"sha256":"2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368"},"initial_weights":{"bytes":375628,"sha256":"f024065e99b1fa7cb98e44e947c77381d44fdd3506a75d01b15b51d6cea8d70b"},"splits":{"bytes":57361,"sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"view_02":{"bytes":22277708,"sha256":"85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb"},"view_03":{"bytes":23650806,"sha256":"98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8"},"view_04":{"bytes":22239100,"sha256":"1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc"},"view_05":{"bytes":24113006,"sha256":"6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda"},"view_06":{"bytes":19897861,"sha256":"8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080"},"view_07":{"bytes":22257574,"sha256":"4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9"},"view_08":{"bytes":18319876,"sha256":"2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"157178a36ed5bebdd0971c3643a6720aa377962175614cdde7115ddef5cfbf56","source_witness_sha256":"cf4a2b5e64791dc9e425486917538a699f546306d6a8158217c49f4b6ffdd2b8"},"diagnostic_population":"train_only","elapsed_s":145.59592099301517,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.6491819293594787e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":1766802944,"peak_reserved_bytes":2013265920,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":2400,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":375628,"role":"checkpoint","sha256":"f024065e99b1fa7cb98e44e947c77381d44fdd3506a75d01b15b51d6cea8d70b"},"checkpoint-2400.npz":{"bytes":375628,"role":"checkpoint","sha256":"dd4f4a3e36c0131ea0408d56a964c3108c76a2bd3623032fcb0eb0f051445029"},"history.json":{"bytes":14408,"role":"training_diagnostics","sha256":"9cb74516daa616d25090150594659fc6dafb91fd8532a8ed5646335858559711"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":1238286,"role":"predictions","sha256":"36c2cb58142321331e7ae3c82b0952a23a99dd9c4ecc190e46a16886437a1b2f"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":57361,"role":"training_splits","sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"summary.json":{"bytes":681,"role":"result_summary","sha256":"e899a1f7e36ac773ea01b99479d226c03bfdda14608fabd81b10ffa7908c7550"},"weights.npz":{"bytes":375628,"role":"checkpoint","sha256":"dd4f4a3e36c0131ea0408d56a964c3108c76a2bd3623032fcb0eb0f051445029"}},"per_case_inference_s":[0.008676945988554507,0.00434948701877147,0.003679666988318786,0.004365450004115701,0.004660647013224661,0.004848857002798468,0.5450872099900153,0.021540499001275748,0.009760502987774089,0.5504937139921822,0.5825276169925928,2.925444757012883],"prediction_export_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"4079d810a54ed06bb1f5d4c776431607802f461d6bff74902863c44276f99589","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"9e79ae53b624e966236d00715ea127fbd8703d3b3411fb43363f6691704ec07a","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"558d2a41c5e44c9141c331d2f520220a810653c182a3d21f3b1d06e2bcb89a40","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"d9e3d88cbff27d4286a867685f7fac0c06e0135bf59d3f8bc405bd1a47d1c0e3","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"368e3103711d833424466028a1abe545bbb509efc97d71091e25c40f4c25078f","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"3f79ef98a4c757166a8ab5b42b39713deaf501ae04733d463297cfb3155aad8a","shape":[7,15,14]},"case_006":{"bytes":262144,"dtype":"<f8","sha256":"324a7d87e272580066ca07659abb0f6f86f14425301d33983fba26ceb50a9100","shape":[32,32,32]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"47f785831bcaf5d170a8d218f314967b359600cf53a8bee97bc19418f2399118","shape":[17,23,31]},"case_008":{"bytes":22464,"dtype":"<f8","sha256":"34c281ca10c0b1f0393e8d22fbaa17e77556e3963f42b3d9247af2eebf3cb3cd","shape":[9,13,24]},"case_009":{"bytes":262144,"dtype":"<f8","sha256":"601e10cd9c81f8db6534937dabcc9df2925515b02d3477cae77c471418a30adf","shape":[32,32,32]},"case_010":{"bytes":262144,"dtype":"<f8","sha256":"fc6c069a27d3f2971af0c721d26923adeedcdff514c20856a6455e6c353944e0","shape":[32,32,32]},"case_011":{"bytes":262144,"dtype":"<f8","sha256":"498d17a2ae557b7d187c04bb305678efa1a13ef3b033c4dba166d66e0962713e","shape":[32,32,32]}},"prediction_export_policy":"NPZ samples at most32cells/axis; full prediction_inventory hashes every full-resolution value and is checked exactly on fresh CPU restore","prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"4079d810a54ed06bb1f5d4c776431607802f461d6bff74902863c44276f99589","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"9e79ae53b624e966236d00715ea127fbd8703d3b3411fb43363f6691704ec07a","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"558d2a41c5e44c9141c331d2f520220a810653c182a3d21f3b1d06e2bcb89a40","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"d9e3d88cbff27d4286a867685f7fac0c06e0135bf59d3f8bc405bd1a47d1c0e3","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"368e3103711d833424466028a1abe545bbb509efc97d71091e25c40f4c25078f","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"3f79ef98a4c757166a8ab5b42b39713deaf501ae04733d463297cfb3155aad8a","shape":[7,15,14]},"case_006":{"bytes":2097152,"dtype":"<f8","sha256":"4fc22a2054d26792d35188d44f2896c5d2ee1aaba4d4a11ba7deb180481942f8","shape":[64,64,64]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"47f785831bcaf5d170a8d218f314967b359600cf53a8bee97bc19418f2399118","shape":[17,23,31]},"case_008":{"bytes":44928,"dtype":"<f8","sha256":"884d1f24a72b906e43d457c9dc2ab620f798b553a3727b719c7d688c59ed98ee","shape":[9,13,48]},"case_009":{"bytes":2097152,"dtype":"<f8","sha256":"69aad2c7a4b2123f6781799b24a07e288c89ba56d9ca65dfc80b1a3c629e9cb9","shape":[64,64,64]},"case_010":{"bytes":2097152,"dtype":"<f8","sha256":"3d4a49bd9701053b4813d8fa1deb24239d4b37f4073e68eef4e44467f1c57ad0","shape":[64,64,64]},"case_011":{"bytes":16777216,"dtype":"<f8","sha256":"3883d7e4118dbc30ccfb3e25e8082986b3f49daffc805b9ab4578ca6f248f232","shape":[128,128,128]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":4185882624,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
