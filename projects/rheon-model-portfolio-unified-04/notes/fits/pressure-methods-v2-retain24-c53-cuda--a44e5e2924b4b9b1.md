# Pressure methods · v2-retain24 · C53 CUDA

Retrospective source review · fit-a44e5e2924b4b9b1

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** The retained 370,016-parameter linear fine pressure CNN, with no numerical or learned coarse branch. Shared two-correction fitting/deployment retains original C53 scales and tensor values at initialization. F64 physical residual scaling q=(rho/h)r, gauging and pressure accumulation surround the F32 learned stages. The result is pressure only; two internal corrections do not produce two physical simulation states.

**Purpose.** Retain the full 1,600-example small stream while adding 800 large-grid examples, for 2,400 updates.

**Experiment.** 2,400 CUDA Adam updates, batch one, initial fine rate 1e-05, schedule [{'update': 1801, 'learning_rate': 1e-06}, {'update': 2251, 'learning_rate': 1e-07}], model seed 2026101820, index seed 2026101804. 26 physical training trajectories/52 requests; grid-density counts {'8.0': 13, '16.0': 13}. Actual tier exposure {'16': 1600, '32': 400, '64': 400} and sampled voxel count 58,565,381; 4,800 fine-network root forwards. Executed added coarse/head learning-rate multiplier 1. All inherited manufactured/direction/current-policy banks are disabled; C53 fine initialization and fixed scales are retained. Physical dt remains 1/60 s and fluid density 1000 kg/m3.

**Observed outcome.** Retained final fitting loss 17.198005842201063. Common 128 consumed requests: worst mean-free pressure relative L2 0.0074792795770482565, equation relative L2 0.004229958520469982, median case-median complete GPU pressure call 4.3554532500000001 ms. The captured default call includes its declared numerical/learned stages, CPU input/output, transfers and synchronization. Cached geometry is warmed; setup/first calls/scoring are separate. These are pressure-request observations, not full-step latency, independent holdout evidence or own-history drift. No new model execution is added.

**Lesson.** Distinguish fixed-update expansion from retained-small-exposure expansion. C90 reduces small exposure to 800 while C91 retains 1,600. Larger voxel work and additional optimizer updates are not free improvements in inference efficiency.

**Reproduction.** Retain the exact typed evidence recipe, witnessed model/preprocessing/coarse sources, method specifications, optimizer group policy, runtime, physical views, fixed scales and transformed C53 initialization. Reproduce the exact private tier/replay stream, including preserved small draws and all large-grid voxel work. Numerical stages require their interpolation, gauge and cached-geometry policy; learned stages require gains, gating and any shape-head initialization. Historical publication/restore has CPU neural calls: qualify GPU-only equivalents before rerunning. Clean fitting, new-Rheon generation and long own-history drift remain unqualified; preserve original receipts and historical ancestry.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v2/v2-retain24-study-02/train/receipt.json",
    "sha256": "a44e5e2924b4b9b1a6e24811510cb4c9d67b25924560be52d0bae8279b0e5321"
  },
  "history": {
    "path": "runs/pressure-methods-v2/v2-retain24-study-02/train/history.json",
    "sha256": "379a3d6411aab44f0ba5bd7153352823d4d40515f67f1a2a38a28078d9932edc"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/config.json",
      "sha256": "fb49eec97fb81cc0b86e8d10a41927f3aa6ffdcb2ad59dd39b0941e2d1edac95"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/data.py",
      "sha256": "cf01d003cdf525e78cb3ec2b5e004186fa28c65ecadd63ffea969d85009674f9"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/fine-tuning.json",
      "sha256": "a46e8020b92d4e7835e64a53c9ca484f21f2e70d0259c786003752281a024379"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/methods_view.py",
      "sha256": "3418389fb087ae3036549b49338bc7377291d497fa8636c0978f2f382131294f"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/mixed_order.py",
      "sha256": "fdbbfa8d48001aa8fb6e8d530c0bceca3de6d99caa44b945af77a81ba5dd6de7"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/policy.py",
      "sha256": "83d2f46900b6a8018a02006520c33c6b2b8e2ed5e46dd2fc92337a570b6e36b4"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/portable_policy.py",
      "sha256": "b890bf302e911c024ce54082c06043700f940d2fd812ffdcdc0fbb3d785a2d39"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/train.py",
      "sha256": "5e02e78d3eca1cb7f9eeb04ed0372dc71c8f11da5f26b7cadf629510a929b81b"
    },
    {
      "path": "runs/pressure-methods-v2/v2-retain24-project-02/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v2/v2-retain24-project-02/source-witness.json",
    "sha256": "2ada3b473af0906526f8d0464b5c17e5dfacfb76a6251d4fcdec5e544abdf994"
  },
  "training_population": {
    "path": "runs/pressure-methods-v2/v2-retain24-study-02/train/splits.json",
    "sha256": "31e0d1c2a777b773eb64826681ad1256013ad2fdfa4435a8185f6ef6aed09bf6"
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
      "bytes": 24736178,
      "path": "runs/pressure-methods-v2/shape-training-view-01/dataset-00.npz",
      "sha256": "8b7386e3831f0bc7cf9154cbb4303902347c26bc3e74b5308ebed9de0618fdd7",
      "slot": "dataset"
    },
    {
      "bytes": 24737374,
      "path": "runs/pressure-methods-v2/shape-training-view-01/dataset-01.npz",
      "sha256": "7e0c4d5e8b3fac166303456b30dfda37461be16b4208ab4f8f0c4e6d836898bb",
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
      "bytes": 42433,
      "path": "runs/pressure-methods-v2/shape-training-view-01/manifest.json",
      "sha256": "31e0d1c2a777b773eb64826681ad1256013ad2fdfa4435a8185f6ef6aed09bf6",
      "slot": "splits"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-a44e5e2924b4b9b1",
    "kind": "training_attempt",
    "name": "Pressure methods · v2-retain24 · C53 CUDA",
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
    "receipt_sha256": "a44e5e2924b4b9b1a6e24811510cb4c9d67b25924560be52d0bae8279b0e5321",
    "receipt_paths": [
      "runs/pressure-methods-v2/v2-retain24-study-02/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "824547fb6bbc4d1abfc1629bfafaa092",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods · v2-retain24 · C53 CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2400,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "fb49eec97fb81cc0b86e8d10a41927f3aa6ffdcb2ad59dd39b0941e2d1edac95",
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
          "bytes": 24736178,
          "sha256": "8b7386e3831f0bc7cf9154cbb4303902347c26bc3e74b5308ebed9de0618fdd7"
        },
        "dataset_extra": {
          "bytes": 24737374,
          "sha256": "7e0c4d5e8b3fac166303456b30dfda37461be16b4208ab4f8f0c4e6d836898bb"
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
          "bytes": 42433,
          "sha256": "31e0d1c2a777b773eb64826681ad1256013ad2fdfa4435a8185f6ef6aed09bf6"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "1e3104b3a0a6cab7ee8bd878534fc5f7ac9123e3a941988c39a340a283cf7f06",
      "source_witness_sha256": "2ada3b473af0906526f8d0464b5c17e5dfacfb76a6251d4fcdec5e544abdf994"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-2400.npz",
        "sha256": "785def495aa458199ea6435269d4ca2cf3a94fcbdc632ec0223f7f59c6b133a2",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "785def495aa458199ea6435269d4ca2cf3a94fcbdc632ec0223f7f59c6b133a2",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2642690048,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/89"
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
      "receipt": "runs/pressure-methods-v2/v2-retain24-study-02/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v2/v2-retain24-study-02/train/receipt.json",
    "sha256": "a44e5e2924b4b9b1a6e24811510cb4c9d67b25924560be52d0bae8279b0e5321",
    "bytes": 10742
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C91": {
        "name": "v2-retain24",
        "weights_sha256": "785def495aa458199ea6435269d4ca2cf3a94fcbdc632ec0223f7f59c6b133a2",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.004229958520469982,
          "pressure_mean_free_relative_l2": 0.0074792795770482565,
          "pressure_absolute_rms": 0.25945321587467396,
          "pressure_absolute_max": 0.5206813483388621
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0025",
            "native_frame": 15,
            "counts_xyz": [
              9,
              11,
              11
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0032",
            "native_frame": 63,
            "counts_xyz": [
              15,
              6,
              8
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.003301018945571475,
              "pressure_mean_free_relative_l2": 0.0074792795770482565,
              "pressure_absolute_rms": 0.25945321587467396,
              "pressure_absolute_max": 0.5206813483388621
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.004229958520469982,
              "pressure_mean_free_relative_l2": 0.006131981225467506,
              "pressure_absolute_rms": 0.1254197085709302,
              "pressure_absolute_max": 0.2815983116521361
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C91/results.json",
          "sha256": "b11c7cd88405f089d5721e3d04f6d1ac726ee3ed53a8a4c01df7980ea654c622"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"v2-retain24","early_weight":0.05,"initial_weights_sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":262144,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Balanced spatial tiers; private RNG; fixed update budget.","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":1,"native_frame":63,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":0,"native_frame":15,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":0,"native_frame":15,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.0625,"counts_xyz":[10,5,7],"frame":1,"native_frame":63,"parent":11,"split":"train","trajectory_id":"baseline--water-train-0011"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":15,"parent":18,"split":"train","trajectory_id":"large-train-64-cpm8"},{"cell_size_m":0.125,"counts_xyz":[31,23,17],"frame":1,"native_frame":63,"parent":20,"split":"train","trajectory_id":"v2-train-00-cpm8"},{"cell_size_m":0.125,"counts_xyz":[61,39,27],"frame":0,"native_frame":15,"parent":22,"split":"train","trajectory_id":"v2-train-01-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[48,13,9],"frame":1,"native_frame":63,"parent":25,"split":"train","trajectory_id":"v2-train-02-cpm16"}],"closure":{"experiment_sha256":"fb49eec97fb81cc0b86e8d10a41927f3aa6ffdcb2ad59dd39b0941e2d1edac95","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":24736178,"sha256":"8b7386e3831f0bc7cf9154cbb4303902347c26bc3e74b5308ebed9de0618fdd7"},"dataset_extra":{"bytes":24737374,"sha256":"7e0c4d5e8b3fac166303456b30dfda37461be16b4208ab4f8f0c4e6d836898bb"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"splits":{"bytes":42433,"sha256":"31e0d1c2a777b773eb64826681ad1256013ad2fdfa4435a8185f6ef6aed09bf6"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"1e3104b3a0a6cab7ee8bd878534fc5f7ac9123e3a941988c39a340a283cf7f06","source_witness_sha256":"2ada3b473af0906526f8d0464b5c17e5dfacfb76a6251d4fcdec5e544abdf994"},"diagnostic_population":"train_only","elapsed_s":74.93763023600332,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":3.4819762007032296e-06,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":291296768,"peak_reserved_bytes":379584512,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":2400,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"checkpoint-2400.npz":{"bytes":1485452,"role":"checkpoint","sha256":"785def495aa458199ea6435269d4ca2cf3a94fcbdc632ec0223f7f59c6b133a2"},"history.json":{"bytes":12276,"role":"training_diagnostics","sha256":"379a3d6411aab44f0ba5bd7153352823d4d40515f67f1a2a38a28078d9932edc"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":2837102,"role":"predictions","sha256":"990af7e873a133879fe936de3c26a1301870719adbcf138090acf81cf228ea22"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":42433,"role":"training_splits","sha256":"31e0d1c2a777b773eb64826681ad1256013ad2fdfa4435a8185f6ef6aed09bf6"},"summary.json":{"bytes":582,"role":"result_summary","sha256":"1db6854eebf91c78f5817de3d25c15baa6e062f422f4b122c64ac0dfe2a9d0ca"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"785def495aa458199ea6435269d4ca2cf3a94fcbdc632ec0223f7f59c6b133a2"}},"per_case_inference_s":[0.027353201003279537,0.007190239994088188,0.005510289978701621,0.004514489992288873,0.005923396005528048,0.0038316900026984513,0.008017327985726297,0.007220544997835532,0.7475975359848235,0.0467265640036203,0.31802129198331386,0.020552855014102533],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"b89af2439dfe430277ca881455cebc9eab7c6fab36f701ad4d581e277ad8b5dd","shape":[16,16,16]},"case_001":{"bytes":8736,"dtype":"<f8","sha256":"8f2140a358fa32126bd88ef5dada65e85f21aea18b0aff7c4f6246e5469fc097","shape":[13,7,12]},"case_002":{"bytes":7200,"dtype":"<f8","sha256":"13798527ebdbac6013926ecbc0db313f5ee34b4842532df7804ecf6f2cd3b629","shape":[6,15,10]},"case_003":{"bytes":2880,"dtype":"<f8","sha256":"d8107e01e420cdef8206b3f7967e03534fc1eb9960fd288a40226791acbd6d44","shape":[9,5,8]},"case_004":{"bytes":6048,"dtype":"<f8","sha256":"95c7e3e5bd2d6c5aaa1094562cc24497a92c2a44c3fa8305ef58024a70dba7fb","shape":[9,14,6]},"case_005":{"bytes":2800,"dtype":"<f8","sha256":"aa32136b812a53d50eb86b58c16c73ac23113342a68d217a9495f206598e94c4","shape":[7,5,10]},"case_006":{"bytes":9000,"dtype":"<f8","sha256":"0959bd05c61f37e5316dac0c79734b79eabfb1a5d833592de6e26a28270d2156","shape":[15,15,5]},"case_007":{"bytes":11760,"dtype":"<f8","sha256":"cb870c556abc06267c0e6186ad2bc91e4db7bf61e5f1e36fc4d84b0d8a10f068","shape":[7,15,14]},"case_008":{"bytes":2097152,"dtype":"<f8","sha256":"1d62f7eb63be14217a20522969c8320e8030271d16cfeed28e86adf3813189f8","shape":[64,64,64]},"case_009":{"bytes":96968,"dtype":"<f8","sha256":"ec5b90b256c6d35cafc0533382347cdaedd5846db1172458870540e44080841f","shape":[17,23,31]},"case_010":{"bytes":513864,"dtype":"<f8","sha256":"97b86a696688314d01775949fe45f81e98b5d6337781a8f77f43265401ff3858","shape":[27,39,61]},"case_011":{"bytes":44928,"dtype":"<f8","sha256":"1553724c10aac13c7fcd872a3a12805aa3e182b875971ff7e2bf02d0f60c75e9","shape":[9,13,48]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2642690048,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
