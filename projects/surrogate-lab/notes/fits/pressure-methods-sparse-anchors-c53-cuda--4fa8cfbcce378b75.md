# Pressure methods · sparse-anchors · C53 CUDA

Retrospective source review · fit-4fa8cfbcce378b75

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A hybrid pressure request with at most four anchor cells per axis. It computes a full global cosine transform and all-mode eigenvalue division, evaluates exact pressure at selected anchors, then interpolates and applies the 370,016-parameter learned fine correction while pinning its change to zero at anchors. The anchors are not independent local voxel solves; their global computation must count. F64 physical residual scaling q=(rho/h)r, gauging and pressure accumulation surround the F32 learned stages. The result is pressure only; two internal corrections do not produce two physical simulation states.

**Purpose.** Test numerical pressure anchors plus learned infill while preserving exact anchor values.

**Experiment.** 1,600 CUDA Adam updates, batch one, initial fine rate 1e-05, schedule [{'update': 1201, 'learning_rate': 1e-06}, {'update': 1501, 'learning_rate': 1e-07}], model seed 2026101803, index seed 2026101804. 20 physical training trajectories/40 requests; grid-density counts {'8.0': 10, '16.0': 10}. Actual tier exposure {'32': 534, '64': 533, '16': 533} and sampled voxel count 157,852,608; 3,200 fine-network root forwards. Executed added coarse/head learning-rate multiplier 1. All inherited manufactured/direction/current-policy banks are disabled; C53 fine initialization and fixed scales are retained. Physical dt remains 1/60 s and fluid density 1000 kg/m3.

**Observed outcome.** Retained final fitting loss 16.128998989017585. Common 128 consumed requests: worst mean-free pressure relative L2 0.0028033986828929153, equation relative L2 0.015850705750819708, median case-median complete GPU pressure call 6.5794192499999999 ms. The captured default call includes its declared numerical/learned stages, CPU input/output, transfers and synchronization. Cached geometry is warmed; setup/first calls/scoring are separate. These are pressure-request observations, not full-step latency, independent holdout evidence or own-history drift. No new model execution is added.

**Lesson.** Sparse returned anchors do not mean sparse global work. The full transform/division/selected inverse and infill belong to the solver. Accuracy at anchors does not guarantee fine-grid accuracy or long rollout stability.

**Reproduction.** Retain the exact typed evidence recipe, witnessed model/preprocessing/coarse sources, method specifications, optimizer group policy, runtime, physical views, fixed scales and transformed C53 initialization. Reproduce the exact private tier/replay stream, including preserved small draws and all large-grid voxel work. Numerical stages require their interpolation, gauge and cached-geometry policy; learned stages require gains, gating and any shape-head initialization. Historical publication/restore has CPU neural calls: qualify GPU-only equivalents before rerunning. Clean fitting, new-Rheon generation and long own-history drift remain unqualified; preserve original receipts and historical ancestry.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v1/sparse-anchors-study-01/train/receipt.json",
    "sha256": "4fa8cfbcce378b75a609facbbd505d3b5195b264a81cbc2b5138610039951892"
  },
  "history": {
    "path": "runs/pressure-methods-v1/sparse-anchors-study-01/train/history.json",
    "sha256": "89e86a21a0acd13ae6a5041b57707fb90653bebb33b89b71ed2e3b0e9e994c5b"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/coarse.py",
      "sha256": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/config.json",
      "sha256": "60a0d0c8050b39ceb04b24ae3041fbd2c6e9a5063a17172f05ba931c066c573c"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/data.py",
      "sha256": "cf01d003cdf525e78cb3ec2b5e004186fa28c65ecadd63ffea969d85009674f9"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/fine-tuning.json",
      "sha256": "cd8f93fb98b44884b2287abc17d792baa554db427c68b9fadb3275c3832062f1"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/hybrid-method.json",
      "sha256": "94f9e66f0028c1fa553a66952b6ace36a6f5b2bf049fed95d03d1802e147d015"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/methods_view.py",
      "sha256": "3418389fb087ae3036549b49338bc7377291d497fa8636c0978f2f382131294f"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/mixed_order.py",
      "sha256": "3df84663563940223c95d0367b10c08b8cd53b36961bbd31656f39399d612f3c"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/model.py",
      "sha256": "eb889562bf4bb8b41d1183582515aaa45a8fb439190f8d9ab617d36a05b79072"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/physical_residual.py",
      "sha256": "8327721a87e6bbfc1cf0e5b92d253f4c6044a82d11adcaa19b0df0b12ff3e139"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/policy.py",
      "sha256": "83d2f46900b6a8018a02006520c33c6b2b8e2ed5e46dd2fc92337a570b6e36b4"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/portable_policy.py",
      "sha256": "b890bf302e911c024ce54082c06043700f940d2fd812ffdcdc0fbb3d785a2d39"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/train.py",
      "sha256": "5e02e78d3eca1cb7f9eeb04ed0372dc71c8f11da5f26b7cadf629510a929b81b"
    },
    {
      "path": "runs/pressure-methods-v1/sparse-anchors-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v1/sparse-anchors-project-01/source-witness.json",
    "sha256": "a22d7d6d073bfa6fee86be8b11bcfd4cb7f14baada576226e6c0b579f7d98879"
  },
  "training_population": {
    "path": "runs/pressure-methods-v1/sparse-anchors-study-01/train/splits.json",
    "sha256": "39f803cf02f6dab67af9189298bb151a235fb9b51172a542208fefc97bf128d1"
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
      "bytes": 19487086,
      "path": "runs/pressure-methods-v1/domain-training-view-01/dataset-00.npz",
      "sha256": "c12c4d88d0a35979fd5fd1045a295239b2fac0bfae0bbf95410439f0331e9971",
      "slot": "dataset"
    },
    {
      "bytes": 19488006,
      "path": "runs/pressure-methods-v1/domain-training-view-01/dataset-01.npz",
      "sha256": "f2dbef26b98a6d2d044626ebca0248efd251970342b1abece67a081aa5b15ef5",
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
      "bytes": 33281,
      "path": "runs/pressure-methods-v1/domain-training-view-01/manifest.json",
      "sha256": "39f803cf02f6dab67af9189298bb151a235fb9b51172a542208fefc97bf128d1",
      "slot": "splits"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-4fa8cfbcce378b75",
    "kind": "training_attempt",
    "name": "Pressure methods · sparse-anchors · C53 CUDA",
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
    "receipt_sha256": "4fa8cfbcce378b75a609facbbd505d3b5195b264a81cbc2b5138610039951892",
    "receipt_paths": [
      "runs/pressure-methods-v1/sparse-anchors-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "8006717afe604050b58c9049f7270d9e",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods · sparse-anchors · C53 CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1600,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "60a0d0c8050b39ceb04b24ae3041fbd2c6e9a5063a17172f05ba931c066c573c",
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
          "bytes": 19487086,
          "sha256": "c12c4d88d0a35979fd5fd1045a295239b2fac0bfae0bbf95410439f0331e9971"
        },
        "dataset_extra": {
          "bytes": 19488006,
          "sha256": "f2dbef26b98a6d2d044626ebca0248efd251970342b1abece67a081aa5b15ef5"
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
          "bytes": 33281,
          "sha256": "39f803cf02f6dab67af9189298bb151a235fb9b51172a542208fefc97bf128d1"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "8b70793af829b8a0ee81380fc59aa19901da271e9de5da1c1954a575e87de9f1",
      "source_witness_sha256": "a22d7d6d073bfa6fee86be8b11bcfd4cb7f14baada576226e6c0b579f7d98879"
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
        "sha256": "949f4fefe168b0ec61f063bdb921284e260cb714d746f297be60b6f5ef548839",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "949f4fefe168b0ec61f063bdb921284e260cb714d746f297be60b6f5ef548839",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2838102016,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/82"
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
      "receipt": "runs/pressure-methods-v1/sparse-anchors-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v1/sparse-anchors-study-01/train/receipt.json",
    "sha256": "4fa8cfbcce378b75a609facbbd505d3b5195b264a81cbc2b5138610039951892",
    "bytes": 11358
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C84": {
        "name": "sparse-anchors",
        "weights_sha256": "949f4fefe168b0ec61f063bdb921284e260cb714d746f297be60b6f5ef548839",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.015850705750819708,
          "pressure_mean_free_relative_l2": 0.0028033986828929153,
          "pressure_absolute_rms": 0.25036708458322127,
          "pressure_absolute_max": 0.6404937316870873
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0028",
            "native_frame": 63,
            "counts_xyz": [
              13,
              10,
              11
            ],
            "cell_size_m": 0.125
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0054",
            "native_frame": 63,
            "counts_xyz": [
              15,
              12,
              5
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.015850705750819708,
              "pressure_mean_free_relative_l2": 0.0028033986828929153,
              "pressure_absolute_rms": 0.25036708458322127,
              "pressure_absolute_max": 0.6404937316870873
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.01479870291025593,
              "pressure_mean_free_relative_l2": 0.001514562067191928,
              "pressure_absolute_rms": 0.047938302863812884,
              "pressure_absolute_max": 0.09058800185399841
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C84/results.json",
          "sha256": "0497fa1f2527d00adb491a45c39078f2dbba20d01b31a393faab53ef2168856f"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"sparse-anchors","early_weight":0.05,"initial_weights_sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":262144,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Same expanded-domain tier sampler and fixed optimizer-update budget","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"numerical_training_method":{"anchors":"At most 4 selected fine cells per axis, including domain endpoints","coarse_nodes_per_axis":4,"computation":"Full global DCT and all-mode division, selected inverse only; no local-solve speed claim","constraint":"Fine neural correction is zero at every exact anchor","cost":"Count geometry setup, restriction/transform, solve/selected inverse, prolongation, residual and neural inference","kind":"sparse-anchors","numerical_operator":"Same sealed fine equation; no reference labels","placement":"Numerical stage before each neural pass"},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.0625,"counts_xyz":[12,6,11],"frame":0,"native_frame":15,"parent":3,"split":"train","trajectory_id":"baseline--water-train-0003"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":1,"native_frame":63,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":0,"native_frame":15,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":1,"native_frame":63,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.0625,"counts_xyz":[32,32,32],"frame":0,"native_frame":15,"parent":17,"split":"train","trajectory_id":"large-train-32-cpm16"},{"cell_size_m":0.0625,"counts_xyz":[64,64,64],"frame":1,"native_frame":63,"parent":19,"split":"train","trajectory_id":"large-train-64-cpm16"}],"closure":{"experiment_sha256":"60a0d0c8050b39ceb04b24ae3041fbd2c6e9a5063a17172f05ba931c066c573c","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":19487086,"sha256":"c12c4d88d0a35979fd5fd1045a295239b2fac0bfae0bbf95410439f0331e9971"},"dataset_extra":{"bytes":19488006,"sha256":"f2dbef26b98a6d2d044626ebca0248efd251970342b1abece67a081aa5b15ef5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"splits":{"bytes":33281,"sha256":"39f803cf02f6dab67af9189298bb151a235fb9b51172a542208fefc97bf128d1"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"8b70793af829b8a0ee81380fc59aa19901da271e9de5da1c1954a575e87de9f1","source_witness_sha256":"a22d7d6d073bfa6fee86be8b11bcfd4cb7f14baada576226e6c0b579f7d98879"},"diagnostic_population":"train_only","elapsed_s":78.83491740899626,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.0445928609215116e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":359975936,"peak_reserved_bytes":434110464,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":1600,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"checkpoint-1600.npz":{"bytes":1485452,"role":"checkpoint","sha256":"949f4fefe168b0ec61f063bdb921284e260cb714d746f297be60b6f5ef548839"},"history.json":{"bytes":10534,"role":"training_diagnostics","sha256":"89e86a21a0acd13ae6a5041b57707fb90653bebb33b89b71ed2e3b0e9e994c5b"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":2472878,"role":"predictions","sha256":"f9cfa3392547329fede3f58a91c612ade0dde65505d375fb77aeacbbb95ea970"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":33281,"role":"training_splits","sha256":"39f803cf02f6dab67af9189298bb151a235fb9b51172a542208fefc97bf128d1"},"summary.json":{"bytes":521,"role":"result_summary","sha256":"8b1fb6961f564f831f8ff97b81832235a852f53f9bb48dd480d1480562ff3ae9"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"949f4fefe168b0ec61f063bdb921284e260cb714d746f297be60b6f5ef548839"}},"per_case_inference_s":[0.03443894599331543,0.008609479002188891,0.008817291993182153,0.009128036996116862,0.007228589995065704,0.008787317987298593,0.01776680297916755,0.010522744996706024,0.012338572007138282,0.011409953993279487,0.15683699498185888,0.9431611459876876],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"4a20fa1cc2f9a120aeb045b5068bbb09100a03dad20c552bd8a7100e07df02ce","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"9b89da777d4b47c4f1c23085d30a7d9448430410ff746f27915328b3204aeb18","shape":[6,10,16]},"case_002":{"bytes":6336,"dtype":"<f8","sha256":"2f9b0431c80d724d59480f60099975825b30fc89f176ea65de2a377383f3ba43","shape":[11,6,12]},"case_003":{"bytes":5280,"dtype":"<f8","sha256":"5518f5add35d56fa42d5a365b54683d3cef5cf2dcfc9e2f3402413e6c5725d3c","shape":[6,10,11]},"case_004":{"bytes":2880,"dtype":"<f8","sha256":"04e0eeb5f8200b618965c4c23c4c1456eaecf433099d8549f72121b91febaf48","shape":[9,5,8]},"case_005":{"bytes":6912,"dtype":"<f8","sha256":"8c732d157ff6a3393412a38efde9847f77da7524fbf33694af910fde4be346cd","shape":[9,8,12]},"case_006":{"bytes":20480,"dtype":"<f8","sha256":"0eb3561860c1ed9d9321a3f29e214375d356c8d3676f0bfafdefc59f4572aac3","shape":[16,16,10]},"case_007":{"bytes":7488,"dtype":"<f8","sha256":"088d23ad44036ba8f714365b7b960f12c094d8674c28c98f5078acb929e10b1e","shape":[12,13,6]},"case_008":{"bytes":9000,"dtype":"<f8","sha256":"df2f7ff064f7650b9561396e41de751222e86ce70b7497d4045edc63e65c23d5","shape":[15,15,5]},"case_009":{"bytes":11760,"dtype":"<f8","sha256":"e7042d612e604ce37c0eb616a45fdd79ee324960b7e07cf89fa0ac13b2f862d5","shape":[7,15,14]},"case_010":{"bytes":262144,"dtype":"<f8","sha256":"b08e3a864c98b1fafa3a246d5316040543e4c90a58193ab81a3112395d8946e5","shape":[32,32,32]},"case_011":{"bytes":2097152,"dtype":"<f8","sha256":"46ddb6ea53d31d5debf497c263cb00b4430dcea4554014c252ff61322eab09d4","shape":[64,64,64]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2838102016,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
