# Pressure methods · spacing-zero-feature · C53 CUDA

Retrospective source review · fit-642c8584c484925e

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** The retained fine CNN gains a second input column that is identically zero. Both matched architectures have 370,232 learned coefficients; the new 8-by-1-by-3-by-3-by-3 convolution column begins at exact zero. All original C53 tensor values are retained. Spacing means voxel size, not fluid mass density. F64 physical residual scaling q=(rho/h)r, gauging and pressure accumulation surround the F32 learned stages. The result is pressure only; two internal corrections do not produce two physical simulation states.

**Purpose.** Compare explicit spacing conditioning against an identical two-channel zero-feature control on the same native spacing-varied data.

**Experiment.** 1,600 CUDA Adam updates, batch one, initial fine rate 1e-05, schedule [{'update': 1201, 'learning_rate': 1e-06}, {'update': 1501, 'learning_rate': 1e-07}], model seed 2026101803, index seed 2026101804. 64 physical training trajectories/128 requests; grid-density counts {'4.0': 16, '8.0': 16, '16.0': 16, '32.0': 16}. Actual tier exposure {'16': 1600} and sampled voxel count 1,823,416; 3,200 fine-network root forwards. Executed added coarse/head learning-rate multiplier 1. All inherited manufactured/direction/current-policy banks are disabled; C53 fine initialization and fixed scales are retained. Physical dt remains 1/60 s and fluid density 1000 kg/m3.

**Observed outcome.** Retained final fitting loss 15.07487912255287. Common 128 consumed requests: worst mean-free pressure relative L2 0.0068284570893911902, equation relative L2 0.0049618307371428957, median case-median complete GPU pressure call 4.4517242499999998 ms. The captured default call includes its declared numerical/learned stages, CPU input/output, transfers and synchronization. Cached geometry is warmed; setup/first calls/scoring are separate. These are pressure-request observations, not full-step latency, independent holdout evidence or own-history drift. No new model execution is added.

**Lesson.** Physical q scaling already uses h; this ablation tests an additional learned spacing interaction. Equal parameter count and matched zero initialization are essential. The extra field remains homogeneous in residual amplitude and does not add an independently simulated scene.

**Reproduction.** Retain the exact typed evidence recipe, witnessed model/preprocessing/coarse sources, method specifications, optimizer group policy, runtime, physical views, fixed scales and transformed C53 initialization. Reproduce the exact private tier/replay stream, including preserved small draws and all large-grid voxel work. Numerical stages require their interpolation, gauge and cached-geometry policy; learned stages require gains, gating and any shape-head initialization. Historical publication/restore has CPU neural calls: qualify GPU-only equivalents before rerunning. Clean fitting, new-Rheon generation and long own-history drift remain unqualified; preserve original receipts and historical ancestry.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v1/spacing-zero-feature-study-01/train/receipt.json",
    "sha256": "642c8584c484925e944e76ca85489b4d7da9c0cff263bca41b566456c1fc7884"
  },
  "history": {
    "path": "runs/pressure-methods-v1/spacing-zero-feature-study-01/train/history.json",
    "sha256": "02fcf4a1c97d1bda5df503805ff72d8d6d4d1d3015ef7feee4373737418c4c28"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/config.json",
      "sha256": "699d4bd3d91cf634d3fd7a591f0abd5d850f2aa8c9e74ed20589804afa9cdef6"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/data.py",
      "sha256": "cf01d003cdf525e78cb3ec2b5e004186fa28c65ecadd63ffea969d85009674f9"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/fine-tuning.json",
      "sha256": "e3cf1b8ded466226efb36f62569f219258f55869760ebd9aa9a4980ecd845ae3"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/methods_view.py",
      "sha256": "3418389fb087ae3036549b49338bc7377291d497fa8636c0978f2f382131294f"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/mixed_order.py",
      "sha256": "3df84663563940223c95d0367b10c08b8cd53b36961bbd31656f39399d612f3c"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/model.py",
      "sha256": "7160b1358f4ee9bba44ff3dbd9d0ab3f1210ce893633dc93b69a6a3fc0df6ca7"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/physical_residual.py",
      "sha256": "05b8760db7e8e8704732ef3260cd418dcb6befe00a040bc8162f5e514277086e"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/policy.py",
      "sha256": "83d2f46900b6a8018a02006520c33c6b2b8e2ed5e46dd2fc92337a570b6e36b4"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/portable_policy.py",
      "sha256": "b890bf302e911c024ce54082c06043700f940d2fd812ffdcdc0fbb3d785a2d39"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/spacing-method.json",
      "sha256": "98604d043ffab50a0ff0f23e04656985fbe58ef5bfebc907e5e004405c22e5c4"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/train.py",
      "sha256": "5e02e78d3eca1cb7f9eeb04ed0372dc71c8f11da5f26b7cadf629510a929b81b"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v1/spacing-zero-feature-project-01/source-witness.json",
    "sha256": "9cfbd78580d966f85befb7476fcd41ce9a37d67fec946af80720efd0f798444a"
  },
  "training_population": {
    "path": "runs/pressure-methods-v1/spacing-zero-feature-study-01/train/splits.json",
    "sha256": "f5852926feb6a5fc917919b91ebfe53d34a5a072189a71aadce43a5763405f99"
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
      "bytes": 2442774,
      "path": "runs/pressure-methods-v1/spacing-training-view-01/dataset-00.npz",
      "sha256": "a50efda58cb892ca4df9875f01b82a7e0f52409397207de059db6ad5acc4beb5",
      "slot": "dataset"
    },
    {
      "bytes": 2445718,
      "path": "runs/pressure-methods-v1/spacing-training-view-01/dataset-01.npz",
      "sha256": "6e2094e3089d2a307fa6d32814835f5510364bbd40ebe358551c3140f0415f75",
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
      "bytes": 1486316,
      "path": "runs/pressure-methods-v1/spacing-two-channel-initialization-01/weights.npz",
      "sha256": "521c6ed406ef0267924639301318711078a2ab1b66a47613d32e4403b2e7b6d8",
      "slot": "initial_weights"
    },
    {
      "bytes": 161625,
      "path": "runs/pressure-methods-v1/spacing-training-view-01/manifest.json",
      "sha256": "f5852926feb6a5fc917919b91ebfe53d34a5a072189a71aadce43a5763405f99",
      "slot": "splits"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-642c8584c484925e",
    "kind": "training_attempt",
    "name": "Pressure methods · spacing-zero-feature · C53 CUDA",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-physical-residual-pressure-v1",
      "input_channels": 2,
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
    "receipt_sha256": "642c8584c484925e944e76ca85489b4d7da9c0cff263bca41b566456c1fc7884",
    "receipt_paths": [
      "runs/pressure-methods-v1/spacing-zero-feature-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "cabce71758044605aa192bcc58655c67",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods · spacing-zero-feature · C53 CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1600,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "699d4bd3d91cf634d3fd7a591f0abd5d850f2aa8c9e74ed20589804afa9cdef6",
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
          "bytes": 2442774,
          "sha256": "a50efda58cb892ca4df9875f01b82a7e0f52409397207de059db6ad5acc4beb5"
        },
        "dataset_extra": {
          "bytes": 2445718,
          "sha256": "6e2094e3089d2a307fa6d32814835f5510364bbd40ebe358551c3140f0415f75"
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
          "bytes": 1486316,
          "sha256": "521c6ed406ef0267924639301318711078a2ab1b66a47613d32e4403b2e7b6d8"
        },
        "splits": {
          "bytes": 161625,
          "sha256": "f5852926feb6a5fc917919b91ebfe53d34a5a072189a71aadce43a5763405f99"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "2d73a7a01e2bcca3c1864c24e3e14de89840a41b7717a10d4b8f5dbd4ac43d0e",
      "source_witness_sha256": "9cfbd78580d966f85befb7476fcd41ce9a37d67fec946af80720efd0f798444a"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "521c6ed406ef0267924639301318711078a2ab1b66a47613d32e4403b2e7b6d8",
        "bytes": 1486316,
        "stage": "initial"
      },
      {
        "name": "checkpoint-1600.npz",
        "sha256": "1458e4a41b381c18ec3e56624bdfdf3b036fbff643e17c16c739e11a1a2896df",
        "bytes": 1486316,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "1458e4a41b381c18ec3e56624bdfdf3b036fbff643e17c16c739e11a1a2896df",
        "bytes": 1486316,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2404102144,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/84"
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
      "receipt": "runs/pressure-methods-v1/spacing-zero-feature-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v1/spacing-zero-feature-study-01/train/receipt.json",
    "sha256": "642c8584c484925e944e76ca85489b4d7da9c0cff263bca41b566456c1fc7884",
    "bytes": 10975
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C86": {
        "name": "spacing-zero-feature",
        "weights_sha256": "1458e4a41b381c18ec3e56624bdfdf3b036fbff643e17c16c739e11a1a2896df",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.004961830737142896,
          "pressure_mean_free_relative_l2": 0.00682845708939119,
          "pressure_absolute_rms": 0.36114848767935154,
          "pressure_absolute_max": 0.8048448118287084
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
              "common_equation_relative_l2": 0.003404341248773172,
              "pressure_mean_free_relative_l2": 0.00682845708939119,
              "pressure_absolute_rms": 0.36114848767935154,
              "pressure_absolute_max": 0.8048448118287084
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.004961830737142896,
              "pressure_mean_free_relative_l2": 0.0061410927775658476,
              "pressure_absolute_rms": 0.1375225663440217,
              "pressure_absolute_max": 0.30292499235371073
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C86/results.json",
          "sha256": "d21ad45dc4a842a8197f028bc3ace839365f5a5fcb59dfb7f4f7d5bc7b5f8887"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs","spacing_modulated_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"explicit_spacing_ablation":{"explicit_spacing":false,"feature":"Identically zero second channel","same_architecture_and_initialization":true,"zero_added_convolution_column":true},"fine_tuning":{"arm":"spacing-zero-feature","early_weight":0.05,"initial_weights_sha256":"521c6ed406ef0267924639301318711078a2ab1b66a47613d32e4403b2e7b6d8","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"Original C53 first-convolution column and all other tensors preserved; added column exactly zero","input_channels":2,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":262144,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Balanced spatial tiers; private RNG; fixed update budget.","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q/train_q_RMS; second channel q/train_q_RMS times log2(h/(1/8 m)) or zero","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.25,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"spacing-parent00-cpm4"},{"cell_size_m":0.125,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":5,"split":"train","trajectory_id":"spacing-parent01-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":11,"split":"train","trajectory_id":"spacing-parent02-cpm32"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":17,"split":"train","trajectory_id":"spacing-parent04-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":22,"split":"train","trajectory_id":"spacing-parent05-cpm16"},{"cell_size_m":0.25,"counts_xyz":[13,11,8],"frame":1,"native_frame":63,"parent":28,"split":"train","trajectory_id":"spacing-parent07-cpm4"},{"cell_size_m":0.0625,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":34,"split":"train","trajectory_id":"spacing-parent08-cpm16"},{"cell_size_m":0.25,"counts_xyz":[10,16,16],"frame":1,"native_frame":63,"parent":40,"split":"train","trajectory_id":"spacing-parent10-cpm4"},{"cell_size_m":0.125,"counts_xyz":[10,5,7],"frame":0,"native_frame":15,"parent":45,"split":"train","trajectory_id":"spacing-parent11-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":51,"split":"train","trajectory_id":"spacing-parent12-cpm32"},{"cell_size_m":0.125,"counts_xyz":[6,16,8],"frame":0,"native_frame":15,"parent":57,"split":"train","trajectory_id":"spacing-parent14-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":63,"split":"train","trajectory_id":"spacing-parent15-cpm32"}],"closure":{"experiment_sha256":"699d4bd3d91cf634d3fd7a591f0abd5d850f2aa8c9e74ed20589804afa9cdef6","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":2442774,"sha256":"a50efda58cb892ca4df9875f01b82a7e0f52409397207de059db6ad5acc4beb5"},"dataset_extra":{"bytes":2445718,"sha256":"6e2094e3089d2a307fa6d32814835f5510364bbd40ebe358551c3140f0415f75"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1486316,"sha256":"521c6ed406ef0267924639301318711078a2ab1b66a47613d32e4403b2e7b6d8"},"splits":{"bytes":161625,"sha256":"f5852926feb6a5fc917919b91ebfe53d34a5a072189a71aadce43a5763405f99"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"2d73a7a01e2bcca3c1864c24e3e14de89840a41b7717a10d4b8f5dbd4ac43d0e","source_witness_sha256":"9cfbd78580d966f85befb7476fcd41ce9a37d67fec946af80720efd0f798444a"},"diagnostic_population":"train_only","elapsed_s":57.6267106809828,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.7855809605625836e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":21172736,"peak_reserved_bytes":39845888,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":1600,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1486316,"role":"checkpoint","sha256":"521c6ed406ef0267924639301318711078a2ab1b66a47613d32e4403b2e7b6d8"},"checkpoint-1600.npz":{"bytes":1486316,"role":"checkpoint","sha256":"1458e4a41b381c18ec3e56624bdfdf3b036fbff643e17c16c739e11a1a2896df"},"history.json":{"bytes":10643,"role":"training_diagnostics","sha256":"02fcf4a1c97d1bda5df503805ff72d8d6d4d1d3015ef7feee4373737418c4c28"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129398,"role":"predictions","sha256":"cdea3a8c873cdf3907f1314b1efcead3591c3bc8f91a2ef7c6a77936ae6551ab"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":161625,"role":"training_splits","sha256":"f5852926feb6a5fc917919b91ebfe53d34a5a072189a71aadce43a5763405f99"},"summary.json":{"bytes":945,"role":"result_summary","sha256":"51fe87e0dfcd8ed1e6847e9e761784e8d2707378bc9e9ccb54676cc00c55c5a5"},"weights.npz":{"bytes":1486316,"role":"checkpoint","sha256":"1458e4a41b381c18ec3e56624bdfdf3b036fbff643e17c16c739e11a1a2896df"}},"per_case_inference_s":[0.03554348400211893,0.007228657021187246,0.009160200017504394,0.007312249013921246,0.006507451995275915,0.008371860021725297,0.007646871003089473,0.016702861001249403,0.00537366300704889,0.00902471598237753,0.007136072003049776,0.009851303009781986],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"a1ae19fb4ad5e7d7942139dc919fb86fd8c888d1a89571950c01e2fa9cd96d49","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"a404f14d5e92d0078627d425cc24b4b3f81e0873963a5c92e1e0e3cc5da08f08","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"476a2dc00379799afcc8dee4772f241abe88580c1cb891fb315bca8f25539115","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"74f3f7d3603b298c47ae28588394c6204c7605c63e3dc8978b31d5fdcc3ce572","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"6834fc3a2aff218933d9e242db08966c364997fcd007be9eb28bfaab7aa4939d","shape":[6,10,11]},"case_005":{"bytes":9152,"dtype":"<f8","sha256":"47d0275187cf2938975d6ca7895faff56feff803cfa18b0a5b2bce4d804faca3","shape":[8,11,13]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"b40de74cea41632b24579baaaee35972c3cdc6093b10072468ae42c62009d8fc","shape":[9,8,12]},"case_007":{"bytes":20480,"dtype":"<f8","sha256":"b49186fe4f21930e903f0e9d83ef4c98765eb2a650a52d2b8a8a6668767fc49d","shape":[16,16,10]},"case_008":{"bytes":2800,"dtype":"<f8","sha256":"6f8148a6e8127be293a361ca3a37378c6240a84727858e4c9d9f344fb2f33b7c","shape":[7,5,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"faf67363e82eda67283a0d46b10964e896b3b7319ee5d96f7a9d61fae8a6dfbe","shape":[12,13,6]},"case_010":{"bytes":6144,"dtype":"<f8","sha256":"cd3851cdc95c22b16e76ca1130f742344d1881039ed0a08a2a45c2b74c83ad28","shape":[8,16,6]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"de88382df1a42a6e6fd705d1aa65fbd18e41d748230ae3f6acfcad364de7d28f","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2404102144,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
