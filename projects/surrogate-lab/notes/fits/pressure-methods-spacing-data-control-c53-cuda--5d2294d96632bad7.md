# Pressure methods · spacing-data-control · C53 CUDA

Retrospective source review · fit-5d2294d96632bad7

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 370,016-parameter linear bias-free one-channel F32 multiscale pressure CNN initialized from exact C53 values. Physical residual q=(rho/h)r is normalized using the fixed original scales; F64 equation assembly, residual recomputation, gauging and pressure accumulation surround shared neural corrections. Default deployment executes two pressure corrections and advances no physical simulation state.

**Purpose.** Test native spacing-varied training without adding an explicit spacing feature to the neural input.

**Experiment.** Configured 1,600 CUDA Adam updates, initial rate 1e-05, schedule [{'update': 1201, 'learning_rate': 1e-06}, {'update': 1501, 'learning_rate': 1e-07}], model seed 2026101803, index seed 2026101804. Sixty-four native training trajectories/128 requests reuse 16 original grid/force templates at 4/8/16/32 cells per metre. Counts per grid remain the corresponding original counts, so changing spacing changes physical extents. This is 64 generated variants, not 64 independent template draws.

**Observed outcome.** Retained final fitting loss 15.074962165656023. Retained common 128-request worst mean-free pressure relative L2 0.0068289890784795941, equation relative L2 0.0049617685462717484; median case-median complete GPU pressure call 4.5719997499999998 ms. This covers CPU native input/output, transfers, two corrections and synchronization; setup/scoring are outside. These are consumed pressure requests, not complete simulation steps or independent holdout evidence.

**Lesson.** Distinguish grid density, voxel counts and physical extent. The one-channel network receives physically scaled q; scalar spacing is used outside the CNN. This control does not demonstrate a learned explicit density channel or independent geometry diversity.

**Reproduction.** Retain the exact typed evidence recipe, all source/config/runtime and input/output bindings, C53 warm tensors, fixed normalization and original private sampling. Native-view manifests retain generation receipts, time mappings and target-conversion policy; recreate these using the preserved old reference before separately qualifying new Rheon. The Adam arm also needs its aggregate gradient order and observational archives. Historical publication/restore contains CPU neural queries; qualify a GPU-only equivalent before replay. Clean retraining, full-state drift and new-Rheon regeneration remain unqualified; preserve original receipts and ancestry.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v1/spacing-data-control-study-01/train/receipt.json",
    "sha256": "5d2294d96632bad70d90005cf272c78fe1addc0cb9e44c9c6b0267d9fbf2be84"
  },
  "history": {
    "path": "runs/pressure-methods-v1/spacing-data-control-study-01/train/history.json",
    "sha256": "17bf510f5901bd6fe6ae4f5324defb572338ff258e751615981edc7090c41cf9"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/config.json",
      "sha256": "9f0dedc2fa775669b1cd49f52a349a105abb9e9e0c9babed2b332e75571309bd"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/data.py",
      "sha256": "cf01d003cdf525e78cb3ec2b5e004186fa28c65ecadd63ffea969d85009674f9"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/fine-tuning.json",
      "sha256": "8883fce4852f21ebece371fdb12a3aafcfb147155454f1b710fa98cfe8af63d0"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/methods_view.py",
      "sha256": "3418389fb087ae3036549b49338bc7377291d497fa8636c0978f2f382131294f"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/mixed_order.py",
      "sha256": "3df84663563940223c95d0367b10c08b8cd53b36961bbd31656f39399d612f3c"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/policy.py",
      "sha256": "83d2f46900b6a8018a02006520c33c6b2b8e2ed5e46dd2fc92337a570b6e36b4"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/portable_policy.py",
      "sha256": "b890bf302e911c024ce54082c06043700f940d2fd812ffdcdc0fbb3d785a2d39"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/train.py",
      "sha256": "5e02e78d3eca1cb7f9eeb04ed0372dc71c8f11da5f26b7cadf629510a929b81b"
    },
    {
      "path": "runs/pressure-methods-v1/spacing-data-control-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v1/spacing-data-control-project-01/source-witness.json",
    "sha256": "2b04be4e6fae57f18b202106a2031a5f74f4ec39f446bd63abec64157ba53ac5"
  },
  "training_population": {
    "path": "runs/pressure-methods-v1/spacing-data-control-study-01/train/splits.json",
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
      "bytes": 1485452,
      "path": "runs/pressure-current-policy-v1/current-late-study-03/restore/weights.npz",
      "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
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
    "item_id": "fit-5d2294d96632bad7",
    "kind": "training_attempt",
    "name": "Pressure methods · spacing-data-control · C53 CUDA",
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
    "receipt_sha256": "5d2294d96632bad70d90005cf272c78fe1addc0cb9e44c9c6b0267d9fbf2be84",
    "receipt_paths": [
      "runs/pressure-methods-v1/spacing-data-control-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "d57b1a3a07ae41c68084b40806a961f6",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods · spacing-data-control · C53 CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1600,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "9f0dedc2fa775669b1cd49f52a349a105abb9e9e0c9babed2b332e75571309bd",
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
          "bytes": 1485452,
          "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"
        },
        "splits": {
          "bytes": 161625,
          "sha256": "f5852926feb6a5fc917919b91ebfe53d34a5a072189a71aadce43a5763405f99"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "2d73a7a01e2bcca3c1864c24e3e14de89840a41b7717a10d4b8f5dbd4ac43d0e",
      "source_witness_sha256": "2b04be4e6fae57f18b202106a2031a5f74f4ec39f446bd63abec64157ba53ac5"
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
        "sha256": "8cafbe04182a667fc4de117300977fc62ae255c0cfecd191cb0279096882b9a3",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "8cafbe04182a667fc4de117300977fc62ae255c0cfecd191cb0279096882b9a3",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2403528704,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/80"
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
      "receipt": "runs/pressure-methods-v1/spacing-data-control-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v1/spacing-data-control-study-01/train/receipt.json",
    "sha256": "5d2294d96632bad70d90005cf272c78fe1addc0cb9e44c9c6b0267d9fbf2be84",
    "bytes": 10720
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C82": {
        "name": "spacing-data-control",
        "weights_sha256": "8cafbe04182a667fc4de117300977fc62ae255c0cfecd191cb0279096882b9a3",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.004961768546271748,
          "pressure_mean_free_relative_l2": 0.006828989078479594,
          "pressure_absolute_rms": 0.3611435974028467,
          "pressure_absolute_max": 0.8048407923087169
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
              "common_equation_relative_l2": 0.0034043492563384007,
              "pressure_mean_free_relative_l2": 0.006828989078479594,
              "pressure_absolute_rms": 0.3611435974028467,
              "pressure_absolute_max": 0.8048407923087169
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.004961768546271748,
              "pressure_mean_free_relative_l2": 0.006141034506512195,
              "pressure_absolute_rms": 0.1375221352946671,
              "pressure_absolute_max": 0.3029215274303345
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C82/results.json",
          "sha256": "c81d91c52a698a632e407655e5440d1098e51c158a82df6deaacfbdbe4963c8c"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"spacing-data-control","early_weight":0.05,"initial_weights_sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":262144,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Balanced spatial tiers; private RNG; fixed update budget.","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.25,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"spacing-parent00-cpm4"},{"cell_size_m":0.125,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":5,"split":"train","trajectory_id":"spacing-parent01-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":11,"split":"train","trajectory_id":"spacing-parent02-cpm32"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":17,"split":"train","trajectory_id":"spacing-parent04-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":22,"split":"train","trajectory_id":"spacing-parent05-cpm16"},{"cell_size_m":0.25,"counts_xyz":[13,11,8],"frame":1,"native_frame":63,"parent":28,"split":"train","trajectory_id":"spacing-parent07-cpm4"},{"cell_size_m":0.0625,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":34,"split":"train","trajectory_id":"spacing-parent08-cpm16"},{"cell_size_m":0.25,"counts_xyz":[10,16,16],"frame":1,"native_frame":63,"parent":40,"split":"train","trajectory_id":"spacing-parent10-cpm4"},{"cell_size_m":0.125,"counts_xyz":[10,5,7],"frame":0,"native_frame":15,"parent":45,"split":"train","trajectory_id":"spacing-parent11-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":51,"split":"train","trajectory_id":"spacing-parent12-cpm32"},{"cell_size_m":0.125,"counts_xyz":[6,16,8],"frame":0,"native_frame":15,"parent":57,"split":"train","trajectory_id":"spacing-parent14-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":63,"split":"train","trajectory_id":"spacing-parent15-cpm32"}],"closure":{"experiment_sha256":"9f0dedc2fa775669b1cd49f52a349a105abb9e9e0c9babed2b332e75571309bd","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":2442774,"sha256":"a50efda58cb892ca4df9875f01b82a7e0f52409397207de059db6ad5acc4beb5"},"dataset_extra":{"bytes":2445718,"sha256":"6e2094e3089d2a307fa6d32814835f5510364bbd40ebe358551c3140f0415f75"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"splits":{"bytes":161625,"sha256":"f5852926feb6a5fc917919b91ebfe53d34a5a072189a71aadce43a5763405f99"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"2d73a7a01e2bcca3c1864c24e3e14de89840a41b7717a10d4b8f5dbd4ac43d0e","source_witness_sha256":"2b04be4e6fae57f18b202106a2031a5f74f4ec39f446bd63abec64157ba53ac5"},"diagnostic_population":"train_only","elapsed_s":47.0533947639924,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":4.37734150739999e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":21161984,"peak_reserved_bytes":39845888,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":1600,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"checkpoint-1600.npz":{"bytes":1485452,"role":"checkpoint","sha256":"8cafbe04182a667fc4de117300977fc62ae255c0cfecd191cb0279096882b9a3"},"history.json":{"bytes":10648,"role":"training_diagnostics","sha256":"17bf510f5901bd6fe6ae4f5324defb572338ff258e751615981edc7090c41cf9"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129398,"role":"predictions","sha256":"5fdda4774cc6b647cef891446f6fe603ad7d972add9b4e022badc67a32c3e07f"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":161625,"role":"training_splits","sha256":"f5852926feb6a5fc917919b91ebfe53d34a5a072189a71aadce43a5763405f99"},"summary.json":{"bytes":945,"role":"result_summary","sha256":"1702ecdd5556d6e5eec6aa803ae434044350733bfe2468bf9b73426f839b784c"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"8cafbe04182a667fc4de117300977fc62ae255c0cfecd191cb0279096882b9a3"}},"per_case_inference_s":[0.02997021799092181,0.00575126800686121,0.007190164993517101,0.005677423003362492,0.005065670004114509,0.00650365900946781,0.005936842993833125,0.012861045019235462,0.00414244199055247,0.00700438700732775,0.005567415995756164,0.007612943998537958],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"3bc6843c02af8bd5f3a230f69850703b243135b70049ede36178fa488ac65a38","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"1e090b7e8c8b7dcf36e4d8134421333f951e26dea54f51847386d1c41586f399","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"ba13ba9d63eee0200c9da589d19e4eb268ecfed857d19cdb914be5b4067fc6e4","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"1f43a03e59192a1e882a4e5ddc16a34a96fb2912f21cbf3a1d06a206b3d70be5","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"79d1a648a517bde95c203b7330554691ac55c1099d66d904a7496955fef5642a","shape":[6,10,11]},"case_005":{"bytes":9152,"dtype":"<f8","sha256":"265e97e05546aea33bb552c516163920eb4466016cd6e3685e07be40480c52f5","shape":[8,11,13]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"3d71bc1e8e6706b1aecd5814e0afae7d2345eeaadbb0f3710bc621b5c752a244","shape":[9,8,12]},"case_007":{"bytes":20480,"dtype":"<f8","sha256":"9e7ec7880aebcac295a4830ab11896b899f048eaaf24a58e0978e32ffdb167c4","shape":[16,16,10]},"case_008":{"bytes":2800,"dtype":"<f8","sha256":"2251aaef2f36c688bd5a737fb8ae8fb9cee24a7773cc108952d389a3e0029c79","shape":[7,5,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"a799de751a72916e405a6aa658734e46084c434888d449d133ea6a694a548454","shape":[12,13,6]},"case_010":{"bytes":6144,"dtype":"<f8","sha256":"4a4b4b0768e271226db850fb386057c1d15792aeb2855091491caec992572a0a","shape":[8,16,6]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"4f60504991ace147606a481e1b092b865dcd8272c9ec22325dd31e757b1cd4bc","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2403528704,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
