# Pressure four-pass · c37-u · C37 fine-tune CUDA

Retrospective source review · fit-0595218bf00cecf7

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 370,016-parameter float32 3D pressure U-Net with encoder widths8/16/24/32, bottleneck48 and reversed decoder; physical residual reconstruction is float64. It starts from the exact C37 checkpoint. Training unrolls four pressure corrections on one equation. These corrections are not future physical simulation steps.

**Purpose.** Compare unchanged four-pass fine tuning with attached and detached auxiliary gradient paths under the same initialization, pattern order, depth schedule and optimizer budget. Retain the four-pass control without the auxiliary loss.

**Experiment.** The optimizer uses eight synthetic analytical patterns: one [0,0,1] mode on each of eight rectangular grids, at8or16cells/m. Each gets200exposures, with50at each of four auxiliary depths. The original16train trajectories calibrate amplitude/normalization; they are not the optimizer sample stream. No companion physical, direction or current-policy loss runs in the captured fitting-bank updater. Teacher weight is zero and counted teacher forwards are zero. All arms run1,600updates; U counts6,400network forwards and A/D8,000. The loss combines analytical pressure loss, log-scaled final equation residual and 0.05mean early residual feedback; A/D add0.05auxiliary loss.

**Observed outcome.** C75, on the retained common128pressure requests: worst mean-free pressure relative L2=0.01129882497652308, equation relative L2=0.0072754935123135562. The parent C37 values are 0.05382213638741031 and0.04891706555322945. These are pressure-only request results, not trajectory drift, full-state accuracy or fresh qualified latency.

**Lesson.** This matched experiment changes gradient flow, not architecture or time advancement. Its effective fitting variety is eight single-mode patterns, despite retaining physical datasets and broader feedback declarations. A has the lowest worst pressure error of the three arms, while U has the lowest worst equation residual; no arm dominates both. Training forward counts differ, so equal optimizer updates do not imply equal compute.

**Reproduction.** Use the exact captured project.json/train-recipe.json and evidence.py entrypoint with the original input bindings, source witness, C37 checkpoint, fitting bank and fine-tuning.json. Clean retraining, fresh qualified latency and new-Rheon data equivalence remain unqualified; no checkpoint can be retired on this review alone.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-detached-correction-v1/c37-u-study-01/train/receipt.json",
    "sha256": "0595218bf00cecf7950b802bea3720a49a8e5d69afe4186a1c77466aaf8627c8"
  },
  "history": {
    "path": "runs/pressure-detached-correction-v1/c37-u-study-01/train/history.json",
    "sha256": "5d854f5020fa1aed846ebefd20337ee0c3ca04e8055bae7789eebee3bb000ed0"
  },
  "sources": [
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/config.json",
      "sha256": "9451f8305ce6cf8086c5eb5b34c65b214851e8be22d2bc7508a3499320407b70"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/fine-tuning.json",
      "sha256": "93a739a05b33f75261c6734021bf3d1cfda85d49fb30f39914bc4810d18addbb"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/fine_tuning.py",
      "sha256": "57658249095fd62c38f8d3bcbbe201d5ebe288009f0042001941bca82110795e"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/fit_bank.py",
      "sha256": "bbdbabfc89805410df196dd0724b24347954548dd83790456994ca46a6f5e347"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/model.py",
      "sha256": "a2ebefe6a8ce12c7e6e1e12fa82b835029f364b50e91284f33b60bfe31b0eaee"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/original_fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/original_fit_bank.py",
      "sha256": "e68c1fa6723331af3448edec8294ca0dc67a5b97576a1cd251c5a789e9bf0668"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/train.py",
      "sha256": "b9c98c38dfd22ec77ed1bc6dcbad8e692b276dc96fe9cb5cf928ec2b6be0774b"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c37-u-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-detached-correction-v1/c37-u-project-01/source-witness.json",
    "sha256": "69c6aa4964ecd9028ace3cd520dc7d313c6acf622d98caae20f85d7cece5b717"
  },
  "training_population": {
    "path": "runs/pressure-detached-correction-v1/c37-u-study-01/train/splits.json",
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
      "path": "runs/water-shape-repeatability-v1/shape-one-longer/managed-full/stage-01/artifacts/weights.npz",
      "sha256": "8cd2a0d4a694f44f393ebc8381a407e4195fd9361f1e115060104bd0df60ac8f",
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

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-0595218bf00cecf7",
    "kind": "training_attempt",
    "name": "Pressure four-pass · c37-u · C37 fine-tune CUDA",
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
    "receipt_sha256": "0595218bf00cecf7950b802bea3720a49a8e5d69afe4186a1c77466aaf8627c8",
    "receipt_paths": [
      "runs/pressure-detached-correction-v1/c37-u-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "aa84aa761a764f378c9a053085636797",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure four-pass · c37-u · C37 fine-tune CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1600,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "9451f8305ce6cf8086c5eb5b34c65b214851e8be22d2bc7508a3499320407b70",
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
          "sha256": "8cd2a0d4a694f44f393ebc8381a407e4195fd9361f1e115060104bd0df60ac8f"
        },
        "splits": {
          "bytes": 256361,
          "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "21ed766152674818e22d0abb5ed5540690ab31aa14507d78fcc73a381e5d058a",
      "source_witness_sha256": "69c6aa4964ecd9028ace3cd520dc7d313c6acf622d98caae20f85d7cece5b717"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "8cd2a0d4a694f44f393ebc8381a407e4195fd9361f1e115060104bd0df60ac8f",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-1600.npz",
        "sha256": "ddad267c2eaff2476d8ee10cee4ce6ff9aff686f3ff54a03e89d83baa6e5731c",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "ddad267c2eaff2476d8ee10cee4ce6ff9aff686f3ff54a03e89d83baa6e5731c",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2413834240,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/73"
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
      "receipt": "runs/pressure-detached-correction-v1/c37-u-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-detached-correction-v1/c37-u-study-01/train/receipt.json",
    "sha256": "0595218bf00cecf7950b802bea3720a49a8e5d69afe4186a1c77466aaf8627c8",
    "bytes": 59916
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C75": {
        "name": "c37-u",
        "weights_sha256": "ddad267c2eaff2476d8ee10cee4ce6ff9aff686f3ff54a03e89d83baa6e5731c",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.007275493512313556,
          "pressure_mean_free_relative_l2": 0.01129882497652308,
          "pressure_absolute_rms": 0.39659261183755945,
          "pressure_absolute_max": 1.242922241117203
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0032",
            "native_frame": 15,
            "counts_xyz": [
              15,
              6,
              8
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
              "common_equation_relative_l2": 0.007275493512313556,
              "pressure_mean_free_relative_l2": 0.01129882497652308,
              "pressure_absolute_rms": 0.39659261183755945,
              "pressure_absolute_max": 1.242922241117203
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.005592357029643427,
              "pressure_mean_free_relative_l2": 0.007826988357620242,
              "pressure_absolute_rms": 0.19091433955818568,
              "pressure_absolute_max": 0.5564467737839336
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C75/results.json",
          "sha256": "c8ad541698814c29573a8773831582e179d361a7c634c78b35ae7edb1d1f16ca"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"c37-u","early_weight":0.05,"fitting_bank":{"amplitude_reference":"original-train-correction-rms","cache_max_bytes":8388608,"grids":[{"cells_per_metre":16,"counts_xyz":[13,15,16]},{"cells_per_metre":16,"counts_xyz":[9,11,11]},{"cells_per_metre":8,"counts_xyz":[15,9,7]},{"cells_per_metre":16,"counts_xyz":[11,5,15]},{"cells_per_metre":8,"counts_xyz":[7,16,5]},{"cells_per_metre":16,"counts_xyz":[11,12,7]},{"cells_per_metre":8,"counts_xyz":[5,13,9]},{"cells_per_metre":16,"counts_xyz":[16,7,11]}],"modes_zyx":[[0,0,1]],"sampling_seed":2026102101},"initial_weights_sha256":"8cd2a0d4a694f44f393ebc8381a407e4195fd9361f1e115060104bd0df60ac8f","relative_equation_target":1e-06,"rollout_feedback":{"depth_seed":2026101807,"mode":"U","weight":0.05},"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":4},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"9451f8305ce6cf8086c5eb5b34c65b214851e8be22d2bc7508a3499320407b70","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"8cd2a0d4a694f44f393ebc8381a407e4195fd9361f1e115060104bd0df60ac8f"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"21ed766152674818e22d0abb5ed5540690ab31aa14507d78fcc73a381e5d058a","source_witness_sha256":"69c6aa4964ecd9028ace3cd520dc7d313c6acf622d98caae20f85d7cece5b717"},"diagnostic_population":"train_only","elapsed_s":70.43437833499047,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":5.067518937806881e-06,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":19713024,"peak_reserved_bytes":37748736,"tf32":false},"fine_tuning_statistics":{"fundamental_bank":{"amplitude_rms_pa":2.8535734640782584,"cases":[{"cells_per_metre":16,"counts_xyz":[13,15,16],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[9,11,11],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[15,9,7],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[11,5,15],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[7,16,5],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[11,12,7],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[5,13,9],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[16,7,11],"mode_zyx":[0,0,1]}],"description_sha256":"30f7092460b6a649b28e4bf380a5769c2955d1e29461e0f64a6e1d474dab960f","expected_exposures":[200,200,200,200,200,200,200,200],"optimizer_examples":1600,"optimizer_exposures":[200,200,200,200,200,200,200,200],"resident_tensor_bytes":297152,"sampling_order_sha256":"d0ea221a7e03a83673ece7fc9cdf4f5212bb54f3023511f93d95c84e45647899","trace":[{"exposures_per_case":10,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.221833931736812},{"case":0,"passes":2,"relative_equation_residual":0.03230619827954216},{"case":0,"passes":4,"relative_equation_residual":0.0008706267196104194},{"case":0,"passes":8,"relative_equation_residual":5.278338596385261e-07},{"case":1,"passes":1,"relative_equation_residual":0.13289917990912276},{"case":1,"passes":2,"relative_equation_residual":0.01565413185817393},{"case":1,"passes":4,"relative_equation_residual":0.00017874174901663813},{"case":1,"passes":8,"relative_equation_residual":1.4373871620185529e-08},{"case":2,"passes":1,"relative_equation_residual":0.25950606523611475},{"case":2,"passes":2,"relative_equation_residual":0.015441505735926758},{"case":2,"passes":4,"relative_equation_residual":0.00018805877645775723},{"case":2,"passes":8,"relative_equation_residual":1.3223056612159674e-08},{"case":3,"passes":1,"relative_equation_residual":0.17834797077760103},{"case":3,"passes":2,"relative_equation_residual":0.005483883248701832},{"case":3,"passes":4,"relative_equation_residual":1.5403855936145188e-05},{"case":3,"passes":8,"relative_equation_residual":4.2463075674991327e-10},{"case":4,"passes":1,"relative_equation_residual":0.04958205538640434},{"case":4,"passes":2,"relative_equation_residual":0.0035565462133021865},{"case":4,"passes":4,"relative_equation_residual":1.5284533084721214e-05},{"case":4,"passes":8,"relative_equation_residual":3.6338736829295015e-10},{"case":5,"passes":1,"relative_equation_residual":0.17834706575388037},{"case":5,"passes":2,"relative_equation_residual":0.005483851394237078},{"case":5,"passes":4,"relative_equation_residual":1.5404930251520522e-05},{"case":5,"passes":8,"relative_equation_residual":4.246490217163977e-10},{"case":6,"passes":1,"relative_equation_residual":0.08567413281822985},{"case":6,"passes":2,"relative_equation_residual":0.006144428267729855},{"case":6,"passes":4,"relative_equation_residual":3.365508058703982e-05},{"case":6,"passes":8,"relative_equation_residual":1.0160979522382748e-09},{"case":7,"passes":1,"relative_equation_residual":0.20099252949262517},{"case":7,"passes":2,"relative_equation_residual":0.032136632221621345},{"case":7,"passes":4,"relative_equation_residual":0.0007908160012937128},{"case":7,"passes":8,"relative_equation_residual":9.737415110569102e-07}],"update":80},{"exposures_per_case":20,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.1585444511016039},{"case":0,"passes":2,"relative_equation_residual":0.024735705635116897},{"case":0,"passes":4,"relative_equation_residual":0.000610925291980371},{"case":0,"passes":8,"relative_equation_residual":3.7016587948875645e-07},{"case":1,"passes":1,"relative_equation_residual":0.15715717061603862},{"case":1,"passes":2,"relative_equation_residual":0.02363511105755632},{"case":1,"passes":4,"relative_equation_residual":0.000498546531181773},{"case":1,"passes":8,"relative_equation_residual":2.1006102410227039e-07},{"case":2,"passes":1,"relative_equation_residual":0.21247829102510038},{"case":2,"passes":2,"relative_equation_residual":0.007702722246093588},{"case":2,"passes":4,"relative_equation_residual":4.92032029016324e-05},{"case":2,"passes":8,"relative_equation_residual":2.4540652163000383e-09},{"case":3,"passes":1,"relative_equation_residual":0.11116954280609861},{"case":3,"passes":2,"relative_equation_residual":0.003535395490809319},{"case":3,"passes":4,"relative_equation_residual":2.0108079826930464e-05},{"case":3,"passes":8,"relative_equation_residual":4.944323750332293e-10},{"case":4,"passes":1,"relative_equation_residual":0.03989368813264598},{"case":4,"passes":2,"relative_equation_residual":0.0009700444124651901},{"case":4,"passes":4,"relative_equation_residual":7.76506346792253e-07},{"case":4,"passes":8,"relative_equation_residual":1.9548489602751377e-11},{"case":5,"passes":1,"relative_equation_residual":0.1111706638221444},{"case":5,"passes":2,"relative_equation_residual":0.003535338751766055},{"case":5,"passes":4,"relative_equation_residual":2.010766953749654e-05},{"case":5,"passes":8,"relative_equation_residual":4.943738301151281e-10},{"case":6,"passes":1,"relative_equation_residual":0.06826787850189962},{"case":6,"passes":2,"relative_equation_residual":0.003674310592223211},{"case":6,"passes":4,"relative_equation_residual":9.736459942308211e-06},{"case":6,"passes":8,"relative_equation_residual":7.785671602508889e-11},{"case":7,"passes":1,"relative_equation_residual":0.20716746288911275},{"case":7,"passes":2,"relative_equation_residual":0.028467304221595907},{"case":7,"passes":4,"relative_equation_residual":0.00046974892899063537},{"case":7,"passes":8,"relative_equation_residual":2.377412725465671e-07}],"update":160},{"exposures_per_case":30,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.3147234111156015},{"case":0,"passes":2,"relative_equation_residual":0.06598873437925729},{"case":0,"passes":4,"relative_equation_residual":0.0009030739246641504},{"case":0,"passes":8,"relative_equation_residual":1.0509852091548508e-06},{"case":1,"passes":1,"relative_equation_residual":0.2596337872120156},{"case":1,"passes":2,"relative_equation_residual":0.04521114668312657},{"case":1,"passes":4,"relative_equation_residual":0.0005020064949394112},{"case":1,"passes":8,"relative_equation_residual":2.270723844450352e-07},{"case":2,"passes":1,"relative_equation_residual":0.1772440460537666},{"case":2,"passes":2,"relative_equation_residual":0.015831673493546392},{"case":2,"passes":4,"relative_equation_residual":6.156455932414906e-05},{"case":2,"passes":8,"relative_equation_residual":6.664406502603766e-10},{"case":3,"passes":1,"relative_equation_residual":0.1147227582254461},{"case":3,"passes":2,"relative_equation_residual":0.007878991251060487},{"case":3,"passes":4,"relative_equation_residual":1.3519723285314817e-05},{"case":3,"passes":8,"relative_equation_residual":1.39007274811405e-10},{"case":4,"passes":1,"relative_equation_residual":0.04919390751257659},{"case":4,"passes":2,"relative_equation_residual":0.0011057536792557416},{"case":4,"passes":4,"relative_equation_residual":5.797827766234795e-07},{"case":4,"passes":8,"relative_equation_residual":8.989832120622292e-12},{"case":5,"passes":1,"relative_equation_residual":0.11472249250855975},{"case":5,"passes":2,"relative_equation_residual":0.007879066832702324},{"case":5,"passes":4,"relative_equation_residual":1.3519219128792835e-05},{"case":5,"passes":8,"relative_equation_residual":1.3870273391194188e-10},{"case":6,"passes":1,"relative_equation_residual":0.040499443144326484},{"case":6,"passes":2,"relative_equation_residual":0.0018870168051860728},{"case":6,"passes":4,"relative_equation_residual":4.244286061506543e-06},{"case":6,"passes":8,"relative_equation_residual":6.129852328745447e-11},{"case":7,"passes":1,"relative_equation_residual":0.2579797983488944},{"case":7,"passes":2,"relative_equation_residual":0.05444555899680821},{"case":7,"passes":4,"relative_equation_residual":0.0036295596475696026},{"case":7,"passes":8,"relative_equation_residual":1.6134582999643663e-05}],"update":240},{"exposures_per_case":40,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.39668289838003135},{"case":0,"passes":2,"relative_equation_residual":0.1289547566135264},{"case":0,"passes":4,"relative_equation_residual":0.012007555996115379},{"case":0,"passes":8,"relative_equation_residual":0.00010030995864714393},{"case":1,"passes":1,"relative_equation_residual":0.2807504282670877},{"case":1,"passes":2,"relative_equation_residual":0.06298895413210548},{"case":1,"passes":4,"relative_equation_residual":0.0026186064239714516},{"case":1,"passes":8,"relative_equation_residual":4.252606139361763e-06},{"case":2,"passes":1,"relative_equation_residual":0.2663474972890706},{"case":2,"passes":2,"relative_equation_residual":0.05286416707303524},{"case":2,"passes":4,"relative_equation_residual":0.002327798909353497},{"case":2,"passes":8,"relative_equation_residual":4.555034819252834e-06},{"case":3,"passes":1,"relative_equation_residual":0.13450795498460477},{"case":3,"passes":2,"relative_equation_residual":0.018246086042396956},{"case":3,"passes":4,"relative_equation_residual":0.00032220334527603424},{"case":3,"passes":8,"relative_equation_residual":1.0204664390908073e-07},{"case":4,"passes":1,"relative_equation_residual":0.0429926028006539},{"case":4,"passes":2,"relative_equation_residual":0.0014731933834486277},{"case":4,"passes":4,"relative_equation_residual":1.1692503803869566e-06},{"case":4,"passes":8,"relative_equation_residual":2.5989521842943515e-11},{"case":5,"passes":1,"relative_equation_residual":0.13450719366089225},{"case":5,"passes":2,"relative_equation_residual":0.01824613829894999},{"case":5,"passes":4,"relative_equation_residual":0.0003222003916695632},{"case":5,"passes":8,"relative_equation_residual":1.0204630238697335e-07},{"case":6,"passes":1,"relative_equation_residual":0.06429065237603421},{"case":6,"passes":2,"relative_equation_residual":0.003903794679762584},{"case":6,"passes":4,"relative_equation_residual":1.3589504060932706e-05},{"case":6,"passes":8,"relative_equation_residual":2.720249816899219e-09},{"case":7,"passes":1,"relative_equation_residual":0.13049018228921358},{"case":7,"passes":2,"relative_equation_residual":0.0038128667782200767},{"case":7,"passes":4,"relative_equation_residual":4.937542909683885e-06},{"case":7,"passes":8,"relative_equation_residual":1.2003075897415495e-10}],"update":320},{"exposures_per_case":50,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.1354806209957898},{"case":0,"passes":2,"relative_equation_residual":0.017412708586315367},{"case":0,"passes":4,"relative_equation_residual":0.000220765111594168},{"case":0,"passes":8,"relative_equation_residual":3.7205425017620695e-08},{"case":1,"passes":1,"relative_equation_residual":0.12123867942793852},{"case":1,"passes":2,"relative_equation_residual":0.010401484095275606},{"case":1,"passes":4,"relative_equation_residual":7.896999827311914e-05},{"case":1,"passes":8,"relative_equation_residual":5.170823501616963e-09},{"case":2,"passes":1,"relative_equation_residual":0.1366162062697285},{"case":2,"passes":2,"relative_equation_residual":0.01336421281020205},{"case":2,"passes":4,"relative_equation_residual":0.00010340590367795333},{"case":2,"passes":8,"relative_equation_residual":1.902068196627291e-09},{"case":3,"passes":1,"relative_equation_residual":0.07803083217127853},{"case":3,"passes":2,"relative_equation_residual":0.0058081455395877285},{"case":3,"passes":4,"relative_equation_residual":2.8285291802497233e-05},{"case":3,"passes":8,"relative_equation_residual":6.159822423785928e-10},{"case":4,"passes":1,"relative_equation_residual":0.06379973672958907},{"case":4,"passes":2,"relative_equation_residual":0.0022338910965738774},{"case":4,"passes":4,"relative_equation_residual":4.265580237606417e-06},{"case":4,"passes":8,"relative_equation_residual":1.0852878746171188e-10},{"case":5,"passes":1,"relative_equation_residual":0.07803054599796384},{"case":5,"passes":2,"relative_equation_residual":0.005808019849462376},{"case":5,"passes":4,"relative_equation_residual":2.828426926199454e-05},{"case":5,"passes":8,"relative_equation_residual":6.156155832105917e-10},{"case":6,"passes":1,"relative_equation_residual":0.04090100333978021},{"case":6,"passes":2,"relative_equation_residual":0.0014926299058856546},{"case":6,"passes":4,"relative_equation_residual":6.4259682714068305e-06},{"case":6,"passes":8,"relative_equation_residual":1.3414856862398654e-10},{"case":7,"passes":1,"relative_equation_residual":0.19690826655453494},{"case":7,"passes":2,"relative_equation_residual":0.03942510811191853},{"case":7,"passes":4,"relative_equation_residual":0.00172404311694848},{"case":7,"passes":8,"relative_equation_residual":4.064636560675945e-06}],"update":400},{"exposures_per_case":60,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2240970454959664},{"case":0,"passes":2,"relative_equation_residual":0.04371540327468499},{"case":0,"passes":4,"relative_equation_residual":0.0015001812663061709},{"case":0,"passes":8,"relative_equation_residual":1.7635707444703715e-06},{"case":1,"passes":1,"relative_equation_residual":0.15753313253293472},{"case":1,"passes":2,"relative_equation_residual":0.01928813695650518},{"case":1,"passes":4,"relative_equation_residual":0.0002559172292367011},{"case":1,"passes":8,"relative_equation_residual":4.843220565881268e-08},{"case":2,"passes":1,"relative_equation_residual":0.19557107979046515},{"case":2,"passes":2,"relative_equation_residual":0.009050219906898922},{"case":2,"passes":4,"relative_equation_residual":1.4529364812075108e-05},{"case":2,"passes":8,"relative_equation_residual":1.1864325927437882e-10},{"case":3,"passes":1,"relative_equation_residual":0.07952909625619675},{"case":3,"passes":2,"relative_equation_residual":0.002283421214926618},{"case":3,"passes":4,"relative_equation_residual":6.1744635533558895e-06},{"case":3,"passes":8,"relative_equation_residual":2.94878570813365e-11},{"case":4,"passes":1,"relative_equation_residual":0.047404892825674276},{"case":4,"passes":2,"relative_equation_residual":0.0014080128756306393},{"case":4,"passes":4,"relative_equation_residual":2.3780018889093077e-06},{"case":4,"passes":8,"relative_equation_residual":5.245180384947108e-11},{"case":5,"passes":1,"relative_equation_residual":0.07952973183893527},{"case":5,"passes":2,"relative_equation_residual":0.0022834887886312876},{"case":5,"passes":4,"relative_equation_residual":6.173477377217692e-06},{"case":5,"passes":8,"relative_equation_residual":3.0985678807953795e-11},{"case":6,"passes":1,"relative_equation_residual":0.03551498274699151},{"case":6,"passes":2,"relative_equation_residual":0.0007295995580072858},{"case":6,"passes":4,"relative_equation_residual":1.3525374619692333e-06},{"case":6,"passes":8,"relative_equation_residual":7.365354730486004e-11},{"case":7,"passes":1,"relative_equation_residual":0.16017039089542012},{"case":7,"passes":2,"relative_equation_residual":0.01818504968165663},{"case":7,"passes":4,"relative_equation_residual":0.00019047963096314771},{"case":7,"passes":8,"relative_equation_residual":1.963675216138937e-07}],"update":480},{"exposures_per_case":70,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.20785786935612482},{"case":0,"passes":2,"relative_equation_residual":0.032926891762101534},{"case":0,"passes":4,"relative_equation_residual":0.0007440811772369288},{"case":0,"passes":8,"relative_equation_residual":3.845002652982053e-07},{"case":1,"passes":1,"relative_equation_residual":0.17477467984809805},{"case":1,"passes":2,"relative_equation_residual":0.023898362116635333},{"case":1,"passes":4,"relative_equation_residual":0.0004633399283400149},{"case":1,"passes":8,"relative_equation_residual":1.779707073629071e-07},{"case":2,"passes":1,"relative_equation_residual":0.16430401817357837},{"case":2,"passes":2,"relative_equation_residual":0.006638926702107514},{"case":2,"passes":4,"relative_equation_residual":1.9488142195352315e-05},{"case":2,"passes":8,"relative_equation_residual":1.67251919426644e-10},{"case":3,"passes":1,"relative_equation_residual":0.046196715961792825},{"case":3,"passes":2,"relative_equation_residual":0.001497820760552807},{"case":3,"passes":4,"relative_equation_residual":9.36310093964371e-07},{"case":3,"passes":8,"relative_equation_residual":3.562756643235576e-11},{"case":4,"passes":1,"relative_equation_residual":0.03250773397571907},{"case":4,"passes":2,"relative_equation_residual":0.0008994000055693045},{"case":4,"passes":4,"relative_equation_residual":1.1179959411989853e-06},{"case":4,"passes":8,"relative_equation_residual":4.4972875412834114e-11},{"case":5,"passes":1,"relative_equation_residual":0.04620129976501242},{"case":5,"passes":2,"relative_equation_residual":0.0014979499727238441},{"case":5,"passes":4,"relative_equation_residual":9.359525623000074e-07},{"case":5,"passes":8,"relative_equation_residual":1.345673318830071e-11},{"case":6,"passes":1,"relative_equation_residual":0.033169631567696714},{"case":6,"passes":2,"relative_equation_residual":0.0009151603192403973},{"case":6,"passes":4,"relative_equation_residual":9.687100722593035e-07},{"case":6,"passes":8,"relative_equation_residual":1.935346838157302e-10},{"case":7,"passes":1,"relative_equation_residual":0.1806961676243724},{"case":7,"passes":2,"relative_equation_residual":0.022773061457916247},{"case":7,"passes":4,"relative_equation_residual":0.0004238977041842547},{"case":7,"passes":8,"relative_equation_residual":2.0299646336272191e-07}],"update":560},{"exposures_per_case":80,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.19593442763731972},{"case":0,"passes":2,"relative_equation_residual":0.03340260507097487},{"case":0,"passes":4,"relative_equation_residual":0.0007648193537809066},{"case":0,"passes":8,"relative_equation_residual":3.489237195887288e-07},{"case":1,"passes":1,"relative_equation_residual":0.15388902014043462},{"case":1,"passes":2,"relative_equation_residual":0.018236055520405917},{"case":1,"passes":4,"relative_equation_residual":0.00016695831423508265},{"case":1,"passes":8,"relative_equation_residual":5.1311397308220884e-09},{"case":2,"passes":1,"relative_equation_residual":0.18395352958542108},{"case":2,"passes":2,"relative_equation_residual":0.009308348482319076},{"case":2,"passes":4,"relative_equation_residual":4.626798036328414e-05},{"case":2,"passes":8,"relative_equation_residual":1.935684962254638e-09},{"case":3,"passes":1,"relative_equation_residual":0.09879578005535881},{"case":3,"passes":2,"relative_equation_residual":0.005098687260354624},{"case":3,"passes":4,"relative_equation_residual":1.818797756573711e-05},{"case":3,"passes":8,"relative_equation_residual":2.7052429352007273e-10},{"case":4,"passes":1,"relative_equation_residual":0.03718078709796238},{"case":4,"passes":2,"relative_equation_residual":0.0011986589119059444},{"case":4,"passes":4,"relative_equation_residual":9.758370470233547e-07},{"case":4,"passes":8,"relative_equation_residual":2.753257686351523e-10},{"case":5,"passes":1,"relative_equation_residual":0.0987923567622522},{"case":5,"passes":2,"relative_equation_residual":0.005098302108059881},{"case":5,"passes":4,"relative_equation_residual":1.8188340478467834e-05},{"case":5,"passes":8,"relative_equation_residual":2.7070235894943034e-10},{"case":6,"passes":1,"relative_equation_residual":0.027511550272009428},{"case":6,"passes":2,"relative_equation_residual":0.0005448244904321266},{"case":6,"passes":4,"relative_equation_residual":1.936676142767439e-07},{"case":6,"passes":8,"relative_equation_residual":1.0331969688365653e-10},{"case":7,"passes":1,"relative_equation_residual":0.1623211613085909},{"case":7,"passes":2,"relative_equation_residual":0.018081414738086363},{"case":7,"passes":4,"relative_equation_residual":0.00018414827618607564},{"case":7,"passes":8,"relative_equation_residual":8.708114551319334e-08}],"update":640},{"exposures_per_case":90,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.13947676788256538},{"case":0,"passes":2,"relative_equation_residual":0.009636932786685065},{"case":0,"passes":4,"relative_equation_residual":1.695741801995016e-05},{"case":0,"passes":8,"relative_equation_residual":2.51448743605499e-10},{"case":1,"passes":1,"relative_equation_residual":0.08645371775764125},{"case":1,"passes":2,"relative_equation_residual":0.004338698166473574},{"case":1,"passes":4,"relative_equation_residual":2.8188224909605075e-06},{"case":1,"passes":8,"relative_equation_residual":4.8205843786436764e-11},{"case":2,"passes":1,"relative_equation_residual":0.18527985618215315},{"case":2,"passes":2,"relative_equation_residual":0.02097416668308968},{"case":2,"passes":4,"relative_equation_residual":0.00027245045529848547},{"case":2,"passes":8,"relative_equation_residual":4.72312315338342e-08},{"case":3,"passes":1,"relative_equation_residual":0.12773324653147133},{"case":3,"passes":2,"relative_equation_residual":0.013955981016255063},{"case":3,"passes":4,"relative_equation_residual":0.00018446454972855628},{"case":3,"passes":8,"relative_equation_residual":3.217952909831214e-08},{"case":4,"passes":1,"relative_equation_residual":0.08529810796710227},{"case":4,"passes":2,"relative_equation_residual":0.006244072237534146},{"case":4,"passes":4,"relative_equation_residual":3.837444186169372e-05},{"case":4,"passes":8,"relative_equation_residual":1.4611625618313543e-09},{"case":5,"passes":1,"relative_equation_residual":0.12773376164164935},{"case":5,"passes":2,"relative_equation_residual":0.013956034589248458},{"case":5,"passes":4,"relative_equation_residual":0.00018446820530311833},{"case":5,"passes":8,"relative_equation_residual":3.2180546660453416e-08},{"case":6,"passes":1,"relative_equation_residual":0.03753546202407836},{"case":6,"passes":2,"relative_equation_residual":0.001493973302151758},{"case":6,"passes":4,"relative_equation_residual":8.13352223609867e-07},{"case":6,"passes":8,"relative_equation_residual":4.2808283594335664e-11},{"case":7,"passes":1,"relative_equation_residual":0.2837961461046702},{"case":7,"passes":2,"relative_equation_residual":0.07656208092708941},{"case":7,"passes":4,"relative_equation_residual":0.006409969572884233},{"case":7,"passes":8,"relative_equation_residual":4.656124717969541e-05}],"update":720},{"exposures_per_case":100,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.21610998397468006},{"case":0,"passes":2,"relative_equation_residual":0.03154822110229641},{"case":0,"passes":4,"relative_equation_residual":0.0003829317530792909},{"case":0,"passes":8,"relative_equation_residual":1.0240422085673425e-08},{"case":1,"passes":1,"relative_equation_residual":0.11839913344639177},{"case":1,"passes":2,"relative_equation_residual":0.005849589855552994},{"case":1,"passes":4,"relative_equation_residual":1.0080981309892901e-05},{"case":1,"passes":8,"relative_equation_residual":8.1634434090124e-11},{"case":2,"passes":1,"relative_equation_residual":0.14103634768209822},{"case":2,"passes":2,"relative_equation_residual":0.0067578020953361455},{"case":2,"passes":4,"relative_equation_residual":4.923890797932184e-05},{"case":2,"passes":8,"relative_equation_residual":2.8192128408890137e-09},{"case":3,"passes":1,"relative_equation_residual":0.07182594031375686},{"case":3,"passes":2,"relative_equation_residual":0.004993197606204233},{"case":3,"passes":4,"relative_equation_residual":2.693093117651344e-05},{"case":3,"passes":8,"relative_equation_residual":7.521822898337002e-10},{"case":4,"passes":1,"relative_equation_residual":0.061410588982359425},{"case":4,"passes":2,"relative_equation_residual":0.0030358642163616605},{"case":4,"passes":4,"relative_equation_residual":6.266340135488131e-06},{"case":4,"passes":8,"relative_equation_residual":3.228409522604636e-11},{"case":5,"passes":1,"relative_equation_residual":0.07182669318811546},{"case":5,"passes":2,"relative_equation_residual":0.004993271809106206},{"case":5,"passes":4,"relative_equation_residual":2.6932299951370164e-05},{"case":5,"passes":8,"relative_equation_residual":7.526556780201439e-10},{"case":6,"passes":1,"relative_equation_residual":0.020956113977043084},{"case":6,"passes":2,"relative_equation_residual":0.0005045142944333037},{"case":6,"passes":4,"relative_equation_residual":4.102958571530586e-07},{"case":6,"passes":8,"relative_equation_residual":6.019699242798364e-11},{"case":7,"passes":1,"relative_equation_residual":0.1501459167931364},{"case":7,"passes":2,"relative_equation_residual":0.019695720056869172},{"case":7,"passes":4,"relative_equation_residual":0.0004487142056538896},{"case":7,"passes":8,"relative_equation_residual":5.432140964172314e-07}],"update":800},{"exposures_per_case":110,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.23125913455829014},{"case":0,"passes":2,"relative_equation_residual":0.035227814186643586},{"case":0,"passes":4,"relative_equation_residual":0.00048140029447886256},{"case":0,"passes":8,"relative_equation_residual":2.9914155917687736e-08},{"case":1,"passes":1,"relative_equation_residual":0.15205345550003085},{"case":1,"passes":2,"relative_equation_residual":0.014027720774708395},{"case":1,"passes":4,"relative_equation_residual":7.516295766089885e-05},{"case":1,"passes":8,"relative_equation_residual":1.4092158155970535e-09},{"case":2,"passes":1,"relative_equation_residual":0.16147882864390786},{"case":2,"passes":2,"relative_equation_residual":0.0038185669426308833},{"case":2,"passes":4,"relative_equation_residual":1.1024247600080282e-05},{"case":2,"passes":8,"relative_equation_residual":1.5265089363696004e-10},{"case":3,"passes":1,"relative_equation_residual":0.0680946392802518},{"case":3,"passes":2,"relative_equation_residual":0.0014022322471068096},{"case":3,"passes":4,"relative_equation_residual":2.4589912786959947e-06},{"case":3,"passes":8,"relative_equation_residual":8.681565795371262e-12},{"case":4,"passes":1,"relative_equation_residual":0.03072949100382202},{"case":4,"passes":2,"relative_equation_residual":0.0005170616286727816},{"case":4,"passes":4,"relative_equation_residual":5.2324267078200313e-08},{"case":4,"passes":8,"relative_equation_residual":3.019993068245064e-11},{"case":5,"passes":1,"relative_equation_residual":0.06809545807470316},{"case":5,"passes":2,"relative_equation_residual":0.0014022359453125732},{"case":5,"passes":4,"relative_equation_residual":2.4590290240664336e-06},{"case":5,"passes":8,"relative_equation_residual":8.802581968585015e-12},{"case":6,"passes":1,"relative_equation_residual":0.007183728822903885},{"case":6,"passes":2,"relative_equation_residual":5.727114100920166e-05},{"case":6,"passes":4,"relative_equation_residual":8.151051071380846e-09},{"case":6,"passes":8,"relative_equation_residual":6.528236457107339e-11},{"case":7,"passes":1,"relative_equation_residual":0.16606686807616156},{"case":7,"passes":2,"relative_equation_residual":0.018007483021995747},{"case":7,"passes":4,"relative_equation_residual":0.00024632803503457244},{"case":7,"passes":8,"relative_equation_residual":9.08215348183464e-08}],"update":880},{"exposures_per_case":120,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2600140882621399},{"case":0,"passes":2,"relative_equation_residual":0.03924649903976579},{"case":0,"passes":4,"relative_equation_residual":0.00047519448724172143},{"case":0,"passes":8,"relative_equation_residual":6.969770008590777e-09},{"case":1,"passes":1,"relative_equation_residual":0.17210198613976216},{"case":1,"passes":2,"relative_equation_residual":0.016754779721416355},{"case":1,"passes":4,"relative_equation_residual":0.00011346207483406754},{"case":1,"passes":8,"relative_equation_residual":4.633669401593452e-09},{"case":2,"passes":1,"relative_equation_residual":0.17528545717125002},{"case":2,"passes":2,"relative_equation_residual":0.004580341341956362},{"case":2,"passes":4,"relative_equation_residual":3.8850154060014106e-06},{"case":2,"passes":8,"relative_equation_residual":8.039149430458794e-11},{"case":3,"passes":1,"relative_equation_residual":0.062234342578324296},{"case":3,"passes":2,"relative_equation_residual":0.0008943707125701129},{"case":3,"passes":4,"relative_equation_residual":3.2643117266479175e-07},{"case":3,"passes":8,"relative_equation_residual":2.2374800131904923e-11},{"case":4,"passes":1,"relative_equation_residual":0.030567664327693454},{"case":4,"passes":2,"relative_equation_residual":0.0004540487617618377},{"case":4,"passes":4,"relative_equation_residual":1.1449727544636066e-07},{"case":4,"passes":8,"relative_equation_residual":1.2041810258956855e-11},{"case":5,"passes":1,"relative_equation_residual":0.062233550814182706},{"case":5,"passes":2,"relative_equation_residual":0.0008944188699719708},{"case":5,"passes":4,"relative_equation_residual":3.259560625757018e-07},{"case":5,"passes":8,"relative_equation_residual":4.890111234820758e-12},{"case":6,"passes":1,"relative_equation_residual":0.003799402975141799},{"case":6,"passes":2,"relative_equation_residual":0.00012539391506391633},{"case":6,"passes":4,"relative_equation_residual":2.909575191172496e-08},{"case":6,"passes":8,"relative_equation_residual":1.7688176677245172e-10},{"case":7,"passes":1,"relative_equation_residual":0.16027958770461645},{"case":7,"passes":2,"relative_equation_residual":0.014799369007818725},{"case":7,"passes":4,"relative_equation_residual":8.645244782078293e-05},{"case":7,"passes":8,"relative_equation_residual":6.420159829863726e-08}],"update":960},{"exposures_per_case":130,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.26056825229554176},{"case":0,"passes":2,"relative_equation_residual":0.03615121703898664},{"case":0,"passes":4,"relative_equation_residual":0.00026756320111993176},{"case":0,"passes":8,"relative_equation_residual":2.8402334735660723e-08},{"case":1,"passes":1,"relative_equation_residual":0.16995190927131085},{"case":1,"passes":2,"relative_equation_residual":0.015305281970121975},{"case":1,"passes":4,"relative_equation_residual":7.456527169767899e-05},{"case":1,"passes":8,"relative_equation_residual":1.1055649451335955e-09},{"case":2,"passes":1,"relative_equation_residual":0.17161061461349686},{"case":2,"passes":2,"relative_equation_residual":0.0030147846987844656},{"case":2,"passes":4,"relative_equation_residual":7.834996225220898e-06},{"case":2,"passes":8,"relative_equation_residual":5.023072485385494e-11},{"case":3,"passes":1,"relative_equation_residual":0.06049791284378233},{"case":3,"passes":2,"relative_equation_residual":0.0007421688169100526},{"case":3,"passes":4,"relative_equation_residual":1.7594450016932677e-07},{"case":3,"passes":8,"relative_equation_residual":2.4815004997455466e-11},{"case":4,"passes":1,"relative_equation_residual":0.028582461604006668},{"case":4,"passes":2,"relative_equation_residual":0.0004083470220565611},{"case":4,"passes":4,"relative_equation_residual":9.49623730725278e-08},{"case":4,"passes":8,"relative_equation_residual":6.297096514164025e-11},{"case":5,"passes":1,"relative_equation_residual":0.0604979861040597},{"case":5,"passes":2,"relative_equation_residual":0.0007422602767254221},{"case":5,"passes":4,"relative_equation_residual":1.756004954076522e-07},{"case":5,"passes":8,"relative_equation_residual":7.126644652324289e-12},{"case":6,"passes":1,"relative_equation_residual":0.004540458008114219},{"case":6,"passes":2,"relative_equation_residual":5.230352574408897e-05},{"case":6,"passes":4,"relative_equation_residual":1.1625558336609896e-08},{"case":6,"passes":8,"relative_equation_residual":5.1855949585403775e-11},{"case":7,"passes":1,"relative_equation_residual":0.17116574756932007},{"case":7,"passes":2,"relative_equation_residual":0.017255164188072474},{"case":7,"passes":4,"relative_equation_residual":0.00011679447716841896},{"case":7,"passes":8,"relative_equation_residual":3.514267182741153e-08}],"update":1040},{"exposures_per_case":140,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2552808370156815},{"case":0,"passes":2,"relative_equation_residual":0.0301353635573858},{"case":0,"passes":4,"relative_equation_residual":8.171335582897316e-05},{"case":0,"passes":8,"relative_equation_residual":8.12306738423214e-09},{"case":1,"passes":1,"relative_equation_residual":0.16673478939850944},{"case":1,"passes":2,"relative_equation_residual":0.012771195447195896},{"case":1,"passes":4,"relative_equation_residual":2.995521819727401e-05},{"case":1,"passes":8,"relative_equation_residual":1.1307331131680398e-10},{"case":2,"passes":1,"relative_equation_residual":0.17248438591232643},{"case":2,"passes":2,"relative_equation_residual":0.002115224174109277},{"case":2,"passes":4,"relative_equation_residual":9.504481642582477e-07},{"case":2,"passes":8,"relative_equation_residual":1.5043756785720324e-11},{"case":3,"passes":1,"relative_equation_residual":0.06118559657830807},{"case":3,"passes":2,"relative_equation_residual":0.001122798837122571},{"case":3,"passes":4,"relative_equation_residual":4.202995113808641e-07},{"case":3,"passes":8,"relative_equation_residual":2.090561368990864e-11},{"case":4,"passes":1,"relative_equation_residual":0.027698267786329996},{"case":4,"passes":2,"relative_equation_residual":0.00035743759052051573},{"case":4,"passes":4,"relative_equation_residual":4.6838926964004845e-08},{"case":4,"passes":8,"relative_equation_residual":3.248225896767353e-11},{"case":5,"passes":1,"relative_equation_residual":0.06118574070976961},{"case":5,"passes":2,"relative_equation_residual":0.0011228416007480339},{"case":5,"passes":4,"relative_equation_residual":4.2023799076405836e-07},{"case":5,"passes":8,"relative_equation_residual":5.613612440825041e-12},{"case":6,"passes":1,"relative_equation_residual":0.006767198075404145},{"case":6,"passes":2,"relative_equation_residual":4.174628241379684e-05},{"case":6,"passes":4,"relative_equation_residual":2.361078155280158e-08},{"case":6,"passes":8,"relative_equation_residual":7.348767802051148e-11},{"case":7,"passes":1,"relative_equation_residual":0.17700832310105977},{"case":7,"passes":2,"relative_equation_residual":0.019624239253231156},{"case":7,"passes":4,"relative_equation_residual":0.0001428950787148241},{"case":7,"passes":8,"relative_equation_residual":6.491852153250079e-08}],"update":1120},{"exposures_per_case":150,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2721407771334936},{"case":0,"passes":2,"relative_equation_residual":0.03779596549829729},{"case":0,"passes":4,"relative_equation_residual":0.00020512509128945962},{"case":0,"passes":8,"relative_equation_residual":4.3299331697649806e-08},{"case":1,"passes":1,"relative_equation_residual":0.18018660689774763},{"case":1,"passes":2,"relative_equation_residual":0.01645903777929707},{"case":1,"passes":4,"relative_equation_residual":6.99695752290676e-05},{"case":1,"passes":8,"relative_equation_residual":5.276319826168649e-10},{"case":2,"passes":1,"relative_equation_residual":0.17183090391443545},{"case":2,"passes":2,"relative_equation_residual":0.004339215216666013},{"case":2,"passes":4,"relative_equation_residual":1.1933851421073517e-05},{"case":2,"passes":8,"relative_equation_residual":1.2846142135022693e-10},{"case":3,"passes":1,"relative_equation_residual":0.05329594821905461},{"case":3,"passes":2,"relative_equation_residual":0.0004325044665141923},{"case":3,"passes":4,"relative_equation_residual":8.017040079575765e-08},{"case":3,"passes":8,"relative_equation_residual":1.6384465680659043e-11},{"case":4,"passes":1,"relative_equation_residual":0.022807380270941497},{"case":4,"passes":2,"relative_equation_residual":0.0003107210757718616},{"case":4,"passes":4,"relative_equation_residual":4.254566212931814e-08},{"case":4,"passes":8,"relative_equation_residual":1.7784710537899378e-11},{"case":5,"passes":1,"relative_equation_residual":0.05329853379476292},{"case":5,"passes":2,"relative_equation_residual":0.0004325661386530762},{"case":5,"passes":4,"relative_equation_residual":7.996432618245253e-08},{"case":5,"passes":8,"relative_equation_residual":1.6755098372564587e-11},{"case":6,"passes":1,"relative_equation_residual":0.00165657449334698},{"case":6,"passes":2,"relative_equation_residual":2.9609095268706607e-05},{"case":6,"passes":4,"relative_equation_residual":9.838324127937685e-09},{"case":6,"passes":8,"relative_equation_residual":2.048246357568828e-11},{"case":7,"passes":1,"relative_equation_residual":0.16603762798007754},{"case":7,"passes":2,"relative_equation_residual":0.015183142518767052},{"case":7,"passes":4,"relative_equation_residual":7.243977404789706e-05},{"case":7,"passes":8,"relative_equation_residual":7.647227763981438e-09}],"update":1200},{"exposures_per_case":160,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2690795460751867},{"case":0,"passes":2,"relative_equation_residual":0.03535351893983212},{"case":0,"passes":4,"relative_equation_residual":0.00012114908933562298},{"case":0,"passes":8,"relative_equation_residual":2.6177836326912695e-08},{"case":1,"passes":1,"relative_equation_residual":0.17788424905971342},{"case":1,"passes":2,"relative_equation_residual":0.015558649790459938},{"case":1,"passes":4,"relative_equation_residual":5.1845201818231e-05},{"case":1,"passes":8,"relative_equation_residual":6.16169842532625e-11},{"case":2,"passes":1,"relative_equation_residual":0.1711388776706489},{"case":2,"passes":2,"relative_equation_residual":0.003384246722380209},{"case":2,"passes":4,"relative_equation_residual":6.8924274812778034e-06},{"case":2,"passes":8,"relative_equation_residual":6.251575077401899e-11},{"case":3,"passes":1,"relative_equation_residual":0.053247144453305105},{"case":3,"passes":2,"relative_equation_residual":0.00028296537593980645},{"case":3,"passes":4,"relative_equation_residual":3.0274637497194064e-08},{"case":3,"passes":8,"relative_equation_residual":2.416085367871321e-11},{"case":4,"passes":1,"relative_equation_residual":0.0225334247245489},{"case":4,"passes":2,"relative_equation_residual":0.00025419917178928645},{"case":4,"passes":4,"relative_equation_residual":2.0180949043198506e-08},{"case":4,"passes":8,"relative_equation_residual":2.8414149986142715e-11},{"case":5,"passes":1,"relative_equation_residual":0.05324644927392382},{"case":5,"passes":2,"relative_equation_residual":0.0002829726783031156},{"case":5,"passes":4,"relative_equation_residual":2.879058946345714e-08},{"case":5,"passes":8,"relative_equation_residual":7.88758339367898e-12},{"case":6,"passes":1,"relative_equation_residual":0.0007533036721146494},{"case":6,"passes":2,"relative_equation_residual":6.986301440565505e-06},{"case":6,"passes":4,"relative_equation_residual":7.257896334172964e-09},{"case":6,"passes":8,"relative_equation_residual":4.757775529176048e-11},{"case":7,"passes":1,"relative_equation_residual":0.16777180319612506},{"case":7,"passes":2,"relative_equation_residual":0.016287725574029518},{"case":7,"passes":4,"relative_equation_residual":7.927049759803308e-05},{"case":7,"passes":8,"relative_equation_residual":1.712323324550867e-08}],"update":1280},{"exposures_per_case":170,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2720176610734196},{"case":0,"passes":2,"relative_equation_residual":0.03530491275511647},{"case":0,"passes":4,"relative_equation_residual":0.00011120481038289173},{"case":0,"passes":8,"relative_equation_residual":2.230343419279198e-08},{"case":1,"passes":1,"relative_equation_residual":0.17989758636162528},{"case":1,"passes":2,"relative_equation_residual":0.015851281144194446},{"case":1,"passes":4,"relative_equation_residual":5.709890202777165e-05},{"case":1,"passes":8,"relative_equation_residual":9.510077104100814e-11},{"case":2,"passes":1,"relative_equation_residual":0.17273705247270796},{"case":2,"passes":2,"relative_equation_residual":0.0032683854405136463},{"case":2,"passes":4,"relative_equation_residual":1.2296485318425632e-06},{"case":2,"passes":8,"relative_equation_residual":3.1330134273196175e-11},{"case":3,"passes":1,"relative_equation_residual":0.05338344693458649},{"case":3,"passes":2,"relative_equation_residual":0.00024281939014884034},{"case":3,"passes":4,"relative_equation_residual":3.638541459672104e-08},{"case":3,"passes":8,"relative_equation_residual":3.399654909885017e-11},{"case":4,"passes":1,"relative_equation_residual":0.021629669959706097},{"case":4,"passes":2,"relative_equation_residual":0.0001866876757338913},{"case":4,"passes":4,"relative_equation_residual":1.1360757598319778e-08},{"case":4,"passes":8,"relative_equation_residual":1.6696913506433297e-11},{"case":5,"passes":1,"relative_equation_residual":0.05338423818833619},{"case":5,"passes":2,"relative_equation_residual":0.00024276754856858632},{"case":5,"passes":4,"relative_equation_residual":3.387182907421215e-08},{"case":5,"passes":8,"relative_equation_residual":8.136825676807022e-12},{"case":6,"passes":1,"relative_equation_residual":0.0016508275725227374},{"case":6,"passes":2,"relative_equation_residual":1.5238835187369466e-05},{"case":6,"passes":4,"relative_equation_residual":5.219641459194769e-09},{"case":6,"passes":8,"relative_equation_residual":3.584455649026481e-11},{"case":7,"passes":1,"relative_equation_residual":0.16858246227380821},{"case":7,"passes":2,"relative_equation_residual":0.01639623643887104},{"case":7,"passes":4,"relative_equation_residual":7.43285751673238e-05},{"case":7,"passes":8,"relative_equation_residual":3.0916468632099654e-08}],"update":1360},{"exposures_per_case":180,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2739092938185471},{"case":0,"passes":2,"relative_equation_residual":0.03537596299682752},{"case":0,"passes":4,"relative_equation_residual":0.00010672674518731585},{"case":0,"passes":8,"relative_equation_residual":2.053013853238037e-08},{"case":1,"passes":1,"relative_equation_residual":0.18094721908009007},{"case":1,"passes":2,"relative_equation_residual":0.0159274619980926},{"case":1,"passes":4,"relative_equation_residual":5.6091061898244934e-05},{"case":1,"passes":8,"relative_equation_residual":5.5566011139475013e-11},{"case":2,"passes":1,"relative_equation_residual":0.17348542981457102},{"case":2,"passes":2,"relative_equation_residual":0.0033534407207374307},{"case":2,"passes":4,"relative_equation_residual":1.2043890023501191e-06},{"case":2,"passes":8,"relative_equation_residual":1.616799686864215e-11},{"case":3,"passes":1,"relative_equation_residual":0.052754198165895924},{"case":3,"passes":2,"relative_equation_residual":0.00021155010245985868},{"case":3,"passes":4,"relative_equation_residual":4.143239163773448e-08},{"case":3,"passes":8,"relative_equation_residual":3.89667572398611e-11},{"case":4,"passes":1,"relative_equation_residual":0.021425040313095063},{"case":4,"passes":2,"relative_equation_residual":0.000183756432088748},{"case":4,"passes":4,"relative_equation_residual":7.660101127557047e-09},{"case":4,"passes":8,"relative_equation_residual":1.6860216519133026e-11},{"case":5,"passes":1,"relative_equation_residual":0.05275464731746003},{"case":5,"passes":2,"relative_equation_residual":0.0002115748107472361},{"case":5,"passes":4,"relative_equation_residual":3.7705752978157645e-08},{"case":5,"passes":8,"relative_equation_residual":1.1512507050039298e-11},{"case":6,"passes":1,"relative_equation_residual":0.0015863629436362067},{"case":6,"passes":2,"relative_equation_residual":8.183575716068605e-06},{"case":6,"passes":4,"relative_equation_residual":6.123996569416405e-09},{"case":6,"passes":8,"relative_equation_residual":4.776866627578806e-11},{"case":7,"passes":1,"relative_equation_residual":0.16792964305289798},{"case":7,"passes":2,"relative_equation_residual":0.016205208152010393},{"case":7,"passes":4,"relative_equation_residual":6.882575711557923e-05},{"case":7,"passes":8,"relative_equation_residual":3.64180035948177e-08}],"update":1440},{"exposures_per_case":190,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.2755583117988859},{"case":0,"passes":2,"relative_equation_residual":0.035926753704134076},{"case":0,"passes":4,"relative_equation_residual":0.0001123655670834389},{"case":0,"passes":8,"relative_equation_residual":2.3115046112060067e-08},{"case":1,"passes":1,"relative_equation_residual":0.18213466347684087},{"case":1,"passes":2,"relative_equation_residual":0.01617974710080727},{"case":1,"passes":4,"relative_equation_residual":6.024552077969955e-05},{"case":1,"passes":8,"relative_equation_residual":1.6054488651444654e-10},{"case":2,"passes":1,"relative_equation_residual":0.17406789771885778},{"case":2,"passes":2,"relative_equation_residual":0.003576730780083353},{"case":2,"passes":4,"relative_equation_residual":1.242337122641861e-06},{"case":2,"passes":8,"relative_equation_residual":9.425153251642119e-11},{"case":3,"passes":1,"relative_equation_residual":0.05275837738494584},{"case":3,"passes":2,"relative_equation_residual":0.000177908021832069},{"case":3,"passes":4,"relative_equation_residual":3.169695250888867e-08},{"case":3,"passes":8,"relative_equation_residual":1.4326429347182816e-11},{"case":4,"passes":1,"relative_equation_residual":0.021199835132689072},{"case":4,"passes":2,"relative_equation_residual":0.00018667890971233077},{"case":4,"passes":4,"relative_equation_residual":1.111337142476413e-08},{"case":4,"passes":8,"relative_equation_residual":1.0724768768041802e-11},{"case":5,"passes":1,"relative_equation_residual":0.052759615439323275},{"case":5,"passes":2,"relative_equation_residual":0.00017784697936765515},{"case":5,"passes":4,"relative_equation_residual":3.1026853372244894e-08},{"case":5,"passes":8,"relative_equation_residual":1.0623673237920104e-11},{"case":6,"passes":1,"relative_equation_residual":0.0016045973989228472},{"case":6,"passes":2,"relative_equation_residual":8.593450036630455e-06},{"case":6,"passes":4,"relative_equation_residual":7.668388921433695e-09},{"case":6,"passes":8,"relative_equation_residual":3.6718063645807767e-11},{"case":7,"passes":1,"relative_equation_residual":0.16775200561160644},{"case":7,"passes":2,"relative_equation_residual":0.016004989559422263},{"case":7,"passes":4,"relative_equation_residual":6.445462394329379e-05},{"case":7,"passes":8,"relative_equation_residual":3.459781212289969e-08}],"update":1520},{"exposures_per_case":200,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.27561099840643916},{"case":0,"passes":2,"relative_equation_residual":0.035937241960999323},{"case":0,"passes":4,"relative_equation_residual":0.00011242427013762363},{"case":0,"passes":8,"relative_equation_residual":2.3136037514508477e-08},{"case":1,"passes":1,"relative_equation_residual":0.18214054368396018},{"case":1,"passes":2,"relative_equation_residual":0.016180650097917762},{"case":1,"passes":4,"relative_equation_residual":6.031489091889812e-05},{"case":1,"passes":8,"relative_equation_residual":1.6606680130666122e-10},{"case":2,"passes":1,"relative_equation_residual":0.17407924628719768},{"case":2,"passes":2,"relative_equation_residual":0.0035856173029973743},{"case":2,"passes":4,"relative_equation_residual":1.249613694016568e-06},{"case":2,"passes":8,"relative_equation_residual":2.3704695458704157e-11},{"case":3,"passes":1,"relative_equation_residual":0.052764948567511044},{"case":3,"passes":2,"relative_equation_residual":0.0001725234829189855},{"case":3,"passes":4,"relative_equation_residual":3.154933834107116e-08},{"case":3,"passes":8,"relative_equation_residual":2.245046474205739e-11},{"case":4,"passes":1,"relative_equation_residual":0.021168562698451993},{"case":4,"passes":2,"relative_equation_residual":0.00018536126170264397},{"case":4,"passes":4,"relative_equation_residual":1.0920170717941838e-08},{"case":4,"passes":8,"relative_equation_residual":4.869329213540277e-12},{"case":5,"passes":1,"relative_equation_residual":0.05276503856903396},{"case":5,"passes":2,"relative_equation_residual":0.00017253549389685365},{"case":5,"passes":4,"relative_equation_residual":3.00980483689008e-08},{"case":5,"passes":8,"relative_equation_residual":8.26406710041509e-12},{"case":6,"passes":1,"relative_equation_residual":0.0016332660683210448},{"case":6,"passes":2,"relative_equation_residual":1.0081166282015955e-05},{"case":6,"passes":4,"relative_equation_residual":5.331260008066254e-09},{"case":6,"passes":8,"relative_equation_residual":4.369450510302322e-11},{"case":7,"passes":1,"relative_equation_residual":0.16767419533171918},{"case":7,"passes":2,"relative_equation_residual":0.01599705530640498},{"case":7,"passes":4,"relative_equation_residual":6.43857453801095e-05},{"case":7,"passes":8,"relative_equation_residual":3.4801583064277504e-08}],"update":1600}]},"rollout_feedback":{"auxiliary_losses":0,"depth_exposures":[[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50]],"depth_order_sha256":"3adbd1b18830b55ac918ce265241ded42841f53a56341f43b1e333ebd0615257","mode":"U","original_sampling_order_sha256":"d0ea221a7e03a83673ece7fc9cdf4f5212bb54f3023511f93d95c84e45647899"},"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":1600,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"8cd2a0d4a694f44f393ebc8381a407e4195fd9361f1e115060104bd0df60ac8f"},"checkpoint-1600.npz":{"bytes":1485452,"role":"checkpoint","sha256":"ddad267c2eaff2476d8ee10cee4ce6ff9aff686f3ff54a03e89d83baa6e5731c"},"history.json":{"bytes":10171,"role":"training_diagnostics","sha256":"5d854f5020fa1aed846ebefd20337ee0c3ca04e8055bae7789eebee3bb000ed0"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"466562bff40e800e987b8827c55d6616096c6f81e1fa2c84302398375181d537"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":477,"role":"result_summary","sha256":"1f9c41b3f4209b23b58f0ead9ae99371953fffc94125021a7b05896083fa7cca"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"ddad267c2eaff2476d8ee10cee4ce6ff9aff686f3ff54a03e89d83baa6e5731c"}},"per_case_inference_s":[0.021669929003110155,0.004286265990231186,0.005263706989353523,0.0042174310074187815,0.003671255020890385,0.0034518539905548096,0.004369795991806313,0.004609766008798033,0.00970526397577487,0.0053674989903811365,0.006424981023883447,0.005779523984529078],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"ca11c8dd0e6a35e68cd75472ff625233731c7c8074988eb7cecc414f039d7ef3","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"256b4ebf415c2cd649e838d9211be7828660e444c98904a5a7f5d6f2d991d7fd","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"ee7fc05cb1a04448d45bc7435a1efadba4228c2b49e1990495edc398efc324a9","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"301e0e1d8d8c80771068a81243c9ee493cea77590af7766030f574f0052a4dbe","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"6d9627b873c183258eec0deb9c6564412265ded25e001a6d9f0c724c2d5dad92","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"355736c8b6200aee5ef1eca68a038d9783441a6880b1773b050619b8b5f30085","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"52fa0917983e71646f420406f761913a664ce6861b887f0c53cba6c8309d8a46","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"52e49ac9765bcfaba75afe28f387a52f444b97e41031ae01721ec2a0c9c864d3","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"dbf945dfcd8053f916a36dc76b0ff4b6493a17eee844a3cc231a4d5d6ec181cf","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"01e7ad8419d6f0417626ce14bd989b913ecba5e355e16abf3f5600ec5449d12d","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"d5db7e085cbc20ef454eadffdbf44179475ed2572a0286edb1bb0ac75583c8a8","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"d5103e7d4ae4e876547340d666fb0b12cb46131eee64b3c5663cf87dcd746aad","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2413834240,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```


## Follow-up: matched pressure correction budgets

Authored follow-up from the independently rescored 12-checkpoint CUDA matrix. The saved C37 parent uses one pressure correction and this fine-tune uses two. The original default-to-default result therefore combines training and additional work. At matched two/four/eight corrections this fine-tune improves both worst equation and mean-free pressure relative L2 errors. At one correction it improves worst equation error but worsens worst pressure error. These statements concern 128 consumed original pressure equations, not physical time steps, unseen scenes, drift or timing.

| Corrections | Parent worst pressure relative L2 | Fine-tune worst pressure relative L2 |
| --- | ---: | ---: |
| 1 | 0.053822136387410319 | 0.060227956262864098 |
| 2 | 0.015001408901349057 | 0.011298824976523078 |
| 4 | 0.0015792885802435198 | 0.00064562124658342282 |
| 8 | 3.0667357052689975e-05 | 8.6982921816512088e-06 |

Independent evidence: docs/plans/model-portfolio-migration-v1/reports/pressure-correction-work-01.json SHA256 c0dfc8734ac41561d0b4e3138e74b89da51aa54b39b5c275d493a5de1df35033.
