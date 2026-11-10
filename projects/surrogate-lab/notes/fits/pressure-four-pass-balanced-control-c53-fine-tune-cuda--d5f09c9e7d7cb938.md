# Pressure four-pass · balanced-control · C53 fine-tune CUDA

Retrospective source review · fit-d5f09c9e7d7cb938

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 370,016-parameter linear one-channel F32 multiscale pressure CNN initialized from the exact C53 checkpoint. F64 equation assembly scales the residual by rho/h before train-only normalization; shared learned corrections are gauged and accumulated in F64. Fine-tuning explicitly owns its recurrence separately from the captured default pressure inference function. The teacher is a frozen copy of the warm checkpoint, has no optimizer and is absent from deployed inference. These are internal pressure passes, not simultaneous physical simulation steps.

**Purpose.** Establish a new-seed balanced manufactured sampling control for the final low-mode objective comparison.

**Experiment.** 1,500 CUDA Adam updates at initial rate 1e-5, dropping at updates 1000/1350 to 1e-6/1e-7; batch one. Model seed 20261019, physical sample seed 2026101901. Sixteen physical training trajectories/32 frames at 8/16 cells per metre, dt=1/60 s and rho=1000 kg/m3 retain the two manufactured banks and two captured-direction banks. All five streams use 4 student corrections, totaling 30,000 student root forwards. The loss adds log1p(final relative equation squared/1e-12), plus total weight 0.05 over earlier correction stages. Frozen eight-correction teacher pressure has weight 0.1; its cache records 15,400 teacher neural forwards and 5,575 hits. Final low-spatial-mode feedback weight is 0; it does not add a deployed neural pass.

**Observed outcome.** Retained final logged fitting loss 9.3057805933833784. The common 128 consumed requests exercise the captured default TWO-correction deployment: worst mean-free pressure relative L2 0.0066126764718811492, equation relative L2 0.0043937520327551665; median case-median complete GPU pressure call 4.3667747500000003 ms. Its CPU input/output, packing, transfers and synchronization are included; setup/scoring are excluded. Possible host metadata overlap is recorded and its timing effect is unmeasured. Four-correction fitting labels do not establish four-correction benchmark coverage. The native fitting diagnostic is train-only, with one already-consumed wrapper probe. No new inference, full-step or drift claim is made.

**Lesson.** Report training recurrence, default inference recurrence and teacher work separately. Four-pass fitting can change the weights while the saved default model still executes two pressure passes; a label alone cannot identify the measured variant. Comparing C57/C58 isolates fitting recurrence under their shared sample order; C58/C59 additionally isolates teacher supervision. C60/C61 share a later sample stream for the final low-mode ablation, so C59/C60 also changes sampling. A teacher cache saves repeated training work without changing inference cost or adding independent simulation scenes.

**Reproduction.** Use the exact native evidence recipe, fine-tuning contract, C53 initial tensors, original source/config/runtime and all fixed input banks. Preserve model/physical/manufactured/direction sampling streams, selected final checkpoint, teacher cache statistics, training-only objective and explicitly chosen deployment correction count. Historical publication/restore includes CPU neural queries; qualify a GPU-only equivalent before replay. Retain the original diagnostic and configuration suffix unchanged. Clean training, fresh four-correction common benchmarking, new-Rheon regeneration and 600-output own-history drift remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-low-mode-tail-v1/balanced-control-study-01/train/receipt.json",
    "sha256": "d5f09c9e7d7cb93834fe8a38d69a37d1a1ab58785a8970ab62dedc60d9666e9c"
  },
  "history": {
    "path": "runs/pressure-low-mode-tail-v1/balanced-control-study-01/train/history.json",
    "sha256": "ee0807dce9ab651602d6644ae5502b300ded1b8dae970949232da7ebfa7d633d"
  },
  "sources": [
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/config.json",
      "sha256": "c374bcbfdd9d13e3e2589895f12db3473537154fedba5cd26138bbc74e35a5f8"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/fine-tuning.json",
      "sha256": "90e46677dd91d29c4c5038b07702b0ca11e4515eb2bc736e26637add94158577"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/fine_tuning.py",
      "sha256": "376c4f72945704f533c5bd804a40c71560d089fc7bb1f8f822a102654a6ceaaa"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/model.py",
      "sha256": "a2ebefe6a8ce12c7e6e1e12fa82b835029f364b50e91284f33b60bfe31b0eaee"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/original_fine_tuning.py",
      "sha256": "6e8d5e11744c0622f700bc645226d5dd96dbf831dd9bea754328c97dc70ecc12"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/train.py",
      "sha256": "18b5ae6232e7a00760288d37c015905363fdea4b97e0d1815bc12033e0d69e12"
    },
    {
      "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-low-mode-tail-v1/balanced-control-project-01/source-witness.json",
    "sha256": "9c164ccc6246af361af647148bfdaae588b5ec67e28c4bd673d332ef83085a52"
  },
  "training_population": {
    "path": "runs/pressure-low-mode-tail-v1/balanced-control-study-01/train/splits.json",
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

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-d5f09c9e7d7cb938",
    "kind": "training_attempt",
    "name": "Pressure four-pass · balanced-control · C53 fine-tune CUDA",
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
    "receipt_sha256": "d5f09c9e7d7cb93834fe8a38d69a37d1a1ab58785a8970ab62dedc60d9666e9c",
    "receipt_paths": [
      "runs/pressure-low-mode-tail-v1/balanced-control-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "b6ab6203750543a18ad61cd4de588cd4",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure four-pass · balanced-control · C53 fine-tune CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1500,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "c374bcbfdd9d13e3e2589895f12db3473537154fedba5cd26138bbc74e35a5f8",
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
      "sample_order_sha256": "6fb1c513e2db57f19de256b347099538cf1b84c52de210119f56660c15382c0a",
      "source_witness_sha256": "9c164ccc6246af361af647148bfdaae588b5ec67e28c4bd673d332ef83085a52"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-1500.npz",
        "sha256": "c8b72ef97d5c97dfdc4146ad0a37f757da947a2470e0612d1af6081a94c6f8e0",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "c8b72ef97d5c97dfdc4146ad0a37f757da947a2470e0612d1af6081a94c6f8e0",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2471321600,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/60"
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
      "receipt": "runs/pressure-low-mode-tail-v1/balanced-control-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-low-mode-tail-v1/balanced-control-study-01/train/receipt.json",
    "sha256": "d5f09c9e7d7cb93834fe8a38d69a37d1a1ab58785a8970ab62dedc60d9666e9c",
    "bytes": 10677
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C60": {
        "name": "balanced-control",
        "weights_sha256": "c8b72ef97d5c97dfdc4146ad0a37f757da947a2470e0612d1af6081a94c6f8e0",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.0043937520327551665,
          "pressure_mean_free_relative_l2": 0.006612676471881149,
          "pressure_absolute_rms": 0.22107254876930782,
          "pressure_absolute_max": 0.4788523705620135
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
              "common_equation_relative_l2": 0.002655439044992489,
              "pressure_mean_free_relative_l2": 0.006612676471881149,
              "pressure_absolute_rms": 0.22107254876930782,
              "pressure_absolute_max": 0.4788523705620135
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.0043937520327551665,
              "pressure_mean_free_relative_l2": 0.0055099010083436714,
              "pressure_absolute_rms": 0.1265619289306127,
              "pressure_absolute_max": 0.2765087435852974
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C60/results.json",
          "sha256": "d16d7905b65604e83d6203b482c7b73b3f44f463dabf90cf41c42ad69d5f84f0"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"balanced-control","early_weight":0.05,"initial_weights_sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954","low_mode_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"weight":0.0},"relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.1,"training_passes":4},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[13,15,16]},{"cells_per_metre":16,"counts_xyz":[9,11,11]},{"cells_per_metre":8,"counts_xyz":[15,9,7]},{"cells_per_metre":16,"counts_xyz":[11,5,15]},{"cells_per_metre":8,"counts_xyz":[7,16,5]},{"cells_per_metre":16,"counts_xyz":[11,12,7]},{"cells_per_metre":8,"counts_xyz":[5,13,9]},{"cells_per_metre":16,"counts_xyz":[16,7,11]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101907,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"c374bcbfdd9d13e3e2589895f12db3473537154fedba5cd26138bbc74e35a5f8","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"6fb1c513e2db57f19de256b347099538cf1b84c52de210119f56660c15382c0a","source_witness_sha256":"9c164ccc6246af361af647148bfdaae588b5ec67e28c4bd673d332ef83085a52"},"diagnostic_population":"train_only","elapsed_s":322.1315627459844,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":3.58146123673464e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":897565184,"peak_reserved_bytes":973078528,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":5575,"teacher_cache_peak_bytes":131831024,"teacher_forwards":15400},"heldout_evaluations":0,"mode":"train","optimizer_updates":1500,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"checkpoint-1500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"c8b72ef97d5c97dfdc4146ad0a37f757da947a2470e0612d1af6081a94c6f8e0"},"history.json":{"bytes":235791,"role":"training_diagnostics","sha256":"ee0807dce9ab651602d6644ae5502b300ded1b8dae970949232da7ebfa7d633d"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"54425bfb9407a060d3c2d773807470573062744e1720e890c2f0526962b08634"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":477,"role":"result_summary","sha256":"3f0bb8938e32eb3c45f64f6e2478b63768bd929fd1b3e8055248bbe7e981c5d9"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"c8b72ef97d5c97dfdc4146ad0a37f757da947a2470e0612d1af6081a94c6f8e0"}},"per_case_inference_s":[0.020601705997250974,0.003975740022724494,0.005000523990020156,0.003981264977483079,0.0034561670036055148,0.004915283992886543,0.0042166130151599646,0.004339903010986745,0.008841465984005481,0.004896875005215406,0.0059149400040041655,0.005296399991493672],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"4e187fa6d17dab009b709aac68ba488b528ebc703ad20eb885504a4fee3c3be0","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"69447462ce5a62ed67bf7388600f09f14458189bc99126a533c77ab11b472de8","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"f728918e39459cf4691648b635c76d4e8f5fc071c32106b0f9aaaa8921f3839c","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"55b8ccb11811913c8d3c12e0316afb1f18f51f082d63c61b98b8e42670a1f657","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"42ee9427ed183e00e67a984f0d6845b033a2bf8bca4e381231914d9b0c23ade7","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"075e3697a2764b895b913c50cb5f3c7a9f2f4c5740d6f17518a24b8b5023acf0","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"aa09e0969f972f0c15aa482ab7380477c8c706ab3bf92ace3904df9e6e2c2265","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"38e63af1512982449dee10115299f734549b4165b60714d87a2bf3c1b6c8ccf7","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"0e7d885b9aea239a1db2446686ecd31e8190fe8329dab726f174f83a8f9aca28","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"539403e6e53d61aa926101101dbea0bc9af30227b35aeca0648032815d15de7c","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"5bab606df0435c461610c5f7db3a1f7cc0ced1c50d1bb370e6b1b286a8737004","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"fa134d4b7848dd6b07e03ad5733261cdde768b4e85e5b7e35e9e89d35ea3d9e8","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2471321600,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
