# Pressure four-pass · four-pass-physics · C53 fine-tune CUDA

Retrospective source review · fit-e3883f407b0f71ad

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 370,016-parameter linear one-channel F32 multiscale pressure CNN initialized from the exact C53 checkpoint. F64 equation assembly scales the residual by rho/h before train-only normalization; shared learned corrections are gauged and accumulated in F64. Fine-tuning explicitly owns its recurrence separately from the captured default pressure inference function. The teacher is a frozen copy of the warm checkpoint, has no optimizer and is absent from deployed inference. These are internal pressure passes, not simultaneous physical simulation steps.

**Purpose.** Increase differentiable fitting recurrence from two to four corrections while retaining the same warm start, physical samples, budget and no teacher loss.

**Experiment.** 1,500 CUDA Adam updates at initial rate 1e-5, dropping at updates 1000/1350 to 1e-6/1e-7; batch one. Model seed 20261018, physical sample seed 2026101801. Sixteen physical training trajectories/32 frames at 8/16 cells per metre, dt=1/60 s and rho=1000 kg/m3 retain the two manufactured banks and two captured-direction banks. All five streams use 4 student corrections, totaling 30,000 student root forwards. The loss adds log1p(final relative equation squared/1e-12), plus total weight 0.05 over earlier correction stages. Frozen eight-correction teacher pressure has weight 0; its cache records 0 teacher neural forwards and 0 hits. Final low-spatial-mode feedback weight is 0; it does not add a deployed neural pass.

**Observed outcome.** Retained final logged fitting loss 10.852934560405114. The common 128 consumed requests exercise the captured default TWO-correction deployment: worst mean-free pressure relative L2 0.0052917722873446196, equation relative L2 0.0036007486326202921; median case-median complete GPU pressure call 4.4410980000000002 ms. Its CPU input/output, packing, transfers and synchronization are included; setup/scoring are excluded. Possible host metadata overlap is recorded and its timing effect is unmeasured. Four-correction fitting labels do not establish four-correction benchmark coverage. The native fitting diagnostic is train-only, with one already-consumed wrapper probe. No new inference, full-step or drift claim is made.

**Lesson.** Report training recurrence, default inference recurrence and teacher work separately. Four-pass fitting can change the weights while the saved default model still executes two pressure passes; a label alone cannot identify the measured variant. Comparing C57/C58 isolates fitting recurrence under their shared sample order; C58/C59 additionally isolates teacher supervision. C60/C61 share a later sample stream for the final low-mode ablation, so C59/C60 also changes sampling. A teacher cache saves repeated training work without changing inference cost or adding independent simulation scenes.

**Reproduction.** Use the exact native evidence recipe, fine-tuning contract, C53 initial tensors, original source/config/runtime and all fixed input banks. Preserve model/physical/manufactured/direction sampling streams, selected final checkpoint, teacher cache statistics, training-only objective and explicitly chosen deployment correction count. Historical publication/restore includes CPU neural queries; qualify a GPU-only equivalent before replay. Retain the original diagnostic and configuration suffix unchanged. Clean training, fresh four-correction common benchmarking, new-Rheon regeneration and 600-output own-history drift remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-study-01/train/receipt.json",
    "sha256": "e3883f407b0f71ad3f84ff3c7052d5dff568b33a4a2aee4458a474a0b5f5f6f5"
  },
  "history": {
    "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-study-01/train/history.json",
    "sha256": "3a7416a76571e7952fa6b1f48700286a83ec0ad1145a59057789fee115fef112"
  },
  "sources": [
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/config.json",
      "sha256": "189f900062879182057dc51cf08d78d9c8a18be26b70d19b54dcf74bb433bb5e"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/fine-tuning.json",
      "sha256": "fff4ea649e66cfc0a040992ca87688f5221840175efcb6392dd40a80ed34c041"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/fine_tuning.py",
      "sha256": "0e3b2882e9b4e981adc2bc14783e10b24d665048c08b814340d9b5e642cfb96f"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/model.py",
      "sha256": "a2ebefe6a8ce12c7e6e1e12fa82b835029f364b50e91284f33b60bfe31b0eaee"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/train.py",
      "sha256": "18b5ae6232e7a00760288d37c015905363fdea4b97e0d1815bc12033e0d69e12"
    },
    {
      "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-project-01/source-witness.json",
    "sha256": "54e18b54dca1c38fbf04e2aa1890547b8db4f83805bcd2ebf116f0a71c416d86"
  },
  "training_population": {
    "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-study-01/train/splits.json",
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
    "item_id": "fit-e3883f407b0f71ad",
    "kind": "training_attempt",
    "name": "Pressure four-pass · four-pass-physics · C53 fine-tune CUDA",
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
    "receipt_sha256": "e3883f407b0f71ad3f84ff3c7052d5dff568b33a4a2aee4458a474a0b5f5f6f5",
    "receipt_paths": [
      "runs/pressure-four-pass-distillation-v1/four-pass-physics-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "474749c5b43c481f941b7d022d0e7a55",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure four-pass · four-pass-physics · C53 fine-tune CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1500,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "189f900062879182057dc51cf08d78d9c8a18be26b70d19b54dcf74bb433bb5e",
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
      "sample_order_sha256": "c3b90c66310b33f7da786d3c8102b12c8c04574840da9d1b9f8555abbe627128",
      "source_witness_sha256": "54e18b54dca1c38fbf04e2aa1890547b8db4f83805bcd2ebf116f0a71c416d86"
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
        "sha256": "be22b1b31f95c0182cb5506439199a5d5478e11771c021476bfffc7435da861e",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "be22b1b31f95c0182cb5506439199a5d5478e11771c021476bfffc7435da861e",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2464268288,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/58"
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
      "receipt": "runs/pressure-four-pass-distillation-v1/four-pass-physics-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-four-pass-distillation-v1/four-pass-physics-study-01/train/receipt.json",
    "sha256": "e3883f407b0f71ad3f84ff3c7052d5dff568b33a4a2aee4458a474a0b5f5f6f5",
    "bytes": 10504
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C58": {
        "name": "four-pass-physics",
        "weights_sha256": "be22b1b31f95c0182cb5506439199a5d5478e11771c021476bfffc7435da861e",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.003600748632620292,
          "pressure_mean_free_relative_l2": 0.00529177228734462,
          "pressure_absolute_rms": 0.21631472506839305,
          "pressure_absolute_max": 0.48753434260361495
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
            "trajectory_id": "baseline--water-validation-0054",
            "native_frame": 15,
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
              "common_equation_relative_l2": 0.00267840743110739,
              "pressure_mean_free_relative_l2": 0.00529177228734462,
              "pressure_absolute_rms": 0.21631472506839305,
              "pressure_absolute_max": 0.48753434260361495
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.003600748632620292,
              "pressure_mean_free_relative_l2": 0.004760045774349322,
              "pressure_absolute_rms": 0.11941433429547338,
              "pressure_absolute_max": 0.24907133484103522
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C58/results.json",
          "sha256": "6ddd922c80ba4e0a2bd374559ef2c8a52a5e58917feeafb09073e0276ffd9cd7"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"four-pass-physics","early_weight":0.05,"initial_weights_sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":4},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"189f900062879182057dc51cf08d78d9c8a18be26b70d19b54dcf74bb433bb5e","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"c3b90c66310b33f7da786d3c8102b12c8c04574840da9d1b9f8555abbe627128","source_witness_sha256":"54e18b54dca1c38fbf04e2aa1890547b8db4f83805bcd2ebf116f0a71c416d86"},"diagnostic_population":"train_only","elapsed_s":292.33861228098976,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":5.770131990345817e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":844437504,"peak_reserved_bytes":905969664,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":1500,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"checkpoint-1500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"be22b1b31f95c0182cb5506439199a5d5478e11771c021476bfffc7435da861e"},"history.json":{"bytes":209243,"role":"training_diagnostics","sha256":"3a7416a76571e7952fa6b1f48700286a83ec0ad1145a59057789fee115fef112"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"809d12ee05fad09c5e04f75e3eccd9a4c28ab0d189054bc956024beef627b91b"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":478,"role":"result_summary","sha256":"7b40682f8d4f29d41a3a0cf9f10c05c2dfcc1b3337600fb9ee019a151de5f1ed"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"be22b1b31f95c0182cb5506439199a5d5478e11771c021476bfffc7435da861e"}},"per_case_inference_s":[0.021722813980886713,0.004100635997019708,0.005223695014137775,0.004111539979930967,0.003647633013315499,0.0034085080260410905,0.004281728994101286,0.004495255998335779,0.009407845005625859,0.005192201002500951,0.006274793995544314,0.0056232259958051145],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"fe9e6a6ec3d71da22b734e4cc21b3cf5c7753f627f2fbf9e49d40ac938c39ae7","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"14b6989e6a06a69601ecafe6e6938ab038c0e271c1dffca7133aefa7cdd391ff","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"4a2cf6e937e626f21cd881aaf0fa35b3e9a77c15ef49c1fd4d2c6734c9c611f3","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"f7c0d0ad1c04e209f2819f6008cc37806cb285860746fa2f74b51f3dfbfa6b37","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"c5eb57d0ed22050ac04b82fbc923f263d32e9610f90ebeeff17af3f01871cabd","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"5264adfaa3f948feb28dd7fde11fe68d9758e710f2d0430fddeb384f82e751b0","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"2935babe05736f425cd39265c99b798d2abcc533f6a440205f4535e09aed9afc","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"4535bf4f328d831dd54054f64ff34a244fc616239334315035eb5c9aaf56cdb5","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"f22a2772c4fb9297b9a0ec58a4cd4e8f146b7399bf271cca2c2682f6a00cda95","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"c4d18ec9f54024f9ddfc4e9384d19ab079c1ec00ea51280c7cc1959c2ba94966","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"bfd3a1940d3d39b90dbb47f95a933e6033d563391919a67d64b771f8286e5b8e","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"5df0efc2fa810f5a78a9c714c2cda1184c3252985ea015ca58ca4a70365e2af2","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2464268288,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
