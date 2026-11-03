# Pressure four-pass · axis-isolation · C53 fine-tune CUDA

Retrospective source review · fit-b4faace130aadce2

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 370,016-parameter linear, bias-free one-channel F32 multiscale pressure CNN warm-started from exact C53 tensors. F64 sealed cubic-grid equation assembly uses q=(rho/h)r and fixed train-only normalization. Shared neural corrections are gauged and accumulated in F64. The fitting objective unrolls four pressure corrections with final logarithmic equation loss and total earlier-stage weight 0.05; the captured default inference still executes two. These passes do not advance physical simulation time.

**Purpose.** Isolate three axis fundamentals on the single 13x15x16 grid.

**Experiment.** 600 CUDA Adam updates; model seed 20261020, retained physical-index seed 2026102001, initial rate 0.0001 and schedule [{'update': 301, 'learning_rate': 1e-05}, {'update': 451, 'learning_rate': 1e-06}, {'update': 526, 'learning_rate': 1e-07}, {'update': 571, 'learning_rate': 1e-08}]. Original 16 physical trajectories/32 frames calibrate fixed normalization and manufactured pressure amplitude, but no physical scene frame enters the optimizer. Actual fitting uses 3 manufactured fundamental fields and 600 example presentations; all inherited auxiliary banks are disabled. There are 600 optimizer updates and 2,400 student neural root forwards. Physical metadata remains dt=1/60 s and fluid density 1000 kg/m3; grid density means cells per metre, not fluid density. Calibration and consumed diagnostics add no independent optimizer scenes.

**Observed outcome.** Retained final fitting loss 0.27159059603992364. Common 128 consumed pressure requests use default TWO-correction inference: worst mean-free pressure relative L2 0.017042723565476883, equation relative L2 0.0065443352337962845, median case-median complete GPU pressure call 4.4621674999999996 ms. Host input/output, transfers and synchronization are included; setup/scoring are excluded. Fitting-bank 1/2/4/8-correction traces are retained consumed-case equation diagnostics, not independent pressure or full-step benchmarks. No new model execution, fresh four-correction common benchmark or rollout result is added.

**Lesson.** A physical data manifest does not establish physical optimizer exposure. Report calibration data, actual optimizer population, update aggregation, fitting recurrence and measured inference recurrence separately.

**Reproduction.** Retain exact typed native source/config/runtime/input receipts, C53 warm tensors, fine-tuning contract, normalization, final checkpoint and the actual optimizer owner. Recreate manufactured fundamental fields from explicit grid/mode/density definitions and original train-calibrated amplitude; reproduce private balanced cycles and gradient aggregation. For the curriculum preserve its repeated patterns and supplemental category order. Historical publication/restore uses CPU neural queries; qualify a GPU-only equivalent before replay. Clean retraining, intended four-correction common evaluation, new-Rheon regeneration and long own-history drift remain unqualified. Preserve original receipts and ancestry.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-study-01/train/receipt.json",
    "sha256": "b4faace130aadce25ed453efb8235b5412cd65e9ebf6cba508c5da6751e8b8bf"
  },
  "history": {
    "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-study-01/train/history.json",
    "sha256": "bffabe79ccbfad290458f36c751515ec153c223e11ae424a9432831428a27240"
  },
  "sources": [
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/config.json",
      "sha256": "e877abd6874d45d9a93eba20c22e2ec7303db734247e6df6412226dcb0b19545"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/fine-tuning.json",
      "sha256": "928baa1ff6031fbbdccd69181df7cb9e3b7b2fe0e661429900e3ac82cb40ca01"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/fine_tuning.py",
      "sha256": "df32da5a1a8a0b6b90dd4aeaf05a167403bd72ecb2e0913fcb611ed5dd01425c"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/fit_bank.py",
      "sha256": "e68c1fa6723331af3448edec8294ca0dc67a5b97576a1cd251c5a789e9bf0668"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/model.py",
      "sha256": "a2ebefe6a8ce12c7e6e1e12fa82b835029f364b50e91284f33b60bfe31b0eaee"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/original_fine_tuning.py",
      "sha256": "6e8d5e11744c0622f700bc645226d5dd96dbf831dd9bea754328c97dc70ecc12"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/train.py",
      "sha256": "b9c98c38dfd22ec77ed1bc6dcbad8e692b276dc96fe9cb5cf928ec2b6be0774b"
    },
    {
      "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-project-01/source-witness.json",
    "sha256": "f304158beef28151e76335277470c0e6f855411cf1430bd85b8bfc51f56417bb"
  },
  "training_population": {
    "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-study-01/train/splits.json",
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
    "item_id": "fit-b4faace130aadce2",
    "kind": "training_attempt",
    "name": "Pressure four-pass · axis-isolation · C53 fine-tune CUDA",
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
    "receipt_sha256": "b4faace130aadce25ed453efb8235b5412cd65e9ebf6cba508c5da6751e8b8bf",
    "receipt_paths": [
      "runs/pressure-fundamental-bank-fit-v1/axis-isolation-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "06df7e568a8d4cf7bd2190e82c7e531a",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure four-pass · axis-isolation · C53 fine-tune CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 600,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "e877abd6874d45d9a93eba20c22e2ec7303db734247e6df6412226dcb0b19545",
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
      "sample_order_sha256": "a475267c79a740fb24a3a26e5e86b9895c94d60103a92c00453cae665ea28368",
      "source_witness_sha256": "f304158beef28151e76335277470c0e6f855411cf1430bd85b8bfc51f56417bb"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-600.npz",
        "sha256": "bdd4ea01dd7667f84e78924c4372e540da8b36a2b779a88f23e9c51fba4c139d",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "bdd4ea01dd7667f84e78924c4372e540da8b36a2b779a88f23e9c51fba4c139d",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2343317504,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/64"
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
      "receipt": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-fundamental-bank-fit-v1/axis-isolation-study-01/train/receipt.json",
    "sha256": "b4faace130aadce25ed453efb8235b5412cd65e9ebf6cba508c5da6751e8b8bf",
    "bytes": 29878
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C65": {
        "name": "axis-isolation",
        "weights_sha256": "bdd4ea01dd7667f84e78924c4372e540da8b36a2b779a88f23e9c51fba4c139d",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.0065443352337962845,
          "pressure_mean_free_relative_l2": 0.017042723565476883,
          "pressure_absolute_rms": 0.4982930136593178,
          "pressure_absolute_max": 1.0857529310181588
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
            "trajectory_id": "baseline--water-validation-0053",
            "native_frame": 63,
            "counts_xyz": [
              14,
              9,
              11
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.005424838930587478,
              "pressure_mean_free_relative_l2": 0.012708586631455698,
              "pressure_absolute_rms": 0.4982930136593178,
              "pressure_absolute_max": 1.0857529310181588
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.0065443352337962845,
              "pressure_mean_free_relative_l2": 0.017042723565476883,
              "pressure_absolute_rms": 0.27682890326563464,
              "pressure_absolute_max": 0.561360589920027
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C65/results.json",
          "sha256": "fa77b06109d9e9a485df10f9e62ae3a0a2115aea2acfe9b54b7f83e9e5aa31dd"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"axis-isolation","early_weight":0.05,"fitting_bank":{"amplitude_reference":"original-train-correction-rms","cache_max_bytes":8388608,"grids":[{"cells_per_metre":16,"counts_xyz":[13,15,16]}],"modes_zyx":[[0,0,1],[0,1,0],[1,0,0]],"sampling_seed":2026102101},"initial_weights_sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":4},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"e877abd6874d45d9a93eba20c22e2ec7303db734247e6df6412226dcb0b19545","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"a475267c79a740fb24a3a26e5e86b9895c94d60103a92c00453cae665ea28368","source_witness_sha256":"f304158beef28151e76335277470c0e6f855411cf1430bd85b8bfc51f56417bb"},"diagnostic_population":"train_only","elapsed_s":29.971363987016957,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":4.642634935692058e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":17261056,"peak_reserved_bytes":39845888,"tf32":false},"fine_tuning_statistics":{"fundamental_bank":{"amplitude_rms_pa":2.8535734640782584,"cases":[{"cells_per_metre":16,"counts_xyz":[13,15,16],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[13,15,16],"mode_zyx":[0,1,0]},{"cells_per_metre":16,"counts_xyz":[13,15,16],"mode_zyx":[1,0,0]}],"description_sha256":"829115d6601d7772dfdb3a849c9c9aa464be8b754ca7f30baf863a314ae69fd3","expected_exposures":[200,200,200],"optimizer_examples":600,"optimizer_exposures":[200,200,200],"resident_tensor_bytes":299592,"sampling_order_sha256":"bd23f27453178dc881575737a2116bf759f7a128df6c2d7bd0b4c4026d5e56b3","trace":[{"exposures_per_case":10,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.1161664408599918},{"case":0,"passes":2,"relative_equation_residual":0.0029066754437020374},{"case":0,"passes":4,"relative_equation_residual":1.2610076402981284e-05},{"case":0,"passes":8,"relative_equation_residual":3.616547110238269e-10},{"case":1,"passes":1,"relative_equation_residual":0.11922121369657065},{"case":1,"passes":2,"relative_equation_residual":0.004637617797980293},{"case":1,"passes":4,"relative_equation_residual":9.141459560423283e-06},{"case":1,"passes":8,"relative_equation_residual":2.5408893664115556e-10},{"case":2,"passes":1,"relative_equation_residual":0.07942480739837954},{"case":2,"passes":2,"relative_equation_residual":0.004676482912226679},{"case":2,"passes":4,"relative_equation_residual":2.926931694534755e-05},{"case":2,"passes":8,"relative_equation_residual":7.691450697733347e-10}],"update":30},{"exposures_per_case":20,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.10449484666746241},{"case":0,"passes":2,"relative_equation_residual":0.005264048144771758},{"case":0,"passes":4,"relative_equation_residual":1.6219920722074854e-05},{"case":0,"passes":8,"relative_equation_residual":1.7275473994135162e-09},{"case":1,"passes":1,"relative_equation_residual":0.10805783286762465},{"case":1,"passes":2,"relative_equation_residual":0.0040235053912072195},{"case":1,"passes":4,"relative_equation_residual":5.7174511107591065e-06},{"case":1,"passes":8,"relative_equation_residual":1.3158768971246305e-10},{"case":2,"passes":1,"relative_equation_residual":0.13548239644896784},{"case":2,"passes":2,"relative_equation_residual":0.006171378960822107},{"case":2,"passes":4,"relative_equation_residual":2.254384499527868e-05},{"case":2,"passes":8,"relative_equation_residual":7.243185742204934e-10}],"update":60},{"exposures_per_case":30,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.05948886568906396},{"case":0,"passes":2,"relative_equation_residual":0.001845115143040754},{"case":0,"passes":4,"relative_equation_residual":3.982593608567447e-06},{"case":0,"passes":8,"relative_equation_residual":5.234290011005074e-11},{"case":1,"passes":1,"relative_equation_residual":0.062820918043708},{"case":1,"passes":2,"relative_equation_residual":0.0022655573452059923},{"case":1,"passes":4,"relative_equation_residual":2.6757040295557016e-06},{"case":1,"passes":8,"relative_equation_residual":2.52293811658285e-11},{"case":2,"passes":1,"relative_equation_residual":0.050550259940315784},{"case":2,"passes":2,"relative_equation_residual":0.0020393174316774725},{"case":2,"passes":4,"relative_equation_residual":3.1536444736747253e-06},{"case":2,"passes":8,"relative_equation_residual":2.422844728267838e-11}],"update":90},{"exposures_per_case":40,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.06827768196616615},{"case":0,"passes":2,"relative_equation_residual":0.0029014646052047456},{"case":0,"passes":4,"relative_equation_residual":8.169367807317471e-06},{"case":0,"passes":8,"relative_equation_residual":1.177592992569527e-10},{"case":1,"passes":1,"relative_equation_residual":0.06552380117769527},{"case":1,"passes":2,"relative_equation_residual":0.003923650660327998},{"case":1,"passes":4,"relative_equation_residual":1.206890101115203e-05},{"case":1,"passes":8,"relative_equation_residual":1.5204082035141135e-10},{"case":2,"passes":1,"relative_equation_residual":0.05808974926360389},{"case":2,"passes":2,"relative_equation_residual":0.002310255653172168},{"case":2,"passes":4,"relative_equation_residual":9.200781559020798e-06},{"case":2,"passes":8,"relative_equation_residual":2.0345688974027443e-10}],"update":120},{"exposures_per_case":50,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.05877508937297209},{"case":0,"passes":2,"relative_equation_residual":0.0027075500314188987},{"case":0,"passes":4,"relative_equation_residual":5.445779273033165e-06},{"case":0,"passes":8,"relative_equation_residual":3.3415603751217593e-11},{"case":1,"passes":1,"relative_equation_residual":0.044272440341256845},{"case":1,"passes":2,"relative_equation_residual":0.0011031984729489934},{"case":1,"passes":4,"relative_equation_residual":1.421813534532301e-06},{"case":1,"passes":8,"relative_equation_residual":5.597255047776564e-12},{"case":2,"passes":1,"relative_equation_residual":0.03172699245259742},{"case":2,"passes":2,"relative_equation_residual":0.0013856881850069687},{"case":2,"passes":4,"relative_equation_residual":1.256022211242544e-06},{"case":2,"passes":8,"relative_equation_residual":6.3346931015662226e-12}],"update":150},{"exposures_per_case":60,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.03636883987379297},{"case":0,"passes":2,"relative_equation_residual":0.0009065710120338092},{"case":0,"passes":4,"relative_equation_residual":2.280472536494176e-06},{"case":0,"passes":8,"relative_equation_residual":1.4724629742980533e-11},{"case":1,"passes":1,"relative_equation_residual":0.04904777262439074},{"case":1,"passes":2,"relative_equation_residual":0.001656181777549695},{"case":1,"passes":4,"relative_equation_residual":3.3918944055637055e-06},{"case":1,"passes":8,"relative_equation_residual":1.5121446908987385e-11},{"case":2,"passes":1,"relative_equation_residual":0.04362934826927999},{"case":2,"passes":2,"relative_equation_residual":0.0013698792997718452},{"case":2,"passes":4,"relative_equation_residual":8.256670175215084e-07},{"case":2,"passes":8,"relative_equation_residual":6.165796403376449e-12}],"update":180},{"exposures_per_case":70,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.04756279335879824},{"case":0,"passes":2,"relative_equation_residual":0.0028373444400785343},{"case":0,"passes":4,"relative_equation_residual":8.210601777457256e-06},{"case":0,"passes":8,"relative_equation_residual":3.9462251157074864e-11},{"case":1,"passes":1,"relative_equation_residual":0.0391989062630662},{"case":1,"passes":2,"relative_equation_residual":0.0014902282667653439},{"case":1,"passes":4,"relative_equation_residual":1.3012367238960305e-06},{"case":1,"passes":8,"relative_equation_residual":3.2292987776934965e-12},{"case":2,"passes":1,"relative_equation_residual":0.06094649882687547},{"case":2,"passes":2,"relative_equation_residual":0.001800563317734728},{"case":2,"passes":4,"relative_equation_residual":2.1428738793016624e-06},{"case":2,"passes":8,"relative_equation_residual":8.255458275018659e-12}],"update":210},{"exposures_per_case":80,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.05562118611968039},{"case":0,"passes":2,"relative_equation_residual":0.0021629569343550063},{"case":0,"passes":4,"relative_equation_residual":1.85722662170264e-06},{"case":0,"passes":8,"relative_equation_residual":4.834665582437169e-12},{"case":1,"passes":1,"relative_equation_residual":0.09411410768736322},{"case":1,"passes":2,"relative_equation_residual":0.0036982422678384012},{"case":1,"passes":4,"relative_equation_residual":7.741999755435703e-06},{"case":1,"passes":8,"relative_equation_residual":1.2454250110164932e-11},{"case":2,"passes":1,"relative_equation_residual":0.05393278175956657},{"case":2,"passes":2,"relative_equation_residual":0.0021088372319071793},{"case":2,"passes":4,"relative_equation_residual":1.895151669816995e-06},{"case":2,"passes":8,"relative_equation_residual":1.4907917325921475e-11}],"update":240},{"exposures_per_case":90,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.0311463721746635},{"case":0,"passes":2,"relative_equation_residual":0.0008817553152749214},{"case":0,"passes":4,"relative_equation_residual":9.282605399835362e-07},{"case":0,"passes":8,"relative_equation_residual":3.0062943018287414e-12},{"case":1,"passes":1,"relative_equation_residual":0.024569047541096515},{"case":1,"passes":2,"relative_equation_residual":0.0007889438010873323},{"case":1,"passes":4,"relative_equation_residual":1.5438905955970486e-06},{"case":1,"passes":8,"relative_equation_residual":2.8211499904965858e-11},{"case":2,"passes":1,"relative_equation_residual":0.04002990791159682},{"case":2,"passes":2,"relative_equation_residual":0.001352857994868542},{"case":2,"passes":4,"relative_equation_residual":1.9502217130504695e-06},{"case":2,"passes":8,"relative_equation_residual":1.1877070895616391e-11}],"update":270},{"exposures_per_case":100,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.05804128749092626},{"case":0,"passes":2,"relative_equation_residual":0.00337977671157367},{"case":0,"passes":4,"relative_equation_residual":1.2913842284766012e-05},{"case":0,"passes":8,"relative_equation_residual":3.376717738787061e-10},{"case":1,"passes":1,"relative_equation_residual":0.1031419176479314},{"case":1,"passes":2,"relative_equation_residual":0.006128025542617284},{"case":1,"passes":4,"relative_equation_residual":7.974144538761695e-06},{"case":1,"passes":8,"relative_equation_residual":5.2801253369547676e-11},{"case":2,"passes":1,"relative_equation_residual":0.05173189813751038},{"case":2,"passes":2,"relative_equation_residual":0.0009658489609759738},{"case":2,"passes":4,"relative_equation_residual":1.6008826451356907e-06},{"case":2,"passes":8,"relative_equation_residual":2.14672147845586e-11}],"update":300},{"exposures_per_case":110,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.0443484921066947},{"case":0,"passes":2,"relative_equation_residual":0.0016726777605086433},{"case":0,"passes":4,"relative_equation_residual":1.3200445660723044e-06},{"case":0,"passes":8,"relative_equation_residual":3.710331228794976e-12},{"case":1,"passes":1,"relative_equation_residual":0.09147937475117911},{"case":1,"passes":2,"relative_equation_residual":0.003492991781482194},{"case":1,"passes":4,"relative_equation_residual":5.919770097180085e-07},{"case":1,"passes":8,"relative_equation_residual":2.5969095037219155e-12},{"case":2,"passes":1,"relative_equation_residual":0.04228747012360767},{"case":2,"passes":2,"relative_equation_residual":0.0007945887156947085},{"case":2,"passes":4,"relative_equation_residual":1.4529405944982842e-07},{"case":2,"passes":8,"relative_equation_residual":1.3167882509279379e-11}],"update":330},{"exposures_per_case":120,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.01967585903785518},{"case":0,"passes":2,"relative_equation_residual":0.0003996945218376973},{"case":0,"passes":4,"relative_equation_residual":1.6293759672455142e-07},{"case":0,"passes":8,"relative_equation_residual":1.7167782570817946e-12},{"case":1,"passes":1,"relative_equation_residual":0.08408327030251479},{"case":1,"passes":2,"relative_equation_residual":0.003021350563736108},{"case":1,"passes":4,"relative_equation_residual":5.069121495130948e-07},{"case":1,"passes":8,"relative_equation_residual":2.1064904099084303e-12},{"case":2,"passes":1,"relative_equation_residual":0.03319619235285159},{"case":2,"passes":2,"relative_equation_residual":0.00037171567302737217},{"case":2,"passes":4,"relative_equation_residual":1.722700626513903e-07},{"case":2,"passes":8,"relative_equation_residual":5.868822885666609e-12}],"update":360},{"exposures_per_case":130,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.008149715796875885},{"case":0,"passes":2,"relative_equation_residual":0.00016213994742740321},{"case":0,"passes":4,"relative_equation_residual":7.383253532209654e-08},{"case":0,"passes":8,"relative_equation_residual":2.0864226973034904e-12},{"case":1,"passes":1,"relative_equation_residual":0.0768703645113629},{"case":1,"passes":2,"relative_equation_residual":0.002508225441138366},{"case":1,"passes":4,"relative_equation_residual":2.1422681531444062e-07},{"case":1,"passes":8,"relative_equation_residual":3.376175480877732e-12},{"case":2,"passes":1,"relative_equation_residual":0.022718087230296058},{"case":2,"passes":2,"relative_equation_residual":0.0003607705329515577},{"case":2,"passes":4,"relative_equation_residual":8.359731176445944e-08},{"case":2,"passes":8,"relative_equation_residual":1.0192330908123557e-11}],"update":390},{"exposures_per_case":140,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.005746792999757304},{"case":0,"passes":2,"relative_equation_residual":0.0002250544251324617},{"case":0,"passes":4,"relative_equation_residual":3.663671179652946e-07},{"case":0,"passes":8,"relative_equation_residual":2.3500496532105423e-12},{"case":1,"passes":1,"relative_equation_residual":0.07397690258425611},{"case":1,"passes":2,"relative_equation_residual":0.0023527907951047394},{"case":1,"passes":4,"relative_equation_residual":3.768061669267435e-07},{"case":1,"passes":8,"relative_equation_residual":3.773557249116388e-12},{"case":2,"passes":1,"relative_equation_residual":0.01288818570858119},{"case":2,"passes":2,"relative_equation_residual":7.510484868046732e-05},{"case":2,"passes":4,"relative_equation_residual":6.040051708030041e-08},{"case":2,"passes":8,"relative_equation_residual":5.895654872975708e-12}],"update":420},{"exposures_per_case":150,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.004905624033910692},{"case":0,"passes":2,"relative_equation_residual":0.00013784019490598533},{"case":0,"passes":4,"relative_equation_residual":1.2860855152808668e-07},{"case":0,"passes":8,"relative_equation_residual":1.4696447847434769e-12},{"case":1,"passes":1,"relative_equation_residual":0.06906263525544568},{"case":1,"passes":2,"relative_equation_residual":0.0021573072827231195},{"case":1,"passes":4,"relative_equation_residual":1.7839936812631288e-07},{"case":1,"passes":8,"relative_equation_residual":2.8575269127484124e-12},{"case":2,"passes":1,"relative_equation_residual":0.009083739353497548},{"case":2,"passes":2,"relative_equation_residual":9.007692946511304e-05},{"case":2,"passes":4,"relative_equation_residual":7.563041050652491e-08},{"case":2,"passes":8,"relative_equation_residual":6.703463453553984e-12}],"update":450},{"exposures_per_case":160,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.0009033736320415734},{"case":0,"passes":2,"relative_equation_residual":2.7077472628236832e-05},{"case":0,"passes":4,"relative_equation_residual":3.405376926531606e-08},{"case":0,"passes":8,"relative_equation_residual":1.5136485050133117e-12},{"case":1,"passes":1,"relative_equation_residual":0.06946668567621954},{"case":1,"passes":2,"relative_equation_residual":0.0020428919900121953},{"case":1,"passes":4,"relative_equation_residual":1.170611387431197e-07},{"case":1,"passes":8,"relative_equation_residual":4.3717151206075044e-12},{"case":2,"passes":1,"relative_equation_residual":0.006907545500123086},{"case":2,"passes":2,"relative_equation_residual":2.7526657762646718e-05},{"case":2,"passes":4,"relative_equation_residual":5.7805697237195795e-08},{"case":2,"passes":8,"relative_equation_residual":8.887234668389454e-12}],"update":480},{"exposures_per_case":170,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.00027173380704850514},{"case":0,"passes":2,"relative_equation_residual":1.285689376004427e-05},{"case":0,"passes":4,"relative_equation_residual":2.0627682300614368e-08},{"case":0,"passes":8,"relative_equation_residual":1.1327611143479879e-12},{"case":1,"passes":1,"relative_equation_residual":0.06844951890435622},{"case":1,"passes":2,"relative_equation_residual":0.001974616848145506},{"case":1,"passes":4,"relative_equation_residual":7.423035533513511e-08},{"case":1,"passes":8,"relative_equation_residual":2.1545653775776207e-12},{"case":2,"passes":1,"relative_equation_residual":0.006194978563710649},{"case":2,"passes":2,"relative_equation_residual":2.1419288913718097e-05},{"case":2,"passes":4,"relative_equation_residual":5.3190440281467146e-08},{"case":2,"passes":8,"relative_equation_residual":7.443885222727698e-12}],"update":510},{"exposures_per_case":180,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.0002830263320210179},{"case":0,"passes":2,"relative_equation_residual":7.2396102038178585e-06},{"case":0,"passes":4,"relative_equation_residual":1.872617987644046e-08},{"case":0,"passes":8,"relative_equation_residual":9.418268401469091e-13},{"case":1,"passes":1,"relative_equation_residual":0.06802018824385925},{"case":1,"passes":2,"relative_equation_residual":0.0019305718070331266},{"case":1,"passes":4,"relative_equation_residual":7.032823952992639e-08},{"case":1,"passes":8,"relative_equation_residual":3.879781293595471e-12},{"case":2,"passes":1,"relative_equation_residual":0.006017814527741204},{"case":2,"passes":2,"relative_equation_residual":1.8319529513489644e-05},{"case":2,"passes":4,"relative_equation_residual":5.990357786232932e-08},{"case":2,"passes":8,"relative_equation_residual":6.48415863534427e-12}],"update":540},{"exposures_per_case":190,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.0002131855750006871},{"case":0,"passes":2,"relative_equation_residual":7.270886670250472e-06},{"case":0,"passes":4,"relative_equation_residual":2.5552674401220974e-08},{"case":0,"passes":8,"relative_equation_residual":2.399374221525202e-12},{"case":1,"passes":1,"relative_equation_residual":0.06759187360566346},{"case":1,"passes":2,"relative_equation_residual":0.0019149124940209823},{"case":1,"passes":4,"relative_equation_residual":8.174755055627431e-08},{"case":1,"passes":8,"relative_equation_residual":3.3537773759705163e-12},{"case":2,"passes":1,"relative_equation_residual":0.005889123941798978},{"case":2,"passes":2,"relative_equation_residual":1.742962969032296e-05},{"case":2,"passes":4,"relative_equation_residual":5.4666962166337025e-08},{"case":2,"passes":8,"relative_equation_residual":7.716307478641608e-12}],"update":570},{"exposures_per_case":200,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.0003498941861138414},{"case":0,"passes":2,"relative_equation_residual":1.2233748821535322e-05},{"case":0,"passes":4,"relative_equation_residual":3.017973257889898e-08},{"case":0,"passes":8,"relative_equation_residual":3.507907973951513e-12},{"case":1,"passes":1,"relative_equation_residual":0.06772326242153043},{"case":1,"passes":2,"relative_equation_residual":0.00192006526471448},{"case":1,"passes":4,"relative_equation_residual":7.150081549952155e-08},{"case":1,"passes":8,"relative_equation_residual":2.67901511031277e-12},{"case":2,"passes":1,"relative_equation_residual":0.006107083628684456},{"case":2,"passes":2,"relative_equation_residual":1.991736530172503e-05},{"case":2,"passes":4,"relative_equation_residual":6.525757054663608e-08},{"case":2,"passes":8,"relative_equation_residual":8.180799255400651e-12}],"update":600}]},"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":600,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"checkpoint-600.npz":{"bytes":1485452,"role":"checkpoint","sha256":"bdd4ea01dd7667f84e78924c4372e540da8b36a2b779a88f23e9c51fba4c139d"},"history.json":{"bytes":8243,"role":"training_diagnostics","sha256":"bffabe79ccbfad290458f36c751515ec153c223e11ae424a9432831428a27240"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"11e2362ce55ca2d38de49053eafc506b8a811932826aaddbd3c0fce17dd8b909"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":475,"role":"result_summary","sha256":"816c806c0173edd3832dc4b48acbc3f004c64a0d2960a5332e7839c54e6b71b7"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"bdd4ea01dd7667f84e78924c4372e540da8b36a2b779a88f23e9c51fba4c139d"}},"per_case_inference_s":[0.02347941100015305,0.004256576008629054,0.005397485016146675,0.004356506979092956,0.0037715669895987958,0.00355594398570247,0.004625823989044875,0.004711640009190887,0.009647951985243708,0.005324338999344036,0.0063959019898902625,0.005747411021729931],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"ed8dd0b9117baeedba00a55dfed26341ab9c6d3451fcd54e6b518ae9e3d0f495","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"4c0f9bbd6231c7e9b6621d5265bc1b007ba7128cde46f1723cc9861d1d26b640","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"2b9310e10b89034b950cf043e79bd1c4557cf02b339d764f2f238368370b21ea","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"94821115a62dad5039f942c28c297af2ce07ca5a2bd9dddbc7397c9f90727252","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"8a3352faac9f8c032da0876480c0d471ae2ac711b397360c7f4f0df13ee59e33","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"0c2cb2a2c2932bffaaa7b5de188edfd27305db08ed88a964b26324643f9187b3","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"78a6f134b4317001b0c2ea505371067a43d4f2ada51c80113e39b15e0d8997ad","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"9f7b7fda4910aa63824d57487a9c53343003c6d9e159d42b6e5847844c7b5033","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"024acbdb4a377611be9620c974964f06ad8638f2bce7a443f2fb6e94a761ef73","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"d120959ef893e123c63b01933fcacf90052f8d6860b5324e8211e3dffe22d526","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"bce34447687636fbe12d57d7e6d655ebe059cc2b9eb9d64584f9b76c49e70da3","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"22ea8e4413f2c520270d90b18143ad7f37b854bec9dcba014e458abadd202514","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2343317504,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
