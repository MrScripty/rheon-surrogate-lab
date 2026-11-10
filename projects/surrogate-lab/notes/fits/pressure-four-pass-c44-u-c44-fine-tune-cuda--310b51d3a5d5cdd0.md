# Pressure four-pass · c44-u · C44 fine-tune CUDA

Retrospective source review · fit-310b51d3a5d5cdd0

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 370,016-parameter float32 3D pressure U-Net with encoder widths8/16/24/32, bottleneck48 and reversed decoder; physical residual reconstruction is float64. It starts from the exact C44 checkpoint. Training unrolls four pressure corrections on one equation. These corrections are not future physical simulation steps.

**Purpose.** Compare unchanged four-pass fine tuning with attached and detached auxiliary gradient paths under the same initialization, pattern order, depth schedule and optimizer budget. Retain the four-pass control without the auxiliary loss.

**Experiment.** The optimizer uses eight synthetic analytical patterns: one [0,0,1] mode on each of eight rectangular grids, at8or16cells/m. Each gets200exposures, with50at each of four auxiliary depths. The original16train trajectories calibrate amplitude/normalization; they are not the optimizer sample stream. No companion physical, direction or current-policy loss runs in the captured fitting-bank updater. Teacher weight is zero and counted teacher forwards are zero. All arms run1,600updates; U counts6,400network forwards and A/D8,000. The loss combines analytical pressure loss, log-scaled final equation residual and 0.05mean early residual feedback; A/D add0.05auxiliary loss.

**Observed outcome.** C72, on the same retained 128 pressure requests: worst mean-free pressure relative L2 0.022052820523551637, equation relative L2 0.0067723741762331464. Parent C44 is better on both: 0.013450717090965448 and 0.0039826222242511878, respectively. No new timing or own-history drift result is implied.

**Lesson.** All three eight-pattern fine-tunes regress against the already stronger C44 parent on both worst pressure and equation error under the unchanged common benchmark. The matched C37 family improved instead, so that improvement does not transfer automatically to another initialization. Among these C44 continuations, A has the lowest worst pressure error and D the lowest worst equation residual; neither supplies an accuracy reason to replace the parent. A narrow analytical fitting objective can disagree with broad native-case results. The data establish the regression, not its sole cause; preserve the parent and validate matched cases before promotion. Equal update counts also conceal the A/D extra forward. The subsequent matched-work CUDA matrix confirms these parent regressions at every tested one/two/four/eight correction budget. C37 benefits at two/four/eight corrections, while its worst pressure error worsens at one correction; its saved parent default uses one and its fine-tunes use two. Keep metric and actual work explicit.

**Reproduction.** Use the exact captured project.json/train-recipe.json and evidence.py entrypoint with the original input bindings, source witness, C44 checkpoint, fitting bank and fine-tuning.json. Clean retraining, fresh qualified latency and new-Rheon data equivalence remain unqualified; no checkpoint can be retired on this review alone.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-detached-correction-v1/c44-u-study-01/train/receipt.json",
    "sha256": "310b51d3a5d5cdd01aa3a752f5e22b7f44fe9be7444e4d24a45217d525213cb7"
  },
  "history": {
    "path": "runs/pressure-detached-correction-v1/c44-u-study-01/train/history.json",
    "sha256": "277193000cb80acea428ac43a38cf18807a0fe1b93857088bbf686a7a0ce0b72"
  },
  "sources": [
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/config.json",
      "sha256": "656dd8f670c7e3b127ea7e9c2fccb222f29ac079dbdf1023ee095309426757d3"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/fine-tuning.json",
      "sha256": "103ac2e21b89d1b9a1e82e464d3b411ae0ea8bab9d401e48a2cb2aad4b95f3bf"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/fine_tuning.py",
      "sha256": "57658249095fd62c38f8d3bcbbe201d5ebe288009f0042001941bca82110795e"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/fit_bank.py",
      "sha256": "bbdbabfc89805410df196dd0724b24347954548dd83790456994ca46a6f5e347"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/model.py",
      "sha256": "a2ebefe6a8ce12c7e6e1e12fa82b835029f364b50e91284f33b60bfe31b0eaee"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/original_fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/original_fit_bank.py",
      "sha256": "e68c1fa6723331af3448edec8294ca0dc67a5b97576a1cd251c5a789e9bf0668"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/train.py",
      "sha256": "b9c98c38dfd22ec77ed1bc6dcbad8e692b276dc96fe9cb5cf928ec2b6be0774b"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-u-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-detached-correction-v1/c44-u-project-01/source-witness.json",
    "sha256": "d5c15c75670f227638d444f0a6b06facc101f1a773ff65890f67febd746e1c1f"
  },
  "training_population": {
    "path": "runs/pressure-detached-correction-v1/c44-u-study-01/train/splits.json",
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
      "path": "runs/water-late-residual-v1/extra-late/managed-full/stage-01/artifacts/weights.npz",
      "sha256": "a73fbc342ed070390b47535e6b94b535dec158671393b074a17e8a6858c249da",
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
    "item_id": "fit-310b51d3a5d5cdd0",
    "kind": "training_attempt",
    "name": "Pressure four-pass · c44-u · C44 fine-tune CUDA",
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
    "receipt_sha256": "310b51d3a5d5cdd01aa3a752f5e22b7f44fe9be7444e4d24a45217d525213cb7",
    "receipt_paths": [
      "runs/pressure-detached-correction-v1/c44-u-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "d4461f9eceb64c17913101a5bcbf78de",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure four-pass · c44-u · C44 fine-tune CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1600,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "656dd8f670c7e3b127ea7e9c2fccb222f29ac079dbdf1023ee095309426757d3",
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
          "sha256": "a73fbc342ed070390b47535e6b94b535dec158671393b074a17e8a6858c249da"
        },
        "splits": {
          "bytes": 256361,
          "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "21ed766152674818e22d0abb5ed5540690ab31aa14507d78fcc73a381e5d058a",
      "source_witness_sha256": "d5c15c75670f227638d444f0a6b06facc101f1a773ff65890f67febd746e1c1f"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "a73fbc342ed070390b47535e6b94b535dec158671393b074a17e8a6858c249da",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-1600.npz",
        "sha256": "759ce063892ed6c32991214c461f4bbac07e2e924467682f979244c077f548d0",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "759ce063892ed6c32991214c461f4bbac07e2e924467682f979244c077f548d0",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2413580288,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/70"
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
      "receipt": "runs/pressure-detached-correction-v1/c44-u-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-detached-correction-v1/c44-u-study-01/train/receipt.json",
    "sha256": "310b51d3a5d5cdd01aa3a752f5e22b7f44fe9be7444e4d24a45217d525213cb7",
    "bytes": 59960
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C72": {
        "name": "c44-u",
        "weights_sha256": "759ce063892ed6c32991214c461f4bbac07e2e924467682f979244c077f548d0",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.006772374176233146,
          "pressure_mean_free_relative_l2": 0.022052820523551637,
          "pressure_absolute_rms": 0.6909521010101206,
          "pressure_absolute_max": 1.8090553600283028
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0039",
            "native_frame": 15,
            "counts_xyz": [
              10,
              12,
              12
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0003",
            "native_frame": 63,
            "counts_xyz": [
              16,
              6,
              12
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.006496573929133819,
              "pressure_mean_free_relative_l2": 0.01610281160766313,
              "pressure_absolute_rms": 0.6909521010101206,
              "pressure_absolute_max": 1.8090553600283028
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.006772374176233146,
              "pressure_mean_free_relative_l2": 0.022052820523551637,
              "pressure_absolute_rms": 0.38383755856424107,
              "pressure_absolute_max": 0.961041440667763
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C72/results.json",
          "sha256": "31f03f5e88ab70116a902b02afce2cf26053e763640457a8a4858dd331110796"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"c44-u","early_weight":0.05,"fitting_bank":{"amplitude_reference":"original-train-correction-rms","cache_max_bytes":8388608,"grids":[{"cells_per_metre":16,"counts_xyz":[13,15,16]},{"cells_per_metre":16,"counts_xyz":[9,11,11]},{"cells_per_metre":8,"counts_xyz":[15,9,7]},{"cells_per_metre":16,"counts_xyz":[11,5,15]},{"cells_per_metre":8,"counts_xyz":[7,16,5]},{"cells_per_metre":16,"counts_xyz":[11,12,7]},{"cells_per_metre":8,"counts_xyz":[5,13,9]},{"cells_per_metre":16,"counts_xyz":[16,7,11]}],"modes_zyx":[[0,0,1]],"sampling_seed":2026102101},"initial_weights_sha256":"a73fbc342ed070390b47535e6b94b535dec158671393b074a17e8a6858c249da","relative_equation_target":1e-06,"rollout_feedback":{"depth_seed":2026101807,"mode":"U","weight":0.05},"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":4},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"656dd8f670c7e3b127ea7e9c2fccb222f29ac079dbdf1023ee095309426757d3","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"a73fbc342ed070390b47535e6b94b535dec158671393b074a17e8a6858c249da"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"21ed766152674818e22d0abb5ed5540690ab31aa14507d78fcc73a381e5d058a","source_witness_sha256":"d5c15c75670f227638d444f0a6b06facc101f1a773ff65890f67febd746e1c1f"},"diagnostic_population":"train_only","elapsed_s":69.95414663202246,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":3.3990720140764097e-06,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":19713024,"peak_reserved_bytes":37748736,"tf32":false},"fine_tuning_statistics":{"fundamental_bank":{"amplitude_rms_pa":2.8535734640782584,"cases":[{"cells_per_metre":16,"counts_xyz":[13,15,16],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[9,11,11],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[15,9,7],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[11,5,15],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[7,16,5],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[11,12,7],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[5,13,9],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[16,7,11],"mode_zyx":[0,0,1]}],"description_sha256":"30f7092460b6a649b28e4bf380a5769c2955d1e29461e0f64a6e1d474dab960f","expected_exposures":[200,200,200,200,200,200,200,200],"optimizer_examples":1600,"optimizer_exposures":[200,200,200,200,200,200,200,200],"resident_tensor_bytes":297152,"sampling_order_sha256":"d0ea221a7e03a83673ece7fc9cdf4f5212bb54f3023511f93d95c84e45647899","trace":[{"exposures_per_case":10,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.1888443572961813},{"case":0,"passes":2,"relative_equation_residual":0.023379419755499512},{"case":0,"passes":4,"relative_equation_residual":0.00043700302587383003},{"case":0,"passes":8,"relative_equation_residual":1.4716181701827847e-07},{"case":1,"passes":1,"relative_equation_residual":0.0802515665117205},{"case":1,"passes":2,"relative_equation_residual":0.003996933285172855},{"case":1,"passes":4,"relative_equation_residual":1.2067444246845241e-05},{"case":1,"passes":8,"relative_equation_residual":1.574519383871211e-10},{"case":2,"passes":1,"relative_equation_residual":0.18450394302696477},{"case":2,"passes":2,"relative_equation_residual":0.011580361963714224},{"case":2,"passes":4,"relative_equation_residual":9.489927812476851e-05},{"case":2,"passes":8,"relative_equation_residual":8.985605045803356e-09},{"case":3,"passes":1,"relative_equation_residual":0.12753243554907817},{"case":3,"passes":2,"relative_equation_residual":0.011622531302601276},{"case":3,"passes":4,"relative_equation_residual":0.0001092623652030184},{"case":3,"passes":8,"relative_equation_residual":8.911678019845442e-09},{"case":4,"passes":1,"relative_equation_residual":0.04491467712025148},{"case":4,"passes":2,"relative_equation_residual":0.002175351644591873},{"case":4,"passes":4,"relative_equation_residual":4.067589562496392e-06},{"case":4,"passes":8,"relative_equation_residual":1.0893577597973699e-10},{"case":5,"passes":1,"relative_equation_residual":0.12753228283896814},{"case":5,"passes":2,"relative_equation_residual":0.011622454126384577},{"case":5,"passes":4,"relative_equation_residual":0.00010926080564004097},{"case":5,"passes":8,"relative_equation_residual":8.910616770289083e-09},{"case":6,"passes":1,"relative_equation_residual":0.01550336664086193},{"case":6,"passes":2,"relative_equation_residual":0.00029453947367530755},{"case":6,"passes":4,"relative_equation_residual":2.636399343953275e-07},{"case":6,"passes":8,"relative_equation_residual":4.5360670198207633e-11},{"case":7,"passes":1,"relative_equation_residual":0.18225162205862275},{"case":7,"passes":2,"relative_equation_residual":0.03215009983125954},{"case":7,"passes":4,"relative_equation_residual":0.0011667220756991132},{"case":7,"passes":8,"relative_equation_residual":2.27412834038272e-06}],"update":80},{"exposures_per_case":20,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.18654875596872236},{"case":0,"passes":2,"relative_equation_residual":0.02903724926345424},{"case":0,"passes":4,"relative_equation_residual":0.0006807277796771985},{"case":0,"passes":8,"relative_equation_residual":3.494254295966964e-07},{"case":1,"passes":1,"relative_equation_residual":0.10795561317339226},{"case":1,"passes":2,"relative_equation_residual":0.01010864543288284},{"case":1,"passes":4,"relative_equation_residual":9.113287539839252e-05},{"case":1,"passes":8,"relative_equation_residual":5.68202201726135e-09},{"case":2,"passes":1,"relative_equation_residual":0.18535588944360393},{"case":2,"passes":2,"relative_equation_residual":0.003216432985039759},{"case":2,"passes":4,"relative_equation_residual":1.288604368523461e-05},{"case":2,"passes":8,"relative_equation_residual":4.058582813717258e-10},{"case":3,"passes":1,"relative_equation_residual":0.07295015127702324},{"case":3,"passes":2,"relative_equation_residual":0.0031898311957591807},{"case":3,"passes":4,"relative_equation_residual":6.728768992335399e-06},{"case":3,"passes":8,"relative_equation_residual":7.165910062500252e-11},{"case":4,"passes":1,"relative_equation_residual":0.02826353311005507},{"case":4,"passes":2,"relative_equation_residual":0.0002644137912641407},{"case":4,"passes":4,"relative_equation_residual":3.689921436776047e-07},{"case":4,"passes":8,"relative_equation_residual":3.020475706477044e-11},{"case":5,"passes":1,"relative_equation_residual":0.07294752756901203},{"case":5,"passes":2,"relative_equation_residual":0.003189952540378471},{"case":5,"passes":4,"relative_equation_residual":6.729920983293357e-06},{"case":5,"passes":8,"relative_equation_residual":7.007028440331996e-11},{"case":6,"passes":1,"relative_equation_residual":0.029269587936193466},{"case":6,"passes":2,"relative_equation_residual":0.0011890385558346013},{"case":6,"passes":4,"relative_equation_residual":3.867907282879844e-06},{"case":6,"passes":8,"relative_equation_residual":7.935563496845224e-11},{"case":7,"passes":1,"relative_equation_residual":0.16827482177788763},{"case":7,"passes":2,"relative_equation_residual":0.025354225486389107},{"case":7,"passes":4,"relative_equation_residual":0.0006768594642385466},{"case":7,"passes":8,"relative_equation_residual":8.793967909611865e-07}],"update":160},{"exposures_per_case":30,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.29294223636265265},{"case":0,"passes":2,"relative_equation_residual":0.08678688643446189},{"case":0,"passes":4,"relative_equation_residual":0.007704272508488352},{"case":0,"passes":8,"relative_equation_residual":6.026370292753397e-05},{"case":1,"passes":1,"relative_equation_residual":0.17913481959308958},{"case":1,"passes":2,"relative_equation_residual":0.030600273048295757},{"case":1,"passes":4,"relative_equation_residual":0.000911375564007099},{"case":1,"passes":8,"relative_equation_residual":8.026275061205094e-07},{"case":2,"passes":1,"relative_equation_residual":0.19257108296982484},{"case":2,"passes":2,"relative_equation_residual":0.025417133818049623},{"case":2,"passes":4,"relative_equation_residual":0.0006337562071801413},{"case":2,"passes":8,"relative_equation_residual":3.886991898642079e-07},{"case":3,"passes":1,"relative_equation_residual":0.0974078649288496},{"case":3,"passes":2,"relative_equation_residual":0.009263124434199516},{"case":3,"passes":4,"relative_equation_residual":0.00010057491954496767},{"case":3,"passes":8,"relative_equation_residual":1.1745041158400389e-08},{"case":4,"passes":1,"relative_equation_residual":0.060080936596568796},{"case":4,"passes":2,"relative_equation_residual":0.0037548936044736895},{"case":4,"passes":4,"relative_equation_residual":1.6186439522276363e-05},{"case":4,"passes":8,"relative_equation_residual":3.3427987615010553e-10},{"case":5,"passes":1,"relative_equation_residual":0.09740575693832539},{"case":5,"passes":2,"relative_equation_residual":0.009262967377688003},{"case":5,"passes":4,"relative_equation_residual":0.00010057025811664971},{"case":5,"passes":8,"relative_equation_residual":1.1744212112463324e-08},{"case":6,"passes":1,"relative_equation_residual":0.04243389541804167},{"case":6,"passes":2,"relative_equation_residual":0.001947893288364553},{"case":6,"passes":4,"relative_equation_residual":3.5892853887553357e-06},{"case":6,"passes":8,"relative_equation_residual":7.273249269329653e-11},{"case":7,"passes":1,"relative_equation_residual":0.1276663235626365},{"case":7,"passes":2,"relative_equation_residual":0.011773064924659012},{"case":7,"passes":4,"relative_equation_residual":0.0003486945566782065},{"case":7,"passes":8,"relative_equation_residual":3.851516261146427e-07}],"update":240},{"exposures_per_case":40,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.21282892593889885},{"case":0,"passes":2,"relative_equation_residual":0.036617726138356094},{"case":0,"passes":4,"relative_equation_residual":0.0012228126552580037},{"case":0,"passes":8,"relative_equation_residual":1.3467216411660708e-06},{"case":1,"passes":1,"relative_equation_residual":0.11357669206332768},{"case":1,"passes":2,"relative_equation_residual":0.010305631081058314},{"case":1,"passes":4,"relative_equation_residual":8.969688451388586e-05},{"case":1,"passes":8,"relative_equation_residual":6.620908163252418e-09},{"case":2,"passes":1,"relative_equation_residual":0.14053801355022558},{"case":2,"passes":2,"relative_equation_residual":0.006954852423950765},{"case":2,"passes":4,"relative_equation_residual":5.450252181092102e-05},{"case":2,"passes":8,"relative_equation_residual":4.46741216822316e-09},{"case":3,"passes":1,"relative_equation_residual":0.049468275287987425},{"case":3,"passes":2,"relative_equation_residual":0.002991259362700531},{"case":3,"passes":4,"relative_equation_residual":1.962058489413286e-05},{"case":3,"passes":8,"relative_equation_residual":9.125486754982912e-10},{"case":4,"passes":1,"relative_equation_residual":0.02210259550625827},{"case":4,"passes":2,"relative_equation_residual":0.0009243943621912757},{"case":4,"passes":4,"relative_equation_residual":9.239093757456384e-07},{"case":4,"passes":8,"relative_equation_residual":2.0754360581844258e-11},{"case":5,"passes":1,"relative_equation_residual":0.04946736464856834},{"case":5,"passes":2,"relative_equation_residual":0.0029911732687827155},{"case":5,"passes":4,"relative_equation_residual":1.961991579406892e-05},{"case":5,"passes":8,"relative_equation_residual":9.116697740019777e-10},{"case":6,"passes":1,"relative_equation_residual":0.016058458593393055},{"case":6,"passes":2,"relative_equation_residual":0.0005962417892714606},{"case":6,"passes":4,"relative_equation_residual":2.785289518769977e-07},{"case":6,"passes":8,"relative_equation_residual":5.6503335555281755e-11},{"case":7,"passes":1,"relative_equation_residual":0.15985833948330447},{"case":7,"passes":2,"relative_equation_residual":0.03012654851647668},{"case":7,"passes":4,"relative_equation_residual":0.0015242610105592371},{"case":7,"passes":8,"relative_equation_residual":4.61789256204948e-06}],"update":320},{"exposures_per_case":50,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.12807457679802337},{"case":0,"passes":2,"relative_equation_residual":0.006472731200004593},{"case":0,"passes":4,"relative_equation_residual":2.7754668996539066e-05},{"case":0,"passes":8,"relative_equation_residual":4.914064040754235e-10},{"case":1,"passes":1,"relative_equation_residual":0.0743436463679482},{"case":1,"passes":2,"relative_equation_residual":0.0036061159491584584},{"case":1,"passes":4,"relative_equation_residual":1.4012571570194902e-05},{"case":1,"passes":8,"relative_equation_residual":1.781744575451372e-10},{"case":2,"passes":1,"relative_equation_residual":0.16662033626188696},{"case":2,"passes":2,"relative_equation_residual":0.01663252771637197},{"case":2,"passes":4,"relative_equation_residual":0.00015172603045600675},{"case":2,"passes":8,"relative_equation_residual":1.2349059366032694e-08},{"case":3,"passes":1,"relative_equation_residual":0.08866835594546323},{"case":3,"passes":2,"relative_equation_residual":0.0075212852656899015},{"case":3,"passes":4,"relative_equation_residual":4.7472064880152604e-05},{"case":3,"passes":8,"relative_equation_residual":1.8517055414702252e-09},{"case":4,"passes":1,"relative_equation_residual":0.03486481942239937},{"case":4,"passes":2,"relative_equation_residual":0.0014077693135686818},{"case":4,"passes":4,"relative_equation_residual":2.8924328531396623e-06},{"case":4,"passes":8,"relative_equation_residual":1.9586149949055265e-11},{"case":5,"passes":1,"relative_equation_residual":0.08866822260204789},{"case":5,"passes":2,"relative_equation_residual":0.00752118787302709},{"case":5,"passes":4,"relative_equation_residual":4.7473111115752915e-05},{"case":5,"passes":8,"relative_equation_residual":1.8519634120022345e-09},{"case":6,"passes":1,"relative_equation_residual":0.03812584557231538},{"case":6,"passes":2,"relative_equation_residual":0.001206585855032293},{"case":6,"passes":4,"relative_equation_residual":9.369023126213496e-07},{"case":6,"passes":8,"relative_equation_residual":8.48874157567739e-11},{"case":7,"passes":1,"relative_equation_residual":0.20468364531180544},{"case":7,"passes":2,"relative_equation_residual":0.03720727791317943},{"case":7,"passes":4,"relative_equation_residual":0.0010908751295916148},{"case":7,"passes":8,"relative_equation_residual":1.2055139342282301e-06}],"update":400},{"exposures_per_case":60,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.19297444920114065},{"case":0,"passes":2,"relative_equation_residual":0.038980042085898725},{"case":0,"passes":4,"relative_equation_residual":0.0016401755864503265},{"case":0,"passes":8,"relative_equation_residual":2.8479352320941623e-06},{"case":1,"passes":1,"relative_equation_residual":0.10913797915063002},{"case":1,"passes":2,"relative_equation_residual":0.008341256668689253},{"case":1,"passes":4,"relative_equation_residual":4.929937936229502e-05},{"case":1,"passes":8,"relative_equation_residual":8.745516496595158e-10},{"case":2,"passes":1,"relative_equation_residual":0.20336235079722392},{"case":2,"passes":2,"relative_equation_residual":0.020377553036094304},{"case":2,"passes":4,"relative_equation_residual":0.0005888105393015494},{"case":2,"passes":8,"relative_equation_residual":4.127801826685734e-07},{"case":3,"passes":1,"relative_equation_residual":0.07128300164139953},{"case":3,"passes":2,"relative_equation_residual":0.00509456419018546},{"case":3,"passes":4,"relative_equation_residual":2.2650972849733028e-05},{"case":3,"passes":8,"relative_equation_residual":5.885729753274744e-10},{"case":4,"passes":1,"relative_equation_residual":0.024662926984461774},{"case":4,"passes":2,"relative_equation_residual":0.0012303411230602868},{"case":4,"passes":4,"relative_equation_residual":7.210393808813623e-06},{"case":4,"passes":8,"relative_equation_residual":2.1323372829036105e-10},{"case":5,"passes":1,"relative_equation_residual":0.07128243289692557},{"case":5,"passes":2,"relative_equation_residual":0.005094645739813616},{"case":5,"passes":4,"relative_equation_residual":2.2651219880363012e-05},{"case":5,"passes":8,"relative_equation_residual":5.88511789246175e-10},{"case":6,"passes":1,"relative_equation_residual":0.008423997687297199},{"case":6,"passes":2,"relative_equation_residual":0.00041191300892603116},{"case":6,"passes":4,"relative_equation_residual":1.1697429225496194e-06},{"case":6,"passes":8,"relative_equation_residual":3.045861926971236e-11},{"case":7,"passes":1,"relative_equation_residual":0.13879888198467574},{"case":7,"passes":2,"relative_equation_residual":0.012177167170787625},{"case":7,"passes":4,"relative_equation_residual":0.00026112422090751136},{"case":7,"passes":8,"relative_equation_residual":5.4504304919161946e-08}],"update":480},{"exposures_per_case":70,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.12895120231817686},{"case":0,"passes":2,"relative_equation_residual":0.007894432660707742},{"case":0,"passes":4,"relative_equation_residual":9.613526165949727e-05},{"case":0,"passes":8,"relative_equation_residual":9.407497279413538e-09},{"case":1,"passes":1,"relative_equation_residual":0.06136893118988412},{"case":1,"passes":2,"relative_equation_residual":0.0025279523486705312},{"case":1,"passes":4,"relative_equation_residual":6.01425204164312e-06},{"case":1,"passes":8,"relative_equation_residual":1.378619741576941e-10},{"case":2,"passes":1,"relative_equation_residual":0.183186869847774},{"case":2,"passes":2,"relative_equation_residual":0.01573023924807648},{"case":2,"passes":4,"relative_equation_residual":0.00017076186805767897},{"case":2,"passes":8,"relative_equation_residual":1.068618589738744e-08},{"case":3,"passes":1,"relative_equation_residual":0.08190687275449036},{"case":3,"passes":2,"relative_equation_residual":0.006423899672363833},{"case":3,"passes":4,"relative_equation_residual":3.514132137596067e-05},{"case":3,"passes":8,"relative_equation_residual":4.5806896639996673e-10},{"case":4,"passes":1,"relative_equation_residual":0.020605820427593163},{"case":4,"passes":2,"relative_equation_residual":0.0009834732856348737},{"case":4,"passes":4,"relative_equation_residual":2.482315651974929e-06},{"case":4,"passes":8,"relative_equation_residual":4.957317426417113e-11},{"case":5,"passes":1,"relative_equation_residual":0.0819071395547162},{"case":5,"passes":2,"relative_equation_residual":0.006423967865616699},{"case":5,"passes":4,"relative_equation_residual":3.514319361477003e-05},{"case":5,"passes":8,"relative_equation_residual":4.593167816391296e-10},{"case":6,"passes":1,"relative_equation_residual":0.03044078827985576},{"case":6,"passes":2,"relative_equation_residual":0.0010365269556799142},{"case":6,"passes":4,"relative_equation_residual":2.2915647071217536e-06},{"case":6,"passes":8,"relative_equation_residual":3.42945688547383e-11},{"case":7,"passes":1,"relative_equation_residual":0.17888652707726968},{"case":7,"passes":2,"relative_equation_residual":0.02846468714431472},{"case":7,"passes":4,"relative_equation_residual":0.0005926220343827135},{"case":7,"passes":8,"relative_equation_residual":4.784964907185458e-07}],"update":560},{"exposures_per_case":80,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.186521522912211},{"case":0,"passes":2,"relative_equation_residual":0.02681588707502331},{"case":0,"passes":4,"relative_equation_residual":0.0003363794704354511},{"case":0,"passes":8,"relative_equation_residual":2.3285837024649723e-08},{"case":1,"passes":1,"relative_equation_residual":0.15331761339349914},{"case":1,"passes":2,"relative_equation_residual":0.019121161128597266},{"case":1,"passes":4,"relative_equation_residual":0.0002625174835550583},{"case":1,"passes":8,"relative_equation_residual":4.82660967892153e-08},{"case":2,"passes":1,"relative_equation_residual":0.1624799505594076},{"case":2,"passes":2,"relative_equation_residual":0.010882522449301218},{"case":2,"passes":4,"relative_equation_residual":0.00019779497550789747},{"case":2,"passes":8,"relative_equation_residual":1.4586194093616048e-08},{"case":3,"passes":1,"relative_equation_residual":0.05403502318627054},{"case":3,"passes":2,"relative_equation_residual":0.0032610559471532518},{"case":3,"passes":4,"relative_equation_residual":1.8172469401746803e-05},{"case":3,"passes":8,"relative_equation_residual":3.639909230632234e-10},{"case":4,"passes":1,"relative_equation_residual":0.06172657233815181},{"case":4,"passes":2,"relative_equation_residual":0.0038492381746415127},{"case":4,"passes":4,"relative_equation_residual":1.4563632855292011e-05},{"case":4,"passes":8,"relative_equation_residual":6.378713808678763e-10},{"case":5,"passes":1,"relative_equation_residual":0.05403466220922023},{"case":5,"passes":2,"relative_equation_residual":0.0032609493377860777},{"case":5,"passes":4,"relative_equation_residual":1.8171932920023422e-05},{"case":5,"passes":8,"relative_equation_residual":3.649786190393997e-10},{"case":6,"passes":1,"relative_equation_residual":0.04450589440203883},{"case":6,"passes":2,"relative_equation_residual":0.0017550531224847729},{"case":6,"passes":4,"relative_equation_residual":2.491899635655323e-06},{"case":6,"passes":8,"relative_equation_residual":7.450135089715174e-11},{"case":7,"passes":1,"relative_equation_residual":0.14596464562649883},{"case":7,"passes":2,"relative_equation_residual":0.010217470359435486},{"case":7,"passes":4,"relative_equation_residual":8.33219490282166e-06},{"case":7,"passes":8,"relative_equation_residual":2.331667587490108e-10}],"update":640},{"exposures_per_case":90,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.10015216761585043},{"case":0,"passes":2,"relative_equation_residual":0.0036808470715528198},{"case":0,"passes":4,"relative_equation_residual":1.8285334126121904e-05},{"case":0,"passes":8,"relative_equation_residual":1.0514872287154832e-10},{"case":1,"passes":1,"relative_equation_residual":0.042670825938278485},{"case":1,"passes":2,"relative_equation_residual":0.0029158222777290873},{"case":1,"passes":4,"relative_equation_residual":2.2032875636413852e-06},{"case":1,"passes":8,"relative_equation_residual":2.6082728878942596e-11},{"case":2,"passes":1,"relative_equation_residual":0.10583247350022479},{"case":2,"passes":2,"relative_equation_residual":0.008127883207909907},{"case":2,"passes":4,"relative_equation_residual":3.218390115002958e-05},{"case":2,"passes":8,"relative_equation_residual":8.720873946273396e-10},{"case":3,"passes":1,"relative_equation_residual":0.06294814393069645},{"case":3,"passes":2,"relative_equation_residual":0.0018661050596893755},{"case":3,"passes":4,"relative_equation_residual":2.1888064518570062e-06},{"case":3,"passes":8,"relative_equation_residual":4.451669532540127e-11},{"case":4,"passes":1,"relative_equation_residual":0.02592785024338926},{"case":4,"passes":2,"relative_equation_residual":0.0008129673544202706},{"case":4,"passes":4,"relative_equation_residual":1.3081951984820983e-06},{"case":4,"passes":8,"relative_equation_residual":2.8890881071309597e-11},{"case":5,"passes":1,"relative_equation_residual":0.06294768988639336},{"case":5,"passes":2,"relative_equation_residual":0.0018660715776668243},{"case":5,"passes":4,"relative_equation_residual":2.187933922557291e-06},{"case":5,"passes":8,"relative_equation_residual":3.504556487061614e-11},{"case":6,"passes":1,"relative_equation_residual":0.032935219790616976},{"case":6,"passes":2,"relative_equation_residual":0.000758715591310994},{"case":6,"passes":4,"relative_equation_residual":5.350112868668998e-07},{"case":6,"passes":8,"relative_equation_residual":7.268768000462226e-11},{"case":7,"passes":1,"relative_equation_residual":0.21551032772248244},{"case":7,"passes":2,"relative_equation_residual":0.04785982139642315},{"case":7,"passes":4,"relative_equation_residual":0.002611985622173537},{"case":7,"passes":8,"relative_equation_residual":8.719558377104508e-06}],"update":720},{"exposures_per_case":100,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.12253290994303145},{"case":0,"passes":2,"relative_equation_residual":0.005568176211835363},{"case":0,"passes":4,"relative_equation_residual":4.842917959350759e-05},{"case":0,"passes":8,"relative_equation_residual":2.041187308353391e-09},{"case":1,"passes":1,"relative_equation_residual":0.06530157757266937},{"case":1,"passes":2,"relative_equation_residual":0.0024350743979164543},{"case":1,"passes":4,"relative_equation_residual":9.320294013056792e-06},{"case":1,"passes":8,"relative_equation_residual":8.054960363563183e-11},{"case":2,"passes":1,"relative_equation_residual":0.13811502260251599},{"case":2,"passes":2,"relative_equation_residual":0.005916475416067485},{"case":2,"passes":4,"relative_equation_residual":1.4114743891564608e-05},{"case":2,"passes":8,"relative_equation_residual":1.412741981931692e-10},{"case":3,"passes":1,"relative_equation_residual":0.06792611249900792},{"case":3,"passes":2,"relative_equation_residual":0.002887260087801505},{"case":3,"passes":4,"relative_equation_residual":7.839952348270381e-06},{"case":3,"passes":8,"relative_equation_residual":5.935247875625006e-11},{"case":4,"passes":1,"relative_equation_residual":0.05161098975430774},{"case":4,"passes":2,"relative_equation_residual":0.0014555946908010248},{"case":4,"passes":4,"relative_equation_residual":2.25021306325609e-06},{"case":4,"passes":8,"relative_equation_residual":5.593233784492245e-11},{"case":5,"passes":1,"relative_equation_residual":0.06792728752655448},{"case":5,"passes":2,"relative_equation_residual":0.0028873422544630264},{"case":5,"passes":4,"relative_equation_residual":7.83888488391665e-06},{"case":5,"passes":8,"relative_equation_residual":6.332782090184505e-11},{"case":6,"passes":1,"relative_equation_residual":0.025518169462964348},{"case":6,"passes":2,"relative_equation_residual":0.0012221539035513704},{"case":6,"passes":4,"relative_equation_residual":2.653988838438868e-07},{"case":6,"passes":8,"relative_equation_residual":2.0563362888841738e-10},{"case":7,"passes":1,"relative_equation_residual":0.1865350167229015},{"case":7,"passes":2,"relative_equation_residual":0.027136540778329943},{"case":7,"passes":4,"relative_equation_residual":0.0004448721319290567},{"case":7,"passes":8,"relative_equation_residual":2.9733915471912683e-07}],"update":800},{"exposures_per_case":110,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.13897713738849002},{"case":0,"passes":2,"relative_equation_residual":0.006625185340478068},{"case":0,"passes":4,"relative_equation_residual":2.8140609336985994e-05},{"case":0,"passes":8,"relative_equation_residual":4.3450820125866333e-10},{"case":1,"passes":1,"relative_equation_residual":0.062324149891567586},{"case":1,"passes":2,"relative_equation_residual":0.00141619961834192},{"case":1,"passes":4,"relative_equation_residual":9.79375290475933e-07},{"case":1,"passes":8,"relative_equation_residual":3.490701286518928e-11},{"case":2,"passes":1,"relative_equation_residual":0.14480295508515326},{"case":2,"passes":2,"relative_equation_residual":0.0025062540128284352},{"case":2,"passes":4,"relative_equation_residual":7.99828417603625e-06},{"case":2,"passes":8,"relative_equation_residual":4.9851411680651733e-11},{"case":3,"passes":1,"relative_equation_residual":0.03629256227319648},{"case":3,"passes":2,"relative_equation_residual":0.0007401697384231601},{"case":3,"passes":4,"relative_equation_residual":1.7019095031222902e-07},{"case":3,"passes":8,"relative_equation_residual":3.6057814866920516e-11},{"case":4,"passes":1,"relative_equation_residual":0.02959944119181242},{"case":4,"passes":2,"relative_equation_residual":0.0005956874898994181},{"case":4,"passes":4,"relative_equation_residual":6.951426416939864e-07},{"case":4,"passes":8,"relative_equation_residual":4.443364494848194e-11},{"case":5,"passes":1,"relative_equation_residual":0.03629285040770127},{"case":5,"passes":2,"relative_equation_residual":0.0007401589853965174},{"case":5,"passes":4,"relative_equation_residual":1.7114304060381244e-07},{"case":5,"passes":8,"relative_equation_residual":2.3362021706293363e-11},{"case":6,"passes":1,"relative_equation_residual":0.018512195475813738},{"case":6,"passes":2,"relative_equation_residual":0.000562366186322635},{"case":6,"passes":4,"relative_equation_residual":1.533741939906698e-07},{"case":6,"passes":8,"relative_equation_residual":1.8319238212952215e-10},{"case":7,"passes":1,"relative_equation_residual":0.15489807002601683},{"case":7,"passes":2,"relative_equation_residual":0.016938738905689962},{"case":7,"passes":4,"relative_equation_residual":8.625597203280389e-05},{"case":7,"passes":8,"relative_equation_residual":2.0425341984028498e-08}],"update":880},{"exposures_per_case":120,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.15480673702450667},{"case":0,"passes":2,"relative_equation_residual":0.009683680661200303},{"case":0,"passes":4,"relative_equation_residual":1.8175985983089446e-05},{"case":0,"passes":8,"relative_equation_residual":4.24067787324842e-11},{"case":1,"passes":1,"relative_equation_residual":0.07410122329994935},{"case":1,"passes":2,"relative_equation_residual":0.0010521693356018217},{"case":1,"passes":4,"relative_equation_residual":7.785186048986494e-07},{"case":1,"passes":8,"relative_equation_residual":4.4598527088033035e-11},{"case":2,"passes":1,"relative_equation_residual":0.1453163254533288},{"case":2,"passes":2,"relative_equation_residual":0.0018157259984982932},{"case":2,"passes":4,"relative_equation_residual":2.530679370425792e-06},{"case":2,"passes":8,"relative_equation_residual":3.408732001643445e-11},{"case":3,"passes":1,"relative_equation_residual":0.023890875938702563},{"case":3,"passes":2,"relative_equation_residual":0.00017303699200965362},{"case":3,"passes":4,"relative_equation_residual":9.312414146763051e-08},{"case":3,"passes":8,"relative_equation_residual":2.475719247050943e-11},{"case":4,"passes":1,"relative_equation_residual":0.01920682464087382},{"case":4,"passes":2,"relative_equation_residual":0.00041975816920850675},{"case":4,"passes":4,"relative_equation_residual":1.1139384558253588e-07},{"case":4,"passes":8,"relative_equation_residual":6.270441943026426e-11},{"case":5,"passes":1,"relative_equation_residual":0.023889271391261315},{"case":5,"passes":2,"relative_equation_residual":0.00017307609631070982},{"case":5,"passes":4,"relative_equation_residual":9.540985612307689e-08},{"case":5,"passes":8,"relative_equation_residual":2.6563457140848927e-11},{"case":6,"passes":1,"relative_equation_residual":0.013733782641921332},{"case":6,"passes":2,"relative_equation_residual":0.00016277987166556438},{"case":6,"passes":4,"relative_equation_residual":8.323734732282165e-08},{"case":6,"passes":8,"relative_equation_residual":1.3920038950873115e-10},{"case":7,"passes":1,"relative_equation_residual":0.14154504196967732},{"case":7,"passes":2,"relative_equation_residual":0.014300171214086168},{"case":7,"passes":4,"relative_equation_residual":7.287024844907769e-05},{"case":7,"passes":8,"relative_equation_residual":4.086283555093417e-08}],"update":960},{"exposures_per_case":130,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.1514536866917009},{"case":0,"passes":2,"relative_equation_residual":0.008584441320333544},{"case":0,"passes":4,"relative_equation_residual":1.934717686065828e-05},{"case":0,"passes":8,"relative_equation_residual":1.6599558385335096e-10},{"case":1,"passes":1,"relative_equation_residual":0.07287895357006786},{"case":1,"passes":2,"relative_equation_residual":0.0005505957037308997},{"case":1,"passes":4,"relative_equation_residual":1.0741572260074142e-06},{"case":1,"passes":8,"relative_equation_residual":4.17288926583975e-11},{"case":2,"passes":1,"relative_equation_residual":0.1508301024877797},{"case":2,"passes":2,"relative_equation_residual":0.0019307451258295618},{"case":2,"passes":4,"relative_equation_residual":1.676561437194513e-05},{"case":2,"passes":8,"relative_equation_residual":6.835081593572677e-11},{"case":3,"passes":1,"relative_equation_residual":0.02064442608194061},{"case":3,"passes":2,"relative_equation_residual":0.0002234162983741896},{"case":3,"passes":4,"relative_equation_residual":9.735137512794367e-08},{"case":3,"passes":8,"relative_equation_residual":1.6809460982413995e-11},{"case":4,"passes":1,"relative_equation_residual":0.010996675023066593},{"case":4,"passes":2,"relative_equation_residual":8.499790179886082e-05},{"case":4,"passes":4,"relative_equation_residual":5.786722992211224e-08},{"case":4,"passes":8,"relative_equation_residual":8.235614445916191e-11},{"case":5,"passes":1,"relative_equation_residual":0.02064249450996771},{"case":5,"passes":2,"relative_equation_residual":0.00022336087110640847},{"case":5,"passes":4,"relative_equation_residual":9.63329204545261e-08},{"case":5,"passes":8,"relative_equation_residual":1.5064345714748616e-11},{"case":6,"passes":1,"relative_equation_residual":0.01192049390108958},{"case":6,"passes":2,"relative_equation_residual":0.00015185847921320573},{"case":6,"passes":4,"relative_equation_residual":9.175388710390818e-08},{"case":6,"passes":8,"relative_equation_residual":2.108343045166268e-10},{"case":7,"passes":1,"relative_equation_residual":0.1425456273056923},{"case":7,"passes":2,"relative_equation_residual":0.013689739700900408},{"case":7,"passes":4,"relative_equation_residual":5.049463822776866e-05},{"case":7,"passes":8,"relative_equation_residual":5.866899672367941e-09}],"update":1040},{"exposures_per_case":140,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.13155044859310636},{"case":0,"passes":2,"relative_equation_residual":0.00534956767328368},{"case":0,"passes":4,"relative_equation_residual":9.065531953770365e-06},{"case":0,"passes":8,"relative_equation_residual":6.728487418472057e-11},{"case":1,"passes":1,"relative_equation_residual":0.061810960730589},{"case":1,"passes":2,"relative_equation_residual":0.0006488104750323682},{"case":1,"passes":4,"relative_equation_residual":2.0547784870472645e-07},{"case":1,"passes":8,"relative_equation_residual":4.5071516072296376e-11},{"case":2,"passes":1,"relative_equation_residual":0.15254922535719662},{"case":2,"passes":2,"relative_equation_residual":0.0018037480492695732},{"case":2,"passes":4,"relative_equation_residual":3.2961881683055813e-06},{"case":2,"passes":8,"relative_equation_residual":6.624176470884811e-11},{"case":3,"passes":1,"relative_equation_residual":0.021456620575455686},{"case":3,"passes":2,"relative_equation_residual":0.0003836121592469573},{"case":3,"passes":4,"relative_equation_residual":1.6873747180297222e-07},{"case":3,"passes":8,"relative_equation_residual":2.0216284978575192e-11},{"case":4,"passes":1,"relative_equation_residual":0.01188936310790311},{"case":4,"passes":2,"relative_equation_residual":0.00013246756366474676},{"case":4,"passes":4,"relative_equation_residual":7.27243829067494e-08},{"case":4,"passes":8,"relative_equation_residual":6.435399980522812e-11},{"case":5,"passes":1,"relative_equation_residual":0.02145647916330207},{"case":5,"passes":2,"relative_equation_residual":0.00038375569176723735},{"case":5,"passes":4,"relative_equation_residual":1.686055460936885e-07},{"case":5,"passes":8,"relative_equation_residual":2.7870133172919844e-11},{"case":6,"passes":1,"relative_equation_residual":0.007400401451740071},{"case":6,"passes":2,"relative_equation_residual":0.00011979677518702297},{"case":6,"passes":4,"relative_equation_residual":1.225116527692628e-07},{"case":6,"passes":8,"relative_equation_residual":2.816164047430062e-10},{"case":7,"passes":1,"relative_equation_residual":0.1516246483294202},{"case":7,"passes":2,"relative_equation_residual":0.015633687463872527},{"case":7,"passes":4,"relative_equation_residual":6.406705261568238e-05},{"case":7,"passes":8,"relative_equation_residual":2.7753254207615593e-08}],"update":1120},{"exposures_per_case":150,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.13696129945964353},{"case":0,"passes":2,"relative_equation_residual":0.007586246731457555},{"case":0,"passes":4,"relative_equation_residual":1.1227810433245062e-05},{"case":0,"passes":8,"relative_equation_residual":5.5590099905351615e-11},{"case":1,"passes":1,"relative_equation_residual":0.06722915469958791},{"case":1,"passes":2,"relative_equation_residual":0.0007903504645188399},{"case":1,"passes":4,"relative_equation_residual":1.8239419375675775e-06},{"case":1,"passes":8,"relative_equation_residual":3.3986286848763085e-11},{"case":2,"passes":1,"relative_equation_residual":0.15133271035723325},{"case":2,"passes":2,"relative_equation_residual":0.0014300208394370474},{"case":2,"passes":4,"relative_equation_residual":3.74812280464902e-06},{"case":2,"passes":8,"relative_equation_residual":4.006073680232447e-11},{"case":3,"passes":1,"relative_equation_residual":0.01726769225204108},{"case":3,"passes":2,"relative_equation_residual":0.00047455266582930255},{"case":3,"passes":4,"relative_equation_residual":4.481838163475035e-07},{"case":3,"passes":8,"relative_equation_residual":3.811017304975534e-11},{"case":4,"passes":1,"relative_equation_residual":0.012016514094212687},{"case":4,"passes":2,"relative_equation_residual":8.239136637467305e-05},{"case":4,"passes":4,"relative_equation_residual":4.230098818117615e-08},{"case":4,"passes":8,"relative_equation_residual":5.263517915399045e-11},{"case":5,"passes":1,"relative_equation_residual":0.01726774701454581},{"case":5,"passes":2,"relative_equation_residual":0.0004745974776733469},{"case":5,"passes":4,"relative_equation_residual":4.4769632708609156e-07},{"case":5,"passes":8,"relative_equation_residual":1.6582485524454635e-11},{"case":6,"passes":1,"relative_equation_residual":0.006211032861645313},{"case":6,"passes":2,"relative_equation_residual":5.228670982977153e-05},{"case":6,"passes":4,"relative_equation_residual":5.992349983593233e-08},{"case":6,"passes":8,"relative_equation_residual":1.0147673307363722e-10},{"case":7,"passes":1,"relative_equation_residual":0.14483101532639395},{"case":7,"passes":2,"relative_equation_residual":0.013983451980061928},{"case":7,"passes":4,"relative_equation_residual":4.414319930377904e-05},{"case":7,"passes":8,"relative_equation_residual":1.09455476466547e-08}],"update":1200},{"exposures_per_case":160,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.13991430525919968},{"case":0,"passes":2,"relative_equation_residual":0.007778161310900754},{"case":0,"passes":4,"relative_equation_residual":9.333720136225795e-06},{"case":0,"passes":8,"relative_equation_residual":2.994300290527781e-10},{"case":1,"passes":1,"relative_equation_residual":0.069676281993784},{"case":1,"passes":2,"relative_equation_residual":0.0006105877191023851},{"case":1,"passes":4,"relative_equation_residual":2.998054806193843e-07},{"case":1,"passes":8,"relative_equation_residual":4.455843347319618e-11},{"case":2,"passes":1,"relative_equation_residual":0.15175801628642085},{"case":2,"passes":2,"relative_equation_residual":0.0014721041923037052},{"case":2,"passes":4,"relative_equation_residual":3.902787501652948e-06},{"case":2,"passes":8,"relative_equation_residual":7.904989242080473e-11},{"case":3,"passes":1,"relative_equation_residual":0.010803660762871741},{"case":3,"passes":2,"relative_equation_residual":2.9449370440473498e-05},{"case":3,"passes":4,"relative_equation_residual":3.330582017478195e-08},{"case":3,"passes":8,"relative_equation_residual":2.0612820943430136e-11},{"case":4,"passes":1,"relative_equation_residual":0.014609024826713907},{"case":4,"passes":2,"relative_equation_residual":0.00016408132154743706},{"case":4,"passes":4,"relative_equation_residual":9.860288140755851e-08},{"case":4,"passes":8,"relative_equation_residual":3.36189252306325e-11},{"case":5,"passes":1,"relative_equation_residual":0.010803419676690839},{"case":5,"passes":2,"relative_equation_residual":2.9438745545526145e-05},{"case":5,"passes":4,"relative_equation_residual":3.354722268054491e-08},{"case":5,"passes":8,"relative_equation_residual":1.2800637927032465e-11},{"case":6,"passes":1,"relative_equation_residual":0.002641488554019981},{"case":6,"passes":2,"relative_equation_residual":6.326834767316236e-06},{"case":6,"passes":4,"relative_equation_residual":5.179722340532867e-08},{"case":6,"passes":8,"relative_equation_residual":8.232207739262613e-11},{"case":7,"passes":1,"relative_equation_residual":0.1446730228197313},{"case":7,"passes":2,"relative_equation_residual":0.013348463981921205},{"case":7,"passes":4,"relative_equation_residual":3.83876602139266e-05},{"case":7,"passes":8,"relative_equation_residual":5.968068349728304e-09}],"update":1280},{"exposures_per_case":170,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.1399174436176986},{"case":0,"passes":2,"relative_equation_residual":0.007566324393856608},{"case":0,"passes":4,"relative_equation_residual":5.333338008313659e-06},{"case":0,"passes":8,"relative_equation_residual":1.737491231201834e-10},{"case":1,"passes":1,"relative_equation_residual":0.06978586745071326},{"case":1,"passes":2,"relative_equation_residual":0.0006112938296018318},{"case":1,"passes":4,"relative_equation_residual":2.924444074518441e-07},{"case":1,"passes":8,"relative_equation_residual":3.177334590198835e-11},{"case":2,"passes":1,"relative_equation_residual":0.15252576652984204},{"case":2,"passes":2,"relative_equation_residual":0.0014370703580328119},{"case":2,"passes":4,"relative_equation_residual":1.0436465043737418e-06},{"case":2,"passes":8,"relative_equation_residual":4.608274353484016e-11},{"case":3,"passes":1,"relative_equation_residual":0.010575644544109955},{"case":3,"passes":2,"relative_equation_residual":1.9672561373134048e-05},{"case":3,"passes":4,"relative_equation_residual":3.483957735858791e-08},{"case":3,"passes":8,"relative_equation_residual":1.9091888223256958e-11},{"case":4,"passes":1,"relative_equation_residual":0.014222764780543169},{"case":4,"passes":2,"relative_equation_residual":0.0001705180807064762},{"case":4,"passes":4,"relative_equation_residual":1.0218905479622387e-07},{"case":4,"passes":8,"relative_equation_residual":1.7526150869029086e-11},{"case":5,"passes":1,"relative_equation_residual":0.010574174001408384},{"case":5,"passes":2,"relative_equation_residual":1.9676280193950845e-05},{"case":5,"passes":4,"relative_equation_residual":3.45925829523897e-08},{"case":5,"passes":8,"relative_equation_residual":1.9774607410105375e-11},{"case":6,"passes":1,"relative_equation_residual":0.002645072455570366},{"case":6,"passes":2,"relative_equation_residual":6.266078873934564e-06},{"case":6,"passes":4,"relative_equation_residual":5.6028344677234844e-08},{"case":6,"passes":8,"relative_equation_residual":1.0051251372994814e-10},{"case":7,"passes":1,"relative_equation_residual":0.1449431138490152},{"case":7,"passes":2,"relative_equation_residual":0.013355701816609366},{"case":7,"passes":4,"relative_equation_residual":3.786068989437236e-05},{"case":7,"passes":8,"relative_equation_residual":3.988133830200147e-09}],"update":1360},{"exposures_per_case":180,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.14069777155461385},{"case":0,"passes":2,"relative_equation_residual":0.007684925455755356},{"case":0,"passes":4,"relative_equation_residual":5.208466425812275e-06},{"case":0,"passes":8,"relative_equation_residual":1.700560596716616e-10},{"case":1,"passes":1,"relative_equation_residual":0.0699441254615935},{"case":1,"passes":2,"relative_equation_residual":0.0006105526983657503},{"case":1,"passes":4,"relative_equation_residual":2.7907675176627895e-07},{"case":1,"passes":8,"relative_equation_residual":2.0245691961106112e-11},{"case":2,"passes":1,"relative_equation_residual":0.15265694222202797},{"case":2,"passes":2,"relative_equation_residual":0.001477802457458162},{"case":2,"passes":4,"relative_equation_residual":1.001356616781705e-06},{"case":2,"passes":8,"relative_equation_residual":4.314261475599914e-11},{"case":3,"passes":1,"relative_equation_residual":0.010530702301330635},{"case":3,"passes":2,"relative_equation_residual":1.533278856600095e-05},{"case":3,"passes":4,"relative_equation_residual":3.105002262678064e-08},{"case":3,"passes":8,"relative_equation_residual":3.218792268281737e-11},{"case":4,"passes":1,"relative_equation_residual":0.013935080300282123},{"case":4,"passes":2,"relative_equation_residual":0.00015852147776774272},{"case":4,"passes":4,"relative_equation_residual":1.0091326911782413e-07},{"case":4,"passes":8,"relative_equation_residual":3.4432454621151214e-11},{"case":5,"passes":1,"relative_equation_residual":0.01053079598557098},{"case":5,"passes":2,"relative_equation_residual":1.530819620239578e-05},{"case":5,"passes":4,"relative_equation_residual":2.99450987966004e-08},{"case":5,"passes":8,"relative_equation_residual":2.025683829929797e-11},{"case":6,"passes":1,"relative_equation_residual":0.0029470773739796886},{"case":6,"passes":2,"relative_equation_residual":7.413830601918109e-06},{"case":6,"passes":4,"relative_equation_residual":7.173230590081785e-08},{"case":6,"passes":8,"relative_equation_residual":1.8520146602799308e-10},{"case":7,"passes":1,"relative_equation_residual":0.1443703182834512},{"case":7,"passes":2,"relative_equation_residual":0.013239198265248029},{"case":7,"passes":4,"relative_equation_residual":3.7411893813638254e-05},{"case":7,"passes":8,"relative_equation_residual":4.072571559835801e-09}],"update":1440},{"exposures_per_case":190,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.14090253694699373},{"case":0,"passes":2,"relative_equation_residual":0.007728243773149855},{"case":0,"passes":4,"relative_equation_residual":5.169314807332048e-06},{"case":0,"passes":8,"relative_equation_residual":1.7075160448223394e-10},{"case":1,"passes":1,"relative_equation_residual":0.07026142660495698},{"case":1,"passes":2,"relative_equation_residual":0.0006509025208459925},{"case":1,"passes":4,"relative_equation_residual":2.856628609137003e-07},{"case":1,"passes":8,"relative_equation_residual":2.5836205744681662e-11},{"case":2,"passes":1,"relative_equation_residual":0.15272790678145556},{"case":2,"passes":2,"relative_equation_residual":0.0015187277250916405},{"case":2,"passes":4,"relative_equation_residual":8.936629690925619e-07},{"case":2,"passes":8,"relative_equation_residual":5.7212097622938634e-11},{"case":3,"passes":1,"relative_equation_residual":0.01030436297365422},{"case":3,"passes":2,"relative_equation_residual":1.1745300782934496e-05},{"case":3,"passes":4,"relative_equation_residual":2.6585972263670135e-08},{"case":3,"passes":8,"relative_equation_residual":3.0491735221424155e-11},{"case":4,"passes":1,"relative_equation_residual":0.014079289593831302},{"case":4,"passes":2,"relative_equation_residual":0.00016505558799751016},{"case":4,"passes":4,"relative_equation_residual":1.0117491177117237e-07},{"case":4,"passes":8,"relative_equation_residual":3.171893432386177e-11},{"case":5,"passes":1,"relative_equation_residual":0.010303943136194836},{"case":5,"passes":2,"relative_equation_residual":1.1745806404539368e-05},{"case":5,"passes":4,"relative_equation_residual":2.5465280342501058e-08},{"case":5,"passes":8,"relative_equation_residual":2.8177237954866035e-11},{"case":6,"passes":1,"relative_equation_residual":0.002804202456669869},{"case":6,"passes":2,"relative_equation_residual":6.949040272458515e-06},{"case":6,"passes":4,"relative_equation_residual":7.914104872514744e-08},{"case":6,"passes":8,"relative_equation_residual":2.0688089487640745e-10},{"case":7,"passes":1,"relative_equation_residual":0.14428475553737719},{"case":7,"passes":2,"relative_equation_residual":0.013209652764221766},{"case":7,"passes":4,"relative_equation_residual":3.714691829817803e-05},{"case":7,"passes":8,"relative_equation_residual":4.173101272696399e-09}],"update":1520},{"exposures_per_case":200,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.14093581333247726},{"case":0,"passes":2,"relative_equation_residual":0.007735700583729332},{"case":0,"passes":4,"relative_equation_residual":5.164729177634301e-06},{"case":0,"passes":8,"relative_equation_residual":1.7064236908138191e-10},{"case":1,"passes":1,"relative_equation_residual":0.07031538061996762},{"case":1,"passes":2,"relative_equation_residual":0.0006541205065537806},{"case":1,"passes":4,"relative_equation_residual":2.867583136173184e-07},{"case":1,"passes":8,"relative_equation_residual":3.691805543555508e-11},{"case":2,"passes":1,"relative_equation_residual":0.1527275961973562},{"case":2,"passes":2,"relative_equation_residual":0.0015250007058468856},{"case":2,"passes":4,"relative_equation_residual":8.930381167312632e-07},{"case":2,"passes":8,"relative_equation_residual":4.5754702128562844e-11},{"case":3,"passes":1,"relative_equation_residual":0.010285243039054608},{"case":3,"passes":2,"relative_equation_residual":1.134853610523774e-05},{"case":3,"passes":4,"relative_equation_residual":2.5141824242697427e-08},{"case":3,"passes":8,"relative_equation_residual":1.5524524470619733e-11},{"case":4,"passes":1,"relative_equation_residual":0.01409779279439456},{"case":4,"passes":2,"relative_equation_residual":0.00016629599753378963},{"case":4,"passes":4,"relative_equation_residual":1.027255865451802e-07},{"case":4,"passes":8,"relative_equation_residual":2.356912518113395e-11},{"case":5,"passes":1,"relative_equation_residual":0.010284946540685149},{"case":5,"passes":2,"relative_equation_residual":1.1340801870521548e-05},{"case":5,"passes":4,"relative_equation_residual":2.3745159715804012e-08},{"case":5,"passes":8,"relative_equation_residual":1.9866260399291227e-11},{"case":6,"passes":1,"relative_equation_residual":0.0027863886638686483},{"case":6,"passes":2,"relative_equation_residual":6.507485841676666e-06},{"case":6,"passes":4,"relative_equation_residual":5.278449838323377e-08},{"case":6,"passes":8,"relative_equation_residual":7.717764476409969e-11},{"case":7,"passes":1,"relative_equation_residual":0.14426641489782047},{"case":7,"passes":2,"relative_equation_residual":0.013204216695749917},{"case":7,"passes":4,"relative_equation_residual":3.7121952984127015e-05},{"case":7,"passes":8,"relative_equation_residual":4.0952629268066116e-09}],"update":1600}]},"rollout_feedback":{"auxiliary_losses":0,"depth_exposures":[[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50]],"depth_order_sha256":"3adbd1b18830b55ac918ce265241ded42841f53a56341f43b1e333ebd0615257","mode":"U","original_sampling_order_sha256":"d0ea221a7e03a83673ece7fc9cdf4f5212bb54f3023511f93d95c84e45647899"},"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":1600,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"a73fbc342ed070390b47535e6b94b535dec158671393b074a17e8a6858c249da"},"checkpoint-1600.npz":{"bytes":1485452,"role":"checkpoint","sha256":"759ce063892ed6c32991214c461f4bbac07e2e924467682f979244c077f548d0"},"history.json":{"bytes":10184,"role":"training_diagnostics","sha256":"277193000cb80acea428ac43a38cf18807a0fe1b93857088bbf686a7a0ce0b72"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"fc63401e39b8ce05a2746658d397b5dd56d276d95c091607395046662b9db790"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":476,"role":"result_summary","sha256":"96cf75008a4f2d51f95c8f87b95e408a2e30386caac4193da533f482af044c9b"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"759ce063892ed6c32991214c461f4bbac07e2e924467682f979244c077f548d0"}},"per_case_inference_s":[0.02103698201244697,0.004048675997182727,0.00524089700775221,0.004213257023366168,0.0037287099985405803,0.0034620839869603515,0.00441125498036854,0.004582904977723956,0.0094368550053332,0.005189268005779013,0.00619609301793389,0.005638573988107964],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"f9747a9df02c76b1abb81e35bb56526b0aeacad0dc268aff7b857f5281e78f61","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"163efc246aab870eb90a68ebbd61bf0ae0c0a4576a167c78cee2d379273c558f","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"88fb7e33c5fab5d82ca5eb2221aa7b9d920dbbe07d5058cecd2e27652fb845da","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"e982a40d9da899d3dbfcb188d94088a98e7bf8567b8c131a304a243a51a79c92","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"cf3ff598dfdbd766f9926990c1a8c8dacafc4cecbee695bd63370dbe77c32631","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"59b436729a8a5f5921abdcd0c8fd0f09ab72ec3d48281dde0dfa35706066d654","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"151f9fcaa0d4eb77b7657ee8c552a71750468daf5443dc36bc76e45699b53cc8","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"f704d1c4200f874eb9f8aca2de43ab8592c95b5c405a9fcf1c534d6f70806e2a","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"6b36269d1bac2f745c65b592aa3a7692886c909544a8bc5ae46a9a0bd8a863fe","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"51fa8f194c67ca60238d7aae919ba5e2e78178ae35d5bb6f4b516be1a9895717","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"9d576b6bb81e754943d1de1224dfadbc697643a9bc7d94a20e6781ac3e634ce4","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"6feb3bc336bebf70d27d99de82993356a4cdde39bd5ab41fe187184df7ce5220","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2413580288,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
