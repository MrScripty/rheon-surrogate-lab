# Pressure four-pass · c44-d · C44 fine-tune CUDA

Retrospective source review · fit-4dde2944839dc81d

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 370,016-parameter float32 3D pressure U-Net with encoder widths8/16/24/32, bottleneck48 and reversed decoder; physical residual reconstruction is float64. It starts from the exact C44 checkpoint. Training unrolls four pressure corrections on one equation. These corrections are not future physical simulation steps.

**Purpose.** Compare unchanged four-pass fine tuning with attached and detached auxiliary gradient paths under the same initialization, pattern order, depth schedule and optimizer budget. Use the same auxiliary loss but detach the carried pressure before the additional correction.

**Experiment.** The optimizer uses eight synthetic analytical patterns: one [0,0,1] mode on each of eight rectangular grids, at8or16cells/m. Each gets200exposures, with50at each of four auxiliary depths. The original16train trajectories calibrate amplitude/normalization; they are not the optimizer sample stream. No companion physical, direction or current-policy loss runs in the captured fitting-bank updater. Teacher weight is zero and counted teacher forwards are zero. All arms run1,600updates; U counts6,400network forwards and A/D8,000. The loss combines analytical pressure loss, log-scaled final equation residual and 0.05mean early residual feedback; A/D add0.05auxiliary loss.

**Observed outcome.** C74, on the same retained 128 pressure requests: worst mean-free pressure relative L2 0.021523997408947868, equation relative L2 0.0064598383441849877. Parent C44 is better on both: 0.013450717090965448 and 0.0039826222242511878, respectively. No new timing or own-history drift result is implied.

**Lesson.** All three eight-pattern fine-tunes regress against the already stronger C44 parent on both worst pressure and equation error under the unchanged common benchmark. The matched C37 family improved instead, so that improvement does not transfer automatically to another initialization. Among these C44 continuations, A has the lowest worst pressure error and D the lowest worst equation residual; neither supplies an accuracy reason to replace the parent. A narrow analytical fitting objective can disagree with broad native-case results. The data establish the regression, not its sole cause; preserve the parent and validate matched cases before promotion. Equal update counts also conceal the A/D extra forward. The subsequent matched-work CUDA matrix confirms these parent regressions at every tested one/two/four/eight correction budget. C37 benefits at two/four/eight corrections, while its worst pressure error worsens at one correction; its saved parent default uses one and its fine-tunes use two. Keep metric and actual work explicit.

**Reproduction.** Use the exact captured project.json/train-recipe.json and evidence.py entrypoint with the original input bindings, source witness, C44 checkpoint, fitting bank and fine-tuning.json. Clean retraining, fresh qualified latency and new-Rheon data equivalence remain unqualified; no checkpoint can be retired on this review alone.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-detached-correction-v1/c44-d-study-01/train/receipt.json",
    "sha256": "4dde2944839dc81d093da2658a0fd6202c6344febd5a721bae242e90b06b47a5"
  },
  "history": {
    "path": "runs/pressure-detached-correction-v1/c44-d-study-01/train/history.json",
    "sha256": "0416124c4656866426a15242b9e0c548a04c524cecff9376f94db22445afe3a4"
  },
  "sources": [
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/config.json",
      "sha256": "897a09126b1884ccabf2d8da1e92dd12f64300701fe20b72e49e31f3e75d3759"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/fine-tuning.json",
      "sha256": "36dd62a2f93014a6be076fdffb9bec016db61dc8fa18996e911182b446073c1e"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/fine_tuning.py",
      "sha256": "57658249095fd62c38f8d3bcbbe201d5ebe288009f0042001941bca82110795e"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/fit_bank.py",
      "sha256": "bbdbabfc89805410df196dd0724b24347954548dd83790456994ca46a6f5e347"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/model.py",
      "sha256": "a2ebefe6a8ce12c7e6e1e12fa82b835029f364b50e91284f33b60bfe31b0eaee"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/original_fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/original_fit_bank.py",
      "sha256": "e68c1fa6723331af3448edec8294ca0dc67a5b97576a1cd251c5a789e9bf0668"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/train.py",
      "sha256": "b9c98c38dfd22ec77ed1bc6dcbad8e692b276dc96fe9cb5cf928ec2b6be0774b"
    },
    {
      "path": "runs/pressure-detached-correction-v1/c44-d-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-detached-correction-v1/c44-d-project-01/source-witness.json",
    "sha256": "80c7b7f3388ce8ab5821d3389e2d4bc529475743a68b7a16e702b3b09a7250c7"
  },
  "training_population": {
    "path": "runs/pressure-detached-correction-v1/c44-d-study-01/train/splits.json",
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
    "item_id": "fit-4dde2944839dc81d",
    "kind": "training_attempt",
    "name": "Pressure four-pass · c44-d · C44 fine-tune CUDA",
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
    "receipt_sha256": "4dde2944839dc81d093da2658a0fd6202c6344febd5a721bae242e90b06b47a5",
    "receipt_paths": [
      "runs/pressure-detached-correction-v1/c44-d-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "13c030afeb9946f8aa40ef5207fcedf7",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure four-pass · c44-d · C44 fine-tune CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1600,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "897a09126b1884ccabf2d8da1e92dd12f64300701fe20b72e49e31f3e75d3759",
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
      "source_witness_sha256": "80c7b7f3388ce8ab5821d3389e2d4bc529475743a68b7a16e702b3b09a7250c7"
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
        "sha256": "b8ddb4b6660efe3512a017f9da424f718d4824475f7679abf81a6b9c7f66e725",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "b8ddb4b6660efe3512a017f9da424f718d4824475f7679abf81a6b9c7f66e725",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2414006272,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/72"
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
      "receipt": "runs/pressure-detached-correction-v1/c44-d-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-detached-correction-v1/c44-d-study-01/train/receipt.json",
    "sha256": "4dde2944839dc81d093da2658a0fd6202c6344febd5a721bae242e90b06b47a5",
    "bytes": 59974
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C74": {
        "name": "c44-d",
        "weights_sha256": "b8ddb4b6660efe3512a017f9da424f718d4824475f7679abf81a6b9c7f66e725",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.006459838344184988,
          "pressure_mean_free_relative_l2": 0.021523997408947868,
          "pressure_absolute_rms": 0.5687374126600026,
          "pressure_absolute_max": 1.230678312532159
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
              "common_equation_relative_l2": 0.006419398652501287,
              "pressure_mean_free_relative_l2": 0.014262067636273064,
              "pressure_absolute_rms": 0.5687374126600026,
              "pressure_absolute_max": 1.230678312532159
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.006459838344184988,
              "pressure_mean_free_relative_l2": 0.021523997408947868,
              "pressure_absolute_rms": 0.38836453115579306,
              "pressure_absolute_max": 1.0202733635017596
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C74/results.json",
          "sha256": "2d1c8f950744d4f041adeeebd3725e4c766243c604222a77e02ac44d8defa39d"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"c44-d","early_weight":0.05,"fitting_bank":{"amplitude_reference":"original-train-correction-rms","cache_max_bytes":8388608,"grids":[{"cells_per_metre":16,"counts_xyz":[13,15,16]},{"cells_per_metre":16,"counts_xyz":[9,11,11]},{"cells_per_metre":8,"counts_xyz":[15,9,7]},{"cells_per_metre":16,"counts_xyz":[11,5,15]},{"cells_per_metre":8,"counts_xyz":[7,16,5]},{"cells_per_metre":16,"counts_xyz":[11,12,7]},{"cells_per_metre":8,"counts_xyz":[5,13,9]},{"cells_per_metre":16,"counts_xyz":[16,7,11]}],"modes_zyx":[[0,0,1]],"sampling_seed":2026102101},"initial_weights_sha256":"a73fbc342ed070390b47535e6b94b535dec158671393b074a17e8a6858c249da","relative_equation_target":1e-06,"rollout_feedback":{"depth_seed":2026101807,"mode":"D","weight":0.05},"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":4},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"897a09126b1884ccabf2d8da1e92dd12f64300701fe20b72e49e31f3e75d3759","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"a73fbc342ed070390b47535e6b94b535dec158671393b074a17e8a6858c249da"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"21ed766152674818e22d0abb5ed5540690ab31aa14507d78fcc73a381e5d058a","source_witness_sha256":"80c7b7f3388ce8ab5821d3389e2d4bc529475743a68b7a16e702b3b09a7250c7"},"diagnostic_population":"train_only","elapsed_s":83.9064802339999,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":8.746392602176911e-06,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":20238336,"peak_reserved_bytes":39845888,"tf32":false},"fine_tuning_statistics":{"fundamental_bank":{"amplitude_rms_pa":2.8535734640782584,"cases":[{"cells_per_metre":16,"counts_xyz":[13,15,16],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[9,11,11],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[15,9,7],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[11,5,15],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[7,16,5],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[11,12,7],"mode_zyx":[0,0,1]},{"cells_per_metre":8,"counts_xyz":[5,13,9],"mode_zyx":[0,0,1]},{"cells_per_metre":16,"counts_xyz":[16,7,11],"mode_zyx":[0,0,1]}],"description_sha256":"30f7092460b6a649b28e4bf380a5769c2955d1e29461e0f64a6e1d474dab960f","expected_exposures":[200,200,200,200,200,200,200,200],"optimizer_examples":1600,"optimizer_exposures":[200,200,200,200,200,200,200,200],"resident_tensor_bytes":297152,"sampling_order_sha256":"d0ea221a7e03a83673ece7fc9cdf4f5212bb54f3023511f93d95c84e45647899","trace":[{"exposures_per_case":10,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.15211852154692748},{"case":0,"passes":2,"relative_equation_residual":0.0160583988398597},{"case":0,"passes":4,"relative_equation_residual":0.0002366515492396003},{"case":0,"passes":8,"relative_equation_residual":4.5895796629515234e-08},{"case":1,"passes":1,"relative_equation_residual":0.0809873966651877},{"case":1,"passes":2,"relative_equation_residual":0.006312270146015475},{"case":1,"passes":4,"relative_equation_residual":3.871456846475757e-05},{"case":1,"passes":8,"relative_equation_residual":1.7839427609830136e-09},{"case":2,"passes":1,"relative_equation_residual":0.14646219847575545},{"case":2,"passes":2,"relative_equation_residual":0.015037589717488604},{"case":2,"passes":4,"relative_equation_residual":0.00017774245237146661},{"case":2,"passes":8,"relative_equation_residual":3.209734158369848e-08},{"case":3,"passes":1,"relative_equation_residual":0.10269398195564243},{"case":3,"passes":2,"relative_equation_residual":0.010321828163022025},{"case":3,"passes":4,"relative_equation_residual":0.00010212741475691357},{"case":3,"passes":8,"relative_equation_residual":1.0433119154278442e-08},{"case":4,"passes":1,"relative_equation_residual":0.036345402263342125},{"case":4,"passes":2,"relative_equation_residual":0.0017433861842117394},{"case":4,"passes":4,"relative_equation_residual":7.051322103380161e-06},{"case":4,"passes":8,"relative_equation_residual":1.39001519796658e-10},{"case":5,"passes":1,"relative_equation_residual":0.10269471148388926},{"case":5,"passes":2,"relative_equation_residual":0.010321976333670263},{"case":5,"passes":4,"relative_equation_residual":0.00010212991769037677},{"case":5,"passes":8,"relative_equation_residual":1.0433945602798432e-08},{"case":6,"passes":1,"relative_equation_residual":0.011972938961245495},{"case":6,"passes":2,"relative_equation_residual":0.00019801088109502882},{"case":6,"passes":4,"relative_equation_residual":5.396265096026505e-07},{"case":6,"passes":8,"relative_equation_residual":3.5624460277344765e-11},{"case":7,"passes":1,"relative_equation_residual":0.2006769186757138},{"case":7,"passes":2,"relative_equation_residual":0.04576509775355216},{"case":7,"passes":4,"relative_equation_residual":0.0028516800266926816},{"case":7,"passes":8,"relative_equation_residual":1.3245605515554855e-05}],"update":80},{"exposures_per_case":20,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.1946999651536928},{"case":0,"passes":2,"relative_equation_residual":0.018083254011830523},{"case":0,"passes":4,"relative_equation_residual":8.046617875258404e-05},{"case":0,"passes":8,"relative_equation_residual":8.826612416482837e-09},{"case":1,"passes":1,"relative_equation_residual":0.1122498131394687},{"case":1,"passes":2,"relative_equation_residual":0.007498401950311919},{"case":1,"passes":4,"relative_equation_residual":4.3425261325564776e-05},{"case":1,"passes":8,"relative_equation_residual":1.978712567166127e-09},{"case":2,"passes":1,"relative_equation_residual":0.190681173812144},{"case":2,"passes":2,"relative_equation_residual":0.009755244740448041},{"case":2,"passes":4,"relative_equation_residual":2.1638176644324007e-05},{"case":2,"passes":8,"relative_equation_residual":4.1483438760972546e-10},{"case":3,"passes":1,"relative_equation_residual":0.0931168518042325},{"case":3,"passes":2,"relative_equation_residual":0.00528633344765404},{"case":3,"passes":4,"relative_equation_residual":1.1563941745127448e-05},{"case":3,"passes":8,"relative_equation_residual":1.5303903708686508e-10},{"case":4,"passes":1,"relative_equation_residual":0.034892180315598834},{"case":4,"passes":2,"relative_equation_residual":0.0008273251993985852},{"case":4,"passes":4,"relative_equation_residual":1.163926658520539e-06},{"case":4,"passes":8,"relative_equation_residual":4.377585848121675e-11},{"case":5,"passes":1,"relative_equation_residual":0.0931168941232997},{"case":5,"passes":2,"relative_equation_residual":0.005285800948532499},{"case":5,"passes":4,"relative_equation_residual":1.1562968977611799e-05},{"case":5,"passes":8,"relative_equation_residual":1.867276210873898e-10},{"case":6,"passes":1,"relative_equation_residual":0.023895372532371034},{"case":6,"passes":2,"relative_equation_residual":0.0006490906002909956},{"case":6,"passes":4,"relative_equation_residual":1.4492079481220003e-06},{"case":6,"passes":8,"relative_equation_residual":1.387359111284408e-10},{"case":7,"passes":1,"relative_equation_residual":0.1836355914590914},{"case":7,"passes":2,"relative_equation_residual":0.026865822248168136},{"case":7,"passes":4,"relative_equation_residual":0.0005718229058325327},{"case":7,"passes":8,"relative_equation_residual":1.7566727639165022e-07}],"update":160},{"exposures_per_case":30,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.28043720012234424},{"case":0,"passes":2,"relative_equation_residual":0.07409116317798892},{"case":0,"passes":4,"relative_equation_residual":0.004940517974656983},{"case":0,"passes":8,"relative_equation_residual":2.1071114591315663e-05},{"case":1,"passes":1,"relative_equation_residual":0.20241371526622962},{"case":1,"passes":2,"relative_equation_residual":0.03982214529351434},{"case":1,"passes":4,"relative_equation_residual":0.0014795130996378577},{"case":1,"passes":8,"relative_equation_residual":2.0100495213843456e-06},{"case":2,"passes":1,"relative_equation_residual":0.16202543971927805},{"case":2,"passes":2,"relative_equation_residual":0.014505642991133138},{"case":2,"passes":4,"relative_equation_residual":0.00014289847190481666},{"case":2,"passes":8,"relative_equation_residual":3.0936553993003016e-08},{"case":3,"passes":1,"relative_equation_residual":0.07784162423372747},{"case":3,"passes":2,"relative_equation_residual":0.006037783513846545},{"case":3,"passes":4,"relative_equation_residual":7.941541209991535e-05},{"case":3,"passes":8,"relative_equation_residual":5.16986068942464e-09},{"case":4,"passes":1,"relative_equation_residual":0.07241170642937074},{"case":4,"passes":2,"relative_equation_residual":0.004166164825426979},{"case":4,"passes":4,"relative_equation_residual":8.055761981940371e-06},{"case":4,"passes":8,"relative_equation_residual":3.217677839215344e-10},{"case":5,"passes":1,"relative_equation_residual":0.07780509255983696},{"case":5,"passes":2,"relative_equation_residual":0.006033668007924243},{"case":5,"passes":4,"relative_equation_residual":7.941142908741428e-05},{"case":5,"passes":8,"relative_equation_residual":5.167831239782716e-09},{"case":6,"passes":1,"relative_equation_residual":0.05976863876157732},{"case":6,"passes":2,"relative_equation_residual":0.0035616652609210494},{"case":6,"passes":4,"relative_equation_residual":1.2976874604703577e-05},{"case":6,"passes":8,"relative_equation_residual":1.0738106602846178e-09},{"case":7,"passes":1,"relative_equation_residual":0.1775487256292568},{"case":7,"passes":2,"relative_equation_residual":0.023451713215293466},{"case":7,"passes":4,"relative_equation_residual":0.0002849384682355437},{"case":7,"passes":8,"relative_equation_residual":1.2815183254670203e-07}],"update":240},{"exposures_per_case":40,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.18369805764068203},{"case":0,"passes":2,"relative_equation_residual":0.027279320116248555},{"case":0,"passes":4,"relative_equation_residual":0.0005251247502206219},{"case":0,"passes":8,"relative_equation_residual":1.8623357821532181e-07},{"case":1,"passes":1,"relative_equation_residual":0.11599259685266097},{"case":1,"passes":2,"relative_equation_residual":0.011022549372109785},{"case":1,"passes":4,"relative_equation_residual":9.01148202701148e-05},{"case":1,"passes":8,"relative_equation_residual":5.62462500618519e-09},{"case":2,"passes":1,"relative_equation_residual":0.08566449057747712},{"case":2,"passes":2,"relative_equation_residual":0.008108183854840058},{"case":2,"passes":4,"relative_equation_residual":2.5209769742834602e-05},{"case":2,"passes":8,"relative_equation_residual":1.1669243601482493e-09},{"case":3,"passes":1,"relative_equation_residual":0.04941314170129211},{"case":3,"passes":2,"relative_equation_residual":0.002855534002950199},{"case":3,"passes":4,"relative_equation_residual":6.31109679542735e-06},{"case":3,"passes":8,"relative_equation_residual":1.5377044093443204e-09},{"case":4,"passes":1,"relative_equation_residual":0.03126681963579489},{"case":4,"passes":2,"relative_equation_residual":0.001295332809718493},{"case":4,"passes":4,"relative_equation_residual":1.6819825178231034e-06},{"case":4,"passes":8,"relative_equation_residual":2.4287275797856208e-11},{"case":5,"passes":1,"relative_equation_residual":0.04825489479905266},{"case":5,"passes":2,"relative_equation_residual":0.0026312305889712347},{"case":5,"passes":4,"relative_equation_residual":5.163912870887571e-06},{"case":5,"passes":8,"relative_equation_residual":4.6888886634885046e-11},{"case":6,"passes":1,"relative_equation_residual":0.028409546359993404},{"case":6,"passes":2,"relative_equation_residual":0.0005739436830258379},{"case":6,"passes":4,"relative_equation_residual":3.103778235708114e-06},{"case":6,"passes":8,"relative_equation_residual":2.1689432054902897e-09},{"case":7,"passes":1,"relative_equation_residual":0.199863550065482},{"case":7,"passes":2,"relative_equation_residual":0.038242181014939224},{"case":7,"passes":4,"relative_equation_residual":0.0016880900470515308},{"case":7,"passes":8,"relative_equation_residual":3.496732829354753e-06}],"update":320},{"exposures_per_case":50,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.1308521312791168},{"case":0,"passes":2,"relative_equation_residual":0.011399236016325151},{"case":0,"passes":4,"relative_equation_residual":7.97120610692687e-05},{"case":0,"passes":8,"relative_equation_residual":6.951064172838817e-09},{"case":1,"passes":1,"relative_equation_residual":0.07696156791371118},{"case":1,"passes":2,"relative_equation_residual":0.005453376717556507},{"case":1,"passes":4,"relative_equation_residual":3.9168814395546966e-05},{"case":1,"passes":8,"relative_equation_residual":5.156619615544015e-09},{"case":2,"passes":1,"relative_equation_residual":0.1448035540449404},{"case":2,"passes":2,"relative_equation_residual":0.014188757388381008},{"case":2,"passes":4,"relative_equation_residual":0.0002750406309489854},{"case":2,"passes":8,"relative_equation_residual":8.461768158696112e-08},{"case":3,"passes":1,"relative_equation_residual":0.058155409084241313},{"case":3,"passes":2,"relative_equation_residual":0.002474768709696754},{"case":3,"passes":4,"relative_equation_residual":1.1333636718096162e-05},{"case":3,"passes":8,"relative_equation_residual":1.3851279653749813e-09},{"case":4,"passes":1,"relative_equation_residual":0.04405104380014582},{"case":4,"passes":2,"relative_equation_residual":0.002166289985478605},{"case":4,"passes":4,"relative_equation_residual":7.277840771685625e-06},{"case":4,"passes":8,"relative_equation_residual":1.796151106857818e-10},{"case":5,"passes":1,"relative_equation_residual":0.058590464108868895},{"case":5,"passes":2,"relative_equation_residual":0.002392550085929652},{"case":5,"passes":4,"relative_equation_residual":8.598000104190199e-06},{"case":5,"passes":8,"relative_equation_residual":1.9056918539606384e-09},{"case":6,"passes":1,"relative_equation_residual":0.0450814224140542},{"case":6,"passes":2,"relative_equation_residual":0.0018048931597546333},{"case":6,"passes":4,"relative_equation_residual":1.2464582968629202e-05},{"case":6,"passes":8,"relative_equation_residual":8.03065764034013e-09},{"case":7,"passes":1,"relative_equation_residual":0.17776048062935992},{"case":7,"passes":2,"relative_equation_residual":0.026460850678622545},{"case":7,"passes":4,"relative_equation_residual":0.0005916215968912087},{"case":7,"passes":8,"relative_equation_residual":6.297841875986625e-07}],"update":400},{"exposures_per_case":60,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.09503876116505798},{"case":0,"passes":2,"relative_equation_residual":0.004172639515915474},{"case":0,"passes":4,"relative_equation_residual":2.571806795827822e-05},{"case":0,"passes":8,"relative_equation_residual":1.081902313472335e-09},{"case":1,"passes":1,"relative_equation_residual":0.0440499371526246},{"case":1,"passes":2,"relative_equation_residual":0.003902863029396211},{"case":1,"passes":4,"relative_equation_residual":6.186603787517248e-06},{"case":1,"passes":8,"relative_equation_residual":7.356980825419215e-10},{"case":2,"passes":1,"relative_equation_residual":0.1556963061534138},{"case":2,"passes":2,"relative_equation_residual":0.01825013363921031},{"case":2,"passes":4,"relative_equation_residual":0.0002670053245115821},{"case":2,"passes":8,"relative_equation_residual":5.6736809260649305e-08},{"case":3,"passes":1,"relative_equation_residual":0.12039090720222194},{"case":3,"passes":2,"relative_equation_residual":0.014442500393840634},{"case":3,"passes":4,"relative_equation_residual":0.00017900291105931766},{"case":3,"passes":8,"relative_equation_residual":1.3967326915205284e-08},{"case":4,"passes":1,"relative_equation_residual":0.06624841377513016},{"case":4,"passes":2,"relative_equation_residual":0.004007304910362468},{"case":4,"passes":4,"relative_equation_residual":1.8982382942495235e-05},{"case":4,"passes":8,"relative_equation_residual":4.0161059097858854e-10},{"case":5,"passes":1,"relative_equation_residual":0.12029259175317665},{"case":5,"passes":2,"relative_equation_residual":0.014430721740537807},{"case":5,"passes":4,"relative_equation_residual":0.00017838196978845288},{"case":5,"passes":8,"relative_equation_residual":1.4383139088524499e-08},{"case":6,"passes":1,"relative_equation_residual":0.05106444392694168},{"case":6,"passes":2,"relative_equation_residual":0.0020083872259484326},{"case":6,"passes":4,"relative_equation_residual":8.005077030482858e-06},{"case":6,"passes":8,"relative_equation_residual":3.216656515447472e-10},{"case":7,"passes":1,"relative_equation_residual":0.21083633648139322},{"case":7,"passes":2,"relative_equation_residual":0.04561552055886857},{"case":7,"passes":4,"relative_equation_residual":0.0021982520962597974},{"case":7,"passes":8,"relative_equation_residual":5.9880122750819105e-06}],"update":480},{"exposures_per_case":70,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.19704138122251624},{"case":0,"passes":2,"relative_equation_residual":0.041498755884212944},{"case":0,"passes":4,"relative_equation_residual":0.0016293939418701954},{"case":0,"passes":8,"relative_equation_residual":2.0320815655497014e-06},{"case":1,"passes":1,"relative_equation_residual":0.1443905321930071},{"case":1,"passes":2,"relative_equation_residual":0.019192893829199933},{"case":1,"passes":4,"relative_equation_residual":0.0003195862884947386},{"case":1,"passes":8,"relative_equation_residual":9.068904977084004e-08},{"case":2,"passes":1,"relative_equation_residual":0.1713611311895671},{"case":2,"passes":2,"relative_equation_residual":0.01812503534425206},{"case":2,"passes":4,"relative_equation_residual":0.00034605604510134434},{"case":2,"passes":8,"relative_equation_residual":1.2081609246536255e-07},{"case":3,"passes":1,"relative_equation_residual":0.07033401555078601},{"case":3,"passes":2,"relative_equation_residual":0.0042681736144826316},{"case":3,"passes":4,"relative_equation_residual":3.128971546082987e-05},{"case":3,"passes":8,"relative_equation_residual":2.738952239273909e-09},{"case":4,"passes":1,"relative_equation_residual":0.10278098000926471},{"case":4,"passes":2,"relative_equation_residual":0.010976920304231325},{"case":4,"passes":4,"relative_equation_residual":0.00013113440593172352},{"case":4,"passes":8,"relative_equation_residual":1.7541222533817537e-08},{"case":5,"passes":1,"relative_equation_residual":0.07055020298427819},{"case":5,"passes":2,"relative_equation_residual":0.004239899126762702},{"case":5,"passes":4,"relative_equation_residual":3.121297933254869e-05},{"case":5,"passes":8,"relative_equation_residual":2.6747447605733477e-09},{"case":6,"passes":1,"relative_equation_residual":0.07824650985884322},{"case":6,"passes":2,"relative_equation_residual":0.00580869282963947},{"case":6,"passes":4,"relative_equation_residual":2.8353444864445518e-05},{"case":6,"passes":8,"relative_equation_residual":1.445151333153183e-09},{"case":7,"passes":1,"relative_equation_residual":0.14168953922212682},{"case":7,"passes":2,"relative_equation_residual":0.011366550643726483},{"case":7,"passes":4,"relative_equation_residual":7.307154459025278e-05},{"case":7,"passes":8,"relative_equation_residual":5.609758762807697e-09}],"update":560},{"exposures_per_case":80,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.24726216466275933},{"case":0,"passes":2,"relative_equation_residual":0.06031200275360503},{"case":0,"passes":4,"relative_equation_residual":0.0035796313385622297},{"case":0,"passes":8,"relative_equation_residual":1.263419959013137e-05},{"case":1,"passes":1,"relative_equation_residual":0.15632124440329928},{"case":1,"passes":2,"relative_equation_residual":0.02128229340174679},{"case":1,"passes":4,"relative_equation_residual":0.0003726983974169058},{"case":1,"passes":8,"relative_equation_residual":1.1385911109991221e-07},{"case":2,"passes":1,"relative_equation_residual":0.2139604812829265},{"case":2,"passes":2,"relative_equation_residual":0.032856362974716724},{"case":2,"passes":4,"relative_equation_residual":0.0009674513364125882},{"case":2,"passes":8,"relative_equation_residual":1.000511385954121e-06},{"case":3,"passes":1,"relative_equation_residual":0.07157893036831166},{"case":3,"passes":2,"relative_equation_residual":0.0039725636850134706},{"case":3,"passes":4,"relative_equation_residual":2.75507587269022e-05},{"case":3,"passes":8,"relative_equation_residual":1.1900408452889064e-09},{"case":4,"passes":1,"relative_equation_residual":0.08089276403108894},{"case":4,"passes":2,"relative_equation_residual":0.005667765469677074},{"case":4,"passes":4,"relative_equation_residual":2.717978930701482e-05},{"case":4,"passes":8,"relative_equation_residual":7.354632953442293e-10},{"case":5,"passes":1,"relative_equation_residual":0.07164071417789555},{"case":5,"passes":2,"relative_equation_residual":0.003977744280090896},{"case":5,"passes":4,"relative_equation_residual":2.7520934135875373e-05},{"case":5,"passes":8,"relative_equation_residual":1.1760171151186774e-09},{"case":6,"passes":1,"relative_equation_residual":0.059872055309197245},{"case":6,"passes":2,"relative_equation_residual":0.0031859295024846837},{"case":6,"passes":4,"relative_equation_residual":5.105660003846785e-06},{"case":6,"passes":8,"relative_equation_residual":2.384005624461046e-10},{"case":7,"passes":1,"relative_equation_residual":0.10786242800440655},{"case":7,"passes":2,"relative_equation_residual":0.007476946308015884},{"case":7,"passes":4,"relative_equation_residual":4.537517827467111e-05},{"case":7,"passes":8,"relative_equation_residual":1.8793223707600633e-09}],"update":640},{"exposures_per_case":90,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.21054021977047424},{"case":0,"passes":2,"relative_equation_residual":0.042628088507781166},{"case":0,"passes":4,"relative_equation_residual":0.0015543404863986874},{"case":0,"passes":8,"relative_equation_residual":2.0043932582177943e-06},{"case":1,"passes":1,"relative_equation_residual":0.12540296323805705},{"case":1,"passes":2,"relative_equation_residual":0.012015248996232063},{"case":1,"passes":4,"relative_equation_residual":8.771402571096071e-05},{"case":1,"passes":8,"relative_equation_residual":4.7610636974867615e-09},{"case":2,"passes":1,"relative_equation_residual":0.12632947800173963},{"case":2,"passes":2,"relative_equation_residual":0.010536722186366523},{"case":2,"passes":4,"relative_equation_residual":6.0553331473528904e-05},{"case":2,"passes":8,"relative_equation_residual":2.5473495096095298e-09},{"case":3,"passes":1,"relative_equation_residual":0.055423477160359245},{"case":3,"passes":2,"relative_equation_residual":0.0034283425263467175},{"case":3,"passes":4,"relative_equation_residual":4.5951131742769755e-06},{"case":3,"passes":8,"relative_equation_residual":1.2442233470441588e-10},{"case":4,"passes":1,"relative_equation_residual":0.04520216047826076},{"case":4,"passes":2,"relative_equation_residual":0.0015187657524515677},{"case":4,"passes":4,"relative_equation_residual":2.980418343365274e-06},{"case":4,"passes":8,"relative_equation_residual":7.207345059451684e-12},{"case":5,"passes":1,"relative_equation_residual":0.05545965794104786},{"case":5,"passes":2,"relative_equation_residual":0.003416963966575635},{"case":5,"passes":4,"relative_equation_residual":4.5613038697088025e-06},{"case":5,"passes":8,"relative_equation_residual":1.1583006963436772e-10},{"case":6,"passes":1,"relative_equation_residual":0.026706697002551804},{"case":6,"passes":2,"relative_equation_residual":0.000985328383079068},{"case":6,"passes":4,"relative_equation_residual":1.8586439140961051e-06},{"case":6,"passes":8,"relative_equation_residual":1.202128515051552e-10},{"case":7,"passes":1,"relative_equation_residual":0.13777646241656297},{"case":7,"passes":2,"relative_equation_residual":0.020021269810737024},{"case":7,"passes":4,"relative_equation_residual":0.0005921375777905075},{"case":7,"passes":8,"relative_equation_residual":5.827038391649135e-07}],"update":720},{"exposures_per_case":100,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.14561783528754663},{"case":0,"passes":2,"relative_equation_residual":0.017807882565182818},{"case":0,"passes":4,"relative_equation_residual":0.0002866356254811129},{"case":0,"passes":8,"relative_equation_residual":7.512909718209306e-08},{"case":1,"passes":1,"relative_equation_residual":0.08278712855320493},{"case":1,"passes":2,"relative_equation_residual":0.003896626162296524},{"case":1,"passes":4,"relative_equation_residual":1.1233739766625833e-05},{"case":1,"passes":8,"relative_equation_residual":2.961083869625865e-10},{"case":2,"passes":1,"relative_equation_residual":0.15906088291792284},{"case":2,"passes":2,"relative_equation_residual":0.00419864418971219},{"case":2,"passes":4,"relative_equation_residual":1.5075460946600898e-05},{"case":2,"passes":8,"relative_equation_residual":6.34339330321155e-10},{"case":3,"passes":1,"relative_equation_residual":0.06529727868242323},{"case":3,"passes":2,"relative_equation_residual":0.0025318047078465287},{"case":3,"passes":4,"relative_equation_residual":4.999683867625464e-06},{"case":3,"passes":8,"relative_equation_residual":8.486757786978887e-11},{"case":4,"passes":1,"relative_equation_residual":0.03964131557690475},{"case":4,"passes":2,"relative_equation_residual":0.0017432854336350864},{"case":4,"passes":4,"relative_equation_residual":5.2441049253358785e-06},{"case":4,"passes":8,"relative_equation_residual":1.411042552915702e-10},{"case":5,"passes":1,"relative_equation_residual":0.06529921898873453},{"case":5,"passes":2,"relative_equation_residual":0.0025320179811541095},{"case":5,"passes":4,"relative_equation_residual":5.0045246846530746e-06},{"case":5,"passes":8,"relative_equation_residual":1.0534797623840305e-10},{"case":6,"passes":1,"relative_equation_residual":0.015149604049737635},{"case":6,"passes":2,"relative_equation_residual":0.00032784795754276484},{"case":6,"passes":4,"relative_equation_residual":7.450670438754358e-07},{"case":6,"passes":8,"relative_equation_residual":1.5232288016863474e-11},{"case":7,"passes":1,"relative_equation_residual":0.18019916424788598},{"case":7,"passes":2,"relative_equation_residual":0.023369193215094588},{"case":7,"passes":4,"relative_equation_residual":0.00029713492743157414},{"case":7,"passes":8,"relative_equation_residual":5.780776762744586e-08}],"update":800},{"exposures_per_case":110,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.12769865884732523},{"case":0,"passes":2,"relative_equation_residual":0.011393408414582121},{"case":0,"passes":4,"relative_equation_residual":9.008711654148178e-05},{"case":0,"passes":8,"relative_equation_residual":6.116376491035292e-09},{"case":1,"passes":1,"relative_equation_residual":0.06884027264866102},{"case":1,"passes":2,"relative_equation_residual":0.0016401061504931842},{"case":1,"passes":4,"relative_equation_residual":1.4080723699867277e-06},{"case":1,"passes":8,"relative_equation_residual":1.912351583843453e-11},{"case":2,"passes":1,"relative_equation_residual":0.155553137490185},{"case":2,"passes":2,"relative_equation_residual":0.0044863779245874905},{"case":2,"passes":4,"relative_equation_residual":8.127466449167997e-06},{"case":2,"passes":8,"relative_equation_residual":8.161014293733599e-11},{"case":3,"passes":1,"relative_equation_residual":0.04504840698309307},{"case":3,"passes":2,"relative_equation_residual":0.001396588347475086},{"case":3,"passes":4,"relative_equation_residual":7.16648710152922e-07},{"case":3,"passes":8,"relative_equation_residual":1.6444917690160792e-11},{"case":4,"passes":1,"relative_equation_residual":0.022632771122666304},{"case":4,"passes":2,"relative_equation_residual":0.0005608336338896303},{"case":4,"passes":4,"relative_equation_residual":4.3069609177830387e-07},{"case":4,"passes":8,"relative_equation_residual":1.3492711038147704e-12},{"case":5,"passes":1,"relative_equation_residual":0.04505339543559593},{"case":5,"passes":2,"relative_equation_residual":0.0013969767692594917},{"case":5,"passes":4,"relative_equation_residual":7.172078192781771e-07},{"case":5,"passes":8,"relative_equation_residual":3.620123296851811e-11},{"case":6,"passes":1,"relative_equation_residual":0.0049952581053346636},{"case":6,"passes":2,"relative_equation_residual":3.7939429202552924e-05},{"case":6,"passes":4,"relative_equation_residual":1.2238819603452813e-07},{"case":6,"passes":8,"relative_equation_residual":7.001736950393796e-12},{"case":7,"passes":1,"relative_equation_residual":0.17429093598983103},{"case":7,"passes":2,"relative_equation_residual":0.021528341357569907},{"case":7,"passes":4,"relative_equation_residual":0.00011778046598087674},{"case":7,"passes":8,"relative_equation_residual":4.9066129963010765e-08}],"update":880},{"exposures_per_case":120,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.13505954479010174},{"case":0,"passes":2,"relative_equation_residual":0.013658181610864472},{"case":0,"passes":4,"relative_equation_residual":0.00013427863394685884},{"case":0,"passes":8,"relative_equation_residual":1.341176554008151e-08},{"case":1,"passes":1,"relative_equation_residual":0.07308018276729014},{"case":1,"passes":2,"relative_equation_residual":0.0020631984962816595},{"case":1,"passes":4,"relative_equation_residual":2.311947136429751e-06},{"case":1,"passes":8,"relative_equation_residual":1.2413676147357318e-11},{"case":2,"passes":1,"relative_equation_residual":0.15204635019971602},{"case":2,"passes":2,"relative_equation_residual":0.0025065428052565533},{"case":2,"passes":4,"relative_equation_residual":1.2224492747991613e-05},{"case":2,"passes":8,"relative_equation_residual":4.133113033009421e-10},{"case":3,"passes":1,"relative_equation_residual":0.030640770136415935},{"case":3,"passes":2,"relative_equation_residual":0.0008617115792061057},{"case":3,"passes":4,"relative_equation_residual":3.7898518855092817e-07},{"case":3,"passes":8,"relative_equation_residual":1.375507492701851e-11},{"case":4,"passes":1,"relative_equation_residual":0.022621430958767952},{"case":4,"passes":2,"relative_equation_residual":0.0006010529551053626},{"case":4,"passes":4,"relative_equation_residual":4.1908299079367465e-07},{"case":4,"passes":8,"relative_equation_residual":3.059363911085657e-12},{"case":5,"passes":1,"relative_equation_residual":0.030637947614481537},{"case":5,"passes":2,"relative_equation_residual":0.0008612515854883572},{"case":5,"passes":4,"relative_equation_residual":3.801137805417766e-07},{"case":5,"passes":8,"relative_equation_residual":1.0151505925604335e-10},{"case":6,"passes":1,"relative_equation_residual":0.002650514638096869},{"case":6,"passes":2,"relative_equation_residual":7.107727333916098e-05},{"case":6,"passes":4,"relative_equation_residual":1.4842068769789504e-07},{"case":6,"passes":8,"relative_equation_residual":1.6794511255973087e-11},{"case":7,"passes":1,"relative_equation_residual":0.16031481844023873},{"case":7,"passes":2,"relative_equation_residual":0.017617857883154814},{"case":7,"passes":4,"relative_equation_residual":6.692170469381134e-05},{"case":7,"passes":8,"relative_equation_residual":6.046764820605701e-09}],"update":960},{"exposures_per_case":130,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.11954748216475257},{"case":0,"passes":2,"relative_equation_residual":0.008860716483498494},{"case":0,"passes":4,"relative_equation_residual":2.319903143851148e-05},{"case":0,"passes":8,"relative_equation_residual":1.3817487977365076e-10},{"case":1,"passes":1,"relative_equation_residual":0.06463545607785585},{"case":1,"passes":2,"relative_equation_residual":0.0011000261344749824},{"case":1,"passes":4,"relative_equation_residual":7.573340360763622e-07},{"case":1,"passes":8,"relative_equation_residual":1.0976717693808997e-11},{"case":2,"passes":1,"relative_equation_residual":0.14872039305876075},{"case":2,"passes":2,"relative_equation_residual":0.005002188274946074},{"case":2,"passes":4,"relative_equation_residual":9.053093434424202e-06},{"case":2,"passes":8,"relative_equation_residual":1.3336998641711682e-10},{"case":3,"passes":1,"relative_equation_residual":0.030200458799357507},{"case":3,"passes":2,"relative_equation_residual":0.0006167749302578749},{"case":3,"passes":4,"relative_equation_residual":2.0011029261496658e-07},{"case":3,"passes":8,"relative_equation_residual":1.104278185900261e-11},{"case":4,"passes":1,"relative_equation_residual":0.015527309853359157},{"case":4,"passes":2,"relative_equation_residual":0.00047859358137436865},{"case":4,"passes":4,"relative_equation_residual":1.556431129479718e-07},{"case":4,"passes":8,"relative_equation_residual":9.688252419744854e-13},{"case":5,"passes":1,"relative_equation_residual":0.030197127317432616},{"case":5,"passes":2,"relative_equation_residual":0.000617304539901726},{"case":5,"passes":4,"relative_equation_residual":2.0042376603846397e-07},{"case":5,"passes":8,"relative_equation_residual":5.073574008735726e-11},{"case":6,"passes":1,"relative_equation_residual":0.0035837604951138927},{"case":6,"passes":2,"relative_equation_residual":4.1443536251689405e-05},{"case":6,"passes":4,"relative_equation_residual":8.742910174487703e-08},{"case":6,"passes":8,"relative_equation_residual":1.709052283365044e-11},{"case":7,"passes":1,"relative_equation_residual":0.17431840318863698},{"case":7,"passes":2,"relative_equation_residual":0.021302808127415494},{"case":7,"passes":4,"relative_equation_residual":0.00018915872410441972},{"case":7,"passes":8,"relative_equation_residual":9.121370988137561e-08}],"update":1040},{"exposures_per_case":140,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.1273910379202248},{"case":0,"passes":2,"relative_equation_residual":0.010572969170632919},{"case":0,"passes":4,"relative_equation_residual":4.1063528583124146e-05},{"case":0,"passes":8,"relative_equation_residual":1.439468906582801e-10},{"case":1,"passes":1,"relative_equation_residual":0.07405456051686579},{"case":1,"passes":2,"relative_equation_residual":0.0018210328035450204},{"case":1,"passes":4,"relative_equation_residual":7.711516014213208e-07},{"case":1,"passes":8,"relative_equation_residual":1.3785652657991365e-11},{"case":2,"passes":1,"relative_equation_residual":0.15101025099444643},{"case":2,"passes":2,"relative_equation_residual":0.0032911715641435373},{"case":2,"passes":4,"relative_equation_residual":1.636316126717835e-05},{"case":2,"passes":8,"relative_equation_residual":3.310154802235946e-11},{"case":3,"passes":1,"relative_equation_residual":0.020538932641850124},{"case":3,"passes":2,"relative_equation_residual":0.00028452956635336915},{"case":3,"passes":4,"relative_equation_residual":1.2540361599601742e-07},{"case":3,"passes":8,"relative_equation_residual":1.7705272381370455e-11},{"case":4,"passes":1,"relative_equation_residual":0.017465829757949712},{"case":4,"passes":2,"relative_equation_residual":0.00047082938973145124},{"case":4,"passes":4,"relative_equation_residual":3.869288582857469e-07},{"case":4,"passes":8,"relative_equation_residual":4.936873705749792e-12},{"case":5,"passes":1,"relative_equation_residual":0.020550745996324015},{"case":5,"passes":2,"relative_equation_residual":0.0002847690332020373},{"case":5,"passes":4,"relative_equation_residual":1.2384724524493056e-07},{"case":5,"passes":8,"relative_equation_residual":1.1451579144248421e-11},{"case":6,"passes":1,"relative_equation_residual":0.0023642098252002297},{"case":6,"passes":2,"relative_equation_residual":4.849645946274716e-05},{"case":6,"passes":4,"relative_equation_residual":7.694051566893923e-08},{"case":6,"passes":8,"relative_equation_residual":1.596486148762338e-11},{"case":7,"passes":1,"relative_equation_residual":0.16638580929242247},{"case":7,"passes":2,"relative_equation_residual":0.018396422099255103},{"case":7,"passes":4,"relative_equation_residual":0.00013073424701454348},{"case":7,"passes":8,"relative_equation_residual":6.580531455557831e-08}],"update":1120},{"exposures_per_case":150,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.13073867226926},{"case":0,"passes":2,"relative_equation_residual":0.011852139223544706},{"case":0,"passes":4,"relative_equation_residual":8.156324481505816e-05},{"case":0,"passes":8,"relative_equation_residual":3.903561154319379e-09},{"case":1,"passes":1,"relative_equation_residual":0.0760447642878337},{"case":1,"passes":2,"relative_equation_residual":0.001677601541817015},{"case":1,"passes":4,"relative_equation_residual":4.924686345254188e-07},{"case":1,"passes":8,"relative_equation_residual":1.0664887380206168e-11},{"case":2,"passes":1,"relative_equation_residual":0.15618291661771921},{"case":2,"passes":2,"relative_equation_residual":0.002688657699339664},{"case":2,"passes":4,"relative_equation_residual":8.248340261548585e-06},{"case":2,"passes":8,"relative_equation_residual":4.630928689238341e-11},{"case":3,"passes":1,"relative_equation_residual":0.01631255209328158},{"case":3,"passes":2,"relative_equation_residual":0.00018443245717832043},{"case":3,"passes":4,"relative_equation_residual":9.422737915018167e-08},{"case":3,"passes":8,"relative_equation_residual":9.578154338456047e-12},{"case":4,"passes":1,"relative_equation_residual":0.015793172499837756},{"case":4,"passes":2,"relative_equation_residual":0.0003517345094873324},{"case":4,"passes":4,"relative_equation_residual":1.6918215809168708e-07},{"case":4,"passes":8,"relative_equation_residual":1.849261670136942e-12},{"case":5,"passes":1,"relative_equation_residual":0.016319596456091556},{"case":5,"passes":2,"relative_equation_residual":0.00018454961273671732},{"case":5,"passes":4,"relative_equation_residual":9.415639098962597e-08},{"case":5,"passes":8,"relative_equation_residual":6.666754755984711e-11},{"case":6,"passes":1,"relative_equation_residual":0.005457072874557531},{"case":6,"passes":2,"relative_equation_residual":5.5116767103683545e-05},{"case":6,"passes":4,"relative_equation_residual":6.751391774549276e-08},{"case":6,"passes":8,"relative_equation_residual":1.3750130681742005e-11},{"case":7,"passes":1,"relative_equation_residual":0.1596746767313841},{"case":7,"passes":2,"relative_equation_residual":0.016243127100097725},{"case":7,"passes":4,"relative_equation_residual":5.155098141117275e-05},{"case":7,"passes":8,"relative_equation_residual":1.767180387987947e-08}],"update":1200},{"exposures_per_case":160,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.13123563907734165},{"case":0,"passes":2,"relative_equation_residual":0.011702440946287479},{"case":0,"passes":4,"relative_equation_residual":6.94249423719508e-05},{"case":0,"passes":8,"relative_equation_residual":2.3411341576568116e-09},{"case":1,"passes":1,"relative_equation_residual":0.07799115471469695},{"case":1,"passes":2,"relative_equation_residual":0.00194087362280544},{"case":1,"passes":4,"relative_equation_residual":5.504618049170515e-07},{"case":1,"passes":8,"relative_equation_residual":9.758085271244386e-12},{"case":2,"passes":1,"relative_equation_residual":0.15746228861649672},{"case":2,"passes":2,"relative_equation_residual":0.0024919004841184125},{"case":2,"passes":4,"relative_equation_residual":1.6270831286041505e-06},{"case":2,"passes":8,"relative_equation_residual":2.2540890908917254e-11},{"case":3,"passes":1,"relative_equation_residual":0.0144719824134245},{"case":3,"passes":2,"relative_equation_residual":8.431111266167491e-05},{"case":3,"passes":4,"relative_equation_residual":5.528367781956861e-08},{"case":3,"passes":8,"relative_equation_residual":1.9829188646463476e-11},{"case":4,"passes":1,"relative_equation_residual":0.01801583660539675},{"case":4,"passes":2,"relative_equation_residual":0.0004850975044044274},{"case":4,"passes":4,"relative_equation_residual":1.6230382801189134e-07},{"case":4,"passes":8,"relative_equation_residual":2.902904922055176e-12},{"case":5,"passes":1,"relative_equation_residual":0.014475112226293186},{"case":5,"passes":2,"relative_equation_residual":8.426072179605418e-05},{"case":5,"passes":4,"relative_equation_residual":4.974705989835327e-08},{"case":5,"passes":8,"relative_equation_residual":7.81160327453587e-12},{"case":6,"passes":1,"relative_equation_residual":0.0017575294707106077},{"case":6,"passes":2,"relative_equation_residual":1.986171865236821e-05},{"case":6,"passes":4,"relative_equation_residual":4.9292487082811727e-08},{"case":6,"passes":8,"relative_equation_residual":1.0633091653526903e-11},{"case":7,"passes":1,"relative_equation_residual":0.1604230078261197},{"case":7,"passes":2,"relative_equation_residual":0.01595279113836677},{"case":7,"passes":4,"relative_equation_residual":4.5067132562320334e-05},{"case":7,"passes":8,"relative_equation_residual":1.3053423805261276e-08}],"update":1280},{"exposures_per_case":170,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.12982788824290672},{"case":0,"passes":2,"relative_equation_residual":0.01124443286821049},{"case":0,"passes":4,"relative_equation_residual":5.578779826134981e-05},{"case":0,"passes":8,"relative_equation_residual":1.1239095713810968e-09},{"case":1,"passes":1,"relative_equation_residual":0.0781207378626816},{"case":1,"passes":2,"relative_equation_residual":0.002064846902669358},{"case":1,"passes":4,"relative_equation_residual":6.647570571979512e-07},{"case":1,"passes":8,"relative_equation_residual":1.4261020928385682e-11},{"case":2,"passes":1,"relative_equation_residual":0.15853242343134996},{"case":2,"passes":2,"relative_equation_residual":0.002377698329272932},{"case":2,"passes":4,"relative_equation_residual":1.5661166308168162e-06},{"case":2,"passes":8,"relative_equation_residual":1.762857782431583e-11},{"case":3,"passes":1,"relative_equation_residual":0.013928862863804365},{"case":3,"passes":2,"relative_equation_residual":4.194916393820751e-05},{"case":3,"passes":4,"relative_equation_residual":5.28298345553294e-08},{"case":3,"passes":8,"relative_equation_residual":1.433326520258129e-11},{"case":4,"passes":1,"relative_equation_residual":0.019400815237887753},{"case":4,"passes":2,"relative_equation_residual":0.0005715660320973154},{"case":4,"passes":4,"relative_equation_residual":2.2639747903209863e-07},{"case":4,"passes":8,"relative_equation_residual":1.3317280445974886e-12},{"case":5,"passes":1,"relative_equation_residual":0.013931497240804655},{"case":5,"passes":2,"relative_equation_residual":4.1849350756322674e-05},{"case":5,"passes":4,"relative_equation_residual":4.967119418677182e-08},{"case":5,"passes":8,"relative_equation_residual":9.522191405056548e-12},{"case":6,"passes":1,"relative_equation_residual":0.00022638605908907385},{"case":6,"passes":2,"relative_equation_residual":6.947752657062209e-06},{"case":6,"passes":4,"relative_equation_residual":4.0672201493818914e-08},{"case":6,"passes":8,"relative_equation_residual":8.995212998595635e-12},{"case":7,"passes":1,"relative_equation_residual":0.1612881346920031},{"case":7,"passes":2,"relative_equation_residual":0.015873860904607285},{"case":7,"passes":4,"relative_equation_residual":4.2110276041115274e-05},{"case":7,"passes":8,"relative_equation_residual":9.359946322359625e-09}],"update":1360},{"exposures_per_case":180,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.12839392231678157},{"case":0,"passes":2,"relative_equation_residual":0.010794051848917768},{"case":0,"passes":4,"relative_equation_residual":4.640090515480583e-05},{"case":0,"passes":8,"relative_equation_residual":5.814368360355054e-10},{"case":1,"passes":1,"relative_equation_residual":0.07730837200701787},{"case":1,"passes":2,"relative_equation_residual":0.001940237759961313},{"case":1,"passes":4,"relative_equation_residual":6.0019986386914e-07},{"case":1,"passes":8,"relative_equation_residual":1.3805202709207472e-11},{"case":2,"passes":1,"relative_equation_residual":0.1591723929217334},{"case":2,"passes":2,"relative_equation_residual":0.0025434710555930816},{"case":2,"passes":4,"relative_equation_residual":1.3932363146617122e-06},{"case":2,"passes":8,"relative_equation_residual":2.3391032714099185e-11},{"case":3,"passes":1,"relative_equation_residual":0.014131221111066124},{"case":3,"passes":2,"relative_equation_residual":4.8808180352483433e-05},{"case":3,"passes":4,"relative_equation_residual":5.635605959150374e-08},{"case":3,"passes":8,"relative_equation_residual":9.748200849799773e-12},{"case":4,"passes":1,"relative_equation_residual":0.018835559211868132},{"case":4,"passes":2,"relative_equation_residual":0.0005519337973571751},{"case":4,"passes":4,"relative_equation_residual":2.151106121631245e-07},{"case":4,"passes":8,"relative_equation_residual":1.297347098268948e-12},{"case":5,"passes":1,"relative_equation_residual":0.014127124493403587},{"case":5,"passes":2,"relative_equation_residual":4.869683586746714e-05},{"case":5,"passes":4,"relative_equation_residual":5.6184291698151513e-08},{"case":5,"passes":8,"relative_equation_residual":6.0771380754635376e-12},{"case":6,"passes":1,"relative_equation_residual":0.0003481349675210917},{"case":6,"passes":2,"relative_equation_residual":5.441302771860372e-06},{"case":6,"passes":4,"relative_equation_residual":4.630726878172797e-08},{"case":6,"passes":8,"relative_equation_residual":2.0525623647872425e-11},{"case":7,"passes":1,"relative_equation_residual":0.16216742620498434},{"case":7,"passes":2,"relative_equation_residual":0.016080410741101732},{"case":7,"passes":4,"relative_equation_residual":4.302110894057383e-05},{"case":7,"passes":8,"relative_equation_residual":6.54360063851525e-09}],"update":1440},{"exposures_per_case":190,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.1282408341184911},{"case":0,"passes":2,"relative_equation_residual":0.010757581348966058},{"case":0,"passes":4,"relative_equation_residual":4.557527170810889e-05},{"case":0,"passes":8,"relative_equation_residual":5.269908568282002e-10},{"case":1,"passes":1,"relative_equation_residual":0.07739989651022933},{"case":1,"passes":2,"relative_equation_residual":0.0019653358245616606},{"case":1,"passes":4,"relative_equation_residual":5.927609613001586e-07},{"case":1,"passes":8,"relative_equation_residual":1.2499410535547837e-11},{"case":2,"passes":1,"relative_equation_residual":0.1592247886002253},{"case":2,"passes":2,"relative_equation_residual":0.002533330447671315},{"case":2,"passes":4,"relative_equation_residual":1.382154582609525e-06},{"case":2,"passes":8,"relative_equation_residual":2.281601575431299e-11},{"case":3,"passes":1,"relative_equation_residual":0.014033232996030086},{"case":3,"passes":2,"relative_equation_residual":4.532821451480558e-05},{"case":3,"passes":4,"relative_equation_residual":5.644919101807319e-08},{"case":3,"passes":8,"relative_equation_residual":1.1040156826989743e-11},{"case":4,"passes":1,"relative_equation_residual":0.019072066702049112},{"case":4,"passes":2,"relative_equation_residual":0.000562903322843457},{"case":4,"passes":4,"relative_equation_residual":2.215935361344582e-07},{"case":4,"passes":8,"relative_equation_residual":1.999759531052599e-12},{"case":5,"passes":1,"relative_equation_residual":0.014034464429909299},{"case":5,"passes":2,"relative_equation_residual":4.5278940988059624e-05},{"case":5,"passes":4,"relative_equation_residual":5.686263408776625e-08},{"case":5,"passes":8,"relative_equation_residual":1.7466567091915338e-11},{"case":6,"passes":1,"relative_equation_residual":0.00017486725707273063},{"case":6,"passes":2,"relative_equation_residual":5.561315099641871e-06},{"case":6,"passes":4,"relative_equation_residual":3.861507747739859e-08},{"case":6,"passes":8,"relative_equation_residual":9.478147722122026e-12},{"case":7,"passes":1,"relative_equation_residual":0.16224423294983945},{"case":7,"passes":2,"relative_equation_residual":0.01607604162163502},{"case":7,"passes":4,"relative_equation_residual":4.286448026596042e-05},{"case":7,"passes":8,"relative_equation_residual":6.386956777692602e-09}],"update":1520},{"exposures_per_case":200,"measurements":[{"case":0,"passes":1,"relative_equation_residual":0.12822542598776676},{"case":0,"passes":2,"relative_equation_residual":0.010753491825545679},{"case":0,"passes":4,"relative_equation_residual":4.546209944541847e-05},{"case":0,"passes":8,"relative_equation_residual":5.197010127861783e-10},{"case":1,"passes":1,"relative_equation_residual":0.07740648478922513},{"case":1,"passes":2,"relative_equation_residual":0.0019675924035697703},{"case":1,"passes":4,"relative_equation_residual":5.915834415128258e-07},{"case":1,"passes":8,"relative_equation_residual":1.1714261867117453e-11},{"case":2,"passes":1,"relative_equation_residual":0.15922730648957434},{"case":2,"passes":2,"relative_equation_residual":0.0025326880380143087},{"case":2,"passes":4,"relative_equation_residual":1.3835774906058448e-06},{"case":2,"passes":8,"relative_equation_residual":2.3877396958710206e-11},{"case":3,"passes":1,"relative_equation_residual":0.014029028159444503},{"case":3,"passes":2,"relative_equation_residual":4.484026368916801e-05},{"case":3,"passes":4,"relative_equation_residual":5.526822088555695e-08},{"case":3,"passes":8,"relative_equation_residual":1.0851081419726366e-11},{"case":4,"passes":1,"relative_equation_residual":0.01909815029760243},{"case":4,"passes":2,"relative_equation_residual":0.0005639590481496822},{"case":4,"passes":4,"relative_equation_residual":2.247339553526807e-07},{"case":4,"passes":8,"relative_equation_residual":2.0853206694764447e-12},{"case":5,"passes":1,"relative_equation_residual":0.014032895285162793},{"case":5,"passes":2,"relative_equation_residual":4.4925828597668575e-05},{"case":5,"passes":4,"relative_equation_residual":5.0876455809562236e-08},{"case":5,"passes":8,"relative_equation_residual":4.0076630159750676e-11},{"case":6,"passes":1,"relative_equation_residual":0.00014332738010325367},{"case":6,"passes":2,"relative_equation_residual":4.5182343217910675e-06},{"case":6,"passes":4,"relative_equation_residual":3.90181043817416e-08},{"case":6,"passes":8,"relative_equation_residual":8.609422756142638e-12},{"case":7,"passes":1,"relative_equation_residual":0.16224348596940996},{"case":7,"passes":2,"relative_equation_residual":0.01607527787639434},{"case":7,"passes":4,"relative_equation_residual":4.2848390112097235e-05},{"case":7,"passes":8,"relative_equation_residual":6.335303109014001e-09}],"update":1600}]},"rollout_feedback":{"auxiliary_losses":1600,"depth_exposures":[[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50],[50,50,50,50]],"depth_order_sha256":"3adbd1b18830b55ac918ce265241ded42841f53a56341f43b1e333ebd0615257","mode":"D","original_sampling_order_sha256":"d0ea221a7e03a83673ece7fc9cdf4f5212bb54f3023511f93d95c84e45647899"},"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":1600,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"a73fbc342ed070390b47535e6b94b535dec158671393b074a17e8a6858c249da"},"checkpoint-1600.npz":{"bytes":1485452,"role":"checkpoint","sha256":"b8ddb4b6660efe3512a017f9da424f718d4824475f7679abf81a6b9c7f66e725"},"history.json":{"bytes":10183,"role":"training_diagnostics","sha256":"0416124c4656866426a15242b9e0c548a04c524cecff9376f94db22445afe3a4"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"64f7553cd3cc929a7b14f64047b1370a9b8e101ad018e29348c1c56b87572dfe"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":476,"role":"result_summary","sha256":"dad22bef355fcb2e2a8a6ea02677db0622d0aafa268a4708e1d395c23456cb14"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"b8ddb4b6660efe3512a017f9da424f718d4824475f7679abf81a6b9c7f66e725"}},"per_case_inference_s":[0.021958764002192765,0.004211374995065853,0.005414611019659787,0.004252555983839557,0.0037618629867210984,0.0035629000049084425,0.0044926900009159,0.0047491060104221106,0.00958425598219037,0.005284118000417948,0.0063809509738348424,0.005766473012045026],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"5e485e3dbf2ebf8215405614ab3fe139d78e626004f1b9be3d88580f18a0eb7e","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"aaf034b5ab0ac4011da3094fcb162b16b396f0a2a1d5451b2f9688e9a4c3285c","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"63a480e4427883226430b7ea309183170ba59e114d42413f1d1fe3e7f577f684","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"29d5043a890267f4e3176258035362bea64efef1baf8c4917981d71045f06dbc","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"203652c8855ef16ee820f87475654eabb3ae0e785cf61d22fbba76f988547fc4","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"e57675efd9c20329dcb315fa3ec2e2ede0819ec30e3729d4ca6aa8bedae95653","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"988db2a43180682c1a9b696a9bbf929b883bcb61a8da8112801258b9a1089e0f","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"651309ae6236b7b30c41f688d3e06cc72960bdc590657594b90d26e7100a1b6e","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"2dafc3494728e0fafecc4235db9b2d5ffbfe94c5c0861cd9b6265b1dacb62a9a","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"90d6da42f138dac875c92b8077e55ee16e64ead26daa1b76f1176cacb979ab2b","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"6ae9880aa53fa2faa54ab105bc66bd79e40e5d04e8d493f87c9af3c82d040c66","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"d5760bebb462ecd95c2077e907a01d3d6ead9f9707a8473d4f7e9d168f8c1afa","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2414006272,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
