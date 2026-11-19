# water-band-weight001 CUDA

Retrospective source review · fit-85ca1880b88a1591

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel370,016-parameter linear multiscale3D convolutional pressure-correction core with19weight tensors; captured transforms remove GroupNorm, SiLU and biases. Input q=(rho/h)r uses train-only RMS. Two F32 corrections recompute the F64 sealed cubic-voxel Neumann residual and reconstruct scaled cell-zero-gauged F64 pressure. dt is validated, not an independent neural input channel. A training-only separable orthonormal DCT-II penalty retains axes modes0..3, removes DC and supervises63nonconstant spatial modes at the first and final correction. The mean of those two losses is weighted; denominator is the larger of target low-mode energy and0.1squared times full mean-free target energy, floored at1e-30. This emits pressure only, with no added deployed pass or physical advancement.

**Purpose.** Assess low-spatial-mode pressure feedback weight0.01 while retaining the existing two-correction core and broad physical/manufactured/captured-direction training streams.

**Experiment.** 5,000CUDA Adam updates, batch1, learning rate0.001 with drops at3000/4500; model seed20261015 and physical sample seed2026102501. All arms share initial tensor inventories,16physical fitting trajectories/32frames at8/16cells per metre, dt1/60s, axes at most16cells. The native fitting packet contains no validation trajectories. Each update also draws primary1,024fields/eightgrids, supplemental768fields/sixrectangles and one late captured direction among192cases in21balanced groups; stream weights are0.1. These add15,000auxiliary training proposals without adding CFD parents. Existing0.1nonfinal equation contraction and final losses remain, first-pressure weight is0, and low-mode weight is0.01. Native evidence.py preserves the original profile as training-profile.json and emits one already-consumed validation probe after fitting; that probe is not a new optimizer example or held-out claim.

**Observed outcome.** Last logged weighted training objective 0.00068800836697682495; final diagnostic retained under the misleading history key validation 3.6611772358969887e-06, explicitly evaluated on training parents with zero held-out evaluations. Common128-request worst mean-free pressure relative L2 0.0063411958454848514; equation relative L2 0.0040036748300964001. Complete GPU pressure-call median of case medians 4.4625544999999995ms; worst case median 4.8374240000000004ms. Timing includes CPU input/output ownership, validation/packing, transfers, all pressure work and synchronization, excluding setup/scoring. These are pressure-request diagnostics, not full physical-step latency or drift.

**Lesson.** Compared with C49, weight0.01 lowers the common worst pressure error but raises worst equation error. It has the lowest pressure tail among this three-weight sweep, while C51 has a lower equation tail. The training history validation label is a train-only diagnostic.

**Reproduction.** Use original typed native evidence recipe/config/runtime, full witnessed source, exact input/output closures, fixed initial tensors and train-only normalization. Preserve original trainer profile mapping to training-profile.json, train-only diagnostic override, private physical/manufactured/direction sample orders, DCT-II gauge removal and energy floor. Keep the consumed validation probe separate from optimizer data and generalization. The historical evidence wrapper performs CPU neural forwards after CUDA fitting; a GPU-only reproducer must be qualified before rerunning. No clean retraining, optimizer resume or new-Rheon regeneration is qualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-feedback-weight-ablation-v1/weight001-study-02/train/receipt.json",
    "sha256": "85ca1880b88a15918503cd33639e8aac20eb21dd0afc2efe4f191490ecdd29eb"
  },
  "history": {
    "path": "runs/pressure-feedback-weight-ablation-v1/weight001-study-02/train/history.json",
    "sha256": "b1cf80b7586a4ecb9e30c84cfb0735710a33b050c900d2934b2ca5ec2a98df3f"
  },
  "sources": [
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/config.json",
      "sha256": "d30ff5b0046a0cb49fcf9c34f151d639aed3eb05b7a2e89172dab4a8716cd8c3"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/model.py",
      "sha256": "8d7bca5b7296396ffb5316df78c58fb1b3fc5a9776be08409bf6bd09b435bd01"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/policy.py",
      "sha256": "2b4d0fc25fb75a7a2563b042b13d435076391d8f06adb29ad18c220567797eaa"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/train.py",
      "sha256": "d450b3230a379b5a4440a12eb8d0cd133359bd27dbe9765837ef4908fd8f68c4"
    },
    {
      "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-feedback-weight-ablation-v1/weight001-project-01/source-witness.json",
    "sha256": "923388ec284fea9c0c988c1dcf5d1197aa578b001b6d3803abee58b70715616a"
  },
  "training_population": {
    "path": "runs/pressure-feedback-weight-ablation-v1/weight001-study-02/train/splits.json",
    "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
  },
  "training_inputs": [
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
    "item_id": "fit-85ca1880b88a1591",
    "kind": "training_attempt",
    "name": "water-band-weight001 CUDA",
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
    "receipt_sha256": "85ca1880b88a15918503cd33639e8aac20eb21dd0afc2efe4f191490ecdd29eb",
    "receipt_paths": [
      "runs/pressure-feedback-weight-ablation-v1/weight001-study-02/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "a7f26e3d6f2d42d297ebf80d23e09e4c",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "water-band-weight001 CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "d30ff5b0046a0cb49fcf9c34f151d639aed3eb05b7a2e89172dab4a8716cd8c3",
      "inputs": {
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
        "splits": {
          "bytes": 256361,
          "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423",
      "source_witness_sha256": "923388ec284fea9c0c988c1dcf5d1197aa578b001b6d3803abee58b70715616a"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "8503d99bd96e9c8f935c08d8e6a649a73e64c394777453dd8ec26aff929844b9",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "8503d99bd96e9c8f935c08d8e6a649a73e64c394777453dd8ec26aff929844b9",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2604511232,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/49"
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
      "receipt": "runs/pressure-feedback-weight-ablation-v1/weight001-study-02/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-feedback-weight-ablation-v1/weight001-study-02/train/receipt.json",
    "sha256": "85ca1880b88a15918503cd33639e8aac20eb21dd0afc2efe4f191490ecdd29eb",
    "bytes": 9579
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C50": {
        "name": "weight001",
        "weights_sha256": "8503d99bd96e9c8f935c08d8e6a649a73e64c394777453dd8ec26aff929844b9",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.0040036748300964,
          "pressure_mean_free_relative_l2": 0.006341195845484851,
          "pressure_absolute_rms": 0.5902455259376975,
          "pressure_absolute_max": 1.1918246050417551
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
              "common_equation_relative_l2": 0.0029992571866568163,
              "pressure_mean_free_relative_l2": 0.006147568820852045,
              "pressure_absolute_rms": 0.5902455259376975,
              "pressure_absolute_max": 1.1918246050417551
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.0040036748300964,
              "pressure_mean_free_relative_l2": 0.006341195845484851,
              "pressure_absolute_rms": 0.09734821889975577,
              "pressure_absolute_max": 0.252974587888993
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-material-GPU-accuracy-01/C50/results.json",
          "sha256": "096fdc719f3da5db76fe765814ffe7a0c066a027431cfdd7c93618421136b3c5"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.01},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"d30ff5b0046a0cb49fcf9c34f151d639aed3eb05b7a2e89172dab4a8716cd8c3","inputs":{"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"923388ec284fea9c0c988c1dcf5d1197aa578b001b6d3803abee58b70715616a"},"diagnostic_population":"train_only","elapsed_s":397.6460579830018,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.5700300525954844e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":837691904,"peak_reserved_bytes":889192448,"tf32":false},"heldout_evaluations":0,"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"8503d99bd96e9c8f935c08d8e6a649a73e64c394777453dd8ec26aff929844b9"},"history.json":{"bytes":214445,"role":"training_diagnostics","sha256":"b1cf80b7586a4ecb9e30c84cfb0735710a33b050c900d2934b2ca5ec2a98df3f"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"9d76650080463cb76eb9b4b175bd0680ba2c7754235c46cde6e3d248130b4524"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":473,"role":"result_summary","sha256":"06e782b54ba2ef0630193606de60d26069efb9baac39399e90e24a09d816492f"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"8503d99bd96e9c8f935c08d8e6a649a73e64c394777453dd8ec26aff929844b9"}},"per_case_inference_s":[0.02083309000590816,0.003843262995360419,0.005037415990955196,0.0039856520015746355,0.0034796330000972375,0.0032879680074984208,0.004166560989688151,0.0043202929955441505,0.008742999008973129,0.004881457003648393,0.00596450301236473,0.0053984499973012134],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"60993e35b7e7c2bd1f2e2da69c04cd399beeb187eaaa572e9e28d27aeddf56ac","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"641efe066d92f7520285ab775f7ab0199600c21258ae7a1c9c3efea355d62e09","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"0af93db8a3b1fa563cb9bc7bf1fd08e113780433281be52fc2537b9a9b3ef199","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"46ecc2b181e90fc5e74ddae968b40d5403ef19a869bf63628f6819bb93ae3699","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"970916d4fbfb395ccf193930a702ec853980fa4fb205fc0a88558612ad8681d4","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"21b19667aef6472dc6789f7e44c3dd57b54e93cbf26e23e7bfd8d2c6b9a05d17","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"91ac6cb74996a6665e22e5c7098d4a7912a6192a0f7b3f128ba63d524986bdd0","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"67061eb602573237271010031316534f61248efd2aafa35b872d5a56d9ce83f2","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"f9bfd590a03368a31642e4fa5018ce01250c7a209c81daff9131956f55846db5","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"77b95ad49ee99e717c206faeb12687273fe9087f69e1fda0761fae14eec581fc","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"fda6bf3cc269ac6fc7e7a42f113b53c0ae399ca648077fb6baa1ecc2e21c5c8f","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"758f557f8532c62c12a11642c13b2dd31266f0cc1244864bb0229a54eb57a87d","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2604511232,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
