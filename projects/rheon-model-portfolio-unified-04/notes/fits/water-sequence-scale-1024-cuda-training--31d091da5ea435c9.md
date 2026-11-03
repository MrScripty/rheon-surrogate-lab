# water-sequence-scale-1024 · CUDA training

Retrospective source review · fit-31d091da5ea435c9

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 371,729-parameter, 38-tensor F32 multiscale 3D pressure-correction CNN with seven input channels: residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep. Encoder widths 8/16/24/32, bottleneck 48, decoder 32/24/16/8, affine-free four-group GroupNorm, SiLU, global-mean broadcast and nearest-to-skip resizing. The linear head supplies a pressure-scaled cell-zero-gauge correction added to the previous pressure. FV arithmetic/output remain F64. This is a pressure request, not a complete simulation advance. The captured cell budget is 65,536.

**Purpose.** Expand the one-frame-per-scene control to 1,024 training scenes across four independently seeded capture cohorts.

**Experiment.** 10,000 whole-network Adam updates, batch 2, initial learning rate 0.001, seed 20261015 and sampling seed 2026102501. Learning-rate schedule: [{'learning_rate': 0.0001, 'update': 1000}, {'learning_rate': 1e-05, 'update': 6000}]. Gradient norm is clipped at 1.0. 1,024 training trajectories/1,024 saved pressure frames; 64 validation trajectories/128 frames. Actual training densities, in scenes: {'8.0': 512, '16.0': 512}. Native training snapshot indices present: [15, 63]. The sealed-water captures use variable rectangular small domains, randomized Gaussian/tapered force position, direction, amplitude, motion, swirl and temporal variation. The objective combines normalized pressure-field and FV equation losses, with scales fitted only on training frames. Reserved-test input bindings are retained separately and do not enter the optimizer parent selection.

**Observed outcome.** The final sampled training loss is 0.0008809482430043715; the retained validation objective changes from 122.42751181637055 to 0.0027100010549747592. The independently rescored 128-request GPU comparison has worst mean-free pressure relative L2 0.13061603130614149 and equation relative L2 0.20792456142277135. These are different metrics/populations from the logged training loss. This source review adds no latency or own-history drift measurement.

**Lesson.** This increases independent scene count rather than merely repeating the original 256 scenes at more cadences. Its broader fitting population and train-only scales still cover only 8/16 cells per metre, 5–16 cells per axis and one 1/60 s step size. More small-grid scenes do not establish large-grid/high-density or variable-timestep accuracy; compare common errors to the 256-scene control.

**Reproduction.** Use the exact captured project/train recipe, config, source witness, runtime, typed input bindings, normalization and seeded checkpoint inventory. Captured training runs execute CUDA gradients/Adam updates, then perform historical CPU deployment/parity checks. Do not run that recipe unchanged under the current GPU-only neural-forward policy. A clean GPU-only retraining/replay path and new-Rheon data regeneration remain to be qualified. Final-update selection and no optimizer-resume state are explicit; preserve the old physics epoch and original receipts.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-learning-sequence-v1/scale-1024/managed/stage-00/artifacts/receipt.json",
    "sha256": "31d091da5ea435c9c53986601e3472ba7d02b9521ca13bb2a03cedd22564dd80"
  },
  "history": {
    "path": "runs/water-learning-sequence-v1/scale-1024/managed/stage-00/artifacts/history.json",
    "sha256": "e53b17e2efbb4169ea1d92fede4e16b593d0bce98176fcce030511f65c448c69"
  },
  "sources": [
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/config.json",
      "sha256": "6a533dab95d1a58e0117e87c7f16774e66fc13c83987bc85d9df2a4352e5caa5"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/model.py",
      "sha256": "d42453ac9060275c7fc6676eab0e23eee7188cf2604a82714bfd4570104d7c0c"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/policy.py",
      "sha256": "5e6fb00db31f558df0bf3665422c7afd4a6ed14e8608e9514576fb86632858cd"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/qualification.json",
      "sha256": "52e94246663f5d1774edbffb1a9c1cf31bdce3c0cf6aba85b918892a4bb85260"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/test_inference.py",
      "sha256": "43bc71643e362670c960c184b6b1a378f93aba7a6408a2763a960059592a1b56"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/train.py",
      "sha256": "75a6e4281f70646ef0e66fc3b93163b573df05b66e54006ca6630c0051338be5"
    }
  ],
  "source_witness": {
    "path": "runs/water-learning-sequence-v1/scale-1024/project/source-witness.json",
    "sha256": "7ea3f8aa6cbba3f3e8bfb035c2372e959141df96e4295ea01086c92402980303"
  },
  "training_population": {
    "path": "runs/water-learning-sequence-v1/scale-1024/managed/stage-00/artifacts/splits.json",
    "sha256": "a05bc26cff4fcc74c64b60975211f16ede4949edf0e88f29f2a75357d73f8a03"
  },
  "training_inputs": [
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/inputs/dataset.npz",
      "sha256": "3f2fa46bae33df7229fbd15153883cb3fac9dfbda987ca8cff0265d3b582160d",
      "slot": "dataset",
      "bytes": 23033558
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/inputs/dataset_extra.npz",
      "sha256": "da990378d5ae021f3b9b5d2a836ac836d8dc425de6536d438df086afe5042a91",
      "slot": "dataset_extra",
      "bytes": 23083606
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset",
      "bytes": 2561142
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra",
      "bytes": 2564086
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits",
      "bytes": 339677
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/inputs/splits.json",
      "sha256": "a05bc26cff4fcc74c64b60975211f16ede4949edf0e88f29f2a75357d73f8a03",
      "slot": "splits",
      "bytes": 2958876
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset",
      "bytes": 2209622
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra",
      "bytes": 2212566
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-1024/project/inputs/validation_splits.json",
      "sha256": "9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5",
      "slot": "validation_splits",
      "bytes": 372865
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-31d091da5ea435c9",
    "kind": "training_attempt",
    "name": "water-sequence-scale-1024 · CUDA training",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-cubic-groupnorm-model-v3",
      "input_channels": 7,
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
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 65536
    },
    "receipt_sha256": "31d091da5ea435c9c53986601e3472ba7d02b9521ca13bb2a03cedd22564dd80",
    "receipt_paths": [
      "runs/water-learning-sequence-v1/scale-1024/managed/stage-00/artifacts/receipt.json",
      "runs/water-learning-sequence-v1/scale-1024/managed/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/c9c0cc20ce4849f8af6b0c9bf841154f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/c9c0cc20ce4849f8af6b0c9bf841154f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/c9c0cc20ce4849f8af6b0c9bf841154f/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/c9c0cc20ce4849f8af6b0c9bf841154f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/c9c0cc20ce4849f8af6b0c9bf841154f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-c9c0cc20ce4849f8af6b0c9bf841154f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-c9c0cc20ce4849f8af6b0c9bf841154f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-c9c0cc20ce4849f8af6b0c9bf841154f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-c9c0cc20ce4849f8af6b0c9bf841154f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/c9c0cc20ce4849f8af6b0c9bf841154f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/c9c0cc20ce4849f8af6b0c9bf841154f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/c9c0cc20ce4849f8af6b0c9bf841154f/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/c9c0cc20ce4849f8af6b0c9bf841154f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-c9c0cc20ce4849f8af6b0c9bf841154f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-c9c0cc20ce4849f8af6b0c9bf841154f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-c9c0cc20ce4849f8af6b0c9bf841154f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-c9c0cc20ce4849f8af6b0c9bf841154f/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-c9c0cc20ce4849f8af6b0c9bf841154f/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "c9c0cc20ce4849f8af6b0c9bf841154f",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-sequence-scale-1024 · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "6a533dab95d1a58e0117e87c7f16774e66fc13c83987bc85d9df2a4352e5caa5",
      "inputs": {
        "dataset": {
          "bytes": 23033558,
          "sha256": "3f2fa46bae33df7229fbd15153883cb3fac9dfbda987ca8cff0265d3b582160d"
        },
        "dataset_extra": {
          "bytes": 23083606,
          "sha256": "da990378d5ae021f3b9b5d2a836ac836d8dc425de6536d438df086afe5042a91"
        },
        "reserved_test_dataset": {
          "bytes": 2561142,
          "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"
        },
        "reserved_test_dataset_extra": {
          "bytes": 2564086,
          "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"
        },
        "reserved_test_splits": {
          "bytes": 339677,
          "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"
        },
        "splits": {
          "bytes": 2958876,
          "sha256": "a05bc26cff4fcc74c64b60975211f16ede4949edf0e88f29f2a75357d73f8a03"
        },
        "validation_dataset": {
          "bytes": 2209622,
          "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"
        },
        "validation_dataset_extra": {
          "bytes": 2212566,
          "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"
        },
        "validation_splits": {
          "bytes": 372865,
          "sha256": "9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"
        }
      },
      "normalization_sha256": "040612980ad5f962c10bfdcdf1478b45f91b91b98abfa35a2604e04a21db4291",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "3b19af779803209edef4b7cce16e6f8a633f9ecea68f19b63a9e16195431b982",
      "source_witness_sha256": "7ea3f8aa6cbba3f3e8bfb035c2372e959141df96e4295ea01086c92402980303"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c",
        "bytes": 1497594,
        "stage": "initial"
      },
      {
        "name": "checkpoint-10000.npz",
        "sha256": "fac9f59be57f95d9112f623006fd93e1cf26eaaad63c553a86a571cf0ce0aac2",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "3fc3e056dbc76a12a26120853912c4200564d723e968bedad6cb137656efe279",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "fac9f59be57f95d9112f623006fd93e1cf26eaaad63c553a86a571cf0ce0aac2",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2664370176,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/9"
    ],
    "purpose": {
      "status": "source_review_required",
      "evidence": "Captured architecture, training configuration and linked named entries."
    },
    "outcome": {
      "status": "retained_training_evidence_only",
      "fresh_common_benchmark": "pending",
      "scientific_promotion": null
    },
    "lesson": {
      "status": "review_required_no_automatic_causal_claim"
    },
    "reproduction": {
      "receipt": "runs/water-learning-sequence-v1/scale-1024/managed/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-learning-sequence-v1/scale-1024/managed/stage-00/artifacts/receipt.json",
    "sha256": "31d091da5ea435c9c53986601e3472ba7d02b9521ca13bb2a03cedd22564dd80",
    "bytes": 11766
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C10": {
        "name": "scale-1024",
        "weights_sha256": "fac9f59be57f95d9112f623006fd93e1cf26eaaad63c553a86a571cf0ce0aac2",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.20792456142277135,
          "pressure_mean_free_relative_l2": 0.1306160313061415,
          "pressure_absolute_rms": 3.0144807438736345,
          "pressure_absolute_max": 8.050191835855381
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0031",
            "native_frame": 15,
            "counts_xyz": [
              14,
              6,
              16
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0031",
            "native_frame": 15,
            "counts_xyz": [
              14,
              6,
              16
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.14241734161101338,
              "pressure_mean_free_relative_l2": 0.05922347096538763,
              "pressure_absolute_rms": 2.1514575380545216,
              "pressure_absolute_max": 8.050191835855381
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.20792456142277135,
              "pressure_mean_free_relative_l2": 0.1306160313061415,
              "pressure_absolute_rms": 3.0144807438736345,
              "pressure_absolute_max": 7.766396241279139
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C10/results.json",
          "sha256": "a426d289fd506c1deca0cfdc142c69f00b0b6c692d6e3cfef035af8777faaf94"
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
{"architecture":{"activation":"silu","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","schema":"rheon-cubic-groupnorm-model-v3","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,5,6],"frame":0,"native_frame":63,"parent":93,"split":"train","trajectory_id":"baseline--water-train-0093"},{"cell_size_m":0.125,"counts_xyz":[13,13,16],"frame":0,"native_frame":15,"parent":186,"split":"train","trajectory_id":"baseline--water-train-0186"},{"cell_size_m":0.0625,"counts_xyz":[15,13,16],"frame":0,"native_frame":63,"parent":279,"split":"train","trajectory_id":"cohort-1--water-train-0023"},{"cell_size_m":0.125,"counts_xyz":[12,16,8],"frame":0,"native_frame":15,"parent":372,"split":"train","trajectory_id":"cohort-1--water-train-0116"},{"cell_size_m":0.0625,"counts_xyz":[12,13,12],"frame":0,"native_frame":63,"parent":465,"split":"train","trajectory_id":"cohort-1--water-train-0209"},{"cell_size_m":0.125,"counts_xyz":[12,7,16],"frame":0,"native_frame":15,"parent":558,"split":"train","trajectory_id":"cohort-2--water-train-0046"},{"cell_size_m":0.0625,"counts_xyz":[6,6,6],"frame":0,"native_frame":63,"parent":651,"split":"train","trajectory_id":"cohort-2--water-train-0139"},{"cell_size_m":0.125,"counts_xyz":[11,7,9],"frame":0,"native_frame":15,"parent":744,"split":"train","trajectory_id":"cohort-2--water-train-0232"},{"cell_size_m":0.0625,"counts_xyz":[6,11,8],"frame":0,"native_frame":63,"parent":837,"split":"train","trajectory_id":"cohort-3--water-train-0069"},{"cell_size_m":0.125,"counts_xyz":[5,8,14],"frame":0,"native_frame":15,"parent":930,"split":"train","trajectory_id":"cohort-3--water-train-0162"},{"cell_size_m":0.0625,"counts_xyz":[5,15,8],"frame":0,"native_frame":63,"parent":1023,"split":"train","trajectory_id":"cohort-3--water-train-0255"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1024,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":1029,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":1035,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":1041,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":1046,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":1052,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":1058,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":1064,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":1069,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":1075,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":1081,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":1087,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"6a533dab95d1a58e0117e87c7f16774e66fc13c83987bc85d9df2a4352e5caa5","inputs":{"dataset":{"bytes":23033558,"sha256":"3f2fa46bae33df7229fbd15153883cb3fac9dfbda987ca8cff0265d3b582160d"},"dataset_extra":{"bytes":23083606,"sha256":"da990378d5ae021f3b9b5d2a836ac836d8dc425de6536d438df086afe5042a91"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":2958876,"sha256":"a05bc26cff4fcc74c64b60975211f16ede4949edf0e88f29f2a75357d73f8a03"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"040612980ad5f962c10bfdcdf1478b45f91b91b98abfa35a2604e04a21db4291","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3b19af779803209edef4b7cce16e6f8a633f9ecea68f19b63a9e16195431b982","source_witness_sha256":"7ea3f8aa6cbba3f3e8bfb035c2372e959141df96e4295ea01086c92402980303"},"elapsed_s":384.13561277801637,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":0.0003307014985409751,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":32872960,"peak_reserved_bytes":58720256,"tf32":false},"mode":"train","optimizer_updates":10000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-10000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"fac9f59be57f95d9112f623006fd93e1cf26eaaad63c553a86a571cf0ce0aac2"},"checkpoint-2000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"3fc3e056dbc76a12a26120853912c4200564d723e968bedad6cb137656efe279"},"history.json":{"bytes":26992,"role":"training_diagnostics","sha256":"e53b17e2efbb4169ea1d92fede4e16b593d0bce98176fcce030511f65c448c69"},"normalization.json":{"bytes":367209,"role":"preprocessing","sha256":"040612980ad5f962c10bfdcdf1478b45f91b91b98abfa35a2604e04a21db4291"},"predictions.npz":{"bytes":274726,"role":"predictions","sha256":"ec760629b72ad5d55be894616b2d51c80484b083fbc9c6596e7a764f30e97230"},"splits.json":{"bytes":2958876,"role":"training_splits","sha256":"a05bc26cff4fcc74c64b60975211f16ede4949edf0e88f29f2a75357d73f8a03"},"summary.json":{"bytes":10864,"role":"result_summary","sha256":"e8969cdab90970129fed95ff6d34262dbfed966f34d3830ad7416f61cbd4f216"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"fac9f59be57f95d9112f623006fd93e1cf26eaaad63c553a86a571cf0ce0aac2"}},"per_case_inference_s":[0.015720217023044825,0.0021425669838208705,0.007751537021249533,0.007338275026995689,0.0038507400022353977,0.004383116000099108,0.0036177790025249124,0.0017605340108275414,0.0025783420132938772,0.0023646740010008216,0.0029576180095318705,0.0024671489954926074,0.009092844993574545,0.0030798109946772456,0.004822903982130811,0.0031113779987208545,0.00202254400937818,0.003772610012674704,0.0025779909919947386,0.00285934898420237,0.003078609996009618,0.00350060299388133,0.0066262210020795465,0.002481389994500205],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"05f140c2bd7099f9590db49956b683b7d221b535470827a58ac5983c4f45664b","shape":[16,16,16]},"case_001":{"bytes":1200,"dtype":"<f8","sha256":"0b1c2feefee65a79938f34d62df96dacb6e48e1ddc18fbdbfb25360f9be0ee9d","shape":[6,5,5]},"case_002":{"bytes":21632,"dtype":"<f8","sha256":"823f2d0797c8a9a35d01d3d567e75ad6532949d57016490ce5333cdf2edff844","shape":[16,13,13]},"case_003":{"bytes":24960,"dtype":"<f8","sha256":"3185234b228d71721491bff259e59f7527e48a4b070852de5e1ce78d2e66b1d7","shape":[16,13,15]},"case_004":{"bytes":12288,"dtype":"<f8","sha256":"8a7edfb53f61ab67ecf13e5a2c660471082cbaca6c4d10d80e47cb454b853f9f","shape":[8,16,12]},"case_005":{"bytes":14976,"dtype":"<f8","sha256":"205302655f5112363648671d4e7e34a9fe09b558310c55b1972dae05a849f3c8","shape":[12,13,12]},"case_006":{"bytes":10752,"dtype":"<f8","sha256":"1befdbf67482b851e614d274b56555f067fc62d3de62ca5222a4a08efff40cd6","shape":[16,7,12]},"case_007":{"bytes":1728,"dtype":"<f8","sha256":"27f524966a6a87640f2c7cf4c7a0bae970fc2948e38572c5fb5dfad03e6b540c","shape":[6,6,6]},"case_008":{"bytes":5544,"dtype":"<f8","sha256":"233ce23d997c4dadc2d6f01b42db9a627fd2080df9e9e80af891b040627bf10a","shape":[9,7,11]},"case_009":{"bytes":4224,"dtype":"<f8","sha256":"6051402f6a8a834577b6e8242b107777a7820f5361506a7acf949ae840b1ef64","shape":[8,11,6]},"case_010":{"bytes":4480,"dtype":"<f8","sha256":"6762cf8a6bd6dcbe0287ed54b1cfc8a8fe38baba4bfee8e47e6132af15f7b20b","shape":[14,8,5]},"case_011":{"bytes":4800,"dtype":"<f8","sha256":"02c38a0bb6354e5b9b21c5ae97ef949da0de743181b887ac54c80e3b5d1bcb22","shape":[8,15,5]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"690537f37d66b49c68b3fda3105f7e73ed0e76afa59f72e832094cfb798668a7","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"86fbb0556f767a384b96bc9c0a1e60c70dd0628dd5722c9a4e53d324f46dce46","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"29842a21dd6f51d7e03809c1eb9d32b3963821f0611dc934d2e66082861216ab","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"9320a29be31b02babbc4ba07265b3a3fa125e7176d98fe5751f93f4fb8ed0e77","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"ca8f9c5c9f523a328625646ccb5b44aa6da1e51a869a3226f0489f30f7ce2bf7","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"fdfb04f5f376d5e5fda97b55c92355f8874aaa831cf8cdff5be4ad0e3f40070e","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"0d49bc464937f630f69bb5d1846fec06efd4f6bb122cdafc751fcfdff92cd266","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"d7fd2292fab7f05b2b5974f5536053f7bc1ddaaade5168aace71acb654ceadd3","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"dd9ae0ce21ab45eb89fea96301933b5f47720e7fbf82db4cf78c883bce7731b9","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"9fb486e8777e126ee288f57167248fef3bdcba2bc85c3e39aaef59719b3cce2b","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"e532d5e8e2b7352dabef91124ae6c6e1cc40de46f90e0d6882c5a59de4ad5c09","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"2e6f800ed9c192c9cfdd381ddd0d9a7ab6f960dd47dc95225b83fbe6fd0f41f2","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2664370176,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
