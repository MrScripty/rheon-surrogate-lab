# water-sequence-scale-256 · CUDA training

Retrospective source review · fit-c88c7a258cbecbb8

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 371,729-parameter, 38-tensor F32 multiscale 3D pressure-correction CNN with seven input channels: residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep. Encoder widths 8/16/24/32, bottleneck 48, decoder 32/24/16/8, affine-free four-group GroupNorm, SiLU, global-mean broadcast and nearest-to-skip resizing. The linear head supplies a pressure-scaled cell-zero-gauge correction added to the previous pressure. FV arithmetic/output remain F64. This is a pressure request, not a complete simulation advance. The captured cell budget is 65,536.

**Purpose.** Establish a 256-scene, one-frame-per-scene data-scaling control with the long optimization schedule.

**Experiment.** 10,000 whole-network Adam updates, batch 2, initial learning rate 0.001, seed 20261015 and sampling seed 2026102501. Learning-rate schedule: [{'learning_rate': 0.0001, 'update': 1000}, {'learning_rate': 1e-05, 'update': 6000}]. Gradient norm is clipped at 1.0. 256 training trajectories/256 saved pressure frames; 64 validation trajectories/128 frames. Actual training densities, in scenes: {'8.0': 128, '16.0': 128}. Native training snapshot indices present: [15, 63]. The sealed-water captures use variable rectangular small domains, randomized Gaussian/tapered force position, direction, amplitude, motion, swirl and temporal variation. The objective combines normalized pressure-field and FV equation losses, with scales fitted only on training frames. Reserved-test input bindings are retained separately and do not enter the optimizer parent selection.

**Observed outcome.** The final sampled training loss is 0.0019979344262684978; the retained validation objective changes from 68.341156818470211 to 0.0027460474486211894. The independently rescored 128-request GPU comparison has worst mean-free pressure relative L2 0.12126726552775925 and equation relative L2 0.18913792467364093. These are different metrics/populations from the logged training loss. This source review adds no latency or own-history drift measurement.

**Lesson.** One saved frame per scene reduces the training frame count from 512 to 256 while retaining 256 physical scenes. Selected frames include both native indices 15 and 63 across scenes; one frame per scene does not mean one common timestamp. The train-derived scales change with this selection, so compare the same common physical errors rather than logged normalized losses alone.

**Reproduction.** Use the exact captured project/train recipe, config, source witness, runtime, typed input bindings, normalization and seeded checkpoint inventory. Captured training runs execute CUDA gradients/Adam updates, then perform historical CPU deployment/parity checks. Do not run that recipe unchanged under the current GPU-only neural-forward policy. A clean GPU-only retraining/replay path and new-Rheon data regeneration remain to be qualified. Final-update selection and no optimizer-resume state are explicit; preserve the old physics epoch and original receipts.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-learning-sequence-v1/scale-256/managed/stage-00/artifacts/receipt.json",
    "sha256": "c88c7a258cbecbb844ee05fbcb6f75c3047dc31dca7cc055f215a3d9836a40be"
  },
  "history": {
    "path": "runs/water-learning-sequence-v1/scale-256/managed/stage-00/artifacts/history.json",
    "sha256": "d9f6b4bfe5e6bfe9533925554c7402e5184b560fac41805d91ffb72bb5677bb1"
  },
  "sources": [
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/config.json",
      "sha256": "2c9e12aa2b407e5345c99045f46a1a051c0748b8f1f97b46caa617d4e54673a1"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/model.py",
      "sha256": "d42453ac9060275c7fc6676eab0e23eee7188cf2604a82714bfd4570104d7c0c"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/policy.py",
      "sha256": "5e6fb00db31f558df0bf3665422c7afd4a6ed14e8608e9514576fb86632858cd"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/qualification.json",
      "sha256": "b84c4289ef60fcd4288f7247e34998eb51fd5e6597998f83f2f90a9b1a8e7d1e"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/test_inference.py",
      "sha256": "43bc71643e362670c960c184b6b1a378f93aba7a6408a2763a960059592a1b56"
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/train.py",
      "sha256": "75a6e4281f70646ef0e66fc3b93163b573df05b66e54006ca6630c0051338be5"
    }
  ],
  "source_witness": {
    "path": "runs/water-learning-sequence-v1/scale-256/project/source-witness.json",
    "sha256": "de3b09de4438dc74366088d931887955029fc1a44a65eb3b316532fb9bdada8a"
  },
  "training_population": {
    "path": "runs/water-learning-sequence-v1/scale-256/managed/stage-00/artifacts/splits.json",
    "sha256": "95b4db8bc528cdbdf00879b1470833e8c8b979cced21d9dce49aefd89f3d4210"
  },
  "training_inputs": [
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/inputs/dataset.npz",
      "sha256": "87a34a7a473aacb98a377fb8d752da8d560e765b11270c591402170ffdf73918",
      "slot": "dataset",
      "bytes": 7247702
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/inputs/dataset_extra.npz",
      "sha256": "514f2b49525da9206de8f9e9fd0a965c2c567e53370c9681d65e10568f65bcd8",
      "slot": "dataset_extra",
      "bytes": 7262422
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset",
      "bytes": 2561142
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra",
      "bytes": 2564086
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits",
      "bytes": 339677
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/inputs/splits.json",
      "sha256": "95b4db8bc528cdbdf00879b1470833e8c8b979cced21d9dce49aefd89f3d4210",
      "slot": "splits",
      "bytes": 897630
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset",
      "bytes": 2209622
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra",
      "bytes": 2212566
    },
    {
      "path": "runs/water-learning-sequence-v1/scale-256/project/inputs/validation_splits.json",
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
    "item_id": "fit-c88c7a258cbecbb8",
    "kind": "training_attempt",
    "name": "water-sequence-scale-256 · CUDA training",
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
    "receipt_sha256": "c88c7a258cbecbb844ee05fbcb6f75c3047dc31dca7cc055f215a3d9836a40be",
    "receipt_paths": [
      "runs/water-learning-sequence-v1/scale-256/managed/stage-00/artifacts/receipt.json",
      "runs/water-learning-sequence-v1/scale-256/managed/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/a9c6b01204874a1e8d12fe863f216e8d/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/a9c6b01204874a1e8d12fe863f216e8d/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/a9c6b01204874a1e8d12fe863f216e8d/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/a9c6b01204874a1e8d12fe863f216e8d/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/a9c6b01204874a1e8d12fe863f216e8d/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-a9c6b01204874a1e8d12fe863f216e8d/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-a9c6b01204874a1e8d12fe863f216e8d/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-a9c6b01204874a1e8d12fe863f216e8d/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-a9c6b01204874a1e8d12fe863f216e8d/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/a9c6b01204874a1e8d12fe863f216e8d/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/a9c6b01204874a1e8d12fe863f216e8d/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/a9c6b01204874a1e8d12fe863f216e8d/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/a9c6b01204874a1e8d12fe863f216e8d/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-a9c6b01204874a1e8d12fe863f216e8d/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-a9c6b01204874a1e8d12fe863f216e8d/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-a9c6b01204874a1e8d12fe863f216e8d/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-a9c6b01204874a1e8d12fe863f216e8d/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-a9c6b01204874a1e8d12fe863f216e8d/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "a9c6b01204874a1e8d12fe863f216e8d",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-sequence-scale-256 · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "2c9e12aa2b407e5345c99045f46a1a051c0748b8f1f97b46caa617d4e54673a1",
      "inputs": {
        "dataset": {
          "bytes": 7247702,
          "sha256": "87a34a7a473aacb98a377fb8d752da8d560e765b11270c591402170ffdf73918"
        },
        "dataset_extra": {
          "bytes": 7262422,
          "sha256": "514f2b49525da9206de8f9e9fd0a965c2c567e53370c9681d65e10568f65bcd8"
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
          "bytes": 897630,
          "sha256": "95b4db8bc528cdbdf00879b1470833e8c8b979cced21d9dce49aefd89f3d4210"
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
      "normalization_sha256": "110daea256f2fca34a7c4ccf492d77d21beb47bbb88d18022a2a45a238c9e32d",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "7dee63cab8a981d7ad4bdac0956df0c48020feab6cebc4c256aad495bff64781",
      "source_witness_sha256": "de3b09de4438dc74366088d931887955029fc1a44a65eb3b316532fb9bdada8a"
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
        "sha256": "d1ee4f181becb11c901d3cc31d8b9388b5dfe222dbda6593245b6dce48ba3616",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "048d109c85b3ff987138b2fd684d2bc2f11efef26bd2be78214942a18f02ad44",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "d1ee4f181becb11c901d3cc31d8b9388b5dfe222dbda6593245b6dce48ba3616",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2446618624,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/8"
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
      "receipt": "runs/water-learning-sequence-v1/scale-256/managed/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-learning-sequence-v1/scale-256/managed/stage-00/artifacts/receipt.json",
    "sha256": "c88c7a258cbecbb844ee05fbcb6f75c3047dc31dca7cc055f215a3d9836a40be",
    "bytes": 11757
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C09": {
        "name": "scale-256",
        "weights_sha256": "d1ee4f181becb11c901d3cc31d8b9388b5dfe222dbda6593245b6dce48ba3616",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.18913792467364093,
          "pressure_mean_free_relative_l2": 0.12126726552775925,
          "pressure_absolute_rms": 2.9091839230642984,
          "pressure_absolute_max": 12.016184306948258
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
              "common_equation_relative_l2": 0.15585531242786885,
              "pressure_mean_free_relative_l2": 0.06610382353107004,
              "pressure_absolute_rms": 2.9091839230642984,
              "pressure_absolute_max": 12.016184306948258
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.18913792467364093,
              "pressure_mean_free_relative_l2": 0.12126726552775925,
              "pressure_absolute_rms": 2.801590568248296,
              "pressure_absolute_max": 7.192904329269325
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C09/results.json",
          "sha256": "00cc40e5ea40cc9cf8965705b2700960d6fa5794cc548a1293e910151e1af6e7"
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
{"architecture":{"activation":"silu","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","schema":"rheon-cubic-groupnorm-model-v3","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[15,10,14],"frame":0,"native_frame":63,"parent":23,"split":"train","trajectory_id":"baseline--water-train-0023"},{"cell_size_m":0.125,"counts_xyz":[5,14,8],"frame":0,"native_frame":15,"parent":46,"split":"train","trajectory_id":"baseline--water-train-0046"},{"cell_size_m":0.0625,"counts_xyz":[10,15,16],"frame":0,"native_frame":63,"parent":69,"split":"train","trajectory_id":"baseline--water-train-0069"},{"cell_size_m":0.125,"counts_xyz":[14,14,9],"frame":0,"native_frame":15,"parent":92,"split":"train","trajectory_id":"baseline--water-train-0092"},{"cell_size_m":0.0625,"counts_xyz":[9,11,16],"frame":0,"native_frame":63,"parent":115,"split":"train","trajectory_id":"baseline--water-train-0115"},{"cell_size_m":0.0625,"counts_xyz":[13,16,16],"frame":0,"native_frame":63,"parent":139,"split":"train","trajectory_id":"baseline--water-train-0139"},{"cell_size_m":0.125,"counts_xyz":[7,10,12],"frame":0,"native_frame":15,"parent":162,"split":"train","trajectory_id":"baseline--water-train-0162"},{"cell_size_m":0.0625,"counts_xyz":[14,14,6],"frame":0,"native_frame":63,"parent":185,"split":"train","trajectory_id":"baseline--water-train-0185"},{"cell_size_m":0.125,"counts_xyz":[16,7,6],"frame":0,"native_frame":15,"parent":208,"split":"train","trajectory_id":"baseline--water-train-0208"},{"cell_size_m":0.0625,"counts_xyz":[12,7,7],"frame":0,"native_frame":63,"parent":231,"split":"train","trajectory_id":"baseline--water-train-0231"},{"cell_size_m":0.0625,"counts_xyz":[6,16,13],"frame":0,"native_frame":63,"parent":255,"split":"train","trajectory_id":"baseline--water-train-0255"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":256,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":261,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":267,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":273,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":278,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":284,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":290,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":296,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":301,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":307,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":313,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":319,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"2c9e12aa2b407e5345c99045f46a1a051c0748b8f1f97b46caa617d4e54673a1","inputs":{"dataset":{"bytes":7247702,"sha256":"87a34a7a473aacb98a377fb8d752da8d560e765b11270c591402170ffdf73918"},"dataset_extra":{"bytes":7262422,"sha256":"514f2b49525da9206de8f9e9fd0a965c2c567e53370c9681d65e10568f65bcd8"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":897630,"sha256":"95b4db8bc528cdbdf00879b1470833e8c8b979cced21d9dce49aefd89f3d4210"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"110daea256f2fca34a7c4ccf492d77d21beb47bbb88d18022a2a45a238c9e32d","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"7dee63cab8a981d7ad4bdac0956df0c48020feab6cebc4c256aad495bff64781","source_witness_sha256":"de3b09de4438dc74366088d931887955029fc1a44a65eb3b316532fb9bdada8a"},"elapsed_s":386.54521932802163,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":0.0002729663478824307,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":32808448,"peak_reserved_bytes":58720256,"tf32":false},"mode":"train","optimizer_updates":10000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-10000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"d1ee4f181becb11c901d3cc31d8b9388b5dfe222dbda6593245b6dce48ba3616"},"checkpoint-2000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"048d109c85b3ff987138b2fd684d2bc2f11efef26bd2be78214942a18f02ad44"},"history.json":{"bytes":27029,"role":"training_diagnostics","sha256":"d9f6b4bfe5e6bfe9533925554c7402e5184b560fac41805d91ffb72bb5677bb1"},"normalization.json":{"bytes":92400,"role":"preprocessing","sha256":"110daea256f2fca34a7c4ccf492d77d21beb47bbb88d18022a2a45a238c9e32d"},"predictions.npz":{"bytes":298222,"role":"predictions","sha256":"74704217ed07e7c11ead5a6ffdb743bbfa0cecf2d1b0c823331aa9a4c4448df6"},"splits.json":{"bytes":897630,"role":"training_splits","sha256":"95b4db8bc528cdbdf00879b1470833e8c8b979cced21d9dce49aefd89f3d4210"},"summary.json":{"bytes":3318,"role":"result_summary","sha256":"69ddc07be9beceede383c85ba6859db321ebac292b03036764cf54300b732849"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"d1ee4f181becb11c901d3cc31d8b9388b5dfe222dbda6593245b6dce48ba3616"}},"per_case_inference_s":[0.011736382992239669,0.004170661995885894,0.0021961299935355783,0.005664640018949285,0.004598508996423334,0.004186296020634472,0.006335592013783753,0.0028718789981212467,0.0027226719830650836,0.0020093669882044196,0.002083940984448418,0.003510356997139752,0.007129147998057306,0.0027039840060751885,0.004151837987592444,0.0027972969983238727,0.001739575993269682,0.003369745012605563,0.0024123619950842112,0.0025940499908756465,0.0026431680016685277,0.0028306699823588133,0.005700833018636331,0.0022362909803632647],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"7b2cfa00adf7816ad092f61297975e4562893f2013c48d3c83fc45047a841021","shape":[16,16,16]},"case_001":{"bytes":16800,"dtype":"<f8","sha256":"b8a82dbb4d9561d563ea54e0f50338011e36b286d29495b76a999e184b1f68c4","shape":[14,10,15]},"case_002":{"bytes":4480,"dtype":"<f8","sha256":"d7868a768ea8d1c1d5405a8c7a91a04a1784b2d23d33a85bf7b0c6b38288cf1c","shape":[8,14,5]},"case_003":{"bytes":19200,"dtype":"<f8","sha256":"d75871f7f4a2d855741b6326c1a1b290284ac36f70fe652c98149fb78012fe7b","shape":[16,15,10]},"case_004":{"bytes":14112,"dtype":"<f8","sha256":"0b51c390622e3e9b36b5adff08a468c3f0f4b31af250431388f1c613e0e0e2c5","shape":[9,14,14]},"case_005":{"bytes":12672,"dtype":"<f8","sha256":"462ab8e6acffbcb4339749c22a8c3e4b7065f3cd232f598dbe92a15ff86225e3","shape":[16,11,9]},"case_006":{"bytes":26624,"dtype":"<f8","sha256":"0a5983fc10b532870776c7171e475e2f979733a5ae923b52242669c3b422569d","shape":[16,16,13]},"case_007":{"bytes":6720,"dtype":"<f8","sha256":"c8c89b9f494ee8472129056c8c1e0bee2e594d3ebb2aa0339f4c56e1536b4360","shape":[12,10,7]},"case_008":{"bytes":9408,"dtype":"<f8","sha256":"5cb5f991fda7b3dc4fafaa2ad58058828c99e46ce896d8bd990a51f45755fc45","shape":[6,14,14]},"case_009":{"bytes":5376,"dtype":"<f8","sha256":"0259114a7cf490e0f47d5c4acb586d7cb1447cd81e1ffe0843288f53a419eb89","shape":[6,7,16]},"case_010":{"bytes":4704,"dtype":"<f8","sha256":"20238307f53623fee1729421410195da866faf8f677692c4df6ebb7c8dc42545","shape":[7,7,12]},"case_011":{"bytes":9984,"dtype":"<f8","sha256":"2391e0fbefdddd56924903c703339d2b448c5f52901ca2bd823a8f3665996cdd","shape":[13,16,6]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"d0829231e013e795e6b17ce14d0b6c7afba94206491be80766beba82e313e0e4","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"e0d4e70dd6d7ef8c9a222fd516b72f0a831afb3ca25196bf051a6caa71487d53","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"47a35fc741e7f87ff64ab852aa5cf91f36382c42560d268f5cbd1f8665e5ef1d","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"414b384ed447751dd26523c026e5972347c6542e6b781afdce9488e99303bae3","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"6e79aedac3afa57e4c3c10a49999605d02a6562c9f1220bf42fcf6897fab6b7b","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"d73acde091c697dbb18683f2cbd02030f2ae73542c46c18e19d2217f7566495c","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"1476d9d67fea80f9a76159433a0d1d476d2445634ead2956ebb571266ddb4eda","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"d7c7e29944d014c83c919867dc9718795818aa610e9dc8242e12ab0d12fb5a35","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"6b821746f3d047bc613f6ae678f2d5b33ca715427efd9ddb0ef62555df319f61","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"29f9e2e6e1b775d4022236ba34f2a875cba7d9b9d7ed0faa464b107309440fda","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"fac58e0d6b0b333faae8618fef60661da04848b7d769f593312f85845c514841","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"4353b498fb9cbf1ed8b693b1f499467fb2ba4d7870a6953b6389f6fc76a27896","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2446618624,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
