# water-sequence-long-256 · CUDA training

Retrospective source review · fit-83c417946315282d

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 371,729-parameter, 38-tensor F32 multiscale 3D pressure-correction CNN with seven input channels: residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep. Encoder widths 8/16/24/32, bottleneck 48, decoder 32/24/16/8, affine-free four-group GroupNorm, SiLU, global-mean broadcast and nearest-to-skip resizing. The linear head supplies a pressure-scaled cell-zero-gauge correction added to the previous pressure. FV arithmetic/output remain F64. This is a pressure request, not a complete simulation advance. The captured cell budget is 65,536.

**Purpose.** Increase the optimization budget on the original 256-scene, two-frame physical training population.

**Experiment.** 10,000 whole-network Adam updates, batch 2, initial learning rate 0.001, seed 20261015 and sampling seed 2026102501. Learning-rate schedule: [{'learning_rate': 0.0001, 'update': 1000}, {'learning_rate': 1e-05, 'update': 6000}]. Gradient norm is clipped at 1.0. 256 training trajectories/512 saved pressure frames; 64 validation trajectories/128 frames. Actual training densities, in scenes: {'8.0': 128, '16.0': 128}. Native training snapshot indices present: [15, 63]. The sealed-water captures use variable rectangular small domains, randomized Gaussian/tapered force position, direction, amplitude, motion, swirl and temporal variation. The objective combines normalized pressure-field and FV equation losses, with scales fitted only on training frames. Reserved-test input bindings are retained separately and do not enter the optimizer parent selection.

**Observed outcome.** The final sampled training loss is 0.0013831257429727239; the retained validation objective changes from 55.894181807199708 to 0.0025726031022003559. The independently rescored 128-request GPU comparison has worst mean-free pressure relative L2 0.11915490497699312 and equation relative L2 0.19760060785529457. These are different metrics/populations from the logged training loss. This source review adds no latency or own-history drift measurement.

**Lesson.** The physical frames, initial tensors and first 2,000 sampled update batches match C06 despite packet resharding/namespaces. Additional updates reduce the captured validation objective, but the final candidates still have substantial common worst error. This is evidence about fitting budget on small grids, not generalization to larger domains or new timesteps.

**Reproduction.** Use the exact captured project/train recipe, config, source witness, runtime, typed input bindings, normalization and seeded checkpoint inventory. Captured training runs execute CUDA gradients/Adam updates, then perform historical CPU deployment/parity checks. Do not run that recipe unchanged under the current GPU-only neural-forward policy. A clean GPU-only retraining/replay path and new-Rheon data regeneration remain to be qualified. Final-update selection and no optimizer-resume state are explicit; preserve the old physics epoch and original receipts.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-learning-sequence-v1/long-256/managed/stage-00/artifacts/receipt.json",
    "sha256": "83c417946315282d5162232503020161c0e9b6e532c073b7c1e6136727c49718"
  },
  "history": {
    "path": "runs/water-learning-sequence-v1/long-256/managed/stage-00/artifacts/history.json",
    "sha256": "e5e6c6daa9dfc5719f75f025e9e9501ba74900a128ee9ab8b538f8ab878779c7"
  },
  "sources": [
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/config.json",
      "sha256": "046f16f5ac47ee5bc2513fdacf9e7ba06a8ab23b432ca735a6de5c232a7cc077"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/model.py",
      "sha256": "d42453ac9060275c7fc6676eab0e23eee7188cf2604a82714bfd4570104d7c0c"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/policy.py",
      "sha256": "5e6fb00db31f558df0bf3665422c7afd4a6ed14e8608e9514576fb86632858cd"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/qualification.json",
      "sha256": "1259371f47c6010e4eb8693f100dbec4a195657f56dfb183a3044620de824f6f"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/test_inference.py",
      "sha256": "43bc71643e362670c960c184b6b1a378f93aba7a6408a2763a960059592a1b56"
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/train.py",
      "sha256": "75a6e4281f70646ef0e66fc3b93163b573df05b66e54006ca6630c0051338be5"
    }
  ],
  "source_witness": {
    "path": "runs/water-learning-sequence-v1/long-256/project/source-witness.json",
    "sha256": "e6e2a4ff3018f222b5cc7e380eb5af9f8ceff7aa5b6f62d5d60a78cb7e1dd58c"
  },
  "training_population": {
    "path": "runs/water-learning-sequence-v1/long-256/managed/stage-00/artifacts/splits.json",
    "sha256": "babfb21aed2a01c38fa6f735f5c67e2987be0daad1867b200806153adfd6c15e"
  },
  "training_inputs": [
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/inputs/dataset.npz",
      "sha256": "e9b307b72ee95ec95707642f4a6c51deb83b293120dda6cf1bf8489f56faa408",
      "slot": "dataset",
      "bytes": 12157270
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/inputs/dataset_extra.npz",
      "sha256": "ce99031105a4e963770bf4a6f3ec0ff9383e3b01f583cc4137fc35e367695059",
      "slot": "dataset_extra",
      "bytes": 12171990
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset",
      "bytes": 2561142
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra",
      "bytes": 2564086
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits",
      "bytes": 339677
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/inputs/splits.json",
      "sha256": "babfb21aed2a01c38fa6f735f5c67e2987be0daad1867b200806153adfd6c15e",
      "slot": "splits",
      "bytes": 986084
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset",
      "bytes": 2209622
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra",
      "bytes": 2212566
    },
    {
      "path": "runs/water-learning-sequence-v1/long-256/project/inputs/validation_splits.json",
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
    "item_id": "fit-83c417946315282d",
    "kind": "training_attempt",
    "name": "water-sequence-long-256 · CUDA training",
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
    "receipt_sha256": "83c417946315282d5162232503020161c0e9b6e532c073b7c1e6136727c49718",
    "receipt_paths": [
      "runs/water-learning-sequence-v1/long-256/managed/stage-00/artifacts/receipt.json",
      "runs/water-learning-sequence-v1/long-256/managed/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/a63a03c360ab4dc993a4add962f86834/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/a63a03c360ab4dc993a4add962f86834/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/a63a03c360ab4dc993a4add962f86834/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/a63a03c360ab4dc993a4add962f86834/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/a63a03c360ab4dc993a4add962f86834/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-a63a03c360ab4dc993a4add962f86834/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-a63a03c360ab4dc993a4add962f86834/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-a63a03c360ab4dc993a4add962f86834/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-a63a03c360ab4dc993a4add962f86834/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/a63a03c360ab4dc993a4add962f86834/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/a63a03c360ab4dc993a4add962f86834/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/a63a03c360ab4dc993a4add962f86834/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/a63a03c360ab4dc993a4add962f86834/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-a63a03c360ab4dc993a4add962f86834/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-a63a03c360ab4dc993a4add962f86834/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-a63a03c360ab4dc993a4add962f86834/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-a63a03c360ab4dc993a4add962f86834/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-a63a03c360ab4dc993a4add962f86834/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "a63a03c360ab4dc993a4add962f86834",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-sequence-long-256 · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "046f16f5ac47ee5bc2513fdacf9e7ba06a8ab23b432ca735a6de5c232a7cc077",
      "inputs": {
        "dataset": {
          "bytes": 12157270,
          "sha256": "e9b307b72ee95ec95707642f4a6c51deb83b293120dda6cf1bf8489f56faa408"
        },
        "dataset_extra": {
          "bytes": 12171990,
          "sha256": "ce99031105a4e963770bf4a6f3ec0ff9383e3b01f583cc4137fc35e367695059"
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
          "bytes": 986084,
          "sha256": "babfb21aed2a01c38fa6f735f5c67e2987be0daad1867b200806153adfd6c15e"
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
      "normalization_sha256": "74763c28d68a4c6fb2923170bb58cd9a7306d2a955b86417efd2f115cb08922f",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "49a69425b29d438760244abd422e08d3aad97df0b193982669633811f92e50e6",
      "source_witness_sha256": "e6e2a4ff3018f222b5cc7e380eb5af9f8ceff7aa5b6f62d5d60a78cb7e1dd58c"
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
        "sha256": "cdf45b9e5c39c48cbe1518306899f76eb1130d68fecad00b88810bdbb30d400c",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "aa82bd122fde9406d9e6b92114b3e6ba77281f5ee8b9c726aa168dacbd93a62d",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "cdf45b9e5c39c48cbe1518306899f76eb1130d68fecad00b88810bdbb30d400c",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2469720064,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/7"
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
      "receipt": "runs/water-learning-sequence-v1/long-256/managed/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-learning-sequence-v1/long-256/managed/stage-00/artifacts/receipt.json",
    "sha256": "83c417946315282d5162232503020161c0e9b6e532c073b7c1e6136727c49718",
    "bytes": 11755
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C08": {
        "name": "long-256",
        "weights_sha256": "cdf45b9e5c39c48cbe1518306899f76eb1130d68fecad00b88810bdbb30d400c",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.19760060785529457,
          "pressure_mean_free_relative_l2": 0.11915490497699312,
          "pressure_absolute_rms": 3.4564258540750536,
          "pressure_absolute_max": 15.599142497148394
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
              "common_equation_relative_l2": 0.13607598280032127,
              "pressure_mean_free_relative_l2": 0.06921681384320755,
              "pressure_absolute_rms": 3.4564258540750536,
              "pressure_absolute_max": 15.599142497148394
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.19760060785529457,
              "pressure_mean_free_relative_l2": 0.11915490497699312,
              "pressure_absolute_rms": 2.7849767392794678,
              "pressure_absolute_max": 7.2097466593794906
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C08/results.json",
          "sha256": "2cadc40c96a53d5b05c6b47cc94e9e8c17d25a01f94927c9de63a6745f084100"
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
{"architecture":{"activation":"silu","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","schema":"rheon-cubic-groupnorm-model-v3","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[15,10,14],"frame":1,"native_frame":63,"parent":23,"split":"train","trajectory_id":"baseline--water-train-0023"},{"cell_size_m":0.125,"counts_xyz":[5,14,8],"frame":0,"native_frame":15,"parent":46,"split":"train","trajectory_id":"baseline--water-train-0046"},{"cell_size_m":0.0625,"counts_xyz":[10,15,16],"frame":1,"native_frame":63,"parent":69,"split":"train","trajectory_id":"baseline--water-train-0069"},{"cell_size_m":0.125,"counts_xyz":[14,14,9],"frame":0,"native_frame":15,"parent":92,"split":"train","trajectory_id":"baseline--water-train-0092"},{"cell_size_m":0.0625,"counts_xyz":[9,11,16],"frame":1,"native_frame":63,"parent":115,"split":"train","trajectory_id":"baseline--water-train-0115"},{"cell_size_m":0.0625,"counts_xyz":[13,16,16],"frame":0,"native_frame":15,"parent":139,"split":"train","trajectory_id":"baseline--water-train-0139"},{"cell_size_m":0.125,"counts_xyz":[7,10,12],"frame":1,"native_frame":63,"parent":162,"split":"train","trajectory_id":"baseline--water-train-0162"},{"cell_size_m":0.0625,"counts_xyz":[14,14,6],"frame":0,"native_frame":15,"parent":185,"split":"train","trajectory_id":"baseline--water-train-0185"},{"cell_size_m":0.125,"counts_xyz":[16,7,6],"frame":1,"native_frame":63,"parent":208,"split":"train","trajectory_id":"baseline--water-train-0208"},{"cell_size_m":0.0625,"counts_xyz":[12,7,7],"frame":0,"native_frame":15,"parent":231,"split":"train","trajectory_id":"baseline--water-train-0231"},{"cell_size_m":0.0625,"counts_xyz":[6,16,13],"frame":1,"native_frame":63,"parent":255,"split":"train","trajectory_id":"baseline--water-train-0255"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":256,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":261,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":267,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":273,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":278,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":284,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":290,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":296,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":301,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":307,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":313,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":319,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"046f16f5ac47ee5bc2513fdacf9e7ba06a8ab23b432ca735a6de5c232a7cc077","inputs":{"dataset":{"bytes":12157270,"sha256":"e9b307b72ee95ec95707642f4a6c51deb83b293120dda6cf1bf8489f56faa408"},"dataset_extra":{"bytes":12171990,"sha256":"ce99031105a4e963770bf4a6f3ec0ff9383e3b01f583cc4137fc35e367695059"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":986084,"sha256":"babfb21aed2a01c38fa6f735f5c67e2987be0daad1867b200806153adfd6c15e"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"74763c28d68a4c6fb2923170bb58cd9a7306d2a955b86417efd2f115cb08922f","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"49a69425b29d438760244abd422e08d3aad97df0b193982669633811f92e50e6","source_witness_sha256":"e6e2a4ff3018f222b5cc7e380eb5af9f8ceff7aa5b6f62d5d60a78cb7e1dd58c"},"elapsed_s":377.3735397270066,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":0.00030965107718472495,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":32808448,"peak_reserved_bytes":58720256,"tf32":false},"mode":"train","optimizer_updates":10000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-10000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"cdf45b9e5c39c48cbe1518306899f76eb1130d68fecad00b88810bdbb30d400c"},"checkpoint-2000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"aa82bd122fde9406d9e6b92114b3e6ba77281f5ee8b9c726aa168dacbd93a62d"},"history.json":{"bytes":27016,"role":"training_diagnostics","sha256":"e5e6c6daa9dfc5719f75f025e9e9501ba74900a128ee9ab8b538f8ab878779c7"},"normalization.json":{"bytes":92398,"role":"preprocessing","sha256":"74763c28d68a4c6fb2923170bb58cd9a7306d2a955b86417efd2f115cb08922f"},"predictions.npz":{"bytes":298222,"role":"predictions","sha256":"4863f732cb907b2cb331fd8706b1c9e43b83b92711f1dc1b583fc91f0a592eb9"},"splits.json":{"bytes":986084,"role":"training_splits","sha256":"babfb21aed2a01c38fa6f735f5c67e2987be0daad1867b200806153adfd6c15e"},"summary.json":{"bytes":3317,"role":"result_summary","sha256":"ca6692821a88e41f9c769257e242b9c44c61471ecb02daa66707a4fb78ef4412"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"cdf45b9e5c39c48cbe1518306899f76eb1130d68fecad00b88810bdbb30d400c"}},"per_case_inference_s":[0.01595902198459953,0.005890598986297846,0.0027290350117255002,0.006322119996184483,0.00455590701312758,0.004456588008906692,0.008056460996158421,0.0031906549993436784,0.0029809899860993028,0.0021571230026893318,0.0022103619994595647,0.0037644679832737893,0.009234148019459099,0.003472187992883846,0.005075188004411757,0.0034523500071372837,0.0021460040006786585,0.004334728000685573,0.003307083999970928,0.003940463007893413,0.003933863015845418,0.003890471998602152,0.007677660003537312,0.0029187350010033697],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"a68e9d09b04181195d2968338fdb884a936b7714a6a32b31234e71573f778dc6","shape":[16,16,16]},"case_001":{"bytes":16800,"dtype":"<f8","sha256":"32896b0af671b4d16feb0231095e428286a7c47e1d5981e98337e4ee4e3e83e4","shape":[14,10,15]},"case_002":{"bytes":4480,"dtype":"<f8","sha256":"f5d6e832ff5c01f17dfd374fb516fda405cc495f38765f93801bdc4ea8678c1d","shape":[8,14,5]},"case_003":{"bytes":19200,"dtype":"<f8","sha256":"7d057d64b3c3f73018aa42dfd130b2b96613738a1060881b122b3520e25cb795","shape":[16,15,10]},"case_004":{"bytes":14112,"dtype":"<f8","sha256":"1af9b22686329f1991de34ab9895c6e8d50002a8d390caa28b7184a818108573","shape":[9,14,14]},"case_005":{"bytes":12672,"dtype":"<f8","sha256":"eabc091c95a8747091fd9b1f8fa40ee6f1c0d9e72d65060bf187ad90471c8885","shape":[16,11,9]},"case_006":{"bytes":26624,"dtype":"<f8","sha256":"dd4faa1549ff08c9a54a245c7d3584540fc809bf17398e6dd9d475ca6bb6ec20","shape":[16,16,13]},"case_007":{"bytes":6720,"dtype":"<f8","sha256":"2891c0daed7f16dc6361090d08155690c9ee55b37b182a759a2657e071cad1d5","shape":[12,10,7]},"case_008":{"bytes":9408,"dtype":"<f8","sha256":"134550cce1d4dc698337de9e2e79ae444262673402e30f7173b2ec9e57c4fd15","shape":[6,14,14]},"case_009":{"bytes":5376,"dtype":"<f8","sha256":"d0e72ebce9c0967459f5f929646f445cad7f5a9becfab6dd6c6516d5b5d3469a","shape":[6,7,16]},"case_010":{"bytes":4704,"dtype":"<f8","sha256":"b84723f4948d4135ba66fe0019289cb597dbf2618a6409782ebb10afe72780ca","shape":[7,7,12]},"case_011":{"bytes":9984,"dtype":"<f8","sha256":"0a3bb9fd47f9b414eb3b71d26f6f966ff5b15f4ec7850c1f59e04e4eadfd6c22","shape":[13,16,6]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"eae3ed850f1904ce0c6211a5d55ec4ebd8c1620d070dcc310af1da4da0da43ad","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"05fbe978ca459f0dec9dbee574b168f5c5ea7dd4610dfeedea9d6ecba5fb926e","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"30cb7ec6430953c4ce3993098618092e6e4fe23aceeb8f689d5259b624065ea2","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"76540f41c0007bbfc76a74a7012c3b88037d413c15446d9e21b179db5fe9bcce","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"f9df6730cae443d00ca9a3413971279ff5103d6e7e84e566a5fc0af89846c2cc","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"f436ad5b19de941d8ca41cfc2125d7cfb0f610e41975c05ece724c4a46024881","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"e0ab8760e69af4bc63f9bf89f6b50677041f8b2e8d762d5d19eca78b7eac3b88","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"d7ee9975a280068391cebdf765eb548a3d505106bf167c6c77ddf63c60bda555","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"273706ed6b884ba4f658161ff95da381877862272ce1f686a4a7e6aa8c23d3f8","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"5501ea084f769b019a9a2e8edc68d7d87c83deffcf0e856ed81f1e6f8caa23a8","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"8ee7ea5543bfc579129a133a5c14c52e3bac2e6653fb67a2e449eb616abb85cc","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"fc26fb6074c852fb82820073f79b191f346e57ea5028b9396bb27640b9b3b82f","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2469720064,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
