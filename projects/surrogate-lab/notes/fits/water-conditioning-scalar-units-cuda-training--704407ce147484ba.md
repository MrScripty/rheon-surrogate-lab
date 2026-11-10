# water-conditioning-scalar-units · CUDA training

Retrospective source review · fit-704407ce147484ba

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** The same seven-channel multiscale 3D pressure-correction CNN as C07: 371,729 F32 parameters in 38 tensors, encoder widths 8/16/24/32, bottleneck 48, decoder widths 32/24/16/8, SiLU and a linear pressure head. Residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep are inputs. The cell-zero-gauged correction is added to previous pressure using F64 reconstruction and FV arithmetic. One call proposes pressure; it does not advance a simulation. Affine-free four-group GroupNorm remains in each convolutional block. The rho/dt channels use the fixed physical units described in the purpose; residual/output amplitude is not normalized per sample.

**Purpose.** Condition the scalar input channels in physical units: divide fluid density by 1,000 kg/m3 and timestep by 1/60 s, leaving fixed training RMS amplitude scales.

**Experiment.** 10,000 whole-network Adam updates, batch 1, initial learning rate 0.001, schedule [{'learning_rate': 0.0001, 'update': 1000}, {'learning_rate': 1e-05, 'update': 6000}], seed 20261015, PCG64 sample seed 2026102501 and gradient clipping at norm 1.0. Training uses just native frame 15 from one 16x16x16 scene at 8 cells/metre and a 1/60 s physical timestep. The 64 validation scenes supply 128 pressure frames at 8/16 cells per metre, with small rectangular grids. The equal-weight objective combines normalized squared pressure-field and FV equation errors. Preprocessing RMS scales are fitted on the one training frame; reserved-test bindings remain separate. The final configured update is selected; no optimizer-resume state is saved.

**Observed outcome.** The last sampled training loss is 8.7632781216849229e-07; captured validation objective 37.254603733204192 initially and 0.075574614718326494 finally. On the common 128 native pressure requests, worst mean-free pressure relative L2 is 1.6662777694431767 and equation relative L2 is 0.44035031269039998. These common errors differ from the training objective. This review adds no timing measurement or drift result.

**Lesson.** Dimensionless scalar features are plausible conditioning, but this captured run worsens common worst pressure relative L2 above 1.6 while still fitting its training frame. The observation does not establish a universal failure of physical scaling; it establishes that this change did not improve this trained candidate.

**Reproduction.** Use the captured typed train recipe, exact config/source witness/runtime, original input slots, train-only normalization and seeded tensor inventory. The historical recipe trains on CUDA then performs CPU deployment/parity forwards. A clean reproduction must first qualify a GPU-only path under the current execution policy. Retain the old-Rheon capture epoch and exact receipts; new-Rheon regeneration, clean retraining and timestep/large-grid generalization remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-conditioning-v1/scalar-units/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "704407ce147484baa4fb89c57b4453aeedb4d61726d25bb2af4012f64975fef1"
  },
  "history": {
    "path": "runs/water-conditioning-v1/scalar-units/managed-full/stage-00/artifacts/history.json",
    "sha256": "c2c1915a816318dd8a18f2789959aad98ff9729abb90b6434ac93e128b9d710f"
  },
  "sources": [
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/config.json",
      "sha256": "0cf75bbf67b015a3d039519a05613aa493786128c5763391e05271c7653d8123"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/model.py",
      "sha256": "d9759cde4e1f1657bb67b396360ec6e24c4189791b53eb59c25c29617e20ac08"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/policy.py",
      "sha256": "e8de9d33488d6ada0a2dd29d4f29e7fbf65f43340adfc5735ef3a8836795cac8"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/qualification.json",
      "sha256": "ae529ebed59434a4910e039a7225bd1fc413e5f534d44029b5a695789266545c"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/train.py",
      "sha256": "0895b4c162a923fc98409518259c7d2d197cd82de6b225a1541e0e359383e4a2"
    }
  ],
  "source_witness": {
    "path": "runs/water-conditioning-v1/scalar-units/project/source-witness.json",
    "sha256": "0a1e778e066731ac6617abddf85a0f9a68fdbf95fbd0848bdd815ae01ff1d11e"
  },
  "training_population": {
    "path": "runs/water-conditioning-v1/scalar-units/managed-full/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset",
      "bytes": 2275660
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra",
      "bytes": 2278650
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset",
      "bytes": 2561142
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra",
      "bytes": 2564086
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits",
      "bytes": 339677
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits",
      "bytes": 374928
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset",
      "bytes": 2209622
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra",
      "bytes": 2212566
    },
    {
      "path": "runs/water-conditioning-v1/scalar-units/project/inputs/validation_splits.json",
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
    "item_id": "fit-704407ce147484ba",
    "kind": "training_attempt",
    "name": "water-conditioning-scalar-units · CUDA training",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-conditioned-cubic-groupnorm-v1",
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
    "receipt_sha256": "704407ce147484baa4fb89c57b4453aeedb4d61726d25bb2af4012f64975fef1",
    "receipt_paths": [
      "runs/water-conditioning-v1/scalar-units/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-conditioning-v1/scalar-units/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/dbf23b411a674a058def8ad63ce7f3fe/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/dbf23b411a674a058def8ad63ce7f3fe/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/dbf23b411a674a058def8ad63ce7f3fe/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/dbf23b411a674a058def8ad63ce7f3fe/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/dbf23b411a674a058def8ad63ce7f3fe/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-dbf23b411a674a058def8ad63ce7f3fe/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-dbf23b411a674a058def8ad63ce7f3fe/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-dbf23b411a674a058def8ad63ce7f3fe/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-dbf23b411a674a058def8ad63ce7f3fe/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/dbf23b411a674a058def8ad63ce7f3fe/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/dbf23b411a674a058def8ad63ce7f3fe/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/dbf23b411a674a058def8ad63ce7f3fe/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/dbf23b411a674a058def8ad63ce7f3fe/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-dbf23b411a674a058def8ad63ce7f3fe/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-dbf23b411a674a058def8ad63ce7f3fe/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-dbf23b411a674a058def8ad63ce7f3fe/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-dbf23b411a674a058def8ad63ce7f3fe/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-dbf23b411a674a058def8ad63ce7f3fe/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "dbf23b411a674a058def8ad63ce7f3fe",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-conditioning-scalar-units · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "0cf75bbf67b015a3d039519a05613aa493786128c5763391e05271c7653d8123",
      "inputs": {
        "dataset": {
          "bytes": 2275660,
          "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"
        },
        "dataset_extra": {
          "bytes": 2278650,
          "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"
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
          "bytes": 374928,
          "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
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
      "normalization_sha256": "59b9e7be6968b788fa5ec3fa189745002d10cd7d747f4d4328f25fe0c359044b",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "b9ce164d30e4101b009fe4be765a070593cfbdd48f897853de159a8c177fabe8",
      "source_witness_sha256": "0a1e778e066731ac6617abddf85a0f9a68fdbf95fbd0848bdd815ae01ff1d11e"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c",
        "bytes": 1497594,
        "stage": "initial"
      },
      {
        "name": "checkpoint-100.npz",
        "sha256": "ae982b7d3bbd1909a6b02d4a35a9c2e1e587442c5d4c9b649d76f5599afc3e07",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "e401f27945ee38776327624d259bc4bda89a75355edfbb5a25436e3bb4e01019",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-10000.npz",
        "sha256": "c10d212b6f665fc2bd1ccb522623c98a91f388a859990af86d521d48bdc3b8cb",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "fc554000d6e648ea0e32dc193a3ab31cea348ef0c5d5d6cbab6ae3beafa49695",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "5d49d3d38941eba7a3992d4d5e281a0ba539da276000185246653ea76c9d390e",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "9c2cb3aa659d4ddf23bed0dc295a2f0f5641a3ba94709f63990e9cb7102a5777",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-6000.npz",
        "sha256": "ce4727607d4b2b49fd8064adf8b8416fd4e65fcd3352e3c6b0dedd38e71685a3",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-8000.npz",
        "sha256": "2540e57feac7f6af00fb0a80c3dcdb80d18583df25d698dfe3a6e88ab126eef0",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "c10d212b6f665fc2bd1ccb522623c98a91f388a859990af86d521d48bdc3b8cb",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2269650944,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/13"
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
      "receipt": "runs/water-conditioning-v1/scalar-units/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-conditioning-v1/scalar-units/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "704407ce147484baa4fb89c57b4453aeedb4d61726d25bb2af4012f64975fef1",
    "bytes": 9364
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C14": {
        "name": "scalar-units",
        "weights_sha256": "c10d212b6f665fc2bd1ccb522623c98a91f388a859990af86d521d48bdc3b8cb",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.4403503126904,
          "pressure_mean_free_relative_l2": 1.6662777694431767,
          "pressure_absolute_rms": 20.21852199273526,
          "pressure_absolute_max": 72.99258804728552
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0019",
            "native_frame": 15,
            "counts_xyz": [
              11,
              5,
              15
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0019",
            "native_frame": 15,
            "counts_xyz": [
              11,
              5,
              15
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.3031474466477495,
              "pressure_mean_free_relative_l2": 0.8469389991930386,
              "pressure_absolute_rms": 20.21852199273526,
              "pressure_absolute_max": 72.99258804728552
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.4403503126904,
              "pressure_mean_free_relative_l2": 1.6662777694431767,
              "pressure_absolute_rms": 8.090840858898204,
              "pressure_absolute_max": 21.882064102156704
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C14/results.json",
          "sha256": "f968dc73278494f59d6976aac6a6f8ce03663435866b9ab7f1475369ec6c4ab0"
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
{"architecture":{"activation":"silu","amplitude":"historical fixed training RMS","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"conditioning":"scalar-units","convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","scalar_feature_units":{"dt_s":0.016666666666666666,"rho_kg_m3":1000.0},"schema":"rheon-conditioned-cubic-groupnorm-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"0cf75bbf67b015a3d039519a05613aa493786128c5763391e05271c7653d8123","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"59b9e7be6968b788fa5ec3fa189745002d10cd7d747f4d4328f25fe0c359044b","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"b9ce164d30e4101b009fe4be765a070593cfbdd48f897853de159a8c177fabe8","source_witness_sha256":"0a1e778e066731ac6617abddf85a0f9a68fdbf95fbd0848bdd815ae01ff1d11e"},"elapsed_s":229.4004932300013,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.5547321518205592e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":8972800,"peak_reserved_bytes":10485760,"tf32":false},"mode":"train","optimizer_updates":10000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-100.npz":{"bytes":1497594,"role":"checkpoint","sha256":"ae982b7d3bbd1909a6b02d4a35a9c2e1e587442c5d4c9b649d76f5599afc3e07"},"checkpoint-1000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"e401f27945ee38776327624d259bc4bda89a75355edfbb5a25436e3bb4e01019"},"checkpoint-10000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"c10d212b6f665fc2bd1ccb522623c98a91f388a859990af86d521d48bdc3b8cb"},"checkpoint-2000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"fc554000d6e648ea0e32dc193a3ab31cea348ef0c5d5d6cbab6ae3beafa49695"},"checkpoint-4000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"5d49d3d38941eba7a3992d4d5e281a0ba539da276000185246653ea76c9d390e"},"checkpoint-500.npz":{"bytes":1497594,"role":"checkpoint","sha256":"9c2cb3aa659d4ddf23bed0dc295a2f0f5641a3ba94709f63990e9cb7102a5777"},"checkpoint-6000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"ce4727607d4b2b49fd8064adf8b8416fd4e65fcd3352e3c6b0dedd38e71685a3"},"checkpoint-8000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2540e57feac7f6af00fb0a80c3dcdb80d18583df25d698dfe3a6e88ab126eef0"},"history.json":{"bytes":73671,"role":"training_diagnostics","sha256":"c2c1915a816318dd8a18f2789959aad98ff9729abb90b6434ac93e128b9d710f"},"normalization.json":{"bytes":1179,"role":"preprocessing","sha256":"59b9e7be6968b788fa5ec3fa189745002d10cd7d747f4d4328f25fe0c359044b"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"e4fea372df786a5cf259c3222de34d9ab29955db226b42ae7d0387d5267ef3a9"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":834,"role":"result_summary","sha256":"ff0c9be73000747c7b4452755c1965d3307ed7a0b04a2d2a136fd144893c0e11"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"c10d212b6f665fc2bd1ccb522623c98a91f388a859990af86d521d48bdc3b8cb"}},"per_case_inference_s":[0.016339680994860828,0.011157892004121095,0.003104562987573445,0.0044724080071318895,0.0030333110189531,0.0019467920064926147,0.003947053017327562,0.00295150198508054,0.00331290002213791,0.0033944540191441774,0.003420620021643117,0.007131927995942533,0.002734056004555896],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"8e50d7f1fd4989bfb56de46474063ed998d5f58066d38c6741833f5ddeb4c947","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"245ee9a4a77f5015005c7f77613e6ea8f9be9d97c4aea02e36f64edefeedfe3f","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"c52bbeb732819425891c0a70e492be943bca0615698afdab8f2b75c2143c8e25","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"f2472318fd9e9bb2b7f5d7cfbc8f035e8ade0a09919ffed4b26e6985e71b867c","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"f6f467a45a880c2b994972ad9cf18e7cd878a6c1983b2460f2a17f6fc90e53a8","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"bb040e0c7dcbda73233fedcfd6de30c408b82fbe99e883c0f6ba685a9d1d2870","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"55019bd5b64990b7f1b1cc27425919d13fcfce3f8ac76ee9014ef0a3b51836b9","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"451d837508508eb4f8b5edef0f0cc0ff352a998fb56cd8c11b01749f096de8c4","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"fa341219db3b59291f6672a8f0769ff5440e1b9ac0acecc66e80cb8a9874c739","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"8c8e0b0dc72d2a42d2ee63f15b4665e3d5ad84cd3062e0560be08909774a7603","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"4a29854954e1b2ab8f3329c73c860b76419377a16196163df3cc9b54b9925f6d","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"a564362e8b481a2a0ac1a3a3fdc96a5fcabf539b4d8ce0c0800cdb6dbe65e948","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"ca3f2bfc1153378ae13e82affec999ff934db6d38c9776714c5023610a7e7446","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2269650944,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
