# Full AI temporal b4-cost-pilot train

Retrospective source review · fit-7f855afaded9ff1a

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 4; each output advances 1/60 second. Full block duration is 4/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 1.02337 | 1.05680 | 0.11509 |
| 8 | 16x16x16 | 1.00322 | 1.03576 | 0.14026 |
| 8 | 32x32x32 | 1.00222 | 1.21507 | 0.40021 |
| 8 | 9x11x11 | 1.00921 | 0.96242 | 0.11762 |
| 16 | 15x6x8 | 1.07161 | 1.10483 | 0.11392 |
| 16 | 16x16x16 | 1.01745 | 0.98957 | 0.13887 |
| 16 | 32x32x32 | 0.998862 | 1.22106 | 0.40363 |
| 16 | 9x11x11 | 1.06172 | 1.06008 | 0.11840 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 046d91f6248a32f47843817c6b8c4d87af71e309336e498b4ac3a50dbf0c2df6; subject: b4-cost-pilot.

**Method.** A 27,704-parameter F32full-state generator producing K=4complete pressure/velocity/tracer states per pass. The same width4 compact current-state encoder, pooled/coarse branch, cheap joint horizon heads and63non-DC learned pressure modes per output as the later protocol2 models. Future prescribed forces enter heads/pooling, and previous pressure is part of the current state. Pressure is previous normalized pressure plus a predicted correction, gauged at the first cell; velocities/tracer are increments from the original block state. Unpacking upcasts pressure toF64but internal prediction isF32. There is no pressure solver or sequential full physical integrator inside the network.

**Purpose.** Measure fitting/restoration cost and the original two-block mechanism with only16updates, rather than train a competitive K4model.

**Experiment.** Original16train-only scenes/2064states at8/16cells per metre and fixed1/60s, with four shapes including32cubed and rectangular domains. All original data/runtime slots match the previously independently decoded corpus. Fit uses16CUDA Adam updates, batch1, private model/sample seeds and the recorded schedule. First8updates supervise one block, later updates supervise two, for24network calls/96physical training predictions. The second call directly consumes predicted[:,-1]without deployed physical unpack/repack. All closure/typed recipe/output bytes, full initial/final tensors and PCG64sample/work counts verify.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 1.07161, 1.00471 and 0.0354775. Median time is 4.25476 ms per K4 block (1.06369 ms per physical output); the worst case median is 1.36027 ms/output. The retained F64 N1 composition has better worst pressure error and 8.25 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** This retained design has higher worst pressure error and median GPU time than the matched F64 composition. Its method and fitting purpose remain recorded above; these measurements compare distinct designs and do not show that one checkpoint deteriorated. Judge complete physical outputs rather than a pressure-only component or a training-loss value.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-pair-01/train/receipt.json",
    "sha256": "7f855afaded9ff1a4a75b91c6a8fb029a58ebd77e832ad11e33772d91a869f21"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-pair-01/train/history.json",
    "sha256": "b43e3883fa162508265948fe2b959cf3410b56936845d6a5203089f8b58b9f0b"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/adaptive_pool.py",
      "sha256": "759eb0e374518052bae51b6f96c89ea6465de5b0160ed5a94b05802d034cd96e"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/bundle_model.py",
      "sha256": "add1f6eda47133318eae626497bde98e36cad02e48fcf7d4a1b252149fca3a05"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/bundle_model_v5.py",
      "sha256": "e1191613b019e951d29121c66e4844bedf610531271961430b6dce29a7eb6c27"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/temporal_train.py",
      "sha256": "1ff25ff005ab5b681d48f19bdfd1582f11c1f554ef2fde667dd638c0c22bd509"
    }
  ],
  "training_inputs": [
    {
      "bytes": 37465,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/manifest.json",
      "sha256": "bdf54cecf37e5cd9422ee51a1a88ecb22ef9efcfb7a98f1a867ccafb8a95075b",
      "slot": "manifest"
    },
    {
      "bytes": 981,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/normalization.json",
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037",
      "slot": "normalization"
    },
    {
      "bytes": 7989288,
      "path": "runs/dependency-migration-v1/runtime/current-manifest-v2.json",
      "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "slot": "runtime_manifest"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/part-00.bin",
      "sha256": "5f3148974b446d197250057d260276fb3df78a33a3f392c9f64a318896fcdd44",
      "slot": "part_00"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/part-01.bin",
      "sha256": "8d7d017e7d0a9ef828455e6bed1690ae723184351a7daf63a740182acfd700fc",
      "slot": "part_01"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/part-02.bin",
      "sha256": "e03079814e38591f42ff74b52af9858a80a17c6af85ec7cd420c64d7950447cd",
      "slot": "part_02"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/part-03.bin",
      "sha256": "5c59c5301b1741c1c7299cac50a5a69b29af212e0030df11c83d1952f2379c1f",
      "slot": "part_03"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/part-04.bin",
      "sha256": "24134714c26931dfbb54d69cf0ce6571ca40e83922be48542210c516770ad4bb",
      "slot": "part_04"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/part-05.bin",
      "sha256": "8aa71a0ef99e94d7ee8638924f9714c1c87c166bef115cb44c4b67cd94ebd1e7",
      "slot": "part_05"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/part-06.bin",
      "sha256": "7297165d24117593602caa726a16ad0c4d3004b06bb7ba642018440d5d255cc5",
      "slot": "part_06"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/part-07.bin",
      "sha256": "51b0e01f901362a6d9220a5f05c96c76b8808ca0fcf3c9ce706b60d70a161265",
      "slot": "part_07"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/part-08.bin",
      "sha256": "151b18d70962cf6564730d36040ce1f98a97c97387ea70bf6a7c8e92a955fb53",
      "slot": "part_08"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/part-09.bin",
      "sha256": "7e539998779bc21cc5ad0eed7747688253412b861a4d179f701e379778fb4ff6",
      "slot": "part_09"
    },
    {
      "bytes": 1247280,
      "path": "runs/full-ai-next-priorities-v1/temporal-training-view-02/part-10.bin",
      "sha256": "8d92e916e9c29a577b9e40bab774904c76d4f337a303802e6ad79d7bf5e98b75",
      "slot": "part_10"
    }
  ]
}
```

## Previous source-review statements

These statements describe the earlier source-review stage. The matched results above update evaluation status while retaining the original interpretation.

```json
{
  "lesson": "Keep this16-update smoke/cost fit distinct from10,000-update scientific candidates. Its K4shape and successful receipt do not imply trained accuracy.",
  "outcome": "Original fit completes its declared16updates; final logged composite objective 22.655643463134766. Parameter count is independently rederived. Exact initial tensors match the corresponding later protocol2joint fit. No new common GPU accuracy, full-block latency or own-history drift is measured by this source review. Consumed-training diagnostics are not an independent holdout.",
  "reproduction": "Preserve exact captured model, objective, trainer, config, typed recipe/runtime, original16scene F32archive and normalization, private seeds/order and the historical feedback branch. Reproducing an old fit must not silently repair its raw-tensor training feedback. The original publication/restore recipes execute CPU neural probes; qualify a GPU-only reproduction before rerunning. New-Rheon regeneration, clean retraining and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-7f855afaded9ff1a",
    "kind": "training_attempt",
    "name": "Full AI temporal b4-cost-pilot train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "7f855afaded9ff1a4a75b91c6a8fb029a58ebd77e832ad11e33772d91a869f21",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b4-cost-pilot-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "69ccaf18a3fd4831b124b7dd7095cd06",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b4-cost-pilot train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 16,
    "parameter_fits": null,
    "declared_parameter_count": 27704,
    "training_closure": {
      "config_sha256": "ad633453dec7599deab717502222c425423250a3f2e23d46cb8f0623c1b31129",
      "inputs": {
        "manifest": {
          "bytes": 37465,
          "sha256": "bdf54cecf37e5cd9422ee51a1a88ecb22ef9efcfb7a98f1a867ccafb8a95075b"
        },
        "normalization": {
          "bytes": 981,
          "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        },
        "part_00": {
          "bytes": 25165824,
          "sha256": "5f3148974b446d197250057d260276fb3df78a33a3f392c9f64a318896fcdd44"
        },
        "part_01": {
          "bytes": 25165824,
          "sha256": "8d7d017e7d0a9ef828455e6bed1690ae723184351a7daf63a740182acfd700fc"
        },
        "part_02": {
          "bytes": 25165824,
          "sha256": "e03079814e38591f42ff74b52af9858a80a17c6af85ec7cd420c64d7950447cd"
        },
        "part_03": {
          "bytes": 25165824,
          "sha256": "5c59c5301b1741c1c7299cac50a5a69b29af212e0030df11c83d1952f2379c1f"
        },
        "part_04": {
          "bytes": 25165824,
          "sha256": "24134714c26931dfbb54d69cf0ce6571ca40e83922be48542210c516770ad4bb"
        },
        "part_05": {
          "bytes": 25165824,
          "sha256": "8aa71a0ef99e94d7ee8638924f9714c1c87c166bef115cb44c4b67cd94ebd1e7"
        },
        "part_06": {
          "bytes": 25165824,
          "sha256": "7297165d24117593602caa726a16ad0c4d3004b06bb7ba642018440d5d255cc5"
        },
        "part_07": {
          "bytes": 25165824,
          "sha256": "51b0e01f901362a6d9220a5f05c96c76b8808ca0fcf3c9ce706b60d70a161265"
        },
        "part_08": {
          "bytes": 25165824,
          "sha256": "151b18d70962cf6564730d36040ce1f98a97c97387ea70bf6a7c8e92a955fb53"
        },
        "part_09": {
          "bytes": 25165824,
          "sha256": "7e539998779bc21cc5ad0eed7747688253412b861a4d179f701e379778fb4ff6"
        },
        "part_10": {
          "bytes": 1247280,
          "sha256": "8d92e916e9c29a577b9e40bab774904c76d4f337a303802e6ad79d7bf5e98b75"
        }
      },
      "sources": {
        "adaptive_pool.py": "759eb0e374518052bae51b6f96c89ea6465de5b0160ed5a94b05802d034cd96e",
        "bundle_model.py": "add1f6eda47133318eae626497bde98e36cad02e48fcf7d4a1b252149fca3a05",
        "bundle_model_v5.py": "e1191613b019e951d29121c66e4844bedf610531271961430b6dce29a7eb6c27",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
        "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
        "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
        "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
        "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
        "temporal_train.py": "1ff25ff005ab5b681d48f19bdfd1582f11c1f554ef2fde667dd638c0c22bd509"
      },
      "horizons": 4,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "d81f786c48d49b8b4226603083378ea20faf3f052b9e177b75fe984403b13915",
        "bytes": 114438,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "bcbec1003ceb6b4c95c838c793f2a91e123a2fedb39510aebf1e31f070170c31",
        "bytes": 114438,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [],
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
      "receipt": "runs/full-ai-next-priorities-v1/b4-cost-pilot-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-pair-01/train/receipt.json",
    "sha256": "7f855afaded9ff1a4a75b91c6a8fb029a58ebd77e832ad11e33772d91a869f21",
    "bytes": 9789
  },
  "bound_training_configuration": {
    "config_sha256": "ad633453dec7599deab717502222c425423250a3f2e23d46cb8f0623c1b31129",
    "graph_id": "d1dc5ad3dff74fa28ebb82e94c1dfe21",
    "graph_document_sha256": "2698bef47e8385d72471144cfc8c41c05c9c525f638c507ed3618239a3d03c3d",
    "recipe_path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/train-recipe.json",
    "recipe_sha256": "352e7b42613d1f27bdcb086eacc76cd942a354da4fb82be94a2b3d09b58fe27e",
    "config_path": "runs/full-ai-next-priorities-v1/b4-cost-pilot-project-03/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-v1",
      "identifier": "b4-cost-pilot",
      "horizons": 4,
      "width": 4,
      "updates": 16,
      "single_block_updates": 8,
      "learning_rate": 0.001,
      "schedule": [],
      "seed": 2026101921,
      "sampling_seed": 2026101922,
      "wall_seconds": 1800,
      "physical_context": {
        "dt_s": 0.016666666666666666,
        "rho_kg_m3": 1000.0,
        "mu_pa_s": 0.001,
        "boundaries": "closed regular grid"
      },
      "initialization": "Seeded compact spatial encoder and zero future heads",
      "inference": "One shared encoder; joint state outputs; learned pressure modes; no numerical pressure solve",
      "sample_step_range": [
        0,
        120
      ],
      "loss": "Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds",
      "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574"
    }
  },
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {},
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
{
  "schema": "rheon-temporal-training-result-v1",
  "status": "complete",
  "mode": "train",
  "optimizer_updates": 16,
  "parameters": 27704,
  "horizons": 4,
  "closure": {
    "config_sha256": "ad633453dec7599deab717502222c425423250a3f2e23d46cb8f0623c1b31129",
    "inputs": {
      "manifest": {
        "bytes": 37465,
        "sha256": "bdf54cecf37e5cd9422ee51a1a88ecb22ef9efcfb7a98f1a867ccafb8a95075b"
      },
      "normalization": {
        "bytes": 981,
        "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      },
      "part_00": {
        "bytes": 25165824,
        "sha256": "5f3148974b446d197250057d260276fb3df78a33a3f392c9f64a318896fcdd44"
      },
      "part_01": {
        "bytes": 25165824,
        "sha256": "8d7d017e7d0a9ef828455e6bed1690ae723184351a7daf63a740182acfd700fc"
      },
      "part_02": {
        "bytes": 25165824,
        "sha256": "e03079814e38591f42ff74b52af9858a80a17c6af85ec7cd420c64d7950447cd"
      },
      "part_03": {
        "bytes": 25165824,
        "sha256": "5c59c5301b1741c1c7299cac50a5a69b29af212e0030df11c83d1952f2379c1f"
      },
      "part_04": {
        "bytes": 25165824,
        "sha256": "24134714c26931dfbb54d69cf0ce6571ca40e83922be48542210c516770ad4bb"
      },
      "part_05": {
        "bytes": 25165824,
        "sha256": "8aa71a0ef99e94d7ee8638924f9714c1c87c166bef115cb44c4b67cd94ebd1e7"
      },
      "part_06": {
        "bytes": 25165824,
        "sha256": "7297165d24117593602caa726a16ad0c4d3004b06bb7ba642018440d5d255cc5"
      },
      "part_07": {
        "bytes": 25165824,
        "sha256": "51b0e01f901362a6d9220a5f05c96c76b8808ca0fcf3c9ce706b60d70a161265"
      },
      "part_08": {
        "bytes": 25165824,
        "sha256": "151b18d70962cf6564730d36040ce1f98a97c97387ea70bf6a7c8e92a955fb53"
      },
      "part_09": {
        "bytes": 25165824,
        "sha256": "7e539998779bc21cc5ad0eed7747688253412b861a4d179f701e379778fb4ff6"
      },
      "part_10": {
        "bytes": 1247280,
        "sha256": "8d92e916e9c29a577b9e40bab774904c76d4f337a303802e6ad79d7bf5e98b75"
      }
    },
    "sources": {
      "adaptive_pool.py": "759eb0e374518052bae51b6f96c89ea6465de5b0160ed5a94b05802d034cd96e",
      "bundle_model.py": "add1f6eda47133318eae626497bde98e36cad02e48fcf7d4a1b252149fca3a05",
      "bundle_model_v5.py": "e1191613b019e951d29121c66e4844bedf610531271961430b6dce29a7eb6c27",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
      "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
      "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
      "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
      "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
      "temporal_train.py": "1ff25ff005ab5b681d48f19bdfd1582f11c1f554ef2fde667dd638c0c22bd509"
    },
    "horizons": 4,
    "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
  },
  "initial_inventory": {
    "fine.0.weight": {
      "shape": [
        4,
        15,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "276fab71a35e9d01434054ae06afe7ae89f27814f5090710a49707949100bc5d"
    },
    "fine.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "cd3e170d7e0040ad5c528fcdcf3722da5c9b8bc6290cf3f5460f68d385b42804"
    },
    "fine.2.weight": {
      "shape": [
        4,
        4,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "bd798e219c571f295e1b0eb221b5f16a52811dd6df951931cd2ae8fe58956f1f"
    },
    "fine.2.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "0539e1dbc57e9b90acde78c5b06b837c5b2ed18c893d4075902364b5aacc3bec"
    },
    "coarse.0.weight": {
      "shape": [
        8,
        4,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "60df8e59f0ac7cae33e0dd96aa719105abf67cb23efd99d7ed35a580d017ae2a"
    },
    "coarse.0.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "d1023094a700071fa42f428394cff4bf75371b95ce2e27a2cee52d14f36398bb"
    },
    "decode.0.weight": {
      "shape": [
        4,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "da722ce7fce97515e3e7f891b1e38104857a88e7abef3bea3579df7de8689ffb"
    },
    "decode.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "ac7fd358698bb965aa0a90ed615e248623bb4729c0b92f8956f1c4ad4cc810ad"
    },
    "head.weight": {
      "shape": [
        20,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "bfe492baf731a0dbf6e1e050f5bc3fe8c1b049383194dcdf82f023bfa409f462"
    },
    "head.bias": {
      "shape": [
        20
      ],
      "dtype": "<f4",
      "sha256": "5b6fb58e61fa475939767d68a446f97f1bff02c0e5935a3ea8bb51e6515783d8"
    },
    "modal.0.weight": {
      "shape": [
        32,
        540
      ],
      "dtype": "<f4",
      "sha256": "f08e861cd1074802a29775d747de11424c3bdabe3b26e96fd5d25705e38c58f5"
    },
    "modal.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "67374f7ae8ab8105893fd8174324ca7b1877f81ee6eab33ab2c7152cc7a5509a"
    },
    "modal.2.weight": {
      "shape": [
        252,
        32
      ],
      "dtype": "<f4",
      "sha256": "47a22accc6fb0aedaf4a5cf8014bdd569ead0efad399fdb8de134235e7c9bb10"
    },
    "modal.2.bias": {
      "shape": [
        252
      ],
      "dtype": "<f4",
      "sha256": "a9e080790cc3b9e7a988324b4ec1cc1432a8f43cf897c78318943a0670f171f6"
    }
  },
  "final_inventory": {
    "fine.0.weight": {
      "shape": [
        4,
        15,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "9a81d3db1fe6b82b5a41181798742d0928fbe4663bdb825c429d0a372ffc4493"
    },
    "fine.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "64c50336f71f178a1030a0588042dd9657c73861627fc18309d8adf250a2e408"
    },
    "fine.2.weight": {
      "shape": [
        4,
        4,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "b0d1a1e6502e29f948c287d874f7442f2923b78e20d471cac393a3619d007bd9"
    },
    "fine.2.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "06301775c264616fb7f00dcc92a8387cbfedff00546d8158c8984ae3df46521e"
    },
    "coarse.0.weight": {
      "shape": [
        8,
        4,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "0066c04b47dd5fcaa6cba954b1ee685956fa563e837bbe1961ac159081dea075"
    },
    "coarse.0.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "9223610a3cd7330da3566219da967709815979915ae226621ddfe8bab8e53b81"
    },
    "decode.0.weight": {
      "shape": [
        4,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "bd347691ad581c9252c290ab51b86077a8d1d5c5f671b7abc8e72dcce0eb4249"
    },
    "decode.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "9187ced96c839bf80b2e5f2ebec9fb907f60a7088a9b00e6ddb87b61908ab260"
    },
    "head.weight": {
      "shape": [
        20,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "fce661648fa856a027e2f283156fa7901b7414b8bed11174129a48574d565b91"
    },
    "head.bias": {
      "shape": [
        20
      ],
      "dtype": "<f4",
      "sha256": "9eb2f0afe96d98ef2643ad5b4d42cc94cf4b6a58a8ad0632b7ca35de3ab5b796"
    },
    "modal.0.weight": {
      "shape": [
        32,
        540
      ],
      "dtype": "<f4",
      "sha256": "1497c91e0b968e5980ad250c2d912c516dcb1662df7c308d201e755a7cf8dd0a"
    },
    "modal.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "5c61ef80510fb4552dcaf5b3f7ce0cdc9aaf041421cc578f1990b17b1620168f"
    },
    "modal.2.weight": {
      "shape": [
        252,
        32
      ],
      "dtype": "<f4",
      "sha256": "de4c290ffcc0aabe83f7d3a14bbd83278189b2ef8e1063bbe013b9021d92b1cb"
    },
    "modal.2.bias": {
      "shape": [
        252
      ],
      "dtype": "<f4",
      "sha256": "2a1490a66957bfb334808d179506e19e2e4bc9bf11dd6a037370f341776bb7b9"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 827038720
  },
  "outputs": {
    "probe.npy": {
      "bytes": 393168,
      "sha256": "699c299c604a5a2b532e3ca17cdb1115e3b5047dbcb6a9db15c09406f7f3d000"
    },
    "weights.npz": {
      "bytes": 114438,
      "sha256": "d81f786c48d49b8b4226603083378ea20faf3f052b9e177b75fe984403b13915"
    },
    "initial-weights.npz": {
      "bytes": 114438,
      "sha256": "bcbec1003ceb6b4c95c838c793f2a91e123a2fedb39510aebf1e31f070170c31"
    },
    "history.json": {
      "bytes": 6495,
      "sha256": "b43e3883fa162508265948fe2b959cf3410b56936845d6a5203089f8b58b9f0b"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 19.64176252100151,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-v1",
  "identifier": "b4-cost-pilot",
  "horizons": 4,
  "width": 4,
  "updates": 16,
  "single_block_updates": 8,
  "learning_rate": 0.001,
  "schedule": [],
  "seed": 2026101921,
  "sampling_seed": 2026101922,
  "wall_seconds": 1800,
  "physical_context": {
    "dt_s": 0.016666666666666666,
    "rho_kg_m3": 1000.0,
    "mu_pa_s": 0.001,
    "boundaries": "closed regular grid"
  },
  "initialization": "Seeded compact spatial encoder and zero future heads",
  "inference": "One shared encoder; joint state outputs; learned pressure modes; no numerical pressure solve",
  "sample_step_range": [
    0,
    120
  ],
  "loss": "Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds",
  "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574"
}

```
