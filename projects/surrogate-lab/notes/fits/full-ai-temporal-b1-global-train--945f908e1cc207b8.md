# Full AI temporal b1-global train

Retrospective source review · fit-945f908e1cc207b8

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 1; each output advances 1/60 second. Full block duration is 1/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.999198 | 4.41180 | 0.25071 |
| 8 | 16x16x16 | 0.998991 | 4.51327 | 0.27377 |
| 8 | 32x32x32 | 0.999874 | 4.68326 | 0.58808 |
| 8 | 9x11x11 | 1.00055 | 4.58349 | 0.25054 |
| 16 | 15x6x8 | 0.995374 | 4.43668 | 0.24466 |
| 16 | 16x16x16 | 1.0003 | 4.62383 | 0.26408 |
| 16 | 32x32x32 | 0.998913 | 4.65065 | 0.58521 |
| 16 | 9x11x11 | 1.00298 | 4.37777 | 0.25399 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 72069eca9de4b6389615baa5c863b99f128efc57cea50b83b056de51a702ba48; subject: b1-global.

**Method.** A 170,728-parameter F32full-state generator producing K=1complete pressure/velocity/tracer states per pass. A width12 encoder, two learned global cosine-channel mixing blocks, a3×3decoder and one joint state head. This changes capacity and spatial architecture together; there is no analytical pressure inverse. Pressure is previous normalized pressure plus a predicted correction, gauged at the first cell; velocities/tracer are increments from the original block state. Unpacking upcasts pressure toF64but internal prediction isF32. There is no pressure solver or sequential full physical integrator inside the network.

**Purpose.** Test greater shared/global capacity at K1 under the existing logarithmic objective.

**Experiment.** Original16train-only scenes/2064states at8/16cells per metre and fixed1/60s, with four shapes including32cubed and rectangular domains. All original data/runtime slots match the previously independently decoded corpus. Fit uses10000CUDA Adam updates, batch1, private model/sample seeds and the recorded schedule. First2000updates supervise one block, later updates supervise two, for18000network calls/18000physical training predictions. The second call uses physical reconstruction and deployed mean-centered re-encoding of its own endpoint. All closure/typed recipe/output bytes, full initial/final tensors and PCG64sample/work counts verify.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 1.00298, 0.689072 and 0.00345536. Median time is 4.49046 ms per K1 block (4.49046 ms per physical output); the worst case median is 5.00501 ms/output. The retained F64 N1 composition has better worst pressure error and 17.38 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** This retained design has higher worst pressure error and median GPU time than the matched F64 composition. Its method and fitting purpose remain recorded above; these measurements compare distinct designs and do not show that one checkpoint deteriorated. Judge complete physical outputs rather than a pressure-only component or a training-loss value.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-global-pair-01/train/receipt.json",
    "sha256": "945f908e1cc207b8e2531b4d168a2792376af0fc98659d4c755079f016de1c08"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b1-global-pair-01/train/history.json",
    "sha256": "76024f2d00cc4045a918199f665056eea70fd5356c112dc4724f22722ae74af5"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b1-global-project-01/bundle_model_global.py",
      "sha256": "44528675b4a3ee6ce126959d23cdd2792136923ec098a6034bc3050934def5f0"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-global-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-global-project-01/learned_global.py",
      "sha256": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-global-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-global-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-global-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-global-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-global-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-global-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-global-project-01/temporal_train_global.py",
      "sha256": "cf2b191950cb1a5c557673879fc8cdaa4985539e37ae32e1fd9cd725a6beece4"
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
  "lesson": "Capacity and global representation change together, so comparisons cannot assign a gain solely to global mixing. Zero output heads preserve the initial persistence function but not the smaller model’s hidden weights or training cost.",
  "outcome": "Original fit completes its declared10000updates; final logged composite objective 13.852665901184082. Parameter count is independently rederived. The larger global model has a different parameter inventory. No new common GPU accuracy, full-block latency or own-history drift is measured by this source review. Consumed-training diagnostics are not an independent holdout.",
  "reproduction": "Preserve exact captured model, objective, trainer, config, typed recipe/runtime, original16scene F32archive and normalization, private seeds/order and the historical feedback branch. Reproducing an old fit must not silently repair its raw-tensor training feedback. The original publication/restore recipes execute CPU neural probes; qualify a GPU-only reproduction before rerunning. New-Rheon regeneration, clean retraining and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-945f908e1cc207b8",
    "kind": "training_attempt",
    "name": "Full AI temporal b1-global train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "945f908e1cc207b8e2531b4d168a2792376af0fc98659d4c755079f016de1c08",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b1-global-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "a84df114e108496bab755ce30602b8df",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b1-global train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 170728,
    "training_closure": {
      "config_sha256": "2a20aed938ae885588143d49211167b2fafb6d6483804edbef275ba565276579",
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
        "bundle_model_global.py": "44528675b4a3ee6ce126959d23cdd2792136923ec098a6034bc3050934def5f0",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "learned_global.py": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b",
        "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
        "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
        "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
        "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
        "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
        "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
        "temporal_train_global.py": "cf2b191950cb1a5c557673879fc8cdaa4985539e37ae32e1fd9cd725a6beece4"
      },
      "horizons": 1,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "6dd1119b452f97c44d6d570ba30ea4f8ff33bf073c875355bfcc99c0727e947d",
        "bytes": 689530,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "52cb1b00f74ab5ae17b675a9dd8dfd47c0197cce10710ca423bdfe91f9ec91c9",
        "bytes": 689530,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/5"
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
      "receipt": "runs/full-ai-next-priorities-v1/b1-global-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-global-pair-01/train/receipt.json",
    "sha256": "945f908e1cc207b8e2531b4d168a2792376af0fc98659d4c755079f016de1c08",
    "bytes": 14088
  },
  "bound_training_configuration": {
    "config_sha256": "2a20aed938ae885588143d49211167b2fafb6d6483804edbef275ba565276579",
    "graph_id": "ab60c14cae714125b59117d59857edd7",
    "graph_document_sha256": "edf386e6e37c0bc1744bfa4409e510bb076c41ff46ea63ff2e8102447d9b88a3",
    "recipe_path": "runs/full-ai-next-priorities-v1/b1-global-project-01/train-recipe.json",
    "recipe_sha256": "41a0d07f591115adf917ce1ccf390b683bf213825258920982a5d950f382b36e",
    "config_path": "runs/full-ai-next-priorities-v1/b1-global-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-global-v1",
      "feedback_protocol": 2,
      "identifier": "b1-global",
      "horizons": 1,
      "width": 12,
      "updates": 10000,
      "single_block_updates": 2000,
      "learning_rate": 0.001,
      "schedule": [
        {
          "update": 6001,
          "learning_rate": 0.0003
        },
        {
          "update": 9001,
          "learning_rate": 0.0001
        }
      ],
      "seed": 2026101921,
      "sampling_seed": 2026101922,
      "wall_seconds": 1800,
      "physical_context": {
        "dt_s": 0.016666666666666666,
        "rho_kg_m3": 1000.0,
        "mu_pa_s": 0.001,
        "boundaries": "closed regular grid"
      },
      "initialization": "Seeded width12 shared encoder, two learned global mixing blocks, zero joint output heads; same persistence function as original control",
      "inference": "One shared encoder; joint state outputs; learned pressure modes; no numerical pressure solve",
      "sample_step_range": [
        0,
        120
      ],
      "loss": "Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds",
      "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
      "capacity_followup_registration_sha256": "d4d417f9c74dd39c4818db8fcdd8df8f3bf4bcc8f97152f2668d809830581c91",
      "full_fit_registration_sha256": "08ed048038db26f1838a6e74f82ae521c848f06b954a42aaf3000a17794d0510"
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
  "optimizer_updates": 10000,
  "parameters": 170728,
  "horizons": 1,
  "closure": {
    "config_sha256": "2a20aed938ae885588143d49211167b2fafb6d6483804edbef275ba565276579",
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
      "bundle_model_global.py": "44528675b4a3ee6ce126959d23cdd2792136923ec098a6034bc3050934def5f0",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "learned_global.py": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b",
      "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
      "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
      "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
      "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
      "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
      "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
      "temporal_train_global.py": "cf2b191950cb1a5c557673879fc8cdaa4985539e37ae32e1fd9cd725a6beece4"
    },
    "horizons": 1,
    "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
  },
  "initial_inventory": {
    "fine.0.weight": {
      "shape": [
        12,
        15,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "0e62c3974ae5376c71d94a305055b10d65e7f2621e4c3755c6da2f51677fe95a"
    },
    "fine.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "5d88c05153cb61e894c620f3fba1f0a7baba29f27a57980fd8ea87dec577926d"
    },
    "fine.2.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "614239865a1252157c547d81d195f12d07a1c207ab84fb6008ffd5773a0d88ff"
    },
    "fine.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "5e4b60390b2c2b656424297dbeb343474c47dd5134ed924c22a70b1bb575b2d3"
    },
    "forcing.weight": {
      "shape": [
        12,
        3,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "4ffd11f5124eec0be8db5a874f9a36f902b9d456c521926ce8f29fde5896a681"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "2d62c4a551750d013c7948d2bdd349f042d362bb80c55616b600e561c646b3a3"
    },
    "local.0.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "0f308e435fd00d6ef7bab6908395ab4f134720414832f7f5fa6d74cd9303222e"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "d6c5ce77f3ec975198c051bc628b29cd3b79772d4689f1cc7421f73ff9339d52"
    },
    "local.1.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "605512ca1598040371b8199f4682b91c907cd9e3e76c30c1092d4791568a10bd"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "0bc2cf72fafc9638d7e070dbbd93adeaf7109c38d7894757aa6c3046fea643f0"
    },
    "global_blocks.0.mixing": {
      "shape": [
        8,
        8,
        8,
        12,
        12
      ],
      "dtype": "<f4",
      "sha256": "2d9873038a20edbfde0e6c47021fdc24f11a2fdaedb09b9b9f81bca10d88e04a"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "0c66fb4111760ed8f11858a1767622c9a203e83fb3f0cfa6e266170b7c7f0be1"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "ed701fef7acb90523a9d6e23a8ef2c9065a4de30b5568196aa41edb815822acc"
    },
    "global_blocks.0.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "f7b586904e3678145aa47e4232587c913139cef0102d6d8e9276fc80c35cbad3"
    },
    "global_blocks.0.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "1a0295f4bf5986c5f74eca9153a6a4cb10b073a01a76ba4a457fd862c78966a4"
    },
    "global_blocks.1.mixing": {
      "shape": [
        8,
        8,
        8,
        12,
        12
      ],
      "dtype": "<f4",
      "sha256": "065f64961d90ebcd7fe27aec8054c44a85d3d2fefc88f6db238d26821ee5fe04"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "afffa14dd0e6b4dc14527262f6a4f7176bf872b8bcaf9ca2d4c93ebd0968d0ab"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "c87b1ba28b12bb88e08d34648fe91e9532052a96128c9e78dc3f2f4fdc20bbcd"
    },
    "global_blocks.1.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "f7b586904e3678145aa47e4232587c913139cef0102d6d8e9276fc80c35cbad3"
    },
    "global_blocks.1.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "1a0295f4bf5986c5f74eca9153a6a4cb10b073a01a76ba4a457fd862c78966a4"
    },
    "decode.0.weight": {
      "shape": [
        12,
        24,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "eef98092a5e5c5e3e60b7f9d47461c1d769210dc0657cad18fbcba1def1b2f5c"
    },
    "decode.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "6e3f4f19c5bca1a78153c829e481540f1debf3c21ab0f34f807fb8662efd9c72"
    },
    "head.weight": {
      "shape": [
        5,
        15,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "d13d4a8b3b8add19b5970157f09d00c12cbda4fed4d74d8493156523f7069b66"
    },
    "head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "de47c9b27eb8d300dbb5f2c353e632c393262cf06340c4fa7f1b40c4cbd36f90"
    }
  },
  "final_inventory": {
    "fine.0.weight": {
      "shape": [
        12,
        15,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "3d45fc1a0c376d4fec5fc9de63a72513c7c8205d90196617e7a518088414809e"
    },
    "fine.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "0b094e81156bea96d332b97cacf3d6c498b6c926930e05ee780d1d0c929ebd5b"
    },
    "fine.2.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "945be4416baf7df1d0c0ed1ffcb34906f67273720240c901b35f0ab0a8302669"
    },
    "fine.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "42c654f3d9962936aa63d46d9ac26040cdbca05aa1e1a5702203cec0e9b91f76"
    },
    "forcing.weight": {
      "shape": [
        12,
        3,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "040a1c7b996f313175206c8c28bd4a67ed253e3cd6bb89e343f145ef223b31cb"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "8f59880fdb8a7b9888948f1fdc882c5b3ae94101cea628aff0c3e6481a753f51"
    },
    "local.0.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "054f1343a838c9119364b9f6b8ad86d749d4612804a4666b1daee1172a4c75f9"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "f5f692b6de8779abf2f007edac913486e376820d73781c224708fc6b577d10bd"
    },
    "local.1.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "5873558ff7285341db9f86105fba2ffdc7e2d6d13c0657997a50e520a07d207f"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "78765268042ab107ba534c8fd87c756b9eca884b0160cabba7cb0c3511e0eefc"
    },
    "global_blocks.0.mixing": {
      "shape": [
        8,
        8,
        8,
        12,
        12
      ],
      "dtype": "<f4",
      "sha256": "3a5217a3c51fcbb65f0bf477a9c87a0cc5c74e5d117bcc0de46d6dce9d2370ab"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "ec1b1690c9113b10dcadd929957d59d58177d580394c5509ce7c8ff33c4e7505"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "a90406b8fe51af1c3f728734f2a379fb2335a31835bc159dbd284e1b375ca8ce"
    },
    "global_blocks.0.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "d7508ada84ff2c86c575b33052de5b831d9f2b3adcfba1c7f66fa02b4c0362ec"
    },
    "global_blocks.0.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "d02bd0ecbf8d0934b956191c5cb623ca4bb9f4d3c13e7335428df1e8cceb90d0"
    },
    "global_blocks.1.mixing": {
      "shape": [
        8,
        8,
        8,
        12,
        12
      ],
      "dtype": "<f4",
      "sha256": "0ae6bf63028ad63c23a8e990741f7070164e5757e21c36302481f632d2235a4f"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "eef09acc4e5475475009c2e7e315fb49c44efb599747cb79b9c639d99ee5a8ee"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "a59038de04f761e4678967978ccbe41c049cb63c5e903f3322549921bcb113c2"
    },
    "global_blocks.1.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "8ad82068c847750ee0b1739bd41ca469fd73ad5ada0ba07decbb5b689421de31"
    },
    "global_blocks.1.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "c44b95b9d46f2c31cfd2bbb25c885feccca731ee1566595aab11532cf67c9143"
    },
    "decode.0.weight": {
      "shape": [
        12,
        24,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "8ddeead08c3ef808133a9890607c1e5c6d1d484cfe513490811d7bffdd6e2d45"
    },
    "decode.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "f42955d9838da3676db2dc99331e2310e6eeab97c5fdeea6286fc53352a06dbe"
    },
    "head.weight": {
      "shape": [
        5,
        15,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "b895fb0383c01ea5a5b248b3018aa635c87714fcf8b20e08023fa59734059438"
    },
    "head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "6615bec2842db8e68eb82f13d77d0a6896bcf0efdf4454b68ecf2358bba72065"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 862834688
  },
  "outputs": {
    "probe.npy": {
      "bytes": 98388,
      "sha256": "355398df128da04bb5eabea2302b021f95222e1faba4176f9b81a602d886cdd4"
    },
    "weights.npz": {
      "bytes": 689530,
      "sha256": "6dd1119b452f97c44d6d570ba30ea4f8ff33bf073c875355bfcc99c0727e947d"
    },
    "initial-weights.npz": {
      "bytes": 689530,
      "sha256": "52cb1b00f74ab5ae17b675a9dd8dfd47c0197cce10710ca423bdfe91f9ec91c9"
    },
    "history.json": {
      "bytes": 273812,
      "sha256": "76024f2d00cc4045a918199f665056eea70fd5356c112dc4724f22722ae74af5"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 312.67503412096994,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-global-v1",
  "feedback_protocol": 2,
  "identifier": "b1-global",
  "horizons": 1,
  "width": 12,
  "updates": 10000,
  "single_block_updates": 2000,
  "learning_rate": 0.001,
  "schedule": [
    {
      "update": 6001,
      "learning_rate": 0.0003
    },
    {
      "update": 9001,
      "learning_rate": 0.0001
    }
  ],
  "seed": 2026101921,
  "sampling_seed": 2026101922,
  "wall_seconds": 1800,
  "physical_context": {
    "dt_s": 0.016666666666666666,
    "rho_kg_m3": 1000.0,
    "mu_pa_s": 0.001,
    "boundaries": "closed regular grid"
  },
  "initialization": "Seeded width12 shared encoder, two learned global mixing blocks, zero joint output heads; same persistence function as original control",
  "inference": "One shared encoder; joint state outputs; learned pressure modes; no numerical pressure solve",
  "sample_step_range": [
    0,
    120
  ],
  "loss": "Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds",
  "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
  "capacity_followup_registration_sha256": "d4d417f9c74dd39c4818db8fcdd8df8f3bf4bcc8f97152f2668d809830581c91",
  "full_fit_registration_sha256": "08ed048038db26f1838a6e74f82ae521c848f06b954a42aaf3000a17794d0510"
}

```
