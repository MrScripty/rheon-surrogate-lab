# Full AI temporal b2-endpoint-mse-fullstate train

Retrospective source review · fit-c95673ba8f0efddf

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 2; each output advances 1/60 second. Full block duration is 2/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.258281 | 2.36050 | 0.15940 |
| 8 | 16x16x16 | 0.229769 | 2.71291 | 0.19605 |
| 8 | 32x32x32 | 0.314903 | 2.56419 | 0.47160 |
| 8 | 9x11x11 | 0.149797 | 2.28224 | 0.16498 |
| 16 | 15x6x8 | 0.148569 | 2.32149 | 0.16040 |
| 16 | 16x16x16 | 0.190956 | 2.52786 | 0.19602 |
| 16 | 32x32x32 | 0.12778 | 2.84191 | 0.47723 |
| 16 | 9x11x11 | 0.169948 | 2.34759 | 0.16505 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 480d903fd12f64c7130577e176878296432640a98f7d5b9471f0d8929f641841; subject: b2-endpoint-mse-fullstate.

**Method.** A163,068-parameter F32width12K2endpoint-conditioned generator with one shared spatial/global encoder, fresh predicted endpoint and joint intermediate decoder. It retains the audited fixed train-only force/increment scaling, excludes previous pressure, and predicts all five fields without a numerical pressure solver. Its model implementation, initial tensors and deployed feedback match the earlier scaled endpoint; only the fitted objective changes. Velocity/tracer use the existing full-state relative squared errors.

**Purpose.** Use a matched MSE pair to isolate whether supervising velocity/tracer in typical-step units improves predictive increments when full-state field magnitude can hide small changes. A separate full-state MSE arm controls the removal of the earlier logarithmic robust objective.

**Experiment.** Both original fits use the same16train-only scenes/2064states,8/16cells per metre, four shapes including32cubed/rectangles, fixed1/60s outputs, and2/60s per block. Original source/typed recipe/input/output closures, all initial/final tensors, exact seeded initialization and PCG64sample stream verify. Both train10,000CUDAupdates with identical learning-rate schedule and2000one-block then8000two-own-history-block curriculum,18,000encoder calls. Both replace prior log1p-scaled pressure/equation/field/physical terms with MSE terms; between these two arms only velocity/tracer denominators differ. Pressure/equation and divergence/mass/energy/bounds coefficients remain common to the pair.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 0.314903, 0.992451 and 0.0173157. Median time is 4.83014 ms per K2 block (2.41507 ms per physical output); the worst case median is 3.00597 ms/output. The retained F64 N1 composition has better worst pressure error and 13.47 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** Sharing an encoder, endpoint or retained features does not by itself establish a useful accuracy/speed tradeoff. This fit remains worse than the matched F64 composition in both worst pressure error and median GPU time. Account for every intermediate output and reconstruction cost; K outputs advance K full 1/60-second intervals.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-pair-01/train/receipt.json",
    "sha256": "c95673ba8f0efddf8a86243b17611347d23ae37f15bb6c1719f2a05748478013"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-pair-01/train/history.json",
    "sha256": "2dbc18cf5deae3cf79c281bad8cd79d5650b35d0d334c425a11321483589652a"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/bundle_model_sharing.py",
      "sha256": "3490bcffb6bfc9c19bb61613c0bd64065968adb27fa32a3e475f1e5e53850716"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/bundle_model_step_scaled.py",
      "sha256": "3d60ebb20e4b1cba76e2710358f23a026ee446fa5d378789eb27ce377096300c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/learned_global.py",
      "sha256": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/temporal_objective_increment.py",
      "sha256": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/temporal_objective_mse.py",
      "sha256": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/temporal_train_increment_supervision.py",
      "sha256": "e92d81c83fa6979568db6f895ca93c9ff4257290e5888a4912e0480c4764e931"
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
  "lesson": "Comparing the two MSE arms isolates the choice of field-error scale. Comparing either arm directly with the earlier step-scaled fit changes the robust objective as well, so an observed gain cannot be assigned solely to increment supervision. Fixed increment units do not supply variable timestep conditioning or longer temporal training, nor guarantee accurate intermediate states.",
  "outcome": "Original training completed10,000updates; final recorded composite MSE objective 0.011267013847827911. The two arms preserve exact untrained parameters and source code but fit different final tensors under their selected denominators. Fresh common full-field GPU scores, block latency and own-history drift remain pending. Loss values across these modes or the earlier logarithmic objective are not directly comparable performance metrics.",
  "reproduction": "Use the exact captured config/typed recipe/runtime/model and MSE objective source, the original16scene F32archive/normalization, fixed step-scales.json and selected full_state/increments mode. Retain common private seeds/order, schedule and deployed own-history reconstruction. Original training publication/restore uses CPU neural probes; qualify a GPU-only recipe before retraining. New-Rheon regeneration, clean retraining and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-c95673ba8f0efddf",
    "kind": "training_attempt",
    "name": "Full AI temporal b2-endpoint-mse-fullstate train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "c95673ba8f0efddf8a86243b17611347d23ae37f15bb6c1719f2a05748478013",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "310f58e301314016bdca88af4efd7ae5",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b2-endpoint-mse-fullstate train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 163068,
    "training_closure": {
      "config_sha256": "bc76a56f5f36b2004ab2638855567150e3c95791a2cf38ddcba12a9e1bbe0e24",
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
        "bundle_model_sharing.py": "3490bcffb6bfc9c19bb61613c0bd64065968adb27fa32a3e475f1e5e53850716",
        "bundle_model_step_scaled.py": "3d60ebb20e4b1cba76e2710358f23a026ee446fa5d378789eb27ce377096300c",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "learned_global.py": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b",
        "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
        "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
        "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
        "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
        "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
        "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
        "temporal_objective_increment.py": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e",
        "temporal_objective_mse.py": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136",
        "temporal_train_increment_supervision.py": "e92d81c83fa6979568db6f895ca93c9ff4257290e5888a4912e0480c4764e931"
      },
      "horizons": 2,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "cf925967fcdb939362096342f1770b196696582b62bc2885490a2558966908b9",
        "bytes": 659506,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "2ebab7ca1c29fc8e12a41b0c488c19172f0604d40e5d5302bf77fee2a5180b97",
        "bytes": 659506,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/14"
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
      "receipt": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-pair-01/train/receipt.json",
    "sha256": "c95673ba8f0efddf8a86243b17611347d23ae37f15bb6c1719f2a05748478013",
    "bytes": 15296
  },
  "bound_training_configuration": {
    "config_sha256": "bc76a56f5f36b2004ab2638855567150e3c95791a2cf38ddcba12a9e1bbe0e24",
    "graph_id": "3da0547a89b44027be4f3a5b8e32bac5",
    "graph_document_sha256": "f85b06cb96a644a448bb8d3fbf2833141e7194903222fdadb2045c4f347aceaf",
    "recipe_path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/train-recipe.json",
    "recipe_sha256": "0830170ac036c0a7fae420aa68d9100c4b420d0474f680e2b8ced0d5a58a7fa0",
    "config_path": "runs/full-ai-next-priorities-v1/b2-endpoint-mse-fullstate-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-increment-supervision-v1",
      "supervision_mode": "full_state",
      "architecture": "endpoint",
      "feedback_protocol": 2,
      "identifier": "b2-endpoint-mse-fullstate",
      "horizons": 2,
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
      "initialization": "Seeded width12 shared global encoder; zero fresh-pressure output heads; previous pressure excluded",
      "inference": "One shared encoder; predicted-endpoint intermediate decoder or retained-feature latent updates; no numerical solve",
      "sample_step_range": [
        0,
        120
      ],
      "loss": "Matched MSE pressure/equation/physical terms; relative full-state velocity/tracer errors",
      "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
      "sharing_registration_sha256": "e9c99cba7100d06d05138e1a18dbf8707f7bdfae61806ed11cb5498d7fc45451",
      "step_scale_statistics_sha256": "5f976a2e19e77cf7b46bf701b9e7c7cd65856d8f391db138020c8855592dad04",
      "original_normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037",
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
  "parameters": 163068,
  "horizons": 2,
  "closure": {
    "config_sha256": "bc76a56f5f36b2004ab2638855567150e3c95791a2cf38ddcba12a9e1bbe0e24",
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
      "bundle_model_sharing.py": "3490bcffb6bfc9c19bb61613c0bd64065968adb27fa32a3e475f1e5e53850716",
      "bundle_model_step_scaled.py": "3d60ebb20e4b1cba76e2710358f23a026ee446fa5d378789eb27ce377096300c",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "learned_global.py": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b",
      "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
      "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
      "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
      "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
      "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
      "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
      "temporal_objective_increment.py": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e",
      "temporal_objective_mse.py": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136",
      "temporal_train_increment_supervision.py": "e92d81c83fa6979568db6f895ca93c9ff4257290e5888a4912e0480c4764e931"
    },
    "horizons": 2,
    "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
  },
  "initial_inventory": {
    "fine.0.weight": {
      "shape": [
        12,
        14,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "ca52d873be1ebfa997ad905d4289c330601b3dd5f77b1c486998719e3c4587a7"
    },
    "fine.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "57ee2c2c5a2730fe834d9601368a12ff099f0a9a860a06d207488286007191ea"
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
      "sha256": "039f3665565bad65dcfe798c7f8295070b18f739fbaf6950aa5911ed277662fc"
    },
    "fine.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "11dd48eb07c8e265cd8215a58c1911a55c98de4df251cea7d6420887a1d2e2c8"
    },
    "forcing.weight": {
      "shape": [
        12,
        6,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "b82fed101efdcf97448e0e407a36e3b476932f42fdb3648b4a39a04d89aa4643"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "d6df8c40c5b38a6829e3e91226893e4df44e6b66a3f332e5a79eb9132ea3e53e"
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
      "sha256": "0ea8c63eb174ac26a3013301f955a62f54cabb1232c09cd17ec4a9c81a1e55cb"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "a1d3db045151e95bdcbf882be3adf7da7875c903c2f60be9167d21bcedc73aa0"
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
      "sha256": "961fd9bbfe5b76a65abf34d4c980ff2fadd91a8f88e45039d26250d036540494"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "27d54173ffdc1523f5846554091f85b8c5cd282c21c2f9c88014a6c0b44e1981"
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
      "sha256": "456c1314691ead7c71a8e8350633858efaa77adbae50f7c65b2bec18da7327f4"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "4757ba378385354c9fafb5df4a22cd9acd5f8b9d85ce47f70a9230b99c134f77"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "63e74b0077d46a179a5d48db065b52505bb2de798d3283ebda9ddd34f19e43e1"
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
      "sha256": "1ec457c9c051c6c543d396cb9cc9ced124fad19088b805dc30ea274dc387bbb2"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "6fc447b94d622974c4c476f4ce41f15e08a03d7a66b45e75f88a78e0136a2350"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "d936db8c4581395f49fab23c90abf41a1c81083d7c3fa4fafc1587e804847c96"
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
    "endpoint_head.weight": {
      "shape": [
        5,
        18,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "d3df611a0ed2e328b050d285287637c60643ba96ec09e4aaefaad7f2cd114b77"
    },
    "endpoint_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "de47c9b27eb8d300dbb5f2c353e632c393262cf06340c4fa7f1b40c4cbd36f90"
    },
    "intermediate.0.weight": {
      "shape": [
        12,
        27,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "d3a5924b561acea8f46ab464d25283063922f0a589bd28a35973c3f0639af339"
    },
    "intermediate.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "4694e9299e744458266a263f825f2cfd0470541a62d7c2df282f155068fbaee7"
    },
    "intermediate_head.weight": {
      "shape": [
        5,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "2dfba633817046c7f559ed4b93076048435f7e1a90f14eb8035c04b9ebae2537"
    },
    "intermediate_head.bias": {
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
        14,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "cc27865b85ef88f4e551905740df82b3c8b88ea81103ac8b4268df0c0460369e"
    },
    "fine.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "718a30c8bf48b84209b33a4cb0b06acebb6928c1660bbed79208ab72d5c13454"
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
      "sha256": "af5d253f2535dd468f01d9331a3af9082e40cf67697513f42f07171f5d5264dc"
    },
    "fine.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "aed497a9226966e96bf257d2239084fec9875cb7c51e8284885a73d51015a05b"
    },
    "forcing.weight": {
      "shape": [
        12,
        6,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "a0d9f4a5e1c879e059aaa23cae459c007edee3ff82e18ea1a4356cef8fea6e85"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "ca97d5b1146ae8e61c63691ccb61eddc7f025aff71f839d91d82f46be540a2f5"
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
      "sha256": "f8fe1d22906b8ef8ad6791ec77ac987b76fcf281f17aa8218f941ada6addde73"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "cdf69f0557c4e930dbd4fd4d780b30683eb97317768a1c607d46697266cf9bf3"
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
      "sha256": "41c9f8ef2ce1834a19230acc2a3b8bfff359148b8301a5ccc3ed49d636ca5231"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "4c183529ef818827673d70e31a11d9244416d7b1050404242c285a0133108d99"
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
      "sha256": "3f405e51599ccb6ea552a91763efbf27f77eb728cdae34cfb6ff608da4beedc7"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "966e81016b1a0d07c4e3916747006fc8b57471d0300abcf2ae793bd67dd0759b"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "7a64fc01b86a6d0c045935bf50101e64120f98d8b463d89953cb2109feae7bdc"
    },
    "global_blocks.0.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "b58dd0bb65b68136715168f629dea82d81aae84d2ec67e0e08f57569cb94ed3f"
    },
    "global_blocks.0.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "b8756216ccbdcd6a8f0391de20c9589579f8178eefffd4530789d9cc41dfa813"
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
      "sha256": "31f2b6a2c163945b4363ae4aa19b9b9bae4bc79f786693fe85ae78775faa69c3"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "a56dc817e587c44c6b4f55365da137cfd69ec0f9aaabc6af692b83e9069beff2"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "6964a3fa88079dc7e8447d408ba7cc9b6400fa80b637a3424edf10af4edb7e94"
    },
    "global_blocks.1.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "d459fb2625e4feb62c31c2e7d2347170ab97d220eb5196d4e0cc4cec45ceb849"
    },
    "global_blocks.1.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "ea1c84ce862224d903a16202b979b3f54af9a0226d93078c78593e769c48554e"
    },
    "endpoint_head.weight": {
      "shape": [
        5,
        18,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "9bc37ecbccfbdd616d001491c23b6564bbd9c10d2130cb7a2acff80e7efe50a9"
    },
    "endpoint_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "40dd54f402dafd9582aae51254f0b48528de39fa78ed8d88e6fba8ab63f29fea"
    },
    "intermediate.0.weight": {
      "shape": [
        12,
        27,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "d733a8d43921adf25f17bdd4da36009401c3c2e1fecc1264f268d695ff3b676f"
    },
    "intermediate.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "f7355c5632dcd2ce94fc795c467c880fede29626739fc266fcfa6f9d0cea12ba"
    },
    "intermediate_head.weight": {
      "shape": [
        5,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "7823d94b6c08e502a41e69799548cf7f523723e857edf6138934f7095e4bf76e"
    },
    "intermediate_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "16b1c22b4b7e7506b370b9ab06405f1a672d35326d6630a629051ed02a723aba"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 874935808
  },
  "outputs": {
    "probe.npy": {
      "bytes": 196648,
      "sha256": "a06ff5ec26d0ae3a43b6810a0eb6ee4d72b858a5afbb405b6273a98c1aba49d0"
    },
    "weights.npz": {
      "bytes": 659506,
      "sha256": "cf925967fcdb939362096342f1770b196696582b62bc2885490a2558966908b9"
    },
    "initial-weights.npz": {
      "bytes": 659506,
      "sha256": "2ebab7ca1c29fc8e12a41b0c488c19172f0604d40e5d5302bf77fee2a5180b97"
    },
    "history.json": {
      "bytes": 275540,
      "sha256": "2dbc18cf5deae3cf79c281bad8cd79d5650b35d0d334c425a11321483589652a"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 301.53182676801225,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-increment-supervision-v1",
  "supervision_mode": "full_state",
  "architecture": "endpoint",
  "feedback_protocol": 2,
  "identifier": "b2-endpoint-mse-fullstate",
  "horizons": 2,
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
  "initialization": "Seeded width12 shared global encoder; zero fresh-pressure output heads; previous pressure excluded",
  "inference": "One shared encoder; predicted-endpoint intermediate decoder or retained-feature latent updates; no numerical solve",
  "sample_step_range": [
    0,
    120
  ],
  "loss": "Matched MSE pressure/equation/physical terms; relative full-state velocity/tracer errors",
  "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
  "sharing_registration_sha256": "e9c99cba7100d06d05138e1a18dbf8707f7bdfae61806ed11cb5498d7fc45451",
  "step_scale_statistics_sha256": "5f976a2e19e77cf7b46bf701b9e7c7cd65856d8f391db138020c8855592dad04",
  "original_normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037",
  "full_fit_registration_sha256": "08ed048038db26f1838a6e74f82ae521c848f06b954a42aaf3000a17794d0510"
}

```
