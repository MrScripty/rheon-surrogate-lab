# Full AI temporal b2-endpoint-curl-increment train

Retrospective source review · fit-a47365bce83a4ce1

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 2; each output advances 1/60 second. Full block duration is 2/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.323275 | 2.56580 | 0.15940 |
| 8 | 16x16x16 | 0.2693 | 2.57230 | 0.19605 |
| 8 | 32x32x32 | 0.601521 | 2.76491 | 0.47160 |
| 8 | 9x11x11 | 0.184872 | 2.57111 | 0.16498 |
| 16 | 15x6x8 | 0.330767 | 2.88084 | 0.16040 |
| 16 | 16x16x16 | 0.54655 | 2.73129 | 0.19602 |
| 16 | 32x32x32 | 0.299665 | 2.75303 | 0.47723 |
| 16 | 9x11x11 | 0.341164 | 2.59380 | 0.16505 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 480d903fd12f64c7130577e176878296432640a98f7d5b9471f0d8929f641841; subject: b2-endpoint-curl-increment.

**Method.** A163,068-parameter F32width12K2endpoint-conditioned generator with the same learned tensors, shared global encoder and fixed train-only scales as the increment-MSE control. Its three raw velocity channels represent edge potentials. A fixed masked discrete curl in F64forms staggered velocity increments, scaled by the horizon RMS and minimum grid dimension, then added to current velocity and cast toF32. Pressure/tracer readouts remain direct learned predictions. No new trainable coefficient, pressure solver or post-prediction velocity projection is added.

**Purpose.** Represent learned velocity increments so their ideal discrete divergence cancels by construction and wall-normal increments vanish, while retaining the same network capacity and matched objective/data/initialization.

**Experiment.** Original16consumed scenes/2064states at8/16cells per metre, four shapes including32cubed/rectangles, fixed1/60s output intervals and2/60s per block.10,000CUDAupdates use the same private seeds/order, learning-rate schedule and2000one-block then8000two-own-history-block curriculum as the increment-MSE endpoint control. Full closure/typed recipe/input/output identities, exact initial tensors, complete final tensors and18,000encoder-call counts verify. Independent NumPy edge/face incidence checks on three rectangular grids confirm the ideal F64curl/divergence identity and zero normal-wall increments; they are not a new neural execution or Torch-output equivalence test.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 0.601521, 0.664356 and 0.0060343. Median time is 5.30700 ms per K2 block (2.65350 ms per physical output); the worst case median is 3.16712 ms/output. The retained F64 N1 composition has better worst pressure error and 14.80 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** Sharing an encoder, endpoint or retained features does not by itself establish a useful accuracy/speed tradeoff. This fit remains worse than the matched F64 composition in both worst pressure error and median GPU time. Account for every intermediate output and reconstruction cost; K outputs advance K full 1/60-second intervals.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-pair-01/train/receipt.json",
    "sha256": "a47365bce83a4ce13e3e438285a01040f4ec025a2b4583548707ec4397dc1ac8"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-pair-01/train/history.json",
    "sha256": "a29dbf96a8587d8e8ee1cd7f46b993c941442ea1a058e956d8c3f46cff3e25e3"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/bundle_model_curl.py",
      "sha256": "075ad797b7c78a4f9a793c1b71370a7a02c399768a069c06fc7f431d4481f90c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/bundle_model_sharing.py",
      "sha256": "3490bcffb6bfc9c19bb61613c0bd64065968adb27fa32a3e475f1e5e53850716"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/bundle_model_step_scaled.py",
      "sha256": "3d60ebb20e4b1cba76e2710358f23a026ee446fa5d378789eb27ce377096300c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/learned_global.py",
      "sha256": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/temporal_objective_increment.py",
      "sha256": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/temporal_objective_mse.py",
      "sha256": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/temporal_train_curl.py",
      "sha256": "0624d654920660504b03192af98d15aea59e4f85bd2a34379377ba5046033c7e"
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
  "lesson": "A divergence-free increment preserves existing divergence rather than removing it; F32addition/rounding can also introduce a residual. The representation does not guarantee pressure-equation consistency, tracer accuracy or low trajectory drift. Pressure/tracer heads remain learned separately. Compare actual full-state fields and divergence, not the mathematical identity alone.",
  "outcome": "Original training completes10,000updates; final composite objective 0.47428551316261292. The learned parameter count and initialization match the control exactly. Independent ideal-readout identities pass, but fresh complete-model GPU errors, block timing and600-output drift remain pending.",
  "reproduction": "Preserve exact curl masking/incidence/readout code, endpoint network/config/objective/typed recipe/runtime, original16scene F32archive/normalization and fixed statistics, private seeds/order and deployed feedback. There are no additional learned curl weights. Original final publication/restore runs CPU neural probes; qualify a GPU-only recipe first. Clean retraining, new-Rheon data generation, fresh common benchmarking and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-a47365bce83a4ce1",
    "kind": "training_attempt",
    "name": "Full AI temporal b2-endpoint-curl-increment train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "a47365bce83a4ce13e3e438285a01040f4ec025a2b4583548707ec4397dc1ac8",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "c8c342dc87954eab9891bb02b630b17d",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b2-endpoint-curl-increment train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 163068,
    "training_closure": {
      "config_sha256": "b11a5a489b8090bcfa45a8e77c28d9505232e195db4fd028a9e15880303675ed",
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
        "bundle_model_curl.py": "075ad797b7c78a4f9a793c1b71370a7a02c399768a069c06fc7f431d4481f90c",
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
        "temporal_train_curl.py": "0624d654920660504b03192af98d15aea59e4f85bd2a34379377ba5046033c7e"
      },
      "horizons": 2,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "e632db3f22f01645f397e9fcdaa5f6dc279f717f9993915b13572b349c5cae6e",
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
      "temporal-models/16"
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
      "receipt": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-pair-01/train/receipt.json",
    "sha256": "a47365bce83a4ce13e3e438285a01040f4ec025a2b4583548707ec4397dc1ac8",
    "bytes": 15376
  },
  "bound_training_configuration": {
    "config_sha256": "b11a5a489b8090bcfa45a8e77c28d9505232e195db4fd028a9e15880303675ed",
    "graph_id": "cde1ed46339a46bd9ecf2310f2415ba5",
    "graph_document_sha256": "fc5dfedd791a65e102cde6668e110640ae5fed635d4a78ab0937acae13dce608",
    "recipe_path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/train-recipe.json",
    "recipe_sha256": "65a651b1fc5a3e20b6b3bd97ecbeefddcff3680aa86e77a161491d527ae6a592",
    "config_path": "runs/full-ai-next-priorities-v1/b2-endpoint-curl-increment-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-curl-readout-v1",
      "velocity_representation": "learned_edge_potential_fixed_curl",
      "supervision_mode": "increments",
      "architecture": "endpoint",
      "feedback_protocol": 2,
      "identifier": "b2-endpoint-curl-increment",
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
      "loss": "Matched MSE pressure/equation/physical terms; velocity/tracer errors in fixed aggregate native-step units",
      "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
      "sharing_registration_sha256": "05df829aadc24ca8d17544b729694f5715018f9bec1eb33fefb3d1e6d0fd79fa",
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
    "config_sha256": "b11a5a489b8090bcfa45a8e77c28d9505232e195db4fd028a9e15880303675ed",
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
      "bundle_model_curl.py": "075ad797b7c78a4f9a793c1b71370a7a02c399768a069c06fc7f431d4481f90c",
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
      "temporal_train_curl.py": "0624d654920660504b03192af98d15aea59e4f85bd2a34379377ba5046033c7e"
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
      "sha256": "898db81acf30ffe4f2343e0209cd475ac88ea6ddc809ba295247ec5e0c7b62bb"
    },
    "fine.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "bfa934c002e0ac0f99347d8ca0405a97e8023771e5b8aaca5697ede3cf82d693"
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
      "sha256": "6190502554a7e82c33fc751e88c3cb190123bad841b9b1fe1d77f22ac989b513"
    },
    "fine.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "1d48e1a5c667ffdebd7f75240b55d25997ff5f9bf61ac4fbc0d1dc976c94309d"
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
      "sha256": "a730e563c5d47c23d88e65139585456f6139b6b610fa834c5abd357ca42057c5"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "7d6e677327336b645046d971fcf315b16613e787cbd28ea57e8f242ca0f1c356"
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
      "sha256": "3e216195f57298a2178b6b42ea53de2021039ed0b29887ebbdc1da6d53b712ba"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "39526dcfea5894e2d624fd037665a5e8fddb7eba5b8a9f9cc58697523751756c"
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
      "sha256": "04ca91c582bedb7dd294b3221108453045609a7abdb0092cc4d298b0c4661455"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "2d36dbfe47ae1f21088486ed5bc8021331a23033e214565fa330209d82e0b6fa"
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
      "sha256": "9eaa36ce29615b59c369f61f6ebd67225a126b2a192334c4c3f0fc69e61f6aa8"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "8c556fef4172f2f175a7f6a84437c2d100af71c287fca32293ec527fb6e3f1ea"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "c281803f2936f2c94b9e9a3ee382dd4e15cc6be4ff4b17a470d8fef42ede8827"
    },
    "global_blocks.0.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "5f06eb6809211cd1685a0cae9d54f7f518df2c21b3cb42ac1a15ee21428f0367"
    },
    "global_blocks.0.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "a28733f0d07c7ff4a4d3de335912f6d78228de5c463576935bea36de6c094238"
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
      "sha256": "67b30abdd30b4abc46166acef6884b63534d8ca7f8d5ebb39b0e78ec60375cf8"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "b70415a5595566b622be0b44774e87eecae8bffb7d49749192d4f17b4a2d291b"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "c8320b525dd7555b2f89c929b4a67304a5b62be3a387e175094c14d422b6acb6"
    },
    "global_blocks.1.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "040a20561887d7794982fbb6d4a53562cbac3c9ac256fbc1999e4a9cadd0a827"
    },
    "global_blocks.1.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "4e6ef724e899ade7ca3b26f87b865851b70faa4c034f038ed08fe242f91456ad"
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
      "sha256": "1a11d00c3900d95272796d8b9cb17d75d618a765c24ad1f1a20a11bf44f61818"
    },
    "endpoint_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "26fefb40ab084c71dbdd9965a9ca1f81d376a6c62d040469c692aa0a897e0c4b"
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
      "sha256": "2d0c6279d56f7ed857112dc5a5abebdcff86cd79ea07a4d1e25f41af7cbe3890"
    },
    "intermediate.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "1e4c41e248d97e72b46bc1e6f5f08f15972cc804477158e48d6fc8df3e8c4035"
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
      "sha256": "114f10e0dc1b066f7d172cfd86c9145d4e7eeb425c36af25cd3cfd7ae7fd92c8"
    },
    "intermediate_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "98661c94ae361892e455df6f21caa5883ac7112717fb3b83c249acf9f908e16f"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 876773376
  },
  "outputs": {
    "probe.npy": {
      "bytes": 196648,
      "sha256": "9089e7c4f59686f1ed25fa13e35639b4bdf50895d5dffb4c28e138ed9294d215"
    },
    "weights.npz": {
      "bytes": 659506,
      "sha256": "e632db3f22f01645f397e9fcdaa5f6dc279f717f9993915b13572b349c5cae6e"
    },
    "initial-weights.npz": {
      "bytes": 659506,
      "sha256": "2ebab7ca1c29fc8e12a41b0c488c19172f0604d40e5d5302bf77fee2a5180b97"
    },
    "history.json": {
      "bytes": 273106,
      "sha256": "a29dbf96a8587d8e8ee1cd7f46b993c941442ea1a058e956d8c3f46cff3e25e3"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 339.6178560589906,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-curl-readout-v1",
  "velocity_representation": "learned_edge_potential_fixed_curl",
  "supervision_mode": "increments",
  "architecture": "endpoint",
  "feedback_protocol": 2,
  "identifier": "b2-endpoint-curl-increment",
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
  "loss": "Matched MSE pressure/equation/physical terms; velocity/tracer errors in fixed aggregate native-step units",
  "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
  "sharing_registration_sha256": "05df829aadc24ca8d17544b729694f5715018f9bec1eb33fefb3d1e6d0fd79fa",
  "step_scale_statistics_sha256": "5f976a2e19e77cf7b46bf701b9e7c7cd65856d8f391db138020c8855592dad04",
  "original_normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037",
  "full_fit_registration_sha256": "08ed048038db26f1838a6e74f82ae521c848f06b954a42aaf3000a17794d0510"
}

```
