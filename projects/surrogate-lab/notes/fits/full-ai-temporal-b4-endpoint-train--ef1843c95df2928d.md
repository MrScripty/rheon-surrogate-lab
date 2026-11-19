# Full AI temporal b4-endpoint train

Retrospective source review · fit-ef1843c95df2928d

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 4; each output advances 1/60 second. Full block duration is 4/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.708872 | 1.14559 | 0.11509 |
| 8 | 16x16x16 | 0.422664 | 1.19069 | 0.14026 |
| 8 | 32x32x32 | 0.758847 | 1.33619 | 0.40021 |
| 8 | 9x11x11 | 0.29538 | 1.18316 | 0.11762 |
| 16 | 15x6x8 | 0.563136 | 1.22083 | 0.11392 |
| 16 | 16x16x16 | 0.700204 | 1.20676 | 0.13887 |
| 16 | 32x32x32 | 0.803162 | 1.35543 | 0.40363 |
| 16 | 9x11x11 | 0.734132 | 1.15552 | 0.11840 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 046d91f6248a32f47843817c6b8c4d87af71e309336e498b4ac3a50dbf0c2df6; subject: b4-endpoint.

**Method.** A 163,372-parameter F32 width12 full-state generator for K=4. Current velocity/tracer plus geometry/current known force enter one spatial encoder; previous pressure is deliberately excluded. All K prescribed forces enter retained features. Two learned global blocks mix up to8cosine modes per axis across12latent channels, with learned7→32→144mode/geometry-conditioned weights; these are neural mixing weights, not inverse pressure eigenvalues. A fresh five-field endpoint head predicts the last state. One shared1x1decoder consumes that predicted endpoint, initial velocity/tracer, hidden features and all forces, then emits all K−1intermediate states jointly. Pressure is fresh raw output with cell-zero gauge; velocity/tracer increments always refer to the initial block state. Velocity walls are masked. Output packing upcasts pressure toF64, but prediction isF32. No numerical pressure solve or full physical integrator is called.

**Purpose.** Test the accurate-endpoint/cheap-intermediate architecture for 4 physical outputs per pass, with a learned predicted endpoint rather than a supplied future label.

**Experiment.** All six fits use the same captured16train scenes/2064states at8/16cells per metre and fixed1/60s, with16cubed,9x11x11,15x6x8 and32cubed domains. Original F32trajectory shards/normalization/runtime slots match the independently decoded joint-generator corpus. Each fit performs10,000CUDA Adam updates: batch1, model seed2026101921/sample seed2026101922, rate0.001 with drops at6001/9001. First2000updates supervise one block; remaining8000supervise a second block from reconstructed own endpoint, producing18,000network calls. Every horizon receives the original composite field/equation/divergence/mass/energy/bounds objective; no separate high-accuracy-endpoint guarantee is imposed. Same-K endpoint/reuse pairs share exact initial encoder/forcing/global tensors. ChangingKchanges parameter shapes and random-number consumption; full initialization equality acrossKis not claimed.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 0.803162, 0.999691 and 0.0275782. Median time is 4.82007 ms per K4 block (1.20502 ms per physical output); the worst case median is 1.49266 ms/output. The retained F64 N1 composition has better worst pressure error and 9.35 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** Sharing an encoder, endpoint or retained features does not by itself establish a useful accuracy/speed tradeoff. This fit remains worse than the matched F64 composition in both worst pressure error and median GPU time. Account for every intermediate output and reconstruction cost; K outputs advance K full 1/60-second intervals.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b4-endpoint-pair-01/train/receipt.json",
    "sha256": "ef1843c95df2928d862655a18bcb6ee2341c86cbb9838690c96af839bc64c2e4"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b4-endpoint-pair-01/train/history.json",
    "sha256": "d1360287240164cd24135d12e6fd04b687f569cdf6d6de17bff9b43b06b96d9d"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/bundle_model_sharing.py",
      "sha256": "3490bcffb6bfc9c19bb61613c0bd64065968adb27fa32a3e475f1e5e53850716"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/learned_global.py",
      "sha256": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/temporal_train_sharing.py",
      "sha256": "d270a1c6cc54f2ead4ad9120b1cdf0973d4522301859ca61e9856d2b9234fb9e"
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
  "lesson": "This implements the proposed endpoint-conditioned intermediate strategy in the model itself. Its endpoint is predicted, so it receives no future-state leak. Shared global processing and a joint intermediate decoder can save repeated encoding, but equal-horizon training does not certify endpoint accuracy or a speed win. Keep this prototype available for common benchmarking rather than rejecting it from an old target or label.",
  "outcome": "Original fit completed10,000updates; final logged composite objective 18.805782318115234 with2supervised blocks. All original initial/final tensors, source/recipe/input/output identities and sample/work counts verify. No new complete-generator accuracy, GPU latency or own-history drift has been measured in this review. The composite loss cannot be read as pressure relative L2 or an inference-time score.",
  "reproduction": "Use exact original config/typed recipe/runtime/source closure, original16scene F32archive and normalization, zero output heads, private seeds/order and deployed reconstruction between supervised blocks. Preserve pressure-history exclusion, two learned global blocks and the selected endpoint/reuse branch. The older recipes perform CPU neural publication/restore probes; qualify GPU-only reproduction before rerunning. Clean retraining, new-Rheon data generation, fresh comparable performance and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-ef1843c95df2928d",
    "kind": "training_attempt",
    "name": "Full AI temporal b4-endpoint train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "ef1843c95df2928d862655a18bcb6ee2341c86cbb9838690c96af839bc64c2e4",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b4-endpoint-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "01629a4e062e43d5ae8e5a9b03a77a67",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b4-endpoint train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 163372,
    "training_closure": {
      "config_sha256": "48a62ed9a5e29ff33e3a4e12e712c8f735b6194bbfe73e9af5d530868196cd56",
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
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "learned_global.py": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b",
        "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
        "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
        "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
        "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
        "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
        "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
        "temporal_train_sharing.py": "d270a1c6cc54f2ead4ad9120b1cdf0973d4522301859ca61e9856d2b9234fb9e"
      },
      "horizons": 4,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "7205f10436add96f3725d1578142291bbb33a487233bd6efdd20377f6603522e",
        "bytes": 660722,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "2006a6861fdff492262d759839a90453cfcab4017580ca916d86b3a2263e0ff9",
        "bytes": 660722,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/10"
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
      "receipt": "runs/full-ai-next-priorities-v1/b4-endpoint-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b4-endpoint-pair-01/train/receipt.json",
    "sha256": "ef1843c95df2928d862655a18bcb6ee2341c86cbb9838690c96af839bc64c2e4",
    "bytes": 14971
  },
  "bound_training_configuration": {
    "config_sha256": "48a62ed9a5e29ff33e3a4e12e712c8f735b6194bbfe73e9af5d530868196cd56",
    "graph_id": "f5c3f5e590a54befb3458ddf20b1b2f6",
    "graph_document_sha256": "e742240f652eab46581435904b577ce7b161428c24a09d8636afb46293995fd8",
    "recipe_path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/train-recipe.json",
    "recipe_sha256": "58eec37522a5f06618c00745d4daaa4534133c309a47e7712279c621ef942e26",
    "config_path": "runs/full-ai-next-priorities-v1/b4-endpoint-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-sharing-v1",
      "architecture": "endpoint",
      "feedback_protocol": 2,
      "identifier": "b4-endpoint",
      "horizons": 4,
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
      "loss": "Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds",
      "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
      "sharing_registration_sha256": "788464651edff12d25ec3c847b27a4091b84c49382af66d139d563a7e0052983",
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
  "parameters": 163372,
  "horizons": 4,
  "closure": {
    "config_sha256": "48a62ed9a5e29ff33e3a4e12e712c8f735b6194bbfe73e9af5d530868196cd56",
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
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "learned_global.py": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b",
      "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
      "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
      "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
      "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
      "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
      "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
      "temporal_train_sharing.py": "d270a1c6cc54f2ead4ad9120b1cdf0973d4522301859ca61e9856d2b9234fb9e"
    },
    "horizons": 4,
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
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "f142b24f5c2351a3a612a53a5123a8277b385c8d24316ee2a04da235f427f7a1"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "a640be64037bda4bdf28bc3ba09aa9186a68b726b51b3f126f6eaabe7d248013"
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
      "sha256": "c5dee049efd56f93d670a191475c9d07007039ef45e5b4174cc5ebaa066e38bd"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "8c98bc614fdc66738ed0756642616602e3c74138d078aa8430f3acbffbecf0a6"
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
      "sha256": "148f1b5c63958c4d50a1e5870848bba30e91e9f8a23c1d7e687d743a5309df72"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "a95cb0b0edc6dce4f433d641d79b997f154d9ee7d766518e789bb9c776346fec"
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
      "sha256": "ed450da304b213b9dce4f7ec5bf211c07ebd98deeff60690473ee1e70f65159d"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "889904a88ac56899a61e926703939074ded230be64d4eacc72e9bbe1a66c512e"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "9bc7066a47890186b34d4d893d4bc7449f4ee3b573ba56577e12818d69762b71"
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
      "sha256": "fb8aafcc0c9335db69423f054322a46de4c4bbfa55e795ea9e2551c8f46fce59"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "5576849fcc6cec31b6a09525c20333dee9adbf9c370d011627037c8a870974e2"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "62a100fb412b5cd24aa486d25f913c3db79dd0c3da3f3ffc2e2e6342cd40ff6d"
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
        24,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "4b48f21a4b7a02bfbec19ef880a967a02334a3cdcef8ae83de2ef327ba8bc5dd"
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
        33,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "28f8bc3e7e32f83cd2d65b5e520310786caa8844300e66e81979f5bd7ecdd659"
    },
    "intermediate.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "1e814e90f29643253f31345cc2e74302e3cf5ab4fa2271309158b4f5104e1bf2"
    },
    "intermediate_head.weight": {
      "shape": [
        15,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "fe2f74a1e0b16a66452888eb4d734bc455cf1304481bb495d59afa8cf9cae93b"
    },
    "intermediate_head.bias": {
      "shape": [
        15
      ],
      "dtype": "<f4",
      "sha256": "5dcc1b5872dd9ff1c234501f1fefda01f664164e1583c3e1bb3dbea47588ab31"
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
      "sha256": "9b1b7151e4e4e4569e2026b70c1ecbef8afa59729888efe7f13b700445d8844d"
    },
    "fine.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "8c88d26cfaf72a61a67591dbe94c65798c282f6f56c8dc6dfd4d28a5a2ae3669"
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
      "sha256": "1af7920e6dd608caf4331d1d9ff8a610cff761f4eba0ce57804e04eb0ef2958d"
    },
    "fine.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "7667857f7aa1370e6c16cf8ae91f516e99c04a264ed49ae440a0f445c039dd55"
    },
    "forcing.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "8115b19f22eab4da98a214a5526d4dfa9e95892cfc8802001405d3dbeaff8c48"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "bf6a1da006683ce400984e46dbcedd46c3606ab1cbb2ff23f25e951a7baf31c1"
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
      "sha256": "a67c6457369fecb3c343ca18b0324f1b018665ec8d7af0e878a243b4fff5310d"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "04fc9932a8c74757eb65fe037b70e766e6257a974600a3f669cc5d3e4153795e"
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
      "sha256": "583c7666d579aa3ab4da0f518a9cf7e4460a8c44524e698eb5190c49bf986e2a"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "9d904d730b1a78bcd175da101439ded963396b550278db14aef367768cafb248"
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
      "sha256": "cdbfe1c7a581488f4eeda2b7e6a15e110b46446dd17c667ba45dbf688afcdb62"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "5e69d7779170aa03e5004fa2f27b51b2fb29d47004679d54ff5aad04a63b55d8"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "7ff4568785ac3a0fa72f2a456788702a9f304a3b64a2dadf573bdee584c3d4f1"
    },
    "global_blocks.0.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "55a15398ccf22d8642f30228a93e13bf5f520c59c0a7a5295aae2aff672c3821"
    },
    "global_blocks.0.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "9148cbbc78acd792ed0730fe165737d2d55ef3fc6d4f6c2a0df897325771ba41"
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
      "sha256": "862817ab26e0a45821b07e92bf22fa4a913afe880ff052b72919f0552673bc92"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "030ee3e02c90d6e6e8c74479317298a9b9f6cf6062f077d284d1bb8e00c0c436"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "8bf1e88ad3078550b9c0ffc040207173cea3d62b64e999bce1b6408a0c7ac787"
    },
    "global_blocks.1.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "d8500388d5c1d6fcf75655bbbfbbd9c9a060a7b373a161c9f4ef109050a13b67"
    },
    "global_blocks.1.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "49edbed40e4b6dcc873d4e80a2fb05ddda7e8646820f7da5a136e6da36c126ba"
    },
    "endpoint_head.weight": {
      "shape": [
        5,
        24,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "2f54f7a619f49ead9c342bbede317fde6bc661583ac5e50e995347df6d53e2d4"
    },
    "endpoint_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "516c214a254f24d357f03814dd781d2a2ad2673d577c1d76ec9f96c044286520"
    },
    "intermediate.0.weight": {
      "shape": [
        12,
        33,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "a371b6739bdfad340b83159f9c1e270c6d978da5ccf4148627cc7a0c38d67de3"
    },
    "intermediate.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "62149f8e8660136c5c8a70ec02acf06619d8896ab5ab1b8a11f7976fc19e5ca7"
    },
    "intermediate_head.weight": {
      "shape": [
        15,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "5f73d238a6ab0d08d87f05c6286c171ecc250dbd4a723cc8ba73aa812843e43b"
    },
    "intermediate_head.bias": {
      "shape": [
        15
      ],
      "dtype": "<f4",
      "sha256": "5d03da2a2d766596f7a68386e2e19cbf7bf398d63ed51e3edc01ed3f4887950e"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 892495872
  },
  "outputs": {
    "probe.npy": {
      "bytes": 393168,
      "sha256": "42ee5abce74b261cf1eadd0d574d923da813ddadf2ec8a54aa0883a23dae8dfa"
    },
    "weights.npz": {
      "bytes": 660722,
      "sha256": "7205f10436add96f3725d1578142291bbb33a487233bd6efdd20377f6603522e"
    },
    "initial-weights.npz": {
      "bytes": 660722,
      "sha256": "2006a6861fdff492262d759839a90453cfcab4017580ca916d86b3a2263e0ff9"
    },
    "history.json": {
      "bytes": 275141,
      "sha256": "d1360287240164cd24135d12e6fd04b687f569cdf6d6de17bff9b43b06b96d9d"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 303.26558475900674,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-sharing-v1",
  "architecture": "endpoint",
  "feedback_protocol": 2,
  "identifier": "b4-endpoint",
  "horizons": 4,
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
  "loss": "Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds",
  "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
  "sharing_registration_sha256": "788464651edff12d25ec3c847b27a4091b84c49382af66d139d563a7e0052983",
  "full_fit_registration_sha256": "08ed048038db26f1838a6e74f82ae521c848f06b954a42aaf3000a17794d0510"
}

```
