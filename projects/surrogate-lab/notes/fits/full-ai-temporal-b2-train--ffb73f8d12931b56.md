# Full AI temporal b2 train

Retrospective source review · fit-ffb73f8d12931b56

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 2; each output advances 1/60 second. Full block duration is 2/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.969819 | 1.97175 | 0.15940 |
| 8 | 16x16x16 | 0.931013 | 1.85630 | 0.19605 |
| 8 | 32x32x32 | 0.939973 | 1.88214 | 0.47160 |
| 8 | 9x11x11 | 0.96678 | 1.74273 | 0.16498 |
| 16 | 15x6x8 | 0.952717 | 1.91304 | 0.16040 |
| 16 | 16x16x16 | 1.01145 | 1.82241 | 0.19602 |
| 16 | 32x32x32 | 1.039 | 2.02855 | 0.47723 |
| 16 | 9x11x11 | 0.986458 | 1.75472 | 0.16505 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 480d903fd12f64c7130577e176878296432640a98f7d5b9471f0d8929f641841; subject: b2.

**Method.** A 18,132-parameter F32 joint full-state generator for K=2. A shared width4 encoder consumes the padded current five-field state, normalized coordinates, log grid counts/spacing and current known force. One3x3x3 convolution plus1x1 layers produce fine/coarse shared features; inexpensive joint1x1heads also receive all K prescribed force increments. A pooled32hidden-unit branch predicts63nonconstant cosine pressure coefficients per horizon. It emits pressure, three staggered velocities and tracer jointly; pressure is cell-zero gauged, velocity walls are masked, and physical fields are cropped/scaled. Pressure is computed in F32 then cast/gauged in F64 on unpacking; this is not F64 internal pressure prediction. No pressure solver or sequential physical advance is called inside this network.

**Purpose.** Test amortizing one current-state encoding over 2 simultaneous physical outputs at1/60s spacing, compared with the other protocol2 K1/2/3/4 joint-head generators.

**Experiment.** 10,000CUDA Adam updates, batch1, model seed2026101921/sample seed2026101922. Rate0.001 drops to0.0003 at6001 and0.0001 at9001. Sixteen train-only scenes provide129consecutive states each: eight at8cells/metre and eight at16, with16cubed,9x11x11,15x6x8 and32cubed domains, two seeds per grid/density. Each scene spans128/60seconds of retained native simulation. Updates1–2000supervise one K-output block; later updates add a second block from the first block own reconstructed endpoint, yielding18,000network calls and36,000predicted states during fitting. Source state indices are sampled0..120. The loss averages field/equation log-relative-square terms plus divergence, mass, energy and tracer-bound penalties. Known forcing varies in space/time, but its integration and features are fixed at1/60s; no independent dt channel is present. Fitting pressure labels are quantized toF32, with recorded worst conversion relative L2 6.225179570786273e-8; original native evaluation fields remain separate.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 1.039, 0.833693 and 0.0172616. Median time is 3.73962 ms per K2 block (1.86981 ms per physical output); the worst case median is 2.09992 ms/output. The retained F64 N1 composition has better worst pressure error and 10.43 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** This retained design has higher worst pressure error and median GPU time than the matched F64 composition. Its method and fitting purpose remain recorded above; these measurements compare distinct designs and do not show that one checkpoint deteriorated. Judge complete physical outputs rather than a pressure-only component or a training-loss value.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-pair-02/train/receipt.json",
    "sha256": "ffb73f8d12931b56183c33eec2b1cd00b97124878453dd4b6e3d3f8dde8a0c8b"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b2-pair-02/train/history.json",
    "sha256": "5d665fe3c2d8109b2c04e7f6352dbe17c0318cd282f0d1b1335e3c96fbfa6893"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b2-project-02/adaptive_pool.py",
      "sha256": "759eb0e374518052bae51b6f96c89ea6465de5b0160ed5a94b05802d034cd96e"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-project-02/bundle_model.py",
      "sha256": "add1f6eda47133318eae626497bde98e36cad02e48fcf7d4a1b252149fca3a05"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-project-02/bundle_model_v5.py",
      "sha256": "e1191613b019e951d29121c66e4844bedf610531271961430b6dce29a7eb6c27"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-project-02/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-project-02/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-project-02/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-project-02/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-project-02/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-project-02/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-project-02/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-project-02/temporal_train_v2.py",
      "sha256": "3dd30c010d72ff2d5da02efcbefda6fec2eca3b8b35132c084476bf67b35b6bf"
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
  "lesson": "This is a genuine joint-head K=2 forecast: it shares the main encoder rather than calling a single-step network K times. One inference pass spans2/60seconds, while the training curriculum sometimes calls two such blocks for own-history supervision. It does not inherit the near-exact F64 pressure-component anchor: its pressure arithmetic and fitting labels are F32, its outputs include full fluid state, and its input/control interface differs. Fixed1/60s training cannot establish arbitrary timestep, high-density or long-rollout accuracy. Fresh common complete-generator measurements remain required.",
  "outcome": "Original training completed at10,000updates; final logged composite loss 16.928298950195312 with2supervised blocks. Original initial/final tensors and CPU probe are retained and byte-verified. No fresh common complete-generator accuracy, latency or own-history drift result is established by this source review. The training loss is not pressure relative L2, and an original CPU probe is not a current GPU benchmark.",
  "reproduction": "Use the exact typed train recipe, config, eleven trajectory shards, normalization/runtime manifests and complete receipt-bound Python source. Preserve lossless F32 transport, original native reference receipts, zero-initialized heads, private sample order, physical reconstruction between supervised blocks and all loss terms. Historical training and restore recipes execute CPU neural probes, so qualify a GPU-only reproducer before rerunning. This audit performs no neural forward, fit or scene generation; clean retraining and new-Rheon regeneration remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-ffb73f8d12931b56",
    "kind": "training_attempt",
    "name": "Full AI temporal b2 train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "ffb73f8d12931b56183c33eec2b1cd00b97124878453dd4b6e3d3f8dde8a0c8b",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b2-pair-02/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "ae9d1bf415ff45c69a561e369b75bc83",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b2 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 18132,
    "training_closure": {
      "config_sha256": "755b2f60212763f0db94c14a7d5a00500b1e10aa32a16439984f61d07e4e271a",
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
        "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
        "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
        "temporal_train_v2.py": "3dd30c010d72ff2d5da02efcbefda6fec2eca3b8b35132c084476bf67b35b6bf"
      },
      "horizons": 2,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "e5c92d5372f9711a2796292cfc76912830ae29d520bcce5cb09ecc1225838e9a",
        "bytes": 76150,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "7c694803c7b56176bc3b8550c96cd2278d7bb273c43910a6c27c7062ba6f65f5",
        "bytes": 76150,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/1"
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
      "receipt": "runs/full-ai-next-priorities-v1/b2-pair-02/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-pair-02/train/receipt.json",
    "sha256": "ffb73f8d12931b56183c33eec2b1cd00b97124878453dd4b6e3d3f8dde8a0c8b",
    "bytes": 9894
  },
  "bound_training_configuration": {
    "config_sha256": "755b2f60212763f0db94c14a7d5a00500b1e10aa32a16439984f61d07e4e271a",
    "graph_id": "87558712a00d4600b78dfb7516b502f3",
    "graph_document_sha256": "3add64bfecc5e5a77564f903c7c889e94821feed4502c44fccee64b4391302e6",
    "recipe_path": "runs/full-ai-next-priorities-v1/b2-project-02/train-recipe.json",
    "recipe_sha256": "6408193d524830833d4dd13ace5c0df6dc7843b27f5bcf96eadfc2ffe622a3c8",
    "config_path": "runs/full-ai-next-priorities-v1/b2-project-02/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-v2",
      "feedback_protocol": 2,
      "identifier": "b2",
      "horizons": 2,
      "width": 4,
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
      "initialization": "Seeded compact spatial encoder and zero future heads",
      "inference": "One shared encoder; joint state outputs; learned pressure modes; no numerical pressure solve",
      "sample_step_range": [
        0,
        120
      ],
      "loss": "Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds",
      "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
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
  "parameters": 18132,
  "horizons": 2,
  "closure": {
    "config_sha256": "755b2f60212763f0db94c14a7d5a00500b1e10aa32a16439984f61d07e4e271a",
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
      "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
      "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
      "temporal_train_v2.py": "3dd30c010d72ff2d5da02efcbefda6fec2eca3b8b35132c084476bf67b35b6bf"
    },
    "horizons": 2,
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
        10,
        10,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "7a12e561363385e9dfeeab326368731c030ed4b374e7f5897ac819159d2884c5"
    },
    "head.bias": {
      "shape": [
        10
      ],
      "dtype": "<f4",
      "sha256": "2c34ce1df23b838c5abf2a7f6437cca3d3067ed509ff25f11df6b11b582b51eb"
    },
    "modal.0.weight": {
      "shape": [
        32,
        378
      ],
      "dtype": "<f4",
      "sha256": "771292db07e6d510084727a4d6d5fa1e091ce1bb27ff297e17bcefd6f81a6eec"
    },
    "modal.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "4ee0374aebcd5090df41ec5014cd054c74b7e45b8c2fe95a4d39a17e93e52808"
    },
    "modal.2.weight": {
      "shape": [
        126,
        32
      ],
      "dtype": "<f4",
      "sha256": "dbd7e2a1e3209b7914264834f82f4463e09ad552c19280705c64a1fb97c86292"
    },
    "modal.2.bias": {
      "shape": [
        126
      ],
      "dtype": "<f4",
      "sha256": "696bda342649ec9268da57b6a279df6f24b0e857d5e6d0605fd25af95adc3cee"
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
      "sha256": "c8f472e5defe6bca68f9cc6cf5d96d9ff5fb00ba7365d228fa38d8b1b2105efb"
    },
    "fine.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "0d30a884ae91a42dfc16791477c4ae9de38ededd0f1f6dfbabe16ada2cfece2a"
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
      "sha256": "bbbbc07bcb104717587ce372027fbfbdeb5bd7b7ac55cb18018b1234f3d79ad8"
    },
    "fine.2.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "7b08b702419c011489954f35c352b506815cbb374e4e42a11b1c4a59c55edab1"
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
      "sha256": "93fd23c2e99177be7c8f7a2d95d436ef69eab26b58192f8a9e6f9eb6954549bf"
    },
    "coarse.0.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "aa09d7d1047bad1b8ca4586d2d72f94dc558a00cc95b650d7a05ac3f33c6c078"
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
      "sha256": "32a0671b281e5cde8cd37dfbd4158b2488da6d022bee7eaa5e708b749c5be1fc"
    },
    "decode.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "4b40028917b18d2aa3e96cb743ca385baa12fdea96f3796abead3460c793c682"
    },
    "head.weight": {
      "shape": [
        10,
        10,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "de5e82cc595f8cc5d9e3574bf83c83ed099910b1591962202ba39d06cfabd026"
    },
    "head.bias": {
      "shape": [
        10
      ],
      "dtype": "<f4",
      "sha256": "5b6151e7d6066df66cc0afe3cb654624c5a5ee42ddb4678c4ecc0955bad29453"
    },
    "modal.0.weight": {
      "shape": [
        32,
        378
      ],
      "dtype": "<f4",
      "sha256": "dcae97476940ad849bbe47121a88eaeb30d10caaf127843edc68167cfbcf8b10"
    },
    "modal.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "c0fa9832b76243fc4bcbd7a9fa1ef96c64db9de81376e4dcfdab845c27084eee"
    },
    "modal.2.weight": {
      "shape": [
        126,
        32
      ],
      "dtype": "<f4",
      "sha256": "2702ef0a3490e5eefdce12662d214b121bab943aaadb0c514e9885207b16c570"
    },
    "modal.2.bias": {
      "shape": [
        126
      ],
      "dtype": "<f4",
      "sha256": "42ada62d18c76695807bed6dcc05cdfce3d1df85d18577d98d03e9d0f83e63e0"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 831308800
  },
  "outputs": {
    "probe.npy": {
      "bytes": 196648,
      "sha256": "fe1908b532a03c87a381834f3d532fabf50ddda96109902712ae7701d707217b"
    },
    "weights.npz": {
      "bytes": 76150,
      "sha256": "e5c92d5372f9711a2796292cfc76912830ae29d520bcce5cb09ecc1225838e9a"
    },
    "initial-weights.npz": {
      "bytes": 76150,
      "sha256": "7c694803c7b56176bc3b8550c96cd2278d7bb273c43910a6c27c7062ba6f65f5"
    },
    "history.json": {
      "bytes": 271271,
      "sha256": "5d665fe3c2d8109b2c04e7f6352dbe17c0318cd282f0d1b1335e3c96fbfa6893"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 215.99563255399698,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-v2",
  "feedback_protocol": 2,
  "identifier": "b2",
  "horizons": 2,
  "width": 4,
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
  "initialization": "Seeded compact spatial encoder and zero future heads",
  "inference": "One shared encoder; joint state outputs; learned pressure modes; no numerical pressure solve",
  "sample_step_range": [
    0,
    120
  ],
  "loss": "Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds",
  "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
  "full_fit_registration_sha256": "08ed048038db26f1838a6e74f82ae521c848f06b954a42aaf3000a17794d0510"
}

```
