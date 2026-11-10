# Full AI temporal b3 train

Retrospective source review · fit-5a4d0a88ce0803e3

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 3; each output advances 1/60 second. Full block duration is 3/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 1.08732 | 1.21107 | 0.12737 |
| 8 | 16x16x16 | 1.17099 | 1.34353 | 0.15059 |
| 8 | 32x32x32 | 0.947298 | 1.49439 | 0.42505 |
| 8 | 9x11x11 | 1.19107 | 1.30631 | 0.13000 |
| 16 | 15x6x8 | 1.32906 | 1.20734 | 0.12743 |
| 16 | 16x16x16 | 1.15619 | 1.30504 | 0.15102 |
| 16 | 32x32x32 | 1.06651 | 1.47443 | 0.43946 |
| 16 | 9x11x11 | 1.24648 | 1.31008 | 0.12982 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 41c5dc934da61e9de5cdc94a6638c609f4202e5b6fc267eb4fb21dc15a79d761; subject: b3.

**Method.** A 22,903-parameter F32 joint full-state generator for K=3. A shared width4 encoder consumes the padded current five-field state, normalized coordinates, log grid counts/spacing and current known force. One3x3x3 convolution plus1x1 layers produce fine/coarse shared features; inexpensive joint1x1heads also receive all K prescribed force increments. A pooled32hidden-unit branch predicts63nonconstant cosine pressure coefficients per horizon. It emits pressure, three staggered velocities and tracer jointly; pressure is cell-zero gauged, velocity walls are masked, and physical fields are cropped/scaled. Pressure is computed in F32 then cast/gauged in F64 on unpacking; this is not F64 internal pressure prediction. No pressure solver or sequential physical advance is called inside this network.

**Purpose.** Test amortizing one current-state encoding over 3 simultaneous physical outputs at1/60s spacing, compared with the other protocol2 K1/2/3/4 joint-head generators.

**Experiment.** 10,000CUDA Adam updates, batch1, model seed2026101921/sample seed2026101922. Rate0.001 drops to0.0003 at6001 and0.0001 at9001. Sixteen train-only scenes provide129consecutive states each: eight at8cells/metre and eight at16, with16cubed,9x11x11,15x6x8 and32cubed domains, two seeds per grid/density. Each scene spans128/60seconds of retained native simulation. Updates1–2000supervise one K-output block; later updates add a second block from the first block own reconstructed endpoint, yielding18,000network calls and54,000predicted states during fitting. Source state indices are sampled0..120. The loss averages field/equation log-relative-square terms plus divergence, mass, energy and tracer-bound penalties. Known forcing varies in space/time, but its integration and features are fixed at1/60s; no independent dt channel is present. Fitting pressure labels are quantized toF32, with recorded worst conversion relative L2 6.225179570786273e-8; original native evaluation fields remain separate.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 1.32906, 1.00005 and 0.0259991. Median time is 3.93257 ms per K3 block (1.31086 ms per physical output); the worst case median is 1.58502 ms/output. The retained F64 N1 composition has better worst pressure error and 9.29 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** This retained design has higher worst pressure error and median GPU time than the matched F64 composition. Its method and fitting purpose remain recorded above; these measurements compare distinct designs and do not show that one checkpoint deteriorated. Judge complete physical outputs rather than a pressure-only component or a training-loss value.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b3-pair-02/train/receipt.json",
    "sha256": "5a4d0a88ce0803e36ff57d7ae0094fe274953906956268bb892a3984fc318bd7"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b3-pair-02/train/history.json",
    "sha256": "1e04540d309982f6727c1c26cdba7453218b50d5f1b354df5bcd058199c634d4"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b3-project-02/adaptive_pool.py",
      "sha256": "759eb0e374518052bae51b6f96c89ea6465de5b0160ed5a94b05802d034cd96e"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-project-02/bundle_model.py",
      "sha256": "add1f6eda47133318eae626497bde98e36cad02e48fcf7d4a1b252149fca3a05"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-project-02/bundle_model_v5.py",
      "sha256": "e1191613b019e951d29121c66e4844bedf610531271961430b6dce29a7eb6c27"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-project-02/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-project-02/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-project-02/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-project-02/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-project-02/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-project-02/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-project-02/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-project-02/temporal_train_v2.py",
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
  "lesson": "This is a genuine joint-head K=3 forecast: it shares the main encoder rather than calling a single-step network K times. One inference pass spans3/60seconds, while the training curriculum sometimes calls two such blocks for own-history supervision. It does not inherit the near-exact F64 pressure-component anchor: its pressure arithmetic and fitting labels are F32, its outputs include full fluid state, and its input/control interface differs. Fixed1/60s training cannot establish arbitrary timestep, high-density or long-rollout accuracy. Fresh common complete-generator measurements remain required.",
  "outcome": "Original training completed at10,000updates; final logged composite loss 18.076080322265625 with2supervised blocks. Original initial/final tensors and CPU probe are retained and byte-verified. No fresh common complete-generator accuracy, latency or own-history drift result is established by this source review. The training loss is not pressure relative L2, and an original CPU probe is not a current GPU benchmark.",
  "reproduction": "Use the exact typed train recipe, config, eleven trajectory shards, normalization/runtime manifests and complete receipt-bound Python source. Preserve lossless F32 transport, original native reference receipts, zero-initialized heads, private sample order, physical reconstruction between supervised blocks and all loss terms. Historical training and restore recipes execute CPU neural probes, so qualify a GPU-only reproducer before rerunning. This audit performs no neural forward, fit or scene generation; clean retraining and new-Rheon regeneration remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-5a4d0a88ce0803e3",
    "kind": "training_attempt",
    "name": "Full AI temporal b3 train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "5a4d0a88ce0803e36ff57d7ae0094fe274953906956268bb892a3984fc318bd7",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b3-pair-02/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "e035c0a8363b4ceab780711d9c7dd841",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b3 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 22903,
    "training_closure": {
      "config_sha256": "a6b6a68b65ceccdd49bcd97fe66888157b82e47fa82cd3b1b3433c6f94f54d66",
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
      "horizons": 3,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "ee3a926ae143c5225bdbb0c2da5fa19732d6bbbe995eebe0fc1bb20d56c634ac",
        "bytes": 95234,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "ec707db2ccb426b9b30940c1d2b1c5241910483df5d55be1425cead232b9a51d",
        "bytes": 95234,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/2"
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
      "receipt": "runs/full-ai-next-priorities-v1/b3-pair-02/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b3-pair-02/train/receipt.json",
    "sha256": "5a4d0a88ce0803e36ff57d7ae0094fe274953906956268bb892a3984fc318bd7",
    "bytes": 9894
  },
  "bound_training_configuration": {
    "config_sha256": "a6b6a68b65ceccdd49bcd97fe66888157b82e47fa82cd3b1b3433c6f94f54d66",
    "graph_id": "8e3f3887b0374bdbaa4386b6cc7dda53",
    "graph_document_sha256": "51942ea831f4e8fe4503daea03a479f276f4af88b2571566d40709db44d74090",
    "recipe_path": "runs/full-ai-next-priorities-v1/b3-project-02/train-recipe.json",
    "recipe_sha256": "bb94f5747dcda7ae83957504a16dc4f13a5be41af2f9a875312bc222d002e504",
    "config_path": "runs/full-ai-next-priorities-v1/b3-project-02/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-v2",
      "feedback_protocol": 2,
      "identifier": "b3",
      "horizons": 3,
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
  "parameters": 22903,
  "horizons": 3,
  "closure": {
    "config_sha256": "a6b6a68b65ceccdd49bcd97fe66888157b82e47fa82cd3b1b3433c6f94f54d66",
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
    "horizons": 3,
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
        15,
        13,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "679879a045a4fc0469906f91b6410cc32496e53b24d8e743d4b954826b25a6d1"
    },
    "head.bias": {
      "shape": [
        15
      ],
      "dtype": "<f4",
      "sha256": "5dcc1b5872dd9ff1c234501f1fefda01f664164e1583c3e1bb3dbea47588ab31"
    },
    "modal.0.weight": {
      "shape": [
        32,
        459
      ],
      "dtype": "<f4",
      "sha256": "59ebf6d4f32c2ca4ac2a9033dbf2a5d582a3c62b9feb9e409ddab0aa8b79feb2"
    },
    "modal.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "aa80241853ba5d4d6ac6606e2187212bac7e0b9f358cf7b08b7c14c25b4aae39"
    },
    "modal.2.weight": {
      "shape": [
        189,
        32
      ],
      "dtype": "<f4",
      "sha256": "65a64ec4b34680687d98cd473b8d954d240c68f72194c2d45ccfa304c84ba3ef"
    },
    "modal.2.bias": {
      "shape": [
        189
      ],
      "dtype": "<f4",
      "sha256": "dfae75e86c3e349de212f970c922f6c3bf1cf09787ffab21ebae82d2a5df2b2a"
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
      "sha256": "0d964f3bf0f0761fc310266b8a78e0f911318bc9b122aeee232cb13adcb7f50b"
    },
    "fine.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "320f880f44dd40cbfe3a34a3f18d890d5cba9e35afb7b35abd12658fded02af4"
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
      "sha256": "d9cdc3d419ebaf73eb33dedd67c100a161bc2a1f8f5a9069e9e71b93c5ac7ee4"
    },
    "fine.2.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "14de9e7486df12e6e79fd01b5a05097b2180b1f819994301f18bab1abafcbfd1"
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
      "sha256": "c74bec367efc6aede78084a879abc9c90817e417f2e3115db2ff1658bf69bca1"
    },
    "coarse.0.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "416ddc569e0c0a70c07681bbeb7f901d96923202fa295b41201698f8a89e3d21"
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
      "sha256": "ac1c2c2cc4a0bcd3a6e9ff192f8d7363952857c67c3b3830723071245c471159"
    },
    "decode.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "dd3268d0ad4225f9558ee2546776ec8fd3ac20a87b0b99811d669b1d8d7c9893"
    },
    "head.weight": {
      "shape": [
        15,
        13,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "c895d25a1ebd2d44d00ca066fdcc1233cb7937e4f786575e400d414894b9bbba"
    },
    "head.bias": {
      "shape": [
        15
      ],
      "dtype": "<f4",
      "sha256": "fcf30b2a873f53f57d992d347a3c393ba5fb25085a1fa591b3292e395a92aab6"
    },
    "modal.0.weight": {
      "shape": [
        32,
        459
      ],
      "dtype": "<f4",
      "sha256": "ca42f6803e88510b97d6887a001b5c3399005872d81fa162b840769d64dff580"
    },
    "modal.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "456f2f01a111e581bb0648f9c52cce695dc0467793c39d459fc4592f3282d8df"
    },
    "modal.2.weight": {
      "shape": [
        189,
        32
      ],
      "dtype": "<f4",
      "sha256": "d5abe1d5edfbd232dc25aa2df4ef886288239c0135448eda553a427d325c02f1"
    },
    "modal.2.bias": {
      "shape": [
        189
      ],
      "dtype": "<f4",
      "sha256": "7c476cb48046c6cc60ed7e7ed62bfbf6a45cb7a30bb6bc9dfced1a417bfa2f37"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 839880192
  },
  "outputs": {
    "probe.npy": {
      "bytes": 294908,
      "sha256": "b0e8b3a3d891276925248337a9369cabd34e1a2282b0bae4be091ac82f6d43f7"
    },
    "weights.npz": {
      "bytes": 95234,
      "sha256": "ee3a926ae143c5225bdbb0c2da5fa19732d6bbbe995eebe0fc1bb20d56c634ac"
    },
    "initial-weights.npz": {
      "bytes": 95234,
      "sha256": "ec707db2ccb426b9b30940c1d2b1c5241910483df5d55be1425cead232b9a51d"
    },
    "history.json": {
      "bytes": 271101,
      "sha256": "1e04540d309982f6727c1c26cdba7453218b50d5f1b354df5bcd058199c634d4"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 205.98550528596388,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-v2",
  "feedback_protocol": 2,
  "identifier": "b3",
  "horizons": 3,
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
