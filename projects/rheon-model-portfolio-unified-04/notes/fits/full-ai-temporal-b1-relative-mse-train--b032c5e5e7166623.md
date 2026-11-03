# Full AI temporal b1-relative-mse train

Retrospective source review · fit-b032c5e5e7166623

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 1; each output advances 1/60 second. Full block duration is 1/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.998534 | 3.42587 | 0.25071 |
| 8 | 16x16x16 | 0.957282 | 3.54374 | 0.27377 |
| 8 | 32x32x32 | 0.937891 | 3.70641 | 0.58808 |
| 8 | 9x11x11 | 1.03397 | 3.49333 | 0.25054 |
| 16 | 15x6x8 | 0.983446 | 3.40427 | 0.24466 |
| 16 | 16x16x16 | 0.952861 | 3.67656 | 0.26408 |
| 16 | 32x32x32 | 1.00568 | 3.66162 | 0.58521 |
| 16 | 9x11x11 | 1.07768 | 3.32772 | 0.25399 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 72069eca9de4b6389615baa5c863b99f128efc57cea50b83b056de51a702ba48; subject: b1-relative-mse.

**Method.** A 13,391-parameter F32full-state generator producing K=1complete pressure/velocity/tracer states per pass. The same width4 compact current-state encoder, pooled/coarse branch, cheap joint horizon heads and63non-DC learned pressure modes per output as the later protocol2 models. Future prescribed forces enter heads/pooling, and previous pressure is part of the current state. Pressure is previous normalized pressure plus a predicted correction, gauged at the first cell; velocities/tracer are increments from the original block state. Unpacking upcasts pressure toF64but internal prediction isF32. There is no pressure solver or sequential full physical integrator inside the network.

**Purpose.** Test replacing logarithmic error compression with relative MSE while retaining the exact compact K1model and corrected deployed feedback.

**Experiment.** Original16train-only scenes/2064states at8/16cells per metre and fixed1/60s, with four shapes including32cubed and rectangular domains. All original data/runtime slots match the previously independently decoded corpus. Fit uses10000CUDA Adam updates, batch1, private model/sample seeds and the recorded schedule. First2000updates supervise one block, later updates supervise two, for18000network calls/18000physical training predictions. The second call uses physical reconstruction and deployed mean-centered re-encoding of its own endpoint. All closure/typed recipe/output bytes, full initial/final tensors and PCG64sample/work counts verify.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 1.07768, 0.787739 and 0.008617. Median time is 3.55355 ms per K1 block (3.55355 ms per physical output); the worst case median is 4.02240 ms/output. The retained F64 N1 composition has better worst pressure error and 13.75 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** This retained design has higher worst pressure error and median GPU time than the matched F64 composition. Its method and fitting purpose remain recorded above; these measurements compare distinct designs and do not show that one checkpoint deteriorated. Judge complete physical outputs rather than a pressure-only component or a training-loss value.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-pair-01/train/receipt.json",
    "sha256": "b032c5e5e7166623a2a75139cc8ea75dd424eda1daf2cc4cae978f119bf1f5b8"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-pair-01/train/history.json",
    "sha256": "cae7365f4986a75042f7fc314ec4e68d9c3a0eedf00d6d0dc0901e4465c36694"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/adaptive_pool.py",
      "sha256": "759eb0e374518052bae51b6f96c89ea6465de5b0160ed5a94b05802d034cd96e"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/bundle_model.py",
      "sha256": "add1f6eda47133318eae626497bde98e36cad02e48fcf7d4a1b252149fca3a05"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/bundle_model_v5.py",
      "sha256": "e1191613b019e951d29121c66e4844bedf610531271961430b6dce29a7eb6c27"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/temporal_objective.py",
      "sha256": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/temporal_train_v2.py",
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
  "lesson": "The MSE and logarithmic loss scalars have different scales. An objective around0.015versus15is not an accuracy improvement by itself; use the common field/equation benchmark.",
  "outcome": "Original fit completes its declared10000updates; final logged composite objective 0.014938680455088615. Parameter count is independently rederived. Exact initial tensors match the corresponding later protocol2joint fit. No new common GPU accuracy, full-block latency or own-history drift is measured by this source review. Consumed-training diagnostics are not an independent holdout.",
  "reproduction": "Preserve exact captured model, objective, trainer, config, typed recipe/runtime, original16scene F32archive and normalization, private seeds/order and the historical feedback branch. Reproducing an old fit must not silently repair its raw-tensor training feedback. The original publication/restore recipes execute CPU neural probes; qualify a GPU-only reproduction before rerunning. New-Rheon regeneration, clean retraining and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-b032c5e5e7166623",
    "kind": "training_attempt",
    "name": "Full AI temporal b1-relative-mse train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "b032c5e5e7166623a2a75139cc8ea75dd424eda1daf2cc4cae978f119bf1f5b8",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b1-relative-mse-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "6ecc58eaa2454f32aaad08a722185cb5",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b1-relative-mse train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 13391,
    "training_closure": {
      "config_sha256": "8bb847d7b7bada8cec1ffc5c28ef3f419fc59a8dff392aa993c6a9eb68270340",
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
        "temporal_objective.py": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136",
        "temporal_train_v2.py": "3dd30c010d72ff2d5da02efcbefda6fec2eca3b8b35132c084476bf67b35b6bf"
      },
      "horizons": 1,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "e07086a3d790351eb7874c2ef92e46e4cf9adbeee581d26feecb72d9bf8b6074",
        "bytes": 57186,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "1ca0e61fb3a5de21f7ae5c5d1f0d15f774553c38b70052df1571a014eb0f29e7",
        "bytes": 57186,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/4"
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
      "receipt": "runs/full-ai-next-priorities-v1/b1-relative-mse-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-relative-mse-pair-01/train/receipt.json",
    "sha256": "b032c5e5e7166623a2a75139cc8ea75dd424eda1daf2cc4cae978f119bf1f5b8",
    "bytes": 9883
  },
  "bound_training_configuration": {
    "config_sha256": "8bb847d7b7bada8cec1ffc5c28ef3f419fc59a8dff392aa993c6a9eb68270340",
    "graph_id": "caf87c59f2c24b08b5c9125273a1e5ab",
    "graph_document_sha256": "f6f04c241899d12bbe070edb38c41f4f095335c65ecd7aa50279008bea8c2422",
    "recipe_path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/train-recipe.json",
    "recipe_sha256": "3a5d6bf5f1d56bf9cb26ce1b3a851e26999267739218986a5a60ff41956c6b85",
    "config_path": "runs/full-ai-next-priorities-v1/b1-relative-mse-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-v2",
      "feedback_protocol": 2,
      "identifier": "b1-relative-mse",
      "horizons": 1,
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
      "loss": "Relative squared errors without logarithmic compression; all original component weights retained",
      "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
      "followup_registration_sha256": "0725e1ec48dc1967ec2864f048cfcd6ecf87440981f7219bb7320a48e75e6412",
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
  "parameters": 13391,
  "horizons": 1,
  "closure": {
    "config_sha256": "8bb847d7b7bada8cec1ffc5c28ef3f419fc59a8dff392aa993c6a9eb68270340",
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
      "temporal_objective.py": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136",
      "temporal_train_v2.py": "3dd30c010d72ff2d5da02efcbefda6fec2eca3b8b35132c084476bf67b35b6bf"
    },
    "horizons": 1,
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
        5,
        7,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "24045c10c12a89f4c11e3b88ea34558fcdf926a8c1008cd08cc33bc71407c774"
    },
    "head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "de47c9b27eb8d300dbb5f2c353e632c393262cf06340c4fa7f1b40c4cbd36f90"
    },
    "modal.0.weight": {
      "shape": [
        32,
        297
      ],
      "dtype": "<f4",
      "sha256": "976bb42f4de95ce276c72de99aef5ebc515d136eeea8f935b483aa212e5bedfb"
    },
    "modal.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "4ec8ebc7df180ed7dcc125724318919e5f1b5bacacef7412d7f51d4c9db75273"
    },
    "modal.2.weight": {
      "shape": [
        63,
        32
      ],
      "dtype": "<f4",
      "sha256": "05bf9d87a6313c41625d93de63e30bc39be25e54b472f17e99d78cf12fccb3da"
    },
    "modal.2.bias": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "sha256": "913c6e2c32d99e4baff62cf421a494730cb043924f2c6bf46406573b59c641bd"
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
      "sha256": "fd6401003da78f755588382bdd3df768b480b3668f1cb1723714a2b396f7c68d"
    },
    "fine.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "61780767d49c628c26f0afbdd5af784c82be5088a46fcd97c0ee8f6c44f0cb35"
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
      "sha256": "96a80af79e71134c9333c2cb4cb59a2ba3e4bcc3c5abf86fb24f85a3ab1d4cfd"
    },
    "fine.2.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "8b48be52612066e18b537681dce8353d2658017267ef6bd58afc8ae45ba6e1c7"
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
      "sha256": "a8d7cb8e3ffb3fab025f0f26d276c986e66ef892b9c82e655408f8d14ee123dc"
    },
    "coarse.0.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "2c87514275c35c3e8578c03c2fa8ad3e5d53669b89563990a4626c966b07f849"
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
      "sha256": "9385147cbfe603638adfd22e66832abe2ed0c7164d8516324042251aa4ee75db"
    },
    "decode.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "a4d0e9a2680daf202f38434649e526b17644ae4b80b99bfcfefcf2311654354e"
    },
    "head.weight": {
      "shape": [
        5,
        7,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "1d827790f04bd8f7469e1f50e73dde504fb6b282ad18bc6b3b959784c4d9a05c"
    },
    "head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "b86fe003a69ff6f82088fb403b806c0e76c1e528823a2ba2db322ec2825cb935"
    },
    "modal.0.weight": {
      "shape": [
        32,
        297
      ],
      "dtype": "<f4",
      "sha256": "2d48357ded788feb19e3b462c29547fdff8589b149bb4e07a8132f3af62223d6"
    },
    "modal.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "c007ec709e68b30cf4af558f14fb328bcfefe974541df1f4b05e4183a65e29dc"
    },
    "modal.2.weight": {
      "shape": [
        63,
        32
      ],
      "dtype": "<f4",
      "sha256": "220058f7e5d9471b659a41b647cbc544facfe37c2ba042fbcd13e8fce5e66b51"
    },
    "modal.2.bias": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "sha256": "1d913e2ee7c5cbf7afbcaac4893d7f76fe53440dacf6425db0be4d931f87b09e"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 821363712
  },
  "outputs": {
    "probe.npy": {
      "bytes": 98388,
      "sha256": "da7cb16d1cc42ee8a9e024209b65d2accbb2a0be85a5fc455d4345350b4627f2"
    },
    "weights.npz": {
      "bytes": 57186,
      "sha256": "e07086a3d790351eb7874c2ef92e46e4cf9adbeee581d26feecb72d9bf8b6074"
    },
    "initial-weights.npz": {
      "bytes": 57186,
      "sha256": "1ca0e61fb3a5de21f7ae5c5d1f0d15f774553c38b70052df1571a014eb0f29e7"
    },
    "history.json": {
      "bytes": 271347,
      "sha256": "cae7365f4986a75042f7fc314ec4e68d9c3a0eedf00d6d0dc0901e4465c36694"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 195.67144052998628,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-v2",
  "feedback_protocol": 2,
  "identifier": "b1-relative-mse",
  "horizons": 1,
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
  "loss": "Relative squared errors without logarithmic compression; all original component weights retained",
  "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
  "followup_registration_sha256": "0725e1ec48dc1967ec2864f048cfcd6ecf87440981f7219bb7320a48e75e6412",
  "full_fit_registration_sha256": "08ed048038db26f1838a6e74f82ae521c848f06b954a42aaf3000a17794d0510"
}

```
