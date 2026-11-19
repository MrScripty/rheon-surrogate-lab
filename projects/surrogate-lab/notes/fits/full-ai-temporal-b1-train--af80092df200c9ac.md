# Full AI temporal b1 train

Retrospective source review · fit-af80092df200c9ac

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 1; each output advances 1/60 second. Full block duration is 1/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.984018 | 3.52443 | 0.25071 |
| 8 | 16x16x16 | 0.975918 | 3.62764 | 0.27377 |
| 8 | 32x32x32 | 0.968822 | 3.98150 | 0.58808 |
| 8 | 9x11x11 | 1.01245 | 3.65262 | 0.25054 |
| 16 | 15x6x8 | 0.939939 | 3.75849 | 0.24466 |
| 16 | 16x16x16 | 1.03218 | 3.49732 | 0.26408 |
| 16 | 32x32x32 | 1.02133 | 3.95113 | 0.58521 |
| 16 | 9x11x11 | 1.00322 | 3.52026 | 0.25399 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 72069eca9de4b6389615baa5c863b99f128efc57cea50b83b056de51a702ba48; subject: b1-v1.

**Method.** A 13,391-parameter F32full-state generator producing K=1complete pressure/velocity/tracer states per pass. The same width4 compact current-state encoder, pooled/coarse branch, cheap joint horizon heads and63non-DC learned pressure modes per output as the later protocol2 models. Future prescribed forces enter heads/pooling, and previous pressure is part of the current state. Pressure is previous normalized pressure plus a predicted correction, gauged at the first cell; velocities/tracer are increments from the original block state. Unpacking upcasts pressure toF64but internal prediction isF32. There is no pressure solver or sequential full physical integrator inside the network.

**Purpose.** Original fully trained K1direct-generator control before deployment-equivalent feedback encoding was introduced.

**Experiment.** Original16train-only scenes/2064states at8/16cells per metre and fixed1/60s, with four shapes including32cubed and rectangular domains. All original data/runtime slots match the previously independently decoded corpus. Fit uses10000CUDA Adam updates, batch1, private model/sample seeds and the recorded schedule. First2000updates supervise one block, later updates supervise two, for18000network calls/18000physical training predictions. The second call directly consumes predicted[:,-1]without deployed physical unpack/repack. All closure/typed recipe/output bytes, full initial/final tensors and PCG64sample/work counts verify.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 1.03218, 0.75034 and 0.00860459. Median time is 3.67141 ms per K1 block (3.67141 ms per physical output); the worst case median is 4.13148 ms/output. The retained F64 N1 composition has better worst pressure error and 14.21 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** This retained design has higher worst pressure error and median GPU time than the matched F64 composition. Its method and fitting purpose remain recorded above; these measurements compare distinct designs and do not show that one checkpoint deteriorated. Judge complete physical outputs rather than a pressure-only component or a training-loss value.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-pair-01/train/receipt.json",
    "sha256": "af80092df200c9acad88789b6d271ab99e8f0a24c10fdb0181ac86c6af7e6ba8"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b1-pair-01/train/history.json",
    "sha256": "46934e641e5f3967269f984d36cd69c73751d2e44ed05dbd45bf839e200e8263"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b1-project-01/adaptive_pool.py",
      "sha256": "759eb0e374518052bae51b6f96c89ea6465de5b0160ed5a94b05802d034cd96e"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-project-01/bundle_model.py",
      "sha256": "add1f6eda47133318eae626497bde98e36cad02e48fcf7d4a1b252149fca3a05"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-project-01/bundle_model_v5.py",
      "sha256": "e1191613b019e951d29121c66e4844bedf610531271961430b6dce29a7eb6c27"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-project-01/temporal_train.py",
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
  "lesson": "Training’s second block consumes a raw predicted normalized tensor, while deployment reconstructs native physical fields then repacks/mean-centers pressure. Preserve this historical discrepancy when reproducing the fit; the later protocol2 control explicitly changes that boundary.",
  "outcome": "Original fit completes its declared10000updates; final logged composite objective 15.090801239013672. Parameter count is independently rederived. Exact initial tensors match the corresponding later protocol2joint fit. No new common GPU accuracy, full-block latency or own-history drift is measured by this source review. Consumed-training diagnostics are not an independent holdout.",
  "reproduction": "Preserve exact captured model, objective, trainer, config, typed recipe/runtime, original16scene F32archive and normalization, private seeds/order and the historical feedback branch. Reproducing an old fit must not silently repair its raw-tensor training feedback. The original publication/restore recipes execute CPU neural probes; qualify a GPU-only reproduction before rerunning. New-Rheon regeneration, clean retraining and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-af80092df200c9ac",
    "kind": "training_attempt",
    "name": "Full AI temporal b1 train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "af80092df200c9acad88789b6d271ab99e8f0a24c10fdb0181ac86c6af7e6ba8",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b1-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "d57e5abddfa14b1db920cdb4f0d92316",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b1 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 13391,
    "training_closure": {
      "config_sha256": "52b6181467bee85700fdebf73c97f41cc46d00e7e485f43bfa0d022133850905",
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
      "horizons": 1,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "4523e85c021f176883bc596f6ae26babb13a10de1c374d45242021b4240f9197",
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
      "receipt": "runs/full-ai-next-priorities-v1/b1-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-pair-01/train/receipt.json",
    "sha256": "af80092df200c9acad88789b6d271ab99e8f0a24c10fdb0181ac86c6af7e6ba8",
    "bytes": 9782
  },
  "bound_training_configuration": {
    "config_sha256": "52b6181467bee85700fdebf73c97f41cc46d00e7e485f43bfa0d022133850905",
    "graph_id": "fd8cc678be9f4ed889a57defc598e9e7",
    "graph_document_sha256": "c54515027d9786818d12fbf22e98a8101155c9e07bda60f13f2211c1f8c688d4",
    "recipe_path": "runs/full-ai-next-priorities-v1/b1-project-01/train-recipe.json",
    "recipe_sha256": "428269524674db613278775d503e230e2c9d83f15df9820257596b64d69b0937",
    "config_path": "runs/full-ai-next-priorities-v1/b1-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-v1",
      "identifier": "b1",
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
      "loss": "Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds",
      "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
      "full_fit_registration_sha256": "1a6a58168fd6a631f87c443804cf74077b7f76a8f80ecf2ea03a25e92269e438"
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
    "config_sha256": "52b6181467bee85700fdebf73c97f41cc46d00e7e485f43bfa0d022133850905",
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
      "sha256": "11df7517d03bba9c8a538ccef5ad37bf29328235bf1c214049f56b6a6c7b1b86"
    },
    "fine.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "575f7ede0c8dcb0a02ceae9e152fe54c115ed0caa7257b6a1a99cf62a4d7ba66"
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
      "sha256": "1ff7cb9946c3429d4978c5d5cf05dd17f6df50ee80c500ee55c6a74cf41aa153"
    },
    "fine.2.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "2c94fb92eb594fc7bdb3eb631a8b8f6d7165d39e248b41b7279e9d849cf68078"
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
      "sha256": "5ff4d6d7bb5407d6594ac80c529aae48cdb0127cb8a3f66538fb37ec582436d2"
    },
    "coarse.0.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "9ad033fa30dae821a793e057b38a72efe0d07d04d88a94491fdf44cab14e742e"
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
      "sha256": "fd307c8ff27d5072bd312a79de6f8b1152f7ec48ce9cdb75cc5117dcf16fed18"
    },
    "decode.0.bias": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "c10f186a18e96f1ee8bfd0d449d92456d0dcd43797650d4bec2e234679e1ffbf"
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
      "sha256": "431cc106d80d85bb6c7257fe0f8a47b536e73e49b41b9ed6c0e9a0ca6a2c4ff8"
    },
    "head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "b069de889316f427b1d0e5b4e91734c685d35c2f822180856b27b4ed7a96ea09"
    },
    "modal.0.weight": {
      "shape": [
        32,
        297
      ],
      "dtype": "<f4",
      "sha256": "0e1d31e90aaaa3d883570c12fc69155dbb25160e323517b6cf1c743e3155fad0"
    },
    "modal.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "0735698e2eaf34e9ad62db9ae233de678b78fc91a8cb3a081038cb4ad4d86e27"
    },
    "modal.2.weight": {
      "shape": [
        63,
        32
      ],
      "dtype": "<f4",
      "sha256": "be35ad6622aed8bc3791c0443103b7b259b534a94294ab6f614969f6cbd7128e"
    },
    "modal.2.bias": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "sha256": "80fe624d76ba9294a2361cd2bce9056897b1d8f4b9d2fcaa5fc87d9cef905b00"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 820640768
  },
  "outputs": {
    "probe.npy": {
      "bytes": 98388,
      "sha256": "0864ffb0ee6193bc548a7339b61778c73432cdb55f1f42b43ee02b6e52d6e3ea"
    },
    "weights.npz": {
      "bytes": 57186,
      "sha256": "4523e85c021f176883bc596f6ae26babb13a10de1c374d45242021b4240f9197"
    },
    "initial-weights.npz": {
      "bytes": 57186,
      "sha256": "1ca0e61fb3a5de21f7ae5c5d1f0d15f774553c38b70052df1571a014eb0f29e7"
    },
    "history.json": {
      "bytes": 271093,
      "sha256": "46934e641e5f3967269f984d36cd69c73751d2e44ed05dbd45bf839e200e8263"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 225.88652067596558,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-v1",
  "identifier": "b1",
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
  "loss": "Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds",
  "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
  "full_fit_registration_sha256": "1a6a58168fd6a631f87c443804cf74077b7f76a8f80ecf2ea03a25e92269e438"
}

```
