# Full AI temporal b4-reuse train

Retrospective source review · fit-926cf04f81244f0e

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 4; each output advances 1/60 second. Full block duration is 4/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.634168 | 1.22815 | 0.11509 |
| 8 | 16x16x16 | 0.477803 | 1.37380 | 0.14026 |
| 8 | 32x32x32 | 0.731722 | 1.51276 | 0.40021 |
| 8 | 9x11x11 | 0.320243 | 1.23464 | 0.11762 |
| 16 | 15x6x8 | 0.511165 | 1.32370 | 0.11392 |
| 16 | 16x16x16 | 0.760382 | 1.29674 | 0.13887 |
| 16 | 32x32x32 | 0.605807 | 1.41921 | 0.40363 |
| 16 | 9x11x11 | 0.760391 | 1.20478 | 0.11840 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 046d91f6248a32f47843817c6b8c4d87af71e309336e498b4ac3a50dbf0c2df6; subject: b4-reuse.

**Method.** A 163,212-parameter F32 width12 full-state generator for K=4. Current velocity/tracer plus geometry/current known force enter one spatial encoder; previous pressure is deliberately excluded. All K prescribed forces enter retained features. Two learned global blocks mix up to8cosine modes per axis across12latent channels, with learned7→32→144mode/geometry-conditioned weights; these are neural mixing weights, not inverse pressure eigenvalues. A first head predicts the first state. Each later output updates retained latent features using its preceding predicted five-field state and current known force, then applies a shared later head; the spatial encoder/global blocks are not rerun. Pressure is fresh raw output with cell-zero gauge; velocity/tracer increments always refer to the initial block state. Velocity walls are masked. Output packing upcasts pressure toF64, but prediction isF32. No numerical pressure solve or full physical integrator is called.

**Purpose.** Test reusing spatial/global features for 4 outputs through smaller recurrent latent updates instead of repeating the main encoder.

**Experiment.** All six fits use the same captured16train scenes/2064states at8/16cells per metre and fixed1/60s, with16cubed,9x11x11,15x6x8 and32cubed domains. Original F32trajectory shards/normalization/runtime slots match the independently decoded joint-generator corpus. Each fit performs10,000CUDA Adam updates: batch1, model seed2026101921/sample seed2026101922, rate0.001 with drops at6001/9001. First2000updates supervise one block; remaining8000supervise a second block from reconstructed own endpoint, producing18,000network calls. Every horizon receives the original composite field/equation/divergence/mass/energy/bounds objective; no separate high-accuracy-endpoint guarantee is imposed. Same-K endpoint/reuse pairs share exact initial encoder/forcing/global tensors. ChangingKchanges parameter shapes and random-number consumption; full initialization equality acrossKis not claimed.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 0.760391, 0.996917 and 0.0329065. Median time is 5.25789 ms per K4 block (1.31447 ms per physical output); the worst case median is 1.64829 ms/output. The retained F64 N1 composition has better worst pressure error and 10.20 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** Sharing an encoder, endpoint or retained features does not by itself establish a useful accuracy/speed tradeoff. This fit remains worse than the matched F64 composition in both worst pressure error and median GPU time. Account for every intermediate output and reconstruction cost; K outputs advance K full 1/60-second intervals.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b4-reuse-pair-01/train/receipt.json",
    "sha256": "926cf04f81244f0eb6b51b9a249092f04a62a79435b593c893983551eb6d9e8b"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b4-reuse-pair-01/train/history.json",
    "sha256": "06d971d8068ad96526be04d4367dc41e85837d360cf57946d1df81a605c2191b"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/bundle_model_sharing.py",
      "sha256": "3490bcffb6bfc9c19bb61613c0bd64065968adb27fa32a3e475f1e5e53850716"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/learned_global.py",
      "sha256": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/temporal_train_sharing.py",
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
  "lesson": "This implements retained-compute reuse with cheaper later latent updates. It remains sequential in latent predictions, but does not repeat the expensive spatial/global encoder per output. Later physical increments refer to the original block state; previous predicted state influences the latent update. Benchmark all intermediate fields and block/K; do not confuse this with Kfull integrator advances or endpoint interpolation.",
  "outcome": "Original fit completed10,000updates; final logged composite objective 18.871288299560547 with2supervised blocks. All original initial/final tensors, source/recipe/input/output identities and sample/work counts verify. No new complete-generator accuracy, GPU latency or own-history drift has been measured in this review. The composite loss cannot be read as pressure relative L2 or an inference-time score.",
  "reproduction": "Use exact original config/typed recipe/runtime/source closure, original16scene F32archive and normalization, zero output heads, private seeds/order and deployed reconstruction between supervised blocks. Preserve pressure-history exclusion, two learned global blocks and the selected endpoint/reuse branch. The older recipes perform CPU neural publication/restore probes; qualify GPU-only reproduction before rerunning. Clean retraining, new-Rheon data generation, fresh comparable performance and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-926cf04f81244f0e",
    "kind": "training_attempt",
    "name": "Full AI temporal b4-reuse train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "926cf04f81244f0eb6b51b9a249092f04a62a79435b593c893983551eb6d9e8b",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b4-reuse-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "f35927500e42487c84788e701fb34283",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b4-reuse train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 163212,
    "training_closure": {
      "config_sha256": "d1ab6b639ae530554da4ef70d3042186eceff4f499cc21cd87d2d1279fba0eff",
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
        "sha256": "ad109f17771dc530cc2a876f1b30cedca580f7ed6707d7ac79f19d1e1f60c94f",
        "bytes": 660594,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "c7bb98aab4bd02876cf6f0237147c9bd77c482d9bab4942c2c9ce4ec6d0e3782",
        "bytes": 660594,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/11"
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
      "receipt": "runs/full-ai-next-priorities-v1/b4-reuse-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b4-reuse-pair-01/train/receipt.json",
    "sha256": "926cf04f81244f0eb6b51b9a249092f04a62a79435b593c893983551eb6d9e8b",
    "bytes": 15741
  },
  "bound_training_configuration": {
    "config_sha256": "d1ab6b639ae530554da4ef70d3042186eceff4f499cc21cd87d2d1279fba0eff",
    "graph_id": "a344059766624f308bba0a5855ed37fa",
    "graph_document_sha256": "8130165bb8f4c164d1a970c28c3b8efcdabe74fef4cfb9aff69c108a9101bd0b",
    "recipe_path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/train-recipe.json",
    "recipe_sha256": "1ffc4dcab570ed828747601cabfd40a58bba892d719eb638a0c56ae7d6ae514c",
    "config_path": "runs/full-ai-next-priorities-v1/b4-reuse-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-sharing-v1",
      "architecture": "reuse",
      "feedback_protocol": 2,
      "identifier": "b4-reuse",
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
  "parameters": 163212,
  "horizons": 4,
  "closure": {
    "config_sha256": "d1ab6b639ae530554da4ef70d3042186eceff4f499cc21cd87d2d1279fba0eff",
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
    "first_head.weight": {
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
    "first_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "de47c9b27eb8d300dbb5f2c353e632c393262cf06340c4fa7f1b40c4cbd36f90"
    },
    "latent_update.0.weight": {
      "shape": [
        12,
        20,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "653893002ca7fb0159e629e4837da6a67dcf729fa7adab05a444960fb0f4d4c7"
    },
    "latent_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "38a2ea6c23242da05ee2a8093643804b5db28e659479c414e4bed6d8e4b1b70f"
    },
    "latent_update.2.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "70fa4d1bb25314cb4abce15367eda04a0b5e6a2cda04a559f8696595abd3d44f"
    },
    "latent_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "bf6f1d18c6b23ff90559eb04342d249fce1f8bc178386457dbbba1c48c851e1e"
    },
    "later_head.weight": {
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
    "later_head.bias": {
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
      "sha256": "b1b037929d2b58bcd974241f561623d3a1058d6894f65656b809d86bd75e2a8d"
    },
    "fine.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "a0642a997376a1f28b23ec82f16ba18ee7e87cba405130c2def857c191c7d8b4"
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
      "sha256": "9f0124884967009f0fe78dafee4af1d236809d5956e1b5d44fedcf379ef4f95b"
    },
    "fine.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "e6453b6d4fd3cf4a8957b4f9860c4943afe5379fc5a72782b612c1775a5c1016"
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
      "sha256": "4871e055156e89b04759a2af66223e91555962c35ac71c017609908cb47f2be1"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "73d445419c3c23f0f195ed371bb34fda47d001098e0694c3d11a9bd08edbaa26"
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
      "sha256": "e3a31070891b3f0700063b057bf8479b08a64888bd09641fe407decd7c5ef428"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "cd06ed1641bef944dff17257772212330f866c3885387d9f3ca83897bdad7ff6"
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
      "sha256": "4c4ac2a6a8a5821f7ece0d5875c1980aede4180d8557668b9872f3f754de3e07"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "e3f59c6bac9933492107118ed0cba36c6ebd6755176095e1d440530293225f2d"
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
      "sha256": "ed6cdbe6fe9fba2e7240d36314b27fdcf8e7349b6314f5cec62af5002c585dfb"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "8ec0126160cefafb3f071a621369d0df0f8ff56dc9f2a11600686c0be4996c28"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "1a464a7026e50ac6b2f164b94190bbaa4faf8baa0814c485cad5381bb2a2b568"
    },
    "global_blocks.0.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "2e78440a7647d5760ca770c1de613d1da513b683908e263fe443487b15aeef5f"
    },
    "global_blocks.0.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "b01a588a836b5e60f0d55290f37fe32ecd12d2bdbd9acc7be26f3a2c35eeafc5"
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
      "sha256": "4515349e78e66e44f91ab7016964d6c1c0e50600cb8c9ec32e23d441aef599b7"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "04dfa7e04ccce8615db57db641a93553589c92596458851931e774369b747ad0"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "399df6f4f9f88f2eb68ffa13123540f6b6967db533ab94eab3f066862d7f6ac0"
    },
    "global_blocks.1.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "a1d73d1f609897b98ee56aff491273b8bd34492cbfdec9335b9a80e669a2a313"
    },
    "global_blocks.1.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "c77ef5babb7bf73c242a1f2b0f4420865b97d0a70c90ed07d237e1146996eb69"
    },
    "first_head.weight": {
      "shape": [
        5,
        15,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "898c9be51a61b62367e66145dad76f11ad5b7abe5330a7b865b64889829f36b7"
    },
    "first_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "f03096dec37db49c0880e220e7625bd4d5e235e516aac316c59741383ca0b9a9"
    },
    "latent_update.0.weight": {
      "shape": [
        12,
        20,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "ec0f9802a98996215c911fe04a11ab9e8d1a2293fc501ecaf4662509acaffc52"
    },
    "latent_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "5aeaa0eb2b28750d2796811ac6029b74f66ef5905b219c9ed0075ea42bc86c78"
    },
    "latent_update.2.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "b659bfba6bcbc15d0bf860e80788f5b4b5967ba3316d664d06d9f8d8dbe7110a"
    },
    "latent_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "094716aec1c691ec65ab7c68e395d0d94b6cf7ce7555c33241c9b6ee893a1778"
    },
    "later_head.weight": {
      "shape": [
        5,
        15,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "54b794ce524602c81812b2b4cd529febd471eba4bde82a596c41e25b75f3ced9"
    },
    "later_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "97d34f9c567ba6e084f1abb5151a47c6117e418174547b37255f15ee89da15a3"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 925770752
  },
  "outputs": {
    "probe.npy": {
      "bytes": 393168,
      "sha256": "fe37e23521c1ee044b85ae0105941364fbd1ee6e074a2af86882fac01f00b18b"
    },
    "weights.npz": {
      "bytes": 660594,
      "sha256": "ad109f17771dc530cc2a876f1b30cedca580f7ed6707d7ac79f19d1e1f60c94f"
    },
    "initial-weights.npz": {
      "bytes": 660594,
      "sha256": "c7bb98aab4bd02876cf6f0237147c9bd77c482d9bab4942c2c9ce4ec6d0e3782"
    },
    "history.json": {
      "bytes": 274421,
      "sha256": "06d971d8068ad96526be04d4367dc41e85837d360cf57946d1df81a605c2191b"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 328.40343528601807,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-sharing-v1",
  "architecture": "reuse",
  "feedback_protocol": 2,
  "identifier": "b4-reuse",
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
