# Full AI temporal b2-reuse train

Retrospective source review · fit-acb5d76a1aafff40

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 2; each output advances 1/60 second. Full block duration is 2/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.623058 | 2.25571 | 0.15940 |
| 8 | 16x16x16 | 0.301347 | 2.18501 | 0.19605 |
| 8 | 32x32x32 | 0.667604 | 2.65222 | 0.47160 |
| 8 | 9x11x11 | 0.354579 | 2.38791 | 0.16498 |
| 16 | 15x6x8 | 0.426591 | 2.33405 | 0.16040 |
| 16 | 16x16x16 | 0.655808 | 2.31100 | 0.19602 |
| 16 | 32x32x32 | 0.590769 | 2.34248 | 0.47723 |
| 16 | 9x11x11 | 0.740472 | 2.40157 | 0.16505 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 480d903fd12f64c7130577e176878296432640a98f7d5b9471f0d8929f641841; subject: b2-reuse.

**Method.** A 163,140-parameter F32 width12 full-state generator for K=2. Current velocity/tracer plus geometry/current known force enter one spatial encoder; previous pressure is deliberately excluded. All K prescribed forces enter retained features. Two learned global blocks mix up to8cosine modes per axis across12latent channels, with learned7→32→144mode/geometry-conditioned weights; these are neural mixing weights, not inverse pressure eigenvalues. A first head predicts the first state. Each later output updates retained latent features using its preceding predicted five-field state and current known force, then applies a shared later head; the spatial encoder/global blocks are not rerun. Pressure is fresh raw output with cell-zero gauge; velocity/tracer increments always refer to the initial block state. Velocity walls are masked. Output packing upcasts pressure toF64, but prediction isF32. No numerical pressure solve or full physical integrator is called.

**Purpose.** Test reusing spatial/global features for 2 outputs through smaller recurrent latent updates instead of repeating the main encoder.

**Experiment.** All six fits use the same captured16train scenes/2064states at8/16cells per metre and fixed1/60s, with16cubed,9x11x11,15x6x8 and32cubed domains. Original F32trajectory shards/normalization/runtime slots match the independently decoded joint-generator corpus. Each fit performs10,000CUDA Adam updates: batch1, model seed2026101921/sample seed2026101922, rate0.001 with drops at6001/9001. First2000updates supervise one block; remaining8000supervise a second block from reconstructed own endpoint, producing18,000network calls. Every horizon receives the original composite field/equation/divergence/mass/energy/bounds objective; no separate high-accuracy-endpoint guarantee is imposed. Same-K endpoint/reuse pairs share exact initial encoder/forcing/global tensors. ChangingKchanges parameter shapes and random-number consumption; full initialization equality acrossKis not claimed.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 0.740472, 0.982428 and 0.0165032. Median time is 4.69466 ms per K2 block (2.34733 ms per physical output); the worst case median is 2.71809 ms/output. The retained F64 N1 composition has better worst pressure error and 13.09 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** Sharing an encoder, endpoint or retained features does not by itself establish a useful accuracy/speed tradeoff. This fit remains worse than the matched F64 composition in both worst pressure error and median GPU time. Account for every intermediate output and reconstruction cost; K outputs advance K full 1/60-second intervals.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-reuse-pair-01/train/receipt.json",
    "sha256": "acb5d76a1aafff403e5a28cd73bed1c879e5316cf9ea3b05920380b6ce6e99c7"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b2-reuse-pair-01/train/history.json",
    "sha256": "23687a6e0cfd256f2bf4f4ffa3a0e1395d0d293a303d3e6ce6188f2b560d355c"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/bundle_model_sharing.py",
      "sha256": "3490bcffb6bfc9c19bb61613c0bd64065968adb27fa32a3e475f1e5e53850716"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/learned_global.py",
      "sha256": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/temporal_train_sharing.py",
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
  "outcome": "Original fit completed10,000updates; final logged composite objective 18.190547943115234 with2supervised blocks. All original initial/final tensors, source/recipe/input/output identities and sample/work counts verify. No new complete-generator accuracy, GPU latency or own-history drift has been measured in this review. The composite loss cannot be read as pressure relative L2 or an inference-time score.",
  "reproduction": "Use exact original config/typed recipe/runtime/source closure, original16scene F32archive and normalization, zero output heads, private seeds/order and deployed reconstruction between supervised blocks. Preserve pressure-history exclusion, two learned global blocks and the selected endpoint/reuse branch. The older recipes perform CPU neural publication/restore probes; qualify GPU-only reproduction before rerunning. Clean retraining, new-Rheon data generation, fresh comparable performance and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-acb5d76a1aafff40",
    "kind": "training_attempt",
    "name": "Full AI temporal b2-reuse train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "acb5d76a1aafff403e5a28cd73bed1c879e5316cf9ea3b05920380b6ce6e99c7",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b2-reuse-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "05be98bfd7154b348e7e1afd00639264",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b2-reuse train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 163140,
    "training_closure": {
      "config_sha256": "8ace95c08d4db54dcf8f423a30cd73fdf09395decb8d1454be0d57d5ba652d4e",
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
      "horizons": 2,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "3f35d70adaf778dca224d2857f2f0d5403258d7879df29e35a82da1086d42b16",
        "bytes": 660306,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "c1438c788c03f4aeec90c6e6d578fedff77bb400aca3effc9a956f8558236c22",
        "bytes": 660306,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/7"
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
      "receipt": "runs/full-ai-next-priorities-v1/b2-reuse-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-reuse-pair-01/train/receipt.json",
    "sha256": "acb5d76a1aafff403e5a28cd73bed1c879e5316cf9ea3b05920380b6ce6e99c7",
    "bytes": 15737
  },
  "bound_training_configuration": {
    "config_sha256": "8ace95c08d4db54dcf8f423a30cd73fdf09395decb8d1454be0d57d5ba652d4e",
    "graph_id": "2001910892034fba9c01f179ceadb5ba",
    "graph_document_sha256": "185d1f8d493470599585371713287553b12efcd0707c6d641c0a464641071584",
    "recipe_path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/train-recipe.json",
    "recipe_sha256": "0e47a938bae6e81fe93358315a1b0ed77794e92ea5ea48c74e0fa1f8f9fcbbab",
    "config_path": "runs/full-ai-next-priorities-v1/b2-reuse-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-sharing-v1",
      "architecture": "reuse",
      "feedback_protocol": 2,
      "identifier": "b2-reuse",
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
  "parameters": 163140,
  "horizons": 2,
  "closure": {
    "config_sha256": "8ace95c08d4db54dcf8f423a30cd73fdf09395decb8d1454be0d57d5ba652d4e",
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
      "sha256": "7bca5f9fa74fba98060343e6b887bd049230f6a8f7b41763ef631fa8e0f48b25"
    },
    "latent_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "1d6b3c0fcbe2d1601aeec218aa4a01577752cbc281e88ec946a7acce4d9f1848"
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
      "sha256": "580792d31623e6be1b0cca93dbd89a971170df06b4b81a1b81f5461489b03062"
    },
    "latent_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "2c8287ddc163ac78d13cdde6ba0df05da08fcbc6fedeb0d28adcfcef4a8aa527"
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
      "sha256": "20e70ae3b4fe9edfdb480b998bb1e46fc6f677453fdb292765f2fa1b2204dc52"
    },
    "fine.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "6f75d75fbbd2c520029ff21951a8ee75ed666cea70938e38354ab66181f94ed2"
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
      "sha256": "30bfe7626c3603982755d7a60089c676f50d3d1e6d9c71b6d56b1ca6683c9be6"
    },
    "fine.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "e7337317a3b75359b649a5133d04bd4207873e26bb86eb6ee96808fa737043ea"
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
      "sha256": "aa742e99fb532ae7b8e1a8ce1437ac10cbc7e9e81f2216b743eb0be2f548d23d"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "ec706d5748e52aca46d11c46ab46bd0b91de9df1e276d0b34c36332019af48fd"
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
      "sha256": "bcb70f9cb36751d2695ae1627d5296d9387d6a506d31e407f0fae10b16618dc4"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "c86b2502fcb74bf3918e7800819bdb818689eb0126e646663ffb8f8f4df1f9fc"
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
      "sha256": "756ceb800239a395cd7ad7fa5f4e9a043d3a5a4b5c05d7d576a1193542321ec5"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "db9ff952180b8874b1aaef83c6bfd6224a05217a0f2ea8961b36287ed8d0de90"
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
      "sha256": "5c4175d6618b6971667c946dd03ef94b990ad6404a5eb31a9537779cc41f1559"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "42661a66fbb830c6bd2a4db22c7e945481f2ad336810a3a713241849769d6a13"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "db1ac580b7685864c93f17e1c05b1c73174d61bce5e0e35ec71253fdc1a50fa7"
    },
    "global_blocks.0.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "2f08d67b7f501b135e19967dff5f63974688a6180a9ba998a05de276ec524ba5"
    },
    "global_blocks.0.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "9141dbab91dac04ba6b08f5f0dc62a8e193a4ac6ee7cce1f1713893969c3566f"
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
      "sha256": "46d56d8926ac374707b55192d712a2e006b3b5f0f63aefd5ca8438fdcc4ffbd0"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "b8668019039dcf6cd424aafdda847b4ba4d7c41143fc06300ec74385d3671ad7"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "bb72c010e7f327a0236073752e1db4bfd9d82ce5892e47d8dcccdaee0dfaa3ab"
    },
    "global_blocks.1.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "87aace129edd8cd2e05996bb391491c074311a9cc0a24f115ba7de2ec5ceab21"
    },
    "global_blocks.1.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "77db76bf5f3309db29ca83fd64f1f3ae8e9ed99b2ff6004988dbe4bdf4eaa827"
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
      "sha256": "983169504db00383a149c45dcb76ad787c04d12f3f151aaa54ef773d74518540"
    },
    "first_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "946e7991d7c95bb01a890aaabe98b610f3ae864db562a2acd94cf99d868026df"
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
      "sha256": "7dfed836463f4455dc3c4de4fbd7f52ca497910058f0cfdd6686b5e2c790eefa"
    },
    "latent_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "66600d00b7a46b8f88ea3699f6406552f123099d1ee7bc83304909716d570f55"
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
      "sha256": "792a14cca2ec7113d7c02732cbcc702003b1b7073bc00d5f2c7c70016295b481"
    },
    "latent_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "6c40605858227ab02489a0951f2f9300be8fdf4106f052ced242c5af6dc00c9f"
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
      "sha256": "4ea75f10f04f5d79ee817efd8414dba4caf8b212b1a6cdb1f95dab1c6c6fdad1"
    },
    "later_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "93010afbb33ddf37dcfb578adeb8ac15f7c659004c9b62ba014099225f39a9e3"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 874064896
  },
  "outputs": {
    "probe.npy": {
      "bytes": 196648,
      "sha256": "d10cc69b9b3448ab38cb426770d8f5f3edeabce518fdef9fa7c70df1b328fdc5"
    },
    "weights.npz": {
      "bytes": 660306,
      "sha256": "3f35d70adaf778dca224d2857f2f0d5403258d7879df29e35a82da1086d42b16"
    },
    "initial-weights.npz": {
      "bytes": 660306,
      "sha256": "c1438c788c03f4aeec90c6e6d578fedff77bb400aca3effc9a956f8558236c22"
    },
    "history.json": {
      "bytes": 274707,
      "sha256": "23687a6e0cfd256f2bf4f4ffa3a0e1395d0d293a303d3e6ce6188f2b560d355c"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 329.645135393017,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-sharing-v1",
  "architecture": "reuse",
  "feedback_protocol": 2,
  "identifier": "b2-reuse",
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
  "loss": "Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds",
  "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
  "sharing_registration_sha256": "788464651edff12d25ec3c847b27a4091b84c49382af66d139d563a7e0052983",
  "full_fit_registration_sha256": "08ed048038db26f1838a6e74f82ae521c848f06b954a42aaf3000a17794d0510"
}

```
