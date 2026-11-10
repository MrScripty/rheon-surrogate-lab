# Full AI temporal b3-reuse train

Retrospective source review · fit-d02a7f586d20d8b5

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 3; each output advances 1/60 second. Full block duration is 3/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.617925 | 1.68058 | 0.12737 |
| 8 | 16x16x16 | 0.397359 | 1.60924 | 0.15059 |
| 8 | 32x32x32 | 0.679757 | 1.82433 | 0.42505 |
| 8 | 9x11x11 | 0.384729 | 1.69599 | 0.13000 |
| 16 | 15x6x8 | 0.588937 | 1.72878 | 0.12743 |
| 16 | 16x16x16 | 0.7362 | 1.71341 | 0.15102 |
| 16 | 32x32x32 | 0.64936 | 1.80612 | 0.43946 |
| 16 | 9x11x11 | 0.623457 | 1.67977 | 0.12982 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 41c5dc934da61e9de5cdc94a6638c609f4202e5b6fc267eb4fb21dc15a79d761; subject: b3-reuse.

**Method.** A 163,176-parameter F32 width12 full-state generator for K=3. Current velocity/tracer plus geometry/current known force enter one spatial encoder; previous pressure is deliberately excluded. All K prescribed forces enter retained features. Two learned global blocks mix up to8cosine modes per axis across12latent channels, with learned7→32→144mode/geometry-conditioned weights; these are neural mixing weights, not inverse pressure eigenvalues. A first head predicts the first state. Each later output updates retained latent features using its preceding predicted five-field state and current known force, then applies a shared later head; the spatial encoder/global blocks are not rerun. Pressure is fresh raw output with cell-zero gauge; velocity/tracer increments always refer to the initial block state. Velocity walls are masked. Output packing upcasts pressure toF64, but prediction isF32. No numerical pressure solve or full physical integrator is called.

**Purpose.** Test reusing spatial/global features for 3 outputs through smaller recurrent latent updates instead of repeating the main encoder.

**Experiment.** All six fits use the same captured16train scenes/2064states at8/16cells per metre and fixed1/60s, with16cubed,9x11x11,15x6x8 and32cubed domains. Original F32trajectory shards/normalization/runtime slots match the independently decoded joint-generator corpus. Each fit performs10,000CUDA Adam updates: batch1, model seed2026101921/sample seed2026101922, rate0.001 with drops at6001/9001. First2000updates supervise one block; remaining8000supervise a second block from reconstructed own endpoint, producing18,000network calls. Every horizon receives the original composite field/equation/divergence/mass/energy/bounds objective; no separate high-accuracy-endpoint guarantee is imposed. Same-K endpoint/reuse pairs share exact initial encoder/forcing/global tensors. ChangingKchanges parameter shapes and random-number consumption; full initialization equality acrossKis not claimed.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 0.7362, 0.996299 and 0.0247787. Median time is 5.14098 ms per K3 block (1.71366 ms per physical output); the worst case median is 1.93562 ms/output. The retained F64 N1 composition has better worst pressure error and 12.14 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** Sharing an encoder, endpoint or retained features does not by itself establish a useful accuracy/speed tradeoff. This fit remains worse than the matched F64 composition in both worst pressure error and median GPU time. Account for every intermediate output and reconstruction cost; K outputs advance K full 1/60-second intervals.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b3-reuse-pair-01/train/receipt.json",
    "sha256": "d02a7f586d20d8b5e92977b9d1d7bec37288fb491ebb0d0f7bf6940469f21967"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b3-reuse-pair-01/train/history.json",
    "sha256": "ddc29a642421e6bf15564d5e688a430dd2cf084bd1eb2dff5609cf45771534f9"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/bundle_model_sharing.py",
      "sha256": "3490bcffb6bfc9c19bb61613c0bd64065968adb27fa32a3e475f1e5e53850716"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/learned_global.py",
      "sha256": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/temporal_train_sharing.py",
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
  "outcome": "Original fit completed10,000updates; final logged composite objective 18.263519287109375 with2supervised blocks. All original initial/final tensors, source/recipe/input/output identities and sample/work counts verify. No new complete-generator accuracy, GPU latency or own-history drift has been measured in this review. The composite loss cannot be read as pressure relative L2 or an inference-time score.",
  "reproduction": "Use exact original config/typed recipe/runtime/source closure, original16scene F32archive and normalization, zero output heads, private seeds/order and deployed reconstruction between supervised blocks. Preserve pressure-history exclusion, two learned global blocks and the selected endpoint/reuse branch. The older recipes perform CPU neural publication/restore probes; qualify GPU-only reproduction before rerunning. Clean retraining, new-Rheon data generation, fresh comparable performance and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-d02a7f586d20d8b5",
    "kind": "training_attempt",
    "name": "Full AI temporal b3-reuse train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "d02a7f586d20d8b5e92977b9d1d7bec37288fb491ebb0d0f7bf6940469f21967",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b3-reuse-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "ee52015dc0794386b088717ba4224abb",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b3-reuse train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 163176,
    "training_closure": {
      "config_sha256": "4423c2efa270792fa71ce164b71e17aa09e4d7da2dd02677b200535ea66105b6",
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
      "horizons": 3,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "1476e5ddc3705fbc20ec08c7741ce78d8fabcbfd8e6b3b76660a961376db88f8",
        "bytes": 660450,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "00c000a1381b233350c7ceafe5c70f2eb396ebd805f3ed9753f936a70d9fcfc0",
        "bytes": 660450,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/9"
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
      "receipt": "runs/full-ai-next-priorities-v1/b3-reuse-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b3-reuse-pair-01/train/receipt.json",
    "sha256": "d02a7f586d20d8b5e92977b9d1d7bec37288fb491ebb0d0f7bf6940469f21967",
    "bytes": 15738
  },
  "bound_training_configuration": {
    "config_sha256": "4423c2efa270792fa71ce164b71e17aa09e4d7da2dd02677b200535ea66105b6",
    "graph_id": "7eb55cde998d47139ebf11165061b8eb",
    "graph_document_sha256": "6a318c599d6a163aaefd1651d76125915498ea00dc348469ccbc3ef26394fbbf",
    "recipe_path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/train-recipe.json",
    "recipe_sha256": "0a880e67e6fbf10fa34e7cb6bf575ae5183d7976f3455473d7c72b37bf208ea0",
    "config_path": "runs/full-ai-next-priorities-v1/b3-reuse-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-sharing-v1",
      "architecture": "reuse",
      "feedback_protocol": 2,
      "identifier": "b3-reuse",
      "horizons": 3,
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
  "parameters": 163176,
  "horizons": 3,
  "closure": {
    "config_sha256": "4423c2efa270792fa71ce164b71e17aa09e4d7da2dd02677b200535ea66105b6",
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
    "horizons": 3,
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
        9,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "9a0a8809ff0d04a789a3b964618f3ff4563f26437538238c3e28ba3db660db10"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "711f0165c1c46118989b94866e9535bb5148c1510c230e69a9682c47bbeb4632"
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
      "sha256": "32680da155c0904c69c65cf9b1079951a878be90c022c8a643452d762c104937"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "6a1d141b6fe61b111290e26f34cd94b0c6abca4626e04a6ad6334e4ae6aa2d2a"
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
      "sha256": "43e08e2c68a4f1b753cbfefab67478b0365edf6ba9e378a96cee96e58deea523"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "a6c3445403c24f6d6a8ea1da972d6391e5a4a5b329690f3e7b64c445385c18f6"
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
      "sha256": "d2f81937e6485fa2e8d872ebbd60fa4e4308d8a7671b4f6dfce9386ca977a70b"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "81b9a4fc399eb6d774676133e9a98e9b9a000282436da4903a36d8b0e479c1be"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "0772f9c7b80558a946d8e4dabf29b54804b8056e565cb89f730f80ba7a9987ec"
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
      "sha256": "822becfce7c00289e390f0a585486414d199826babc3fa14dcff2cfe2496a079"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "729f66f4a64ede78edbdd65d56ff8ee5e6ebb2b1d8c814455835e1915d857bd9"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "6dbfaaf242f49d447d47114d1557de602135bd6a067a0983d77862df806a3ca6"
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
      "sha256": "7b5e10056f5ba20f2ffc37e641585a374a0ddffb60757839186691b2f2690ff6"
    },
    "latent_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "9c21482ae10247ba19d9ad347cb98f344d22414685bfae1c8d63fbe3ea4f9597"
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
      "sha256": "c0fcb240170d348db677b0243b27545620a13536fd5ea495afc654c1c67746d7"
    },
    "latent_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "ea38515daf7eed9842e6ca22084f9f1289c083a9d7ec0cecce8bb60769496a21"
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
      "sha256": "ba9c054df94b13c46d9a47912e78dd792cb6f5feb1466b7df7e0dc9328e04b6c"
    },
    "fine.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "424f71b15c2e45a5de11e90c24e8d29c2fe2fc38f9140911dc3a58691a5fe00f"
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
      "sha256": "28d49ea5b9f9e51b93dff7c483ceb23f344a6b1c2913beaca1682ff53afb8bca"
    },
    "fine.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "7d6fa4a610634d92d3639563175a27ede3b083467f656c91ce8d579b7044ffa5"
    },
    "forcing.weight": {
      "shape": [
        12,
        9,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "3104b6f5bdd6c19b142206301585adc94e14065e0a524335af6b9730adb597cc"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "6c1244efae9030dda9886ec648524d5b89ed032768c9496937c4afd12f0e98c8"
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
      "sha256": "4aceff353f7045ce820b47274b1add1db1e003439fdb695daac48c1eb2701eda"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "e7d7e717c5b0f2c0451dbcac3381f3d3e33d3305cecda8d18b65c7e648d3c1eb"
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
      "sha256": "fc81e129c6bb4d636a6bfc4affdf80ce6c065913565e8fe8468777e38e7cecfc"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "cc2115fe0cf59bf61115cc517f14801ab203d094513b7867946576f7ea7dd4f7"
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
      "sha256": "ac8d7b60392878432c5c5c774f6826a8c96f2b6d1067c48396687351ba5c49ff"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "239509a6754d217e88bab08e04bafba9e36a3e2dd3ed06bee35e34975eb22dd8"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "2b08090b703ee4fb856a3e5df39afbc961d6ad52363d22b2e12f3d9c00b602b0"
    },
    "global_blocks.0.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "51972a00a25c1ac603dc07beab0716221e81602f6869dad35fda462bbba9254d"
    },
    "global_blocks.0.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "8e7a7148be7fa90efe0602a0c26681e9a9f7127507a5e2d26732d72e36d77c63"
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
      "sha256": "fa0d83186ebef312bd5a4bec4f6165e0cd158d4e6024396c9c5559ccefeebfb1"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "d2db63e402acf86728530cbe3cfcbd0acc41437805088747a88397373586c70b"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "927abf25c737e6eeb190ec6ff241bbc85b9d3d2621a340a2290c7623ccd673e0"
    },
    "global_blocks.1.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "697d1ccf6f2ec4dc7068c117e407d043941c2d487f8c06c29081479cef215ba4"
    },
    "global_blocks.1.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "db55fb2039849bdb23049f89a7119b701bd3c851889646cd2d37bf6582ac0651"
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
      "sha256": "26206372001761f80435fb5fd25a7e45ecf65fc4ab34708cc81cdd6ae6fb8044"
    },
    "first_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "705ddb7122b4bb9bfc3dbb4ee7a92b00d715c46cd0bafd21c9362cf9ae164e96"
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
      "sha256": "c17e5ac9f57132258db23a01eee605398e9d3307fe0e21372fda7596dae4e0d5"
    },
    "latent_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "755501ae1781ba663facec673dd1c012a13a314dbb8429ee223aefec490d05c2"
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
      "sha256": "0eb270ff99079e987a7a3ab257e379005c3df3f6bbfdbf7152ad63a1eb2dc00d"
    },
    "latent_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "1febb59b94813967772912aa2439fd06cb07ebab3620a9b9d821dbeab336cc76"
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
      "sha256": "9f25164a869ca9e3f0b4c3af5d38d8facd2768e8c2ddde8a409f3da32b3bab03"
    },
    "later_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "7b15fdc5f087fd99e2d920c3ee1cc985d507a9b827df7a6ef6b26c6ac1f97921"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 901659648
  },
  "outputs": {
    "probe.npy": {
      "bytes": 294908,
      "sha256": "88404dadb03e79c3aee297297cdf0e7ea1e2a69ba6daec599fd1fcb47197c1ab"
    },
    "weights.npz": {
      "bytes": 660450,
      "sha256": "1476e5ddc3705fbc20ec08c7741ce78d8fabcbfd8e6b3b76660a961376db88f8"
    },
    "initial-weights.npz": {
      "bytes": 660450,
      "sha256": "00c000a1381b233350c7ceafe5c70f2eb396ebd805f3ed9753f936a70d9fcfc0"
    },
    "history.json": {
      "bytes": 275210,
      "sha256": "ddc29a642421e6bf15564d5e688a430dd2cf084bd1eb2dff5609cf45771534f9"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 317.8240408140118,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-sharing-v1",
  "architecture": "reuse",
  "feedback_protocol": 2,
  "identifier": "b3-reuse",
  "horizons": 3,
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
