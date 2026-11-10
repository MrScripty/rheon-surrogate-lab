# Full AI temporal b2-endpoint-step-scaled train

Retrospective source review · fit-e2a657f5ad91acb1

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 2; each output advances 1/60 second. Full block duration is 2/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.517998 | 2.39418 | 0.15940 |
| 8 | 16x16x16 | 0.211065 | 2.36691 | 0.19605 |
| 8 | 32x32x32 | 0.343285 | 2.40208 | 0.47160 |
| 8 | 9x11x11 | 0.167849 | 2.48820 | 0.16498 |
| 16 | 15x6x8 | 0.217267 | 2.40615 | 0.16040 |
| 16 | 16x16x16 | 0.176761 | 2.34104 | 0.19602 |
| 16 | 32x32x32 | 0.19829 | 2.52108 | 0.47723 |
| 16 | 9x11x11 | 0.198914 | 2.51184 | 0.16505 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 480d903fd12f64c7130577e176878296432640a98f7d5b9471f0d8929f641841; subject: b2-endpoint-step-scaled.

**Method.** A 163,068-parameter F32 width12 K2 endpoint generator with the same learned spatial/global encoder and heads as its original unscaled counterpart. Previous pressure is excluded. The prescribed-force channels are divided by fixed train-only RMS 0.021354100957838906. Raw velocity/tracer increments are multiplied by their fixed horizon-specific normalized training RMS before addition to the original block state. Pressure remains a fresh gauged output; physical input/output normalization is unchanged. The endpoint branch predicts its last state and jointly decodes intermediate states; the reuse branch advances retained features using prior predictions. No numerical pressure solve or future query label is used.

**Purpose.** Test whether matching force and increment magnitudes improves trainability of the existing compute-sharing architecture without adding parameters or changing simulated cadence. The name step-scaled refers to increment units, not variable dt or longer-time training.

**Experiment.** Original16train-only scenes/2064states at8/16cells per metre, four shapes including32cubed and rectangular domains, fixed1/60s outputs and K2blocks spanning2/60s. Fixed RMS values are computed once over all training-parent increments at horizons1–4 and prescribed forces. No development/reserved data is used. Two fits retain their corresponding unscaled architecture’s exact seeded initial tensors, optimizer schedule, private sample stream,10,000CUDAupdates and18,000encoder calls under the2000one-block/8000two-own-history-block curriculum. Complete source/typed recipe/input/output identities and initial/final tensor inventories verify.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 0.517998, 1.00252 and 0.0172871. Median time is 4.85508 ms per K2 block (2.42754 ms per physical output); the worst case median is 2.67093 ms/output. The retained F64 N1 composition has better worst pressure error and 13.54 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** Sharing an encoder, endpoint or retained features does not by itself establish a useful accuracy/speed tradeoff. This fit remains worse than the matched F64 composition in both worst pressure error and median GPU time. Account for every intermediate output and reconstruction cost; K outputs advance K full 1/60-second intervals.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-pair-01/train/receipt.json",
    "sha256": "e2a657f5ad91acb16fdf073c93e743256b68937f2b24d974505cefb507853d8c"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-pair-01/train/history.json",
    "sha256": "168d66fa06fbd218ffe1b7f2da76fc67024cb5567acb58178fde1db193e358cf"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/bundle_model_sharing.py",
      "sha256": "3490bcffb6bfc9c19bb61613c0bd64065968adb27fa32a3e475f1e5e53850716"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/bundle_model_step_scaled.py",
      "sha256": "3d60ebb20e4b1cba76e2710358f23a026ee446fa5d378789eb27ce377096300c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/learned_global.py",
      "sha256": "099078b789e728fa8b14acb63327e9942ed90b2daeea4a2c79af2ad3dde26c9b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/temporal_train_step_scaled.py",
      "sha256": "85812c735f9a9e36ea1bb37fb85abe6c7876d9fa0c7836ebd32a78760500cbfa"
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
  "lesson": "This is a controlled conditioning/optimization experiment rather than proof that larger temporal strides generalize. The K2endpoint or latent-reuse structure can amortize encoder work, but fixed scale factors do not certify a high-accuracy endpoint, remove F32 prediction arithmetic, or recover the near-exact F64 anchors. Preserve the candidate for common benchmarking instead of rejecting it from an obsolete timing target.",
  "outcome": "The original fit completed10,000updates; last logged composite objective 16.639377593994141. Independent NumPy decoding checks all80original field hashes and rederives every parent/aggregate increment scale plus known-force RMS. The paired architecture adds zero parameters and preserves exact seeded initialization. Fresh common GPU full-field accuracy, block timing and own-history drift remain unmeasured for this fit. A normalized composite objective is not a pressure relative-error result.",
  "reproduction": "Retain exact captured model/trainer/config/typed recipe/runtime, original16scene F32archive and normalization, the separately source-bound step-scales.json, private seeds/order and deployed feedback reconstruction. The fixed scales are constructor inputs outside the NPZ tensor inventory; weights alone are insufficient. Original publication/restore recipes execute CPU neural probes and require GPU-only qualification before rerunning. Clean retraining, new-Rheon data generation and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-e2a657f5ad91acb1",
    "kind": "training_attempt",
    "name": "Full AI temporal b2-endpoint-step-scaled train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "e2a657f5ad91acb16fdf073c93e743256b68937f2b24d974505cefb507853d8c",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "23430433c12f422887fcb718ca5dcb13",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b2-endpoint-step-scaled train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 163068,
    "training_closure": {
      "config_sha256": "7990644dae2796c8e156e54f459c038a8452dd77fc0056da62d42f8eeb3d8b83",
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
        "temporal_train_step_scaled.py": "85812c735f9a9e36ea1bb37fb85abe6c7876d9fa0c7836ebd32a78760500cbfa"
      },
      "horizons": 2,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "a26d25af5cc9c3c74c87ffe542e471e5cbd33857c98b510952efe5497a6c82c8",
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
      "temporal-models/12"
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
      "receipt": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-pair-01/train/receipt.json",
    "sha256": "e2a657f5ad91acb16fdf073c93e743256b68937f2b24d974505cefb507853d8c",
    "bytes": 15073
  },
  "bound_training_configuration": {
    "config_sha256": "7990644dae2796c8e156e54f459c038a8452dd77fc0056da62d42f8eeb3d8b83",
    "graph_id": "d8d2619dd46545d2b7f7e1638a47ab3c",
    "graph_document_sha256": "8feb35b7b6e1d7111d0f17bba2c52b89ae09cbc97c29515f9ac590521ce75ef1",
    "recipe_path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/train-recipe.json",
    "recipe_sha256": "334cfd13ea9dda72ded9b81bed449ed6f9da049912a9185a94bc75d986a75777",
    "config_path": "runs/full-ai-next-priorities-v1/b2-endpoint-step-scaled-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-step-scaled-v1",
      "architecture": "endpoint",
      "feedback_protocol": 2,
      "identifier": "b2-endpoint-step-scaled",
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
      "sharing_registration_sha256": "39dacd04f68ff5f8d40ea7987d3b6c6c0cfb3e3ce8cb672fdae2d8068d6bb747",
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
    "config_sha256": "7990644dae2796c8e156e54f459c038a8452dd77fc0056da62d42f8eeb3d8b83",
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
      "temporal_train_step_scaled.py": "85812c735f9a9e36ea1bb37fb85abe6c7876d9fa0c7836ebd32a78760500cbfa"
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
      "sha256": "acebbbfd9527d8ad2d57f5e969d3a8998682a2a645576440db065e3b8da915a1"
    },
    "fine.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "8b8d329cffada98ab9a7828f5acce09dadcd7fc3b45af624c446038151bed138"
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
      "sha256": "07436b2a368c24f24c9ad6d7215ea75467945e23bb224520d236aebdc5bc1f5b"
    },
    "fine.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "59e814bf7aa7b015a32016b5e92d2d05554cdadfb7d216f7c89166bcf8bcd463"
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
      "sha256": "bf9a3bd2b0f0f553dc7da2c1df9a764552ea2947ae3f6a34b4f617f86b72b34e"
    },
    "forcing.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "31cccc2217bc0b8ea813406c69a5312ad28af298cd7d07386ba43a7049695398"
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
      "sha256": "3825437bdf4cf195df6c0969d6db4c138c48e55ad644ab2a641a38c0e16bacc7"
    },
    "local.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "2a9434abda9f4c1816d8d1ec7ac92078ff60c31c4f55db1a3d0a7bccd4d34ef4"
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
      "sha256": "7f56d67c14db3e73a22107e9ceafd8fab03123230fff445854c8847ad09237ea"
    },
    "local.1.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "8515d5a121660d1672a09fab6fefbfa71cbfff618d10deb8246989fcff8a991c"
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
      "sha256": "58698e78a7cc6c650a9402d9c2ad9c1ff6ed73c2a1748309fcf4e5bd86ccfa27"
    },
    "global_blocks.0.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "4317221d0b1c4f04db1275b556995a49ce13159f3f5f94000d117bfe0dcc0e34"
    },
    "global_blocks.0.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "f1f24054d9ce4eabf15c5ee46b1aa0bad1d816eeac53ed934322f5fd89426d98"
    },
    "global_blocks.0.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "716902dbfe2df6b654dc014cbfd05ff31ecac5fbf93e0a1951b4506000231928"
    },
    "global_blocks.0.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "119042a4497ff8bed33099155a0815a574c718e287785864f45e4d1cbbedbffa"
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
      "sha256": "4927b17fe99b10b7fd77fc9f566cf00617a051895c4b13382d15259e791e7216"
    },
    "global_blocks.1.geometry_head.0.weight": {
      "shape": [
        32,
        7
      ],
      "dtype": "<f4",
      "sha256": "dcfcbd34f8ace56edf7afe1d6b057a311cc9913c36c95b1b80792cb82b45550e"
    },
    "global_blocks.1.geometry_head.0.bias": {
      "shape": [
        32
      ],
      "dtype": "<f4",
      "sha256": "1a377a43a47aa9dd02c0a43325024871ed116696c3340c78cea2334670d3e1ab"
    },
    "global_blocks.1.geometry_head.2.weight": {
      "shape": [
        144,
        32
      ],
      "dtype": "<f4",
      "sha256": "64c9315e1f637a3b989e7f44b7145a7c2a2913c95b86f6400307f51ae8b0ddf7"
    },
    "global_blocks.1.geometry_head.2.bias": {
      "shape": [
        144
      ],
      "dtype": "<f4",
      "sha256": "982235c4e9c30228aa50e4b48124b4e7f1fb399b0876fee36980642bc0897695"
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
      "sha256": "e9249328591f9ee87c499f9b169268cb8512990613769b9ca39ba615438ff14e"
    },
    "endpoint_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "7a04ff9fd7da1332397f0b19745b6eb64f058ba3644e32b165f848cc080be09b"
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
      "sha256": "53ce4959f7538b2b40423b169890236661d70e7416b907c2aa2f58a6b45b329c"
    },
    "intermediate.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "088c54db2709b3f0a595071b9c55a62f625a8196ff512d0a7276fca295f60a1b"
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
      "sha256": "0aaa5cf9afd32990395e0663242bdc3f55567c3f4b7c40d49069552be64950da"
    },
    "intermediate_head.bias": {
      "shape": [
        5
      ],
      "dtype": "<f4",
      "sha256": "fe09fe998c9a7707cdc58113ad8a4c88c9e7094c23e0185ab687796582c6ae05"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 874939392
  },
  "outputs": {
    "probe.npy": {
      "bytes": 196648,
      "sha256": "32c351f0c1f432d9ad0cf84a4ce68fe3beefdcdef7c115ccce9d87a8abd884b0"
    },
    "weights.npz": {
      "bytes": 659506,
      "sha256": "a26d25af5cc9c3c74c87ffe542e471e5cbd33857c98b510952efe5497a6c82c8"
    },
    "initial-weights.npz": {
      "bytes": 659506,
      "sha256": "2ebab7ca1c29fc8e12a41b0c488c19172f0604d40e5d5302bf77fee2a5180b97"
    },
    "history.json": {
      "bytes": 275408,
      "sha256": "168d66fa06fbd218ffe1b7f2da76fc67024cb5567acb58178fde1db193e358cf"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 300.3621639730409,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-step-scaled-v1",
  "architecture": "endpoint",
  "feedback_protocol": 2,
  "identifier": "b2-endpoint-step-scaled",
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
  "sharing_registration_sha256": "39dacd04f68ff5f8d40ea7987d3b6c6c0cfb3e3ce8cb672fdae2d8068d6bb747",
  "step_scale_statistics_sha256": "5f976a2e19e77cf7b46bf701b9e7c7cd65856d8f391db138020c8855592dad04",
  "original_normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037",
  "full_fit_registration_sha256": "08ed048038db26f1838a6e74f82ae521c848f06b954a42aaf3000a17794d0510"
}

```
