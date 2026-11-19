# Full AI temporal b2-endpoint-pressure-expert train

Retrospective source review · fit-1031d32821fb88cf

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 2; each output advances 1/60 second. Full block duration is 2/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.165148 | 3.25665 | 0.15940 |
| 8 | 16x16x16 | 0.239149 | 5.09857 | 0.19605 |
| 8 | 32x32x32 | 0.193066 | 5.33548 | 0.47160 |
| 8 | 9x11x11 | 0.153098 | 3.53993 | 0.16498 |
| 16 | 15x6x8 | 0.145321 | 3.41815 | 0.16040 |
| 16 | 16x16x16 | 0.209559 | 4.89205 | 0.19602 |
| 16 | 32x32x32 | 0.172123 | 5.29171 | 0.47723 |
| 16 | 9x11x11 | 0.229862 | 3.40345 | 0.16505 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 480d903fd12f64c7130577e176878296432640a98f7d5b9471f0d8929f641841; subject: b2-endpoint-pressure-expert.

**Method.** A width16K2endpoint-conditioned generator with111,650parameters,19,009trainable and92,641frozen. A causal neural request plus learned correction feeds one frozen a-tail8pressure expert per block. The predicted endpoint and shared features condition cheaper intermediate pressure/velocity/tracer heads. Previous pressure and tracer do not feed the fluid encoder. Pressure-expert accuracy is for its supplied predicted request; it does not certify that the request or future state is correct. Velocity uses a learned pressure convolution plus direct known-force accumulation and a learned increment. Tracer uses additive learned updates with multiplicative fluid/tracer features. Pressure-to-velocity scaling uses single dtin every horizon.

**Purpose.** Reuse the more accurate frozen pressure component once per multistep block while learning causal future requests and smaller intermediate-state readouts.

**Experiment.** Original16consumed training scenes/2064states at8/16cells per metre and four shapes including32cubed/rectangles, fixed1/60s output cadence and2/60s block advancement.10,000CUDA Adam updates use batch1, private seeds/order and2000one-block then8000two-own-history-block curriculum. All original nested assets/source/typed recipe/input/output identities, complete initial/final/frozen tensors and work counters verify. The endpoint fits use the fixed-increment MSE objective. The decoder preserves all common seeded fluid/request/pressure tensors but changes its tracer architecture/function, so it is a bundled representation comparison.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 0.239149, 0.401022 and 0.00176105. Median time is 8.20404 ms per K2 block (4.10202 ms per physical output); the worst case median is 5.41193 ms/output. The retained F64 N1 composition has better worst pressure error and 22.88 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** Sharing an encoder, endpoint or retained features does not by itself establish a useful accuracy/speed tradeoff. This fit remains worse than the matched F64 composition in both worst pressure error and median GPU time. Account for every intermediate output and reconstruction cost; K outputs advance K full 1/60-second intervals.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Inherited derived pressure maps are explicitly GPU-built precision variants. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-pair-01/train/receipt.json",
    "sha256": "1031d32821fb88cfebc2e0d1ea6a65661f45501bfbf9b81ab84aa4a1f3d6de09"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-pair-01/train/history.json",
    "sha256": "5544322fe73696373de19851894eb439cf9812464503803ac63df0ce0b560b78"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/bundle_model_pressure_expert.py",
      "sha256": "9cc6527ccf6d273de4d37d8ccff2752159edf040789c74dc1ca85e4382588b75"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/expert-assets/pressure/configuration.json",
      "sha256": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/expert-assets/pressure/fine-weights.npz",
      "sha256": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/expert-assets/pressure/geometry-gains.npz",
      "sha256": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/expert-assets/pressure/receipt.json",
      "sha256": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/expert-assets/request-initialization.json",
      "sha256": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/expert-assets/request-weights.npz",
      "sha256": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/factored_request_model.py",
      "sha256": "39577e2b6b275850f61f2e440e646ab435def01da1d16b5301fb5329e1fec6b5"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/frozen_temporal_pressure_core.py",
      "sha256": "5e83b77ec90ea3709dab8a84975aa068a60c1acf9e364af0a2b8f545e3f49666"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/temporal_objective_increment.py",
      "sha256": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/temporal_objective_mse.py",
      "sha256": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/temporal_train_pressure_expert.py",
      "sha256": "56e3749c0bd87bb864dfa4b8abded09b1790ec7346b0621c0f5794e84de39255"
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
  "lesson": "A high-accuracy pressure component can coexist with inaccurate full-state generation when the predicted request or intermediate decoder is wrong. Benchmark every field/lead against the reference, and distinguish one expert call per block from Kfully recurrent expert calls.",
  "outcome": "Original training completes10,000updates; final composite objective 0.030960345640778542. Independent architecture sums reproduce all parameter counts and frozen coefficients remain exact. No fresh common GPU full-field error, complete block latency, independent holdout or600-output drift result is measured in this review.",
  "reproduction": "Retain exact captured config/typed recipe/runtime/objective and all expert assets, including learned pressure gains/provenance, causal request initialization, fixed statistics and original16scene F32corpus/normalization. Displacement additionally needs frozen point calibration and training-only own-history teacher semantics. Original constructors generate neural unit-response matrices on CPU before .to(device), and publication/restore executes CPU neural probes. Qualify an explicit CUDA-only setup/execution variant before running these recipes. Preserve precision/rounding differences visibly; clean retraining, new-Rheon generation and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-1031d32821fb88cf",
    "kind": "training_attempt",
    "name": "Full AI temporal b2-endpoint-pressure-expert train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "1031d32821fb88cfebc2e0d1ea6a65661f45501bfbf9b81ab84aa4a1f3d6de09",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "a3f5e3593d8249d99fd75d5ee45b26b5",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b2-endpoint-pressure-expert train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 111650,
    "training_closure": {
      "config_sha256": "681df22172d13326b0f6d17187561d805ce5e0ac451382b68fbbfa5171b36193",
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
        "bundle_model_pressure_expert.py": "9cc6527ccf6d273de4d37d8ccff2752159edf040789c74dc1ca85e4382588b75",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "expert-assets/pressure/configuration.json": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1",
        "expert-assets/pressure/fine-weights.npz": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6",
        "expert-assets/pressure/geometry-gains.npz": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d",
        "expert-assets/pressure/receipt.json": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544",
        "expert-assets/request-initialization.json": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab",
        "expert-assets/request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd",
        "factored_request_model.py": "39577e2b6b275850f61f2e440e646ab435def01da1d16b5301fb5329e1fec6b5",
        "frozen_temporal_pressure_core.py": "5e83b77ec90ea3709dab8a84975aa068a60c1acf9e364af0a2b8f545e3f49666",
        "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
        "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
        "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
        "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
        "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
        "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
        "temporal_objective_increment.py": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e",
        "temporal_objective_mse.py": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136",
        "temporal_train_pressure_expert.py": "56e3749c0bd87bb864dfa4b8abded09b1790ec7346b0621c0f5794e84de39255"
      },
      "horizons": 2,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "a3da5ec538e7fbb2a841700c972232a931c53af5229e896b183fcef723f6a27b",
        "bytes": 458556,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "44a6a71345f179992f2621a0358b601fe52000b50db5759189e4ef2e5489f3a8",
        "bytes": 458556,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/17"
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
      "receipt": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-pair-01/train/receipt.json",
    "sha256": "1031d32821fb88cfebc2e0d1ea6a65661f45501bfbf9b81ab84aa4a1f3d6de09",
    "bytes": 29276
  },
  "bound_training_configuration": {
    "config_sha256": "681df22172d13326b0f6d17187561d805ce5e0ac451382b68fbbfa5171b36193",
    "graph_id": "34a32d4d95014d73b466743caf6d9c7e",
    "graph_document_sha256": "2ed5e907bfe932d263878226f83ad729a20d29812ac4613ad8a562b546b80135",
    "recipe_path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/train-recipe.json",
    "recipe_sha256": "0be8804f29e9dcd3e8e5018d1544fe1750e75082f5c8ae112dc8924d4f325911",
    "config_path": "runs/full-ai-next-priorities-v1/b2-endpoint-pressure-expert-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-pressure-expert-v1",
      "velocity_representation": "learned_pressure_conditioned_convolution",
      "supervision_mode": "increments",
      "architecture": "endpoint",
      "feedback_protocol": 2,
      "identifier": "b2-endpoint-pressure-expert",
      "horizons": 2,
      "width": 16,
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
      "initialization": "Frozen a-tail8 pressure expert, managed causal request initialization, seeded width16 fluid/tracer encoders and zero new heads",
      "inference": "One fluid encoder and one frozen pressure expert per block; predicted endpoint plus cheap intermediate decoder or first-prediction feature reuse; learned velocity readout, no numerical solve or post-projection",
      "sample_step_range": [
        0,
        120
      ],
      "loss": "Matched MSE pressure/equation/physical terms; velocity/tracer errors in fixed aggregate native-step units",
      "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
      "sharing_registration_sha256": "ad3f739d27f4a12f0b433d61b34a3c6ce67f4c04dccc55a188f0786559a5581e",
      "step_scale_statistics_sha256": "5f976a2e19e77cf7b46bf701b9e7c7cd65856d8f391db138020c8855592dad04",
      "original_normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037",
      "physical_scales": {
        "pressure": 9.47211357321988,
        "velocity": 0.0064655899998301335,
        "tracer": 0.012995164216288916
      },
      "expert_asset_sha256": {
        "pressure/configuration.json": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1",
        "pressure/fine-weights.npz": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6",
        "pressure/geometry-gains.npz": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d",
        "pressure/receipt.json": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544",
        "request-initialization.json": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab",
        "request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
      },
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
  "parameters": 111650,
  "trainable_parameters": 19009,
  "frozen_parameter_inventory": {
    "pressure_core.fine.encoders.0.layers.0.weight": {
      "shape": [
        4,
        1,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "f37ee2e1dca35520b1efe258b0e9b494da3d1c16ae50aa7a2fe4f811147bfb8f"
    },
    "pressure_core.fine.encoders.0.layers.3.weight": {
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "02337c2d9fbdf1911b2a26172edb5102a611748ac130b22a3c449521f4c0b424"
    },
    "pressure_core.fine.encoders.1.layers.0.weight": {
      "shape": [
        8,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "b41b9311a838779c264cc28b24c78b2497ed657927e4537b08bdc1949ed255fc"
    },
    "pressure_core.fine.encoders.1.layers.3.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "060174e35a0610d5e1d0b0639310767f8b878a3d8e06e7759b286540899af539"
    },
    "pressure_core.fine.encoders.2.layers.0.weight": {
      "shape": [
        12,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "8b543e79d54ee0752c48ea07b4d4ff1f267174ba5e4e1e6d36994bfa10f27e98"
    },
    "pressure_core.fine.encoders.2.layers.3.weight": {
      "shape": [
        12,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "4a7c38eee7eb8eaa4e3890e9e8ce861ab6975c1e1416684b1fb1862d1782b76a"
    },
    "pressure_core.fine.encoders.3.layers.0.weight": {
      "shape": [
        16,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "7932627cfd088d4874ffaa18c62f6b80b84fc4be11c478232bb5cf4e6e3b9d08"
    },
    "pressure_core.fine.encoders.3.layers.3.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "67a1fe4c5403bc257879f8c1b785872619c234839817be792dce919f724ed5de"
    },
    "pressure_core.fine.bottleneck.layers.0.weight": {
      "shape": [
        24,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "2a9862fb35c4f0c618738f42c6af669ce8071a53c576a224ec723fedeb936922"
    },
    "pressure_core.fine.bottleneck.layers.3.weight": {
      "shape": [
        24,
        24,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "d8e092e6be01936d786e8b17a9e604e461ba4af6cb31cde38099893417c90008"
    },
    "pressure_core.fine.decoders.0.layers.0.weight": {
      "shape": [
        16,
        40,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "58d51c78e13a8b2c36b4c035b74dd8a87c2db92b75f8298b629fc5c148a2c02b"
    },
    "pressure_core.fine.decoders.0.layers.3.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "38c9644c2c242ad75ad9070d2ca6e2d2984d3f8e45248a3bd39cd931703e36c2"
    },
    "pressure_core.fine.decoders.1.layers.0.weight": {
      "shape": [
        12,
        28,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "0dd3fffaaf1eb31e81742ac35625c41eb0d61a7577ce0d7e76d5c7e8c538097e"
    },
    "pressure_core.fine.decoders.1.layers.3.weight": {
      "shape": [
        12,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "e662b00c70c4f8eb3fd6af5daefe62d8670faa4381f380ca0952ea272d22e428"
    },
    "pressure_core.fine.decoders.2.layers.0.weight": {
      "shape": [
        8,
        20,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "897216db24db4a4df34d9273d197508f3d0261dfdc5decfafb6982cea0a4337b"
    },
    "pressure_core.fine.decoders.2.layers.3.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "414758deeb72af51fc5c494b967bf9e77167def358b2884608ee0f906080ce38"
    },
    "pressure_core.fine.decoders.3.layers.0.weight": {
      "shape": [
        4,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "817316b8873759212cf3b458109374d35b52a3a77bb3e82aa9e29f83650a0b66"
    },
    "pressure_core.fine.decoders.3.layers.3.weight": {
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "64631b34d1c4b8aa25fde2aad93974bb888bb1ad7166a9b3b4a4cf8bd4cd6c49"
    },
    "pressure_core.fine.head.weight": {
      "shape": [
        1,
        4,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "62c58dcd44aa770b0d67da6cc12dd66057e882a29bc3678a7c1d1bbd6cfd3ee8"
    },
    "request.flux.weight": {
      "shape": [
        1,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "dacf17b5d8b84199c50513a57cecd3233b6049fe116850333618e930e5911041"
    }
  },
  "parent_pressure_learned_parameters": 128938,
  "pressure_expert_calls_per_block": 1,
  "horizons": 2,
  "closure": {
    "config_sha256": "681df22172d13326b0f6d17187561d805ce5e0ac451382b68fbbfa5171b36193",
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
      "bundle_model_pressure_expert.py": "9cc6527ccf6d273de4d37d8ccff2752159edf040789c74dc1ca85e4382588b75",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "expert-assets/pressure/configuration.json": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1",
      "expert-assets/pressure/fine-weights.npz": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6",
      "expert-assets/pressure/geometry-gains.npz": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d",
      "expert-assets/pressure/receipt.json": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544",
      "expert-assets/request-initialization.json": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab",
      "expert-assets/request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd",
      "factored_request_model.py": "39577e2b6b275850f61f2e440e646ab435def01da1d16b5301fb5329e1fec6b5",
      "frozen_temporal_pressure_core.py": "5e83b77ec90ea3709dab8a84975aa068a60c1acf9e364af0a2b8f545e3f49666",
      "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
      "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
      "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
      "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
      "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
      "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
      "temporal_objective_increment.py": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e",
      "temporal_objective_mse.py": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136",
      "temporal_train_pressure_expert.py": "56e3749c0bd87bb864dfa4b8abded09b1790ec7346b0621c0f5794e84de39255"
    },
    "horizons": 2,
    "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
  },
  "initial_inventory": {
    "pressure_core.fine.encoders.0.layers.0.weight": {
      "shape": [
        4,
        1,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "f37ee2e1dca35520b1efe258b0e9b494da3d1c16ae50aa7a2fe4f811147bfb8f"
    },
    "pressure_core.fine.encoders.0.layers.3.weight": {
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "02337c2d9fbdf1911b2a26172edb5102a611748ac130b22a3c449521f4c0b424"
    },
    "pressure_core.fine.encoders.1.layers.0.weight": {
      "shape": [
        8,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "b41b9311a838779c264cc28b24c78b2497ed657927e4537b08bdc1949ed255fc"
    },
    "pressure_core.fine.encoders.1.layers.3.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "060174e35a0610d5e1d0b0639310767f8b878a3d8e06e7759b286540899af539"
    },
    "pressure_core.fine.encoders.2.layers.0.weight": {
      "shape": [
        12,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "8b543e79d54ee0752c48ea07b4d4ff1f267174ba5e4e1e6d36994bfa10f27e98"
    },
    "pressure_core.fine.encoders.2.layers.3.weight": {
      "shape": [
        12,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "4a7c38eee7eb8eaa4e3890e9e8ce861ab6975c1e1416684b1fb1862d1782b76a"
    },
    "pressure_core.fine.encoders.3.layers.0.weight": {
      "shape": [
        16,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "7932627cfd088d4874ffaa18c62f6b80b84fc4be11c478232bb5cf4e6e3b9d08"
    },
    "pressure_core.fine.encoders.3.layers.3.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "67a1fe4c5403bc257879f8c1b785872619c234839817be792dce919f724ed5de"
    },
    "pressure_core.fine.bottleneck.layers.0.weight": {
      "shape": [
        24,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "2a9862fb35c4f0c618738f42c6af669ce8071a53c576a224ec723fedeb936922"
    },
    "pressure_core.fine.bottleneck.layers.3.weight": {
      "shape": [
        24,
        24,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "d8e092e6be01936d786e8b17a9e604e461ba4af6cb31cde38099893417c90008"
    },
    "pressure_core.fine.decoders.0.layers.0.weight": {
      "shape": [
        16,
        40,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "58d51c78e13a8b2c36b4c035b74dd8a87c2db92b75f8298b629fc5c148a2c02b"
    },
    "pressure_core.fine.decoders.0.layers.3.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "38c9644c2c242ad75ad9070d2ca6e2d2984d3f8e45248a3bd39cd931703e36c2"
    },
    "pressure_core.fine.decoders.1.layers.0.weight": {
      "shape": [
        12,
        28,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "0dd3fffaaf1eb31e81742ac35625c41eb0d61a7577ce0d7e76d5c7e8c538097e"
    },
    "pressure_core.fine.decoders.1.layers.3.weight": {
      "shape": [
        12,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "e662b00c70c4f8eb3fd6af5daefe62d8670faa4381f380ca0952ea272d22e428"
    },
    "pressure_core.fine.decoders.2.layers.0.weight": {
      "shape": [
        8,
        20,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "897216db24db4a4df34d9273d197508f3d0261dfdc5decfafb6982cea0a4337b"
    },
    "pressure_core.fine.decoders.2.layers.3.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "414758deeb72af51fc5c494b967bf9e77167def358b2884608ee0f906080ce38"
    },
    "pressure_core.fine.decoders.3.layers.0.weight": {
      "shape": [
        4,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "817316b8873759212cf3b458109374d35b52a3a77bb3e82aa9e29f83650a0b66"
    },
    "pressure_core.fine.decoders.3.layers.3.weight": {
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "64631b34d1c4b8aa25fde2aad93974bb888bb1ad7166a9b3b4a4cf8bd4cd6c49"
    },
    "pressure_core.fine.head.weight": {
      "shape": [
        1,
        4,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "62c58dcd44aa770b0d67da6cc12dd66057e882a29bc3678a7c1d1bbd6cfd3ee8"
    },
    "request.flux.weight": {
      "shape": [
        1,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "dacf17b5d8b84199c50513a57cecd3233b6049fe116850333618e930e5911041"
    },
    "request.left.weight": {
      "shape": [
        16,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "bda0101f0e2163356ecafe087bad39780c023b5a68be5191f816236f3addd63d"
    },
    "request.right.weight": {
      "shape": [
        16,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "aa95dd917b7dc7fd4334d63050a550b1a9381df854ef373fada156ec206eb959"
    },
    "request.mix.weight": {
      "shape": [
        16,
        80,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "bfb4c67335751a70a0114474413e70bfc5ab75d7f41c63bfa13703409453ca52"
    },
    "request.head.weight": {
      "shape": [
        1,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "390bfffbc96e0ca68a67a6d585116525dca3ad742185c7ec2606ffd2cda0cb1d"
    },
    "fluid.0.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "f688aeaecd57098d5e2ddc854bebafe9a527b61dbd144071387e81038fc36698"
    },
    "fluid.0.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "sha256": "7f86e248af2355ea053183e275c548406b1848bfefbd40fbe8de1bf014f071f0"
    },
    "fluid.2.weight": {
      "shape": [
        16,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "43bd352d323e0ef335beabb05b24986d0f67db15bd653b28c76f1819f5923209"
    },
    "fluid.2.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "sha256": "96554feca6448bff9da3c6ad2e4d60138f730041509ca3c698fcf498100cffd5"
    },
    "request_head.weight": {
      "shape": [
        1,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "7636c9f8da9a82f9a9fcd804a25795a9c900e1c7e19f7442a7b113c8ccd64eb2"
    },
    "pressure_velocity.weight": {
      "shape": [
        3,
        1,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "7119f314d3837dac54bf984118de657f6f1e6dfd2a2b2a20510454f762dddf11"
    },
    "velocity_head.weight": {
      "shape": [
        3,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "5d89f056865052bcb89c910d2d62872e029fb273c3db03f8968a52a41593c1b5"
    },
    "tracer.0.weight": {
      "shape": [
        8,
        29,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "f5ab645021a73c2750453d7678d51f0eb1884cbb3e98f5127524c1e9fbc9d3fd"
    },
    "tracer.0.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "455a568f0edf6483b6eed35796b9829a36366c456ec235f701c126292015b77d"
    },
    "tracer_head.weight": {
      "shape": [
        1,
        8,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "middle.0.weight": {
      "shape": [
        16,
        26,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "8fb08e2e08bb81746ab5321fe3b314ce257696776a5cb29d0d06e0c5d6efe61e"
    },
    "middle.0.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "sha256": "c461b988c9a62e10ca6236ede581c57b97ea84e65699058519862db7881b8e55"
    },
    "middle_p.weight": {
      "shape": [
        1,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
    },
    "middle_v.weight": {
      "shape": [
        3,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "5d89f056865052bcb89c910d2d62872e029fb273c3db03f8968a52a41593c1b5"
    },
    "middle_tracer.0.weight": {
      "shape": [
        8,
        19,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "b008f529a33d37050eabd187484ac32041a9af2b4bb918c1985e26f57e7d8d37"
    },
    "middle_tracer.0.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "fb7094f5287804c242a7c7e5351ace4be729b49902d407d75c8bb04dcfc62b7b"
    },
    "middle_t.weight": {
      "shape": [
        1,
        8,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    }
  },
  "final_inventory": {
    "pressure_core.fine.encoders.0.layers.0.weight": {
      "shape": [
        4,
        1,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "f37ee2e1dca35520b1efe258b0e9b494da3d1c16ae50aa7a2fe4f811147bfb8f"
    },
    "pressure_core.fine.encoders.0.layers.3.weight": {
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "02337c2d9fbdf1911b2a26172edb5102a611748ac130b22a3c449521f4c0b424"
    },
    "pressure_core.fine.encoders.1.layers.0.weight": {
      "shape": [
        8,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "b41b9311a838779c264cc28b24c78b2497ed657927e4537b08bdc1949ed255fc"
    },
    "pressure_core.fine.encoders.1.layers.3.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "060174e35a0610d5e1d0b0639310767f8b878a3d8e06e7759b286540899af539"
    },
    "pressure_core.fine.encoders.2.layers.0.weight": {
      "shape": [
        12,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "8b543e79d54ee0752c48ea07b4d4ff1f267174ba5e4e1e6d36994bfa10f27e98"
    },
    "pressure_core.fine.encoders.2.layers.3.weight": {
      "shape": [
        12,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "4a7c38eee7eb8eaa4e3890e9e8ce861ab6975c1e1416684b1fb1862d1782b76a"
    },
    "pressure_core.fine.encoders.3.layers.0.weight": {
      "shape": [
        16,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "7932627cfd088d4874ffaa18c62f6b80b84fc4be11c478232bb5cf4e6e3b9d08"
    },
    "pressure_core.fine.encoders.3.layers.3.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "67a1fe4c5403bc257879f8c1b785872619c234839817be792dce919f724ed5de"
    },
    "pressure_core.fine.bottleneck.layers.0.weight": {
      "shape": [
        24,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "2a9862fb35c4f0c618738f42c6af669ce8071a53c576a224ec723fedeb936922"
    },
    "pressure_core.fine.bottleneck.layers.3.weight": {
      "shape": [
        24,
        24,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "d8e092e6be01936d786e8b17a9e604e461ba4af6cb31cde38099893417c90008"
    },
    "pressure_core.fine.decoders.0.layers.0.weight": {
      "shape": [
        16,
        40,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "58d51c78e13a8b2c36b4c035b74dd8a87c2db92b75f8298b629fc5c148a2c02b"
    },
    "pressure_core.fine.decoders.0.layers.3.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "38c9644c2c242ad75ad9070d2ca6e2d2984d3f8e45248a3bd39cd931703e36c2"
    },
    "pressure_core.fine.decoders.1.layers.0.weight": {
      "shape": [
        12,
        28,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "0dd3fffaaf1eb31e81742ac35625c41eb0d61a7577ce0d7e76d5c7e8c538097e"
    },
    "pressure_core.fine.decoders.1.layers.3.weight": {
      "shape": [
        12,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "e662b00c70c4f8eb3fd6af5daefe62d8670faa4381f380ca0952ea272d22e428"
    },
    "pressure_core.fine.decoders.2.layers.0.weight": {
      "shape": [
        8,
        20,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "897216db24db4a4df34d9273d197508f3d0261dfdc5decfafb6982cea0a4337b"
    },
    "pressure_core.fine.decoders.2.layers.3.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "414758deeb72af51fc5c494b967bf9e77167def358b2884608ee0f906080ce38"
    },
    "pressure_core.fine.decoders.3.layers.0.weight": {
      "shape": [
        4,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "817316b8873759212cf3b458109374d35b52a3a77bb3e82aa9e29f83650a0b66"
    },
    "pressure_core.fine.decoders.3.layers.3.weight": {
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "64631b34d1c4b8aa25fde2aad93974bb888bb1ad7166a9b3b4a4cf8bd4cd6c49"
    },
    "pressure_core.fine.head.weight": {
      "shape": [
        1,
        4,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "62c58dcd44aa770b0d67da6cc12dd66057e882a29bc3678a7c1d1bbd6cfd3ee8"
    },
    "request.flux.weight": {
      "shape": [
        1,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "dacf17b5d8b84199c50513a57cecd3233b6049fe116850333618e930e5911041"
    },
    "request.left.weight": {
      "shape": [
        16,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "433a67faf0886b4268a6306d07b8c8868fd35097b79103e0f25aa0ddaef4a8e0"
    },
    "request.right.weight": {
      "shape": [
        16,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "7a7264bee8ddf37a07eef6cdfb395995ed94f3ae8717628c35d4743e7f7bed4b"
    },
    "request.mix.weight": {
      "shape": [
        16,
        80,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "687ec1372c91ec53584ccc0b43e1a22e2b531b534a1ac016cf6ad69c80a2e504"
    },
    "request.head.weight": {
      "shape": [
        1,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "daab1e3a136165a429849358bd019268ed9e29a6161735ca14d65dd63cd1436b"
    },
    "fluid.0.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "5340133b97040a6f67d8a8eabdbf10841bc411e1ba32592e7e257f2d017b4f0f"
    },
    "fluid.0.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "sha256": "f8c12e743cb44ade311b41829826dd9155939648f8883cd808f2fbaf35906ebd"
    },
    "fluid.2.weight": {
      "shape": [
        16,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "3c70cf9900f94e3e187d769d725e0d6e513aba4924765a4835a36a5e33ce99e1"
    },
    "fluid.2.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "sha256": "b6d1f86aa5a91060337ba0fc647e2d427547618b08c3a160e2b848855e659931"
    },
    "request_head.weight": {
      "shape": [
        1,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "97e355a2ae363ee50d0bbd842568db058299a5c6c85143f00352952987a136fd"
    },
    "pressure_velocity.weight": {
      "shape": [
        3,
        1,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "a16382ed95c29c4e2661491153f21948a419285a8253aedc02d91254b9291ec3"
    },
    "velocity_head.weight": {
      "shape": [
        3,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "10cefc43b76e78710875a26c4dc414b0041dda3ad15c4f2da475fff57a056b03"
    },
    "tracer.0.weight": {
      "shape": [
        8,
        29,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "dae7ebb7047ecb9aac8aa40f52fda4252a59a2b9a2dbeed00d77453d970fe4a4"
    },
    "tracer.0.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "76cbf24908d4499eb8e7f25bd279ebdff082dfe91bf08d203a0bd12183e9ec71"
    },
    "tracer_head.weight": {
      "shape": [
        1,
        8,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "781f92828782838b41e1cebc4b952f68dd325eb518d74657872a1fdd42e3a42a"
    },
    "middle.0.weight": {
      "shape": [
        16,
        26,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "bee05327c1123a988c919ab9672e05dc7e583a8df9c423e386ab833d994e5d61"
    },
    "middle.0.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "sha256": "4871f6c7383b1ece5df7bca159ffe4c84c84e6476879029e1ae44219e4853797"
    },
    "middle_p.weight": {
      "shape": [
        1,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "3898685b6df578a6429019198f92c136c14e5038c089733d59edfe8e94f206da"
    },
    "middle_v.weight": {
      "shape": [
        3,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "dc667b1baf69773af5ef2a4a03b25633dd7a17593d2a34c24e891fbae3142bc3"
    },
    "middle_tracer.0.weight": {
      "shape": [
        8,
        19,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "b4ee86f4d5d74765e696cf5fc9618f97eb6ec3c9291639f761ffe5d0d18005b6"
    },
    "middle_tracer.0.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "5f9d3fdb673bb6aecfaef8fd1e4dba8b3789b4b3b208b157f11f63ce0d66e9e8"
    },
    "middle_t.weight": {
      "shape": [
        1,
        8,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "4625f6d7213d8d74dcf5b5d22df36b884652f5ecd326a6ffab894bc68bc704c3"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 975841280
  },
  "outputs": {
    "probe.npy": {
      "bytes": 196648,
      "sha256": "f8a1de202ea621ff32913dfacfdc84773c28ce46c2b27f8acc56f785bfb62bdb"
    },
    "weights.npz": {
      "bytes": 458556,
      "sha256": "a3da5ec538e7fbb2a841700c972232a931c53af5229e896b183fcef723f6a27b"
    },
    "initial-weights.npz": {
      "bytes": 458556,
      "sha256": "44a6a71345f179992f2621a0358b601fe52000b50db5759189e4ef2e5489f3a8"
    },
    "history.json": {
      "bytes": 279485,
      "sha256": "5544322fe73696373de19851894eb439cf9812464503803ac63df0ce0b560b78"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 570.632488421048,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-pressure-expert-v1",
  "velocity_representation": "learned_pressure_conditioned_convolution",
  "supervision_mode": "increments",
  "architecture": "endpoint",
  "feedback_protocol": 2,
  "identifier": "b2-endpoint-pressure-expert",
  "horizons": 2,
  "width": 16,
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
  "initialization": "Frozen a-tail8 pressure expert, managed causal request initialization, seeded width16 fluid/tracer encoders and zero new heads",
  "inference": "One fluid encoder and one frozen pressure expert per block; predicted endpoint plus cheap intermediate decoder or first-prediction feature reuse; learned velocity readout, no numerical solve or post-projection",
  "sample_step_range": [
    0,
    120
  ],
  "loss": "Matched MSE pressure/equation/physical terms; velocity/tracer errors in fixed aggregate native-step units",
  "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
  "sharing_registration_sha256": "ad3f739d27f4a12f0b433d61b34a3c6ce67f4c04dccc55a188f0786559a5581e",
  "step_scale_statistics_sha256": "5f976a2e19e77cf7b46bf701b9e7c7cd65856d8f391db138020c8855592dad04",
  "original_normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037",
  "physical_scales": {
    "pressure": 9.47211357321988,
    "velocity": 0.0064655899998301335,
    "tracer": 0.012995164216288916
  },
  "expert_asset_sha256": {
    "pressure/configuration.json": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1",
    "pressure/fine-weights.npz": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6",
    "pressure/geometry-gains.npz": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d",
    "pressure/receipt.json": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544",
    "request-initialization.json": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab",
    "request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
  },
  "full_fit_registration_sha256": "08ed048038db26f1838a6e74f82ae521c848f06b954a42aaf3000a17794d0510"
}

```
