# Full AI temporal b2-displacement train

Retrospective source review · fit-273889e7a0201f42

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 2; each output advances 1/60 second. Full block duration is 2/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.00166661 | 4.21012 | 0.15940 |
| 8 | 16x16x16 | 0.000749938 | 4.27635 | 0.19605 |
| 8 | 32x32x32 | 0.00189149 | 9.90429 | 0.47160 |
| 8 | 9x11x11 | 0.00148968 | 4.23400 | 0.16498 |
| 16 | 15x6x8 | 0.00437921 | 4.55077 | 0.16040 |
| 16 | 16x16x16 | 0.00586025 | 4.49891 | 0.19602 |
| 16 | 32x32x32 | 0.00324843 | 10.45753 | 0.47723 |
| 16 | 9x11x11 | 0.00645736 | 4.17533 | 0.16505 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 480d903fd12f64c7130577e176878296432640a98f7d5b9471f0d8929f641841; subject: b2-displacement.

**Method.** A learned-displacement full-state composition with100,456parameters:6,843trainable and93,613frozen. A shared width12encoder precedes K2recurrent neural displacement/transport, learned trainable viscosity, frozen flux, learned pressure and tied momentum. Point coefficients are frozen; a local quadratic/curvature correction predicts shifts used by a fixed convex trilinear carrier readout. Both outputs run pressure separately; pressure is downcast into the F32packed state.

**Purpose.** Introduce supervised neural displacement readouts and a shared carrier reconstruction while retaining learned pressure/flux/momentum components; provide the trainable-viscosity predecessor to the later frozen-viscosity family.

**Experiment.** Original16consumed training scenes/2064states at8/16cells per metre and four shapes including32cubed/rectangles, fixed1/60s output cadence and2/60s block advancement.10,000CUDA Adam updates use batch1, private seeds/order and2000one-block then8000two-own-history-block curriculum. All original nested assets/source/typed recipe/input/output identities, complete initial/final/frozen tensors and work counters verify. The displacement objective adds detached midpoint-backtrace labels from actual entering predicted states in alpha-squared units. The teacher is training-only. Its later learning-rate drop is8001; the endpoint fits drop at6001and9001.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 0.00645736, 0.000291671 and 7.50231e-06. Median time is 8.84679 ms per K2 block (4.42340 ms per physical output); the worst case median is 10.92624 ms/output. The retained F64 N1 composition has better worst pressure error and 24.68 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** This retained design has higher worst pressure error and median GPU time than the matched F64 composition. Its method and fitting purpose remain recorded above; these measurements compare distinct designs and do not show that one checkpoint deteriorated. Judge complete physical outputs rather than a pressure-only component or a training-loss value.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Inherited derived pressure maps are explicitly GPU-built precision variants. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-displacement-pair-01/train/receipt.json",
    "sha256": "273889e7a0201f4275e7dc678b483e70d6e619d63dd8610f0d8e6744e999395a"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b2-displacement-pair-01/train/history.json",
    "sha256": "22e3bb00d2afef5f870cde26670be6810b7a62315df9eac1f8ef95583bd89b2f"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/bundle_model_coupled.py",
      "sha256": "103a7fd05bfff77cca9db22390dcb86ed11eabf2438cdedce20297b9d42c4d01"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/bundle_model_decoder.py",
      "sha256": "84825ec60584f94a0eab4d8afe6fe773374c490a48d3a662df8e73b6d663fb45"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/bundle_model_displacement.py",
      "sha256": "75d352ef7a464da7d2a216e4f970cd0ab52244dc6ce75c5ab0c612968736c131"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/bundle_model_pressure_expert.py",
      "sha256": "9cc6527ccf6d273de4d37d8ccff2752159edf040789c74dc1ca85e4382588b75"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/displacement_readout.py",
      "sha256": "fd684b49468f1769fdd60d8ccb3e42d1983d6d171aba6ee6878ce3a84f822e1d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/expert-assets/coupled-asset-provenance.json",
      "sha256": "4cecfc1a7803f6921f4013ecfbc7fe9f85ba649360de05dd30d25e1a9aeb0407"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/expert-assets/gate-bounds.json",
      "sha256": "99ba099508587fa8cd722af3e6ece4ba46eeb9c0d445000599e6b5bc81ebd498"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/expert-assets/gate-bounds.source.json",
      "sha256": "7ea101b7d0bc4ce14ab4e6689ca3ffad7a9faf256263a463c44822952124b020"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/expert-assets/point-initialization.json",
      "sha256": "0be966772e00fb551c7551e596a82bf96aa2fb336a914e06cfe4b12fd725b0ea"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/expert-assets/point-weights.npz",
      "sha256": "88446b9820c746a458f3fc5da008e1de16822c63c5aaed130088962b9bfc9635"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/expert-assets/pressure/configuration.json",
      "sha256": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/expert-assets/pressure/fine-weights.npz",
      "sha256": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/expert-assets/pressure/geometry-gains.npz",
      "sha256": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/expert-assets/pressure/receipt.json",
      "sha256": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/expert-assets/request-initialization.json",
      "sha256": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/expert-assets/request-weights.npz",
      "sha256": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/factored_request_model.py",
      "sha256": "39577e2b6b275850f61f2e440e646ab435def01da1d16b5301fb5329e1fec6b5"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/frozen_temporal_pressure_core.py",
      "sha256": "5e83b77ec90ea3709dab8a84975aa068a60c1acf9e364af0a2b8f545e3f49666"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/full_learned_pressure_map.py",
      "sha256": "a71dd3279eae019267334b113ba3e069f46149842c44ac7ea045e0ad55f20388"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/temporal_displacement_objective.py",
      "sha256": "29f409f28c325bf35985440f391e75ce30b2c38df69aeacf930d71097e7d1022"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/temporal_objective_increment.py",
      "sha256": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/temporal_objective_mse.py",
      "sha256": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/temporal_train_coupled.py",
      "sha256": "3f8465a23e83a07ae64f22936dc2a57d6370224c083cc293a5341c25fb30a89c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/temporal_train_displacement.py",
      "sha256": "73752b4cb01669739d369938d542a59e4668f507dd9e145d71b6e213abc52dd7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/tied_momentum_readout.py",
      "sha256": "589e4086ed8fca8b9b4df89efa6b2c8b5f59778bb2cbe96f6999f16473e8d53c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/training_displacement_teacher.py",
      "sha256": "bf620c6b177d66ac9e7fc6195173732c791e09cc55240237d5f13f87b8dd12ad"
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
  "lesson": "Learned displacement plus fixed interpolation can preserve a local carrier range, but that is not an end-to-end accuracy or drift guarantee. Training’s numerical teacher does not enter deployed inference. Freezing viscosity in later fits changes the trainable coefficient set and calibration, not this parent’s total parameter count.",
  "outcome": "Original training completes10,000updates; final composite objective 0.0058840701621020427. Independent architecture sums reproduce all parameter counts and frozen coefficients remain exact. No fresh common GPU full-field error, complete block latency, independent holdout or600-output drift result is measured in this review.",
  "reproduction": "Retain exact captured config/typed recipe/runtime/objective and all expert assets, including learned pressure gains/provenance, causal request initialization, fixed statistics and original16scene F32corpus/normalization. Displacement additionally needs frozen point calibration and training-only own-history teacher semantics. Original constructors generate neural unit-response matrices on CPU before .to(device), and publication/restore executes CPU neural probes. Qualify an explicit CUDA-only setup/execution variant before running these recipes. Preserve precision/rounding differences visibly; clean retraining, new-Rheon generation and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-273889e7a0201f42",
    "kind": "training_attempt",
    "name": "Full AI temporal b2-displacement train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "273889e7a0201f4275e7dc678b483e70d6e619d63dd8610f0d8e6744e999395a",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b2-displacement-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "52851a02f7cf4162b983a5be029f71a0",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b2-displacement train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 100456,
    "training_closure": {
      "config_sha256": "6b9b31364281be8d8db62dd5561b260cd0e407162f34e53948d93df19f7b91cc",
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
        "bundle_model_coupled.py": "103a7fd05bfff77cca9db22390dcb86ed11eabf2438cdedce20297b9d42c4d01",
        "bundle_model_decoder.py": "84825ec60584f94a0eab4d8afe6fe773374c490a48d3a662df8e73b6d663fb45",
        "bundle_model_displacement.py": "75d352ef7a464da7d2a216e4f970cd0ab52244dc6ce75c5ab0c612968736c131",
        "bundle_model_pressure_expert.py": "9cc6527ccf6d273de4d37d8ccff2752159edf040789c74dc1ca85e4382588b75",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "displacement_readout.py": "fd684b49468f1769fdd60d8ccb3e42d1983d6d171aba6ee6878ce3a84f822e1d",
        "expert-assets/coupled-asset-provenance.json": "4cecfc1a7803f6921f4013ecfbc7fe9f85ba649360de05dd30d25e1a9aeb0407",
        "expert-assets/gate-bounds.json": "99ba099508587fa8cd722af3e6ece4ba46eeb9c0d445000599e6b5bc81ebd498",
        "expert-assets/gate-bounds.source.json": "7ea101b7d0bc4ce14ab4e6689ca3ffad7a9faf256263a463c44822952124b020",
        "expert-assets/point-initialization.json": "0be966772e00fb551c7551e596a82bf96aa2fb336a914e06cfe4b12fd725b0ea",
        "expert-assets/point-weights.npz": "88446b9820c746a458f3fc5da008e1de16822c63c5aaed130088962b9bfc9635",
        "expert-assets/pressure/configuration.json": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1",
        "expert-assets/pressure/fine-weights.npz": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6",
        "expert-assets/pressure/geometry-gains.npz": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d",
        "expert-assets/pressure/receipt.json": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544",
        "expert-assets/request-initialization.json": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab",
        "expert-assets/request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd",
        "factored_request_model.py": "39577e2b6b275850f61f2e440e646ab435def01da1d16b5301fb5329e1fec6b5",
        "frozen_temporal_pressure_core.py": "5e83b77ec90ea3709dab8a84975aa068a60c1acf9e364af0a2b8f545e3f49666",
        "full_learned_pressure_map.py": "a71dd3279eae019267334b113ba3e069f46149842c44ac7ea045e0ad55f20388",
        "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
        "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
        "temporal_displacement_objective.py": "29f409f28c325bf35985440f391e75ce30b2c38df69aeacf930d71097e7d1022",
        "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
        "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
        "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
        "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
        "temporal_objective_increment.py": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e",
        "temporal_objective_mse.py": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136",
        "temporal_train_coupled.py": "3f8465a23e83a07ae64f22936dc2a57d6370224c083cc293a5341c25fb30a89c",
        "temporal_train_displacement.py": "73752b4cb01669739d369938d542a59e4668f507dd9e145d71b6e213abc52dd7",
        "tied_momentum_readout.py": "589e4086ed8fca8b9b4df89efa6b2c8b5f59778bb2cbe96f6999f16473e8d53c",
        "training_displacement_teacher.py": "bf620c6b177d66ac9e7fc6195173732c791e09cc55240237d5f13f87b8dd12ad"
      },
      "horizons": 2,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "962019c0c2fba6f1689f78324c97b6f2b627dd56a9f7fc4046203eea6c805633",
        "bytes": 412266,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "d7cc8d655c5ade8d50db7ec524434715f3053259d169e4e6d2cd3a141eac541c",
        "bytes": 412266,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/22"
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
      "receipt": "runs/full-ai-next-priorities-v1/b2-displacement-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-displacement-pair-01/train/receipt.json",
    "sha256": "273889e7a0201f4275e7dc678b483e70d6e619d63dd8610f0d8e6744e999395a",
    "bytes": 28420
  },
  "bound_training_configuration": {
    "config_sha256": "6b9b31364281be8d8db62dd5561b260cd0e407162f34e53948d93df19f7b91cc",
    "graph_id": "154ae2a9c18d4df3a83bdc43aab9968c",
    "graph_document_sha256": "c3c8e5a087f641a887b225cc2342b061d770489b018e964be15f7a94c5758edd",
    "recipe_path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/train-recipe.json",
    "recipe_sha256": "6c500ebc3a93a57a05784e3f1da2116b78e4542ac30df0a8994e0844256ee8eb",
    "config_path": "runs/full-ai-next-priorities-v1/b2-displacement-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-displacement-fit-v1",
      "velocity_representation": "shared_neural_flux_and_momentum_coefficients",
      "supervision_mode": "increments",
      "architecture": "displacement",
      "feedback_protocol": 2,
      "identifier": "b2-displacement",
      "horizons": 2,
      "width": 12,
      "updates": 10000,
      "single_block_updates": 2000,
      "learning_rate": 0.001,
      "schedule": [
        {
          "update": 8001,
          "learning_rate": 0.0003
        }
      ],
      "seed": 2026102045,
      "sampling_seed": 2026101922,
      "wall_seconds": 1800,
      "physical_context": {
        "dt_s": 0.016666666666666666,
        "rho_kg_m3": 1000.0,
        "mu_pa_s": 0.001,
        "boundaries": "closed regular grid"
      },
      "initialization": "Fresh shared neural fluid cell, qualified pretrained pressure/flux and managed 972-coefficient point calibration",
      "inference": "One shared encoder, fused learned point/local projections, neural displacements, fixed convex field readouts, learned pressure and momentum; all K states",
      "sample_step_range": [
        0,
        120
      ],
      "loss": "Original step-unit full-state objective plus native-qualified displacement labels in alpha-squared units; labels only after prediction",
      "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
      "sharing_registration_sha256": "0a23b692104cdeb55a802fb7037b42055baf9764a3e25d38d5521ea242337a7a",
      "step_scale_statistics_sha256": "5f976a2e19e77cf7b46bf701b9e7c7cd65856d8f391db138020c8855592dad04",
      "original_normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037",
      "physical_scales": {
        "pressure": 9.47211357321988,
        "velocity": 0.0064655899998301335,
        "tracer": 0.012995164216288916
      },
      "expert_asset_sha256": {
        "coupled-asset-provenance.json": "4cecfc1a7803f6921f4013ecfbc7fe9f85ba649360de05dd30d25e1a9aeb0407",
        "gate-bounds.json": "99ba099508587fa8cd722af3e6ece4ba46eeb9c0d445000599e6b5bc81ebd498",
        "gate-bounds.source.json": "7ea101b7d0bc4ce14ab4e6689ca3ffad7a9faf256263a463c44822952124b020",
        "point-initialization.json": "0be966772e00fb551c7551e596a82bf96aa2fb336a914e06cfe4b12fd725b0ea",
        "point-weights.npz": "88446b9820c746a458f3fc5da008e1de16822c63c5aaed130088962b9bfc9635",
        "pressure/configuration.json": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1",
        "pressure/fine-weights.npz": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6",
        "pressure/geometry-gains.npz": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d",
        "pressure/receipt.json": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544",
        "request-initialization.json": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab",
        "request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
      },
      "decoder_representation": "Neural point displacement plus learned second-order curvature; shared velocity/tracer transport readout",
      "pretrained_nonlinear_request_weights_used": false,
      "displacement_auxiliary_weight": 1.0
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
  "parameters": 100456,
  "trainable_parameters": 6843,
  "frozen_parameter_inventory": {
    "pressure_core.source.fine.encoders.0.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.0.layers.3.weight": {
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
    "pressure_core.source.fine.encoders.1.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.1.layers.3.weight": {
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
    "pressure_core.source.fine.encoders.2.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.2.layers.3.weight": {
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
    "pressure_core.source.fine.encoders.3.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.3.layers.3.weight": {
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
    "pressure_core.source.fine.bottleneck.layers.0.weight": {
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
    "pressure_core.source.fine.bottleneck.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.0.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.0.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.1.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.1.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.2.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.2.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.3.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.3.layers.3.weight": {
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
    "pressure_core.source.fine.head.weight": {
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
    "flux.weight": {
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
    "displacement.point_weight": {
      "shape": [
        12,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "db727a29917fae433efcc201f7771f7220b82090c4944346f3551e723cc9d3b1"
    }
  },
  "parent_pressure_learned_parameters": 128938,
  "pressure_expert_calls_per_block": 2,
  "horizons": 2,
  "closure": {
    "config_sha256": "6b9b31364281be8d8db62dd5561b260cd0e407162f34e53948d93df19f7b91cc",
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
      "bundle_model_coupled.py": "103a7fd05bfff77cca9db22390dcb86ed11eabf2438cdedce20297b9d42c4d01",
      "bundle_model_decoder.py": "84825ec60584f94a0eab4d8afe6fe773374c490a48d3a662df8e73b6d663fb45",
      "bundle_model_displacement.py": "75d352ef7a464da7d2a216e4f970cd0ab52244dc6ce75c5ab0c612968736c131",
      "bundle_model_pressure_expert.py": "9cc6527ccf6d273de4d37d8ccff2752159edf040789c74dc1ca85e4382588b75",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "displacement_readout.py": "fd684b49468f1769fdd60d8ccb3e42d1983d6d171aba6ee6878ce3a84f822e1d",
      "expert-assets/coupled-asset-provenance.json": "4cecfc1a7803f6921f4013ecfbc7fe9f85ba649360de05dd30d25e1a9aeb0407",
      "expert-assets/gate-bounds.json": "99ba099508587fa8cd722af3e6ece4ba46eeb9c0d445000599e6b5bc81ebd498",
      "expert-assets/gate-bounds.source.json": "7ea101b7d0bc4ce14ab4e6689ca3ffad7a9faf256263a463c44822952124b020",
      "expert-assets/point-initialization.json": "0be966772e00fb551c7551e596a82bf96aa2fb336a914e06cfe4b12fd725b0ea",
      "expert-assets/point-weights.npz": "88446b9820c746a458f3fc5da008e1de16822c63c5aaed130088962b9bfc9635",
      "expert-assets/pressure/configuration.json": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1",
      "expert-assets/pressure/fine-weights.npz": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6",
      "expert-assets/pressure/geometry-gains.npz": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d",
      "expert-assets/pressure/receipt.json": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544",
      "expert-assets/request-initialization.json": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab",
      "expert-assets/request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd",
      "factored_request_model.py": "39577e2b6b275850f61f2e440e646ab435def01da1d16b5301fb5329e1fec6b5",
      "frozen_temporal_pressure_core.py": "5e83b77ec90ea3709dab8a84975aa068a60c1acf9e364af0a2b8f545e3f49666",
      "full_learned_pressure_map.py": "a71dd3279eae019267334b113ba3e069f46149842c44ac7ea045e0ad55f20388",
      "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
      "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
      "temporal_displacement_objective.py": "29f409f28c325bf35985440f391e75ce30b2c38df69aeacf930d71097e7d1022",
      "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
      "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
      "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
      "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
      "temporal_objective_increment.py": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e",
      "temporal_objective_mse.py": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136",
      "temporal_train_coupled.py": "3f8465a23e83a07ae64f22936dc2a57d6370224c083cc293a5341c25fb30a89c",
      "temporal_train_displacement.py": "73752b4cb01669739d369938d542a59e4668f507dd9e145d71b6e213abc52dd7",
      "tied_momentum_readout.py": "589e4086ed8fca8b9b4df89efa6b2c8b5f59778bb2cbe96f6999f16473e8d53c",
      "training_displacement_teacher.py": "bf620c6b177d66ac9e7fc6195173732c791e09cc55240237d5f13f87b8dd12ad"
    },
    "horizons": 2,
    "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
  },
  "initial_inventory": {
    "pressure_core.source.fine.encoders.0.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.0.layers.3.weight": {
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
    "pressure_core.source.fine.encoders.1.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.1.layers.3.weight": {
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
    "pressure_core.source.fine.encoders.2.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.2.layers.3.weight": {
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
    "pressure_core.source.fine.encoders.3.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.3.layers.3.weight": {
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
    "pressure_core.source.fine.bottleneck.layers.0.weight": {
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
    "pressure_core.source.fine.bottleneck.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.0.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.0.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.1.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.1.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.2.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.2.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.3.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.3.layers.3.weight": {
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
    "pressure_core.source.fine.head.weight": {
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
    "flux.weight": {
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
    "fluid.0.weight": {
      "shape": [
        12,
        10,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "59a3a163cc7be6ba80796f1f42dd9c4fd2094b63d18e024355cdbbb47095121e"
    },
    "fluid.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "c2835958014a4fad584489fe0bf6f4208c5887d1a2ad1c7618b488842a6e23ba"
    },
    "fluid.2.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "7110237819b8d3ed15c989de17c4dcf5eaf461d82d6c53183be58c41a759fcce"
    },
    "fluid.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "3097d7f75cf5cd6eb5954774e38c046df971114dda8f9a9fbd32e2ef1cfcc6c6"
    },
    "local_update.0.weight": {
      "shape": [
        12,
        15,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "ce82954dc5fdfbdda5fa3f5f0545c3a212af5bb8ef32033a363b82cbf921accd"
    },
    "local_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "11270a095c717ea95d231a93d46a6ce3e921d9ace0957b6b707a60b45d3f76fb"
    },
    "local_update.2.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "cb869bf132a1d6fd7ca6efb88124ec30bddb11ca89ad5129bed914bf3bbc45df"
    },
    "local_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "cd10a4f6beb2cbccb1323068bf9d2272be2040f135c861d952a3ff30f358f5f7"
    },
    "displacement.point_weight": {
      "shape": [
        12,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "db727a29917fae433efcc201f7771f7220b82090c4944346f3551e723cc9d3b1"
    },
    "displacement.local_weight": {
      "shape": [
        24,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "09298699bdab5d332cdc2a3742ec0688a46644b8df03d4c96a49134c84fdf356"
    },
    "displacement.context.weight": {
      "shape": [
        36,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "7636c9f8da9a82f9a9fcd804a25795a9c900e1c7e19f7442a7b113c8ccd64eb2"
    },
    "displacement.context.bias": {
      "shape": [
        36
      ],
      "dtype": "<f4",
      "sha256": "81c611f35bff79491538b2f7cf201c7597a661a5c549633541c62bdc8af1613f"
    },
    "displacement.curvature.weight": {
      "shape": [
        12,
        36,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "7636c9f8da9a82f9a9fcd804a25795a9c900e1c7e19f7442a7b113c8ccd64eb2"
    },
    "viscous_head.weight": {
      "shape": [
        3,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "96c5023576733f88726277165f7fa44936426956bcbcca126d7f1169216e1a4b"
    }
  },
  "final_inventory": {
    "pressure_core.source.fine.encoders.0.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.0.layers.3.weight": {
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
    "pressure_core.source.fine.encoders.1.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.1.layers.3.weight": {
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
    "pressure_core.source.fine.encoders.2.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.2.layers.3.weight": {
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
    "pressure_core.source.fine.encoders.3.layers.0.weight": {
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
    "pressure_core.source.fine.encoders.3.layers.3.weight": {
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
    "pressure_core.source.fine.bottleneck.layers.0.weight": {
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
    "pressure_core.source.fine.bottleneck.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.0.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.0.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.1.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.1.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.2.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.2.layers.3.weight": {
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
    "pressure_core.source.fine.decoders.3.layers.0.weight": {
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
    "pressure_core.source.fine.decoders.3.layers.3.weight": {
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
    "pressure_core.source.fine.head.weight": {
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
    "flux.weight": {
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
    "fluid.0.weight": {
      "shape": [
        12,
        10,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "f78b301a4f69177789fa561b4f1068553991e3be123e931677a680ce11612149"
    },
    "fluid.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "dd14179b2944abf14690fb74bcafbeb386f6816462ca2fcda061850e3780412c"
    },
    "fluid.2.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "fa99e61d6b2ccd19418d1a4559c9a03b97629bac834883107838f2d4a4716b9e"
    },
    "fluid.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "8b13ff4549b5573a017b9497c32d280bb58d8abf4ccc471abf3d7d6699348304"
    },
    "local_update.0.weight": {
      "shape": [
        12,
        15,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "6463d7dd06f46a4aaebc7c7b07b6a9df13c3dcf5e6e8ba408e79edfe909121c4"
    },
    "local_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "155985d53f79eff4e614821c8cb3de192b8effd7e0617ac4c3797db6dc2192e0"
    },
    "local_update.2.weight": {
      "shape": [
        12,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "1513a7345a99cb4b94541202dcfd5229147855f4238f976e335db9cb33f294f8"
    },
    "local_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "21721a0546832a5c6cfb0a29e9a94c7b25792922424c78623d88a504d065855e"
    },
    "displacement.point_weight": {
      "shape": [
        12,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "db727a29917fae433efcc201f7771f7220b82090c4944346f3551e723cc9d3b1"
    },
    "displacement.local_weight": {
      "shape": [
        24,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "51c9a7442d0f50c0792942afa9c16a6ebdb824086dd0b50d7023bbc2fcf88afd"
    },
    "displacement.context.weight": {
      "shape": [
        36,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "9c4512e087246cd1663ec8b4e79f74a770c4c60b31f2ed30f27a1a88e23f73c8"
    },
    "displacement.context.bias": {
      "shape": [
        36
      ],
      "dtype": "<f4",
      "sha256": "34183c5e8a284974efbcbcf5d7e6c5738bbbf9420e6c54588994576e42118b0b"
    },
    "displacement.curvature.weight": {
      "shape": [
        12,
        36,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "c6842ca4958bfb1f7abdde34ab22e53c321c8ca8dd5fc6a1ac69e5c25c16bc95"
    },
    "viscous_head.weight": {
      "shape": [
        3,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "4955bbbe13725e3b2b41763f2919d63c509981831efbd54a66958741bec8b2b7"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 1612733952
  },
  "outputs": {
    "probe.npy": {
      "bytes": 196648,
      "sha256": "a9b06fe18870983a63d6cb55ade36ca2d6505e82874ed5346be66c224a9cd493"
    },
    "weights.npz": {
      "bytes": 412266,
      "sha256": "962019c0c2fba6f1689f78324c97b6f2b627dd56a9f7fc4046203eea6c805633"
    },
    "initial-weights.npz": {
      "bytes": 412266,
      "sha256": "d7cc8d655c5ade8d50db7ec524434715f3053259d169e4e6d2cd3a141eac541c"
    },
    "history.json": {
      "bytes": 287799,
      "sha256": "22e3bb00d2afef5f870cde26670be6810b7a62315df9eac1f8ef95583bd89b2f"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 1094.8842589390115,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-displacement-fit-v1",
  "velocity_representation": "shared_neural_flux_and_momentum_coefficients",
  "supervision_mode": "increments",
  "architecture": "displacement",
  "feedback_protocol": 2,
  "identifier": "b2-displacement",
  "horizons": 2,
  "width": 12,
  "updates": 10000,
  "single_block_updates": 2000,
  "learning_rate": 0.001,
  "schedule": [
    {
      "update": 8001,
      "learning_rate": 0.0003
    }
  ],
  "seed": 2026102045,
  "sampling_seed": 2026101922,
  "wall_seconds": 1800,
  "physical_context": {
    "dt_s": 0.016666666666666666,
    "rho_kg_m3": 1000.0,
    "mu_pa_s": 0.001,
    "boundaries": "closed regular grid"
  },
  "initialization": "Fresh shared neural fluid cell, qualified pretrained pressure/flux and managed 972-coefficient point calibration",
  "inference": "One shared encoder, fused learned point/local projections, neural displacements, fixed convex field readouts, learned pressure and momentum; all K states",
  "sample_step_range": [
    0,
    120
  ],
  "loss": "Original step-unit full-state objective plus native-qualified displacement labels in alpha-squared units; labels only after prediction",
  "track_b_registration_sha256": "204d8b457e73661663eab53ba977d9d792c2185431825cc677626e6050036574",
  "sharing_registration_sha256": "0a23b692104cdeb55a802fb7037b42055baf9764a3e25d38d5521ea242337a7a",
  "step_scale_statistics_sha256": "5f976a2e19e77cf7b46bf701b9e7c7cd65856d8f391db138020c8855592dad04",
  "original_normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037",
  "physical_scales": {
    "pressure": 9.47211357321988,
    "velocity": 0.0064655899998301335,
    "tracer": 0.012995164216288916
  },
  "expert_asset_sha256": {
    "coupled-asset-provenance.json": "4cecfc1a7803f6921f4013ecfbc7fe9f85ba649360de05dd30d25e1a9aeb0407",
    "gate-bounds.json": "99ba099508587fa8cd722af3e6ece4ba46eeb9c0d445000599e6b5bc81ebd498",
    "gate-bounds.source.json": "7ea101b7d0bc4ce14ab4e6689ca3ffad7a9faf256263a463c44822952124b020",
    "point-initialization.json": "0be966772e00fb551c7551e596a82bf96aa2fb336a914e06cfe4b12fd725b0ea",
    "point-weights.npz": "88446b9820c746a458f3fc5da008e1de16822c63c5aaed130088962b9bfc9635",
    "pressure/configuration.json": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1",
    "pressure/fine-weights.npz": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6",
    "pressure/geometry-gains.npz": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d",
    "pressure/receipt.json": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544",
    "request-initialization.json": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab",
    "request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
  },
  "decoder_representation": "Neural point displacement plus learned second-order curvature; shared velocity/tracer transport readout",
  "pretrained_nonlinear_request_weights_used": false,
  "displacement_auxiliary_weight": 1.0
}

```
