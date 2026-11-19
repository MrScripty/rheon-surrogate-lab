# Full AI temporal b2-coupled train

Retrospective source review · fit-950df0d22165f059

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 2; each output advances 1/60 second. Full block duration is 2/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.0355906 | 3.69822 | 0.15940 |
| 8 | 16x16x16 | 0.0134294 | 3.75040 | 0.19605 |
| 8 | 32x32x32 | 0.0196932 | 9.64753 | 0.47160 |
| 8 | 9x11x11 | 0.0471804 | 4.18681 | 0.16498 |
| 16 | 15x6x8 | 0.014983 | 3.78408 | 0.16040 |
| 16 | 16x16x16 | 0.0449481 | 3.82138 | 0.19602 |
| 16 | 32x32x32 | 0.0150786 | 9.43626 | 0.47723 |
| 16 | 9x11x11 | 0.0819544 | 3.94929 | 0.16505 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 480d903fd12f64c7130577e176878296432640a98f7d5b9471f0d8929f641841; subject: b2-coupled.

**Method.** A100,611-parameter coupled full-state composition with7,970trainable and92,641frozen coefficients. One width12fluid encoder is shared across a K2block; each physical step still invokes local transport/viscosity, neural flux, the frozen learned pressure map, tied neural momentum and convex tracer readout. Flux uses81inherited learned coefficients; momentum uses their negative spatial transpose. Pressure inherits the a-tail8spatial network and cached learned modal gains. Small registered grids use F64unit-response compositions;32cubed uses the original frozen component. Output packing passes pressure through F32before F64unpack. These are two actual recurrent advances, not one endpoint plus a free intermediate state.

**Purpose.** Couple learned provisional velocity, pressure request and momentum readouts through shared pretrained flux coefficients while sharing the fluid encoder across physical steps.

**Experiment.** Original16consumed fitting scenes/2064states at8/16cells per metre, four shapes including32cubed, fixed1/60s outputs and2/60s block advancement. Fit performs10000CUDAupdates/18000root calls/36000step readouts. All original source/recipe/input/output identities, expert assets, initial/final/frozen tensors and private sample/work counts verify. The first2000updates supervise one block and later8000supervise a second reconstructed own-history block. Inherited flux provenance records1000updates within its11000-update request parent; pressure provenance remains separate from this fit’s updates.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 0.0819544, 0.00243953 and 0.00175785. Median time is 7.80823 ms per K2 block (3.90411 ms per physical output); the worst case median is 10.49649 ms/output. The retained F64 N1 composition has better worst pressure error and 21.78 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** This retained design has higher worst pressure error and median GPU time than the matched F64 composition. Its method and fitting purpose remain recorded above; these measurements compare distinct designs and do not show that one checkpoint deteriorated. Judge complete physical outputs rather than a pressure-only component or a training-loss value.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Inherited derived pressure maps are explicitly GPU-built precision variants. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-coupled-pair-01/train/receipt.json",
    "sha256": "950df0d22165f059fa952e5efba047718e693374c812f7c4a27d0bbfd86e0f17"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b2-coupled-pair-01/train/history.json",
    "sha256": "5a5ead0bc184388245f19a393cbcf521a69d801223a775e024d0ef7a62897b3c"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/bundle_model_coupled.py",
      "sha256": "103a7fd05bfff77cca9db22390dcb86ed11eabf2438cdedce20297b9d42c4d01"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/bundle_model_decoder.py",
      "sha256": "84825ec60584f94a0eab4d8afe6fe773374c490a48d3a662df8e73b6d663fb45"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/bundle_model_pressure_expert.py",
      "sha256": "9cc6527ccf6d273de4d37d8ccff2752159edf040789c74dc1ca85e4382588b75"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/expert-assets/coupled-asset-provenance.json",
      "sha256": "4cecfc1a7803f6921f4013ecfbc7fe9f85ba649360de05dd30d25e1a9aeb0407"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/expert-assets/gate-bounds.json",
      "sha256": "99ba099508587fa8cd722af3e6ece4ba46eeb9c0d445000599e6b5bc81ebd498"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/expert-assets/gate-bounds.source.json",
      "sha256": "7ea101b7d0bc4ce14ab4e6689ca3ffad7a9faf256263a463c44822952124b020"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/expert-assets/pressure/configuration.json",
      "sha256": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/expert-assets/pressure/fine-weights.npz",
      "sha256": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/expert-assets/pressure/geometry-gains.npz",
      "sha256": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/expert-assets/pressure/receipt.json",
      "sha256": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/expert-assets/request-initialization.json",
      "sha256": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/expert-assets/request-weights.npz",
      "sha256": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/factored_request_model.py",
      "sha256": "39577e2b6b275850f61f2e440e646ab435def01da1d16b5301fb5329e1fec6b5"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/frozen_temporal_pressure_core.py",
      "sha256": "5e83b77ec90ea3709dab8a84975aa068a60c1acf9e364af0a2b8f545e3f49666"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/full_learned_pressure_map.py",
      "sha256": "a71dd3279eae019267334b113ba3e069f46149842c44ac7ea045e0ad55f20388"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/temporal_objective_increment.py",
      "sha256": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/temporal_objective_mse.py",
      "sha256": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/temporal_train_coupled.py",
      "sha256": "3f8465a23e83a07ae64f22936dc2a57d6370224c083cc293a5341c25fb30a89c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/tied_momentum_readout.py",
      "sha256": "589e4086ed8fca8b9b4df89efa6b2c8b5f59778bb2cbe96f6999f16473e8d53c"
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
  "lesson": "Sharing the encoder saves some work, but K2still runs the pressure readout twice. Numerical interpolation and physical unit scalings remain explicit parts of the learned composition. Learned-map construction derives coefficients from neural unit queries rather than a numerical inverse, yet the original constructor executes those queries on CPU before .to(device); do not run it unchanged under GPU-only inference policy. Constructor/setup time and matrix memory must be reported separately from warmed complete-block latency.",
  "outcome": "Original fit completed; final recorded composite objective 0.0071535129100084305. All frozen pressure/flux coefficients remain byte-identical to their inherited assets. Fresh common GPU full-state accuracy/timing and600-output drift are pending; loss values that include an auxiliary term are not a fair accuracy ranking.",
  "reproduction": "Preserve exact captured config/model/objective/typed recipe/runtime, original16scene F32corpus/normalization, step-scales.json and every expert asset including learned gains/bounds/provenance. Continuations also require exact continuation-weights.npz and fresh Adam, not restored optimizer moments. The full learned pressure matrices are nonpersistent derived buffers and must be reconstructed. Qualify CUDA-only constructor unit responses and publication probes before executing old recipes; retain arithmetic differences as an execution variant. Clean retraining, new-Rheon data generation and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-950df0d22165f059",
    "kind": "training_attempt",
    "name": "Full AI temporal b2-coupled train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "950df0d22165f059fa952e5efba047718e693374c812f7c4a27d0bbfd86e0f17",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b2-coupled-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "178f324fe7b045d6b1e58be4113bf270",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b2-coupled train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 100611,
    "training_closure": {
      "config_sha256": "4be803494f24b0ff935f438c97f0b4899752e23959cab51fbbfd36d140dec1bc",
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
        "bundle_model_pressure_expert.py": "9cc6527ccf6d273de4d37d8ccff2752159edf040789c74dc1ca85e4382588b75",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "expert-assets/coupled-asset-provenance.json": "4cecfc1a7803f6921f4013ecfbc7fe9f85ba649360de05dd30d25e1a9aeb0407",
        "expert-assets/gate-bounds.json": "99ba099508587fa8cd722af3e6ece4ba46eeb9c0d445000599e6b5bc81ebd498",
        "expert-assets/gate-bounds.source.json": "7ea101b7d0bc4ce14ab4e6689ca3ffad7a9faf256263a463c44822952124b020",
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
        "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
        "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
        "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
        "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
        "temporal_objective_increment.py": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e",
        "temporal_objective_mse.py": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136",
        "temporal_train_coupled.py": "3f8465a23e83a07ae64f22936dc2a57d6370224c083cc293a5341c25fb30a89c",
        "tied_momentum_readout.py": "589e4086ed8fca8b9b4df89efa6b2c8b5f59778bb2cbe96f6999f16473e8d53c"
      },
      "horizons": 2,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "e47cc3ca55f2392d26ea1916e383583ce99f813d51246f64bdce0b1dc8a6746f",
        "bytes": 414052,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "a3d49b170d8dfc5c87d4ee423e6e52c004b8e22b67b2bc8549954ff9e8141708",
        "bytes": 414052,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/19"
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
      "receipt": "runs/full-ai-next-priorities-v1/b2-coupled-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b2-coupled-pair-01/train/receipt.json",
    "sha256": "950df0d22165f059fa952e5efba047718e693374c812f7c4a27d0bbfd86e0f17",
    "bytes": 29115
  },
  "bound_training_configuration": {
    "config_sha256": "4be803494f24b0ff935f438c97f0b4899752e23959cab51fbbfd36d140dec1bc",
    "graph_id": "7136c8b8667a401684ea513099c96414",
    "graph_document_sha256": "15baaa8d064b27bedf0eead082af834ee9b4f6c2106c3adb7825087198210462",
    "recipe_path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/train-recipe.json",
    "recipe_sha256": "722aabee4472378df1fb1a197038013238bb46dd2afd3579fbd499eac02ec617",
    "config_path": "runs/full-ai-next-priorities-v1/b2-coupled-project-01/config.json",
    "configuration": {
      "schema": "rheon-temporal-fit-coupled-v1",
      "velocity_representation": "shared_neural_flux_and_momentum_coefficients",
      "supervision_mode": "increments",
      "architecture": "coupled",
      "feedback_protocol": 2,
      "identifier": "b2-coupled",
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
      "initialization": "Frozen neural pressure map and calibrated neural flux only; seeded shared width12 cell and zero local transport/viscosity heads",
      "inference": "One shared encoder; Kcheap local updates and global learned pressure/flux/momentum readouts; raw full-state outputs, no native pressure or transport solver or external projection",
      "sample_step_range": [
        0,
        120
      ],
      "loss": "Matched MSE pressure/equation/physical terms; velocity/tracer errors in fixed aggregate native-step units",
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
        "pressure/configuration.json": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1",
        "pressure/fine-weights.npz": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6",
        "pressure/geometry-gains.npz": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d",
        "pressure/receipt.json": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544",
        "request-initialization.json": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab",
        "request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
      },
      "decoder_representation": "Coupled learned provisional velocity, flux and momentum; entering-velocity tracer staging; same shared cell across K",
      "pretrained_nonlinear_request_weights_used": false,
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
  "parameters": 100611,
  "trainable_parameters": 7970,
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
    }
  },
  "parent_pressure_learned_parameters": 128938,
  "pressure_expert_calls_per_block": 2,
  "horizons": 2,
  "closure": {
    "config_sha256": "4be803494f24b0ff935f438c97f0b4899752e23959cab51fbbfd36d140dec1bc",
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
      "bundle_model_pressure_expert.py": "9cc6527ccf6d273de4d37d8ccff2752159edf040789c74dc1ca85e4382588b75",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "expert-assets/coupled-asset-provenance.json": "4cecfc1a7803f6921f4013ecfbc7fe9f85ba649360de05dd30d25e1a9aeb0407",
      "expert-assets/gate-bounds.json": "99ba099508587fa8cd722af3e6ece4ba46eeb9c0d445000599e6b5bc81ebd498",
      "expert-assets/gate-bounds.source.json": "7ea101b7d0bc4ce14ab4e6689ca3ffad7a9faf256263a463c44822952124b020",
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
      "temporal_features.py": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a",
      "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a",
      "temporal_feedback.py": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801",
      "temporal_objective.py": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250",
      "temporal_objective_increment.py": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e",
      "temporal_objective_mse.py": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136",
      "temporal_train_coupled.py": "3f8465a23e83a07ae64f22936dc2a57d6370224c083cc293a5341c25fb30a89c",
      "tied_momentum_readout.py": "589e4086ed8fca8b9b4df89efa6b2c8b5f59778bb2cbe96f6999f16473e8d53c"
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
      "sha256": "3a3274af714eebaa20da2d41fedc50f42d80a82524a477d76b81c3d2d7fcca8c"
    },
    "fluid.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "38859782704b9ea1b7ea08be5dcfb3f706c2c08de31bcbeb5967bf947a64557e"
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
      "sha256": "c6fde81a5016396c10439b4df26c07a36d6d9ac7e8a2b5ce88d6831fb5371faf"
    },
    "fluid.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "e3259d308a357be60f02fb29cf607788f0434f61e9b442bcd640aec5388040d4"
    },
    "left.weight": {
      "shape": [
        12,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "447a4db0cc4b391c47d4d469b97bc34f3d2e3c15ab50de550f799b1ab20fb64a"
    },
    "right.weight": {
      "shape": [
        12,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "408a033c4799fb5068e7a949f7337302fc66edc1cbde14800f7febcf6dacc7d0"
    },
    "local_update.0.weight": {
      "shape": [
        12,
        39,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "9555988a2f983e5ebaf7eaf404e3a476adeb8712d30276b11ca6ac197cb9cd99"
    },
    "local_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "5a5fc3755e149a613a1d4e7539a1ed9c7af5c18835863e98c8b8e6defc73eeac"
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
      "sha256": "8781044759192aab9c6e1de5370a0681fa5118963fb61d2fa03185c0710ee220"
    },
    "local_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "a7591c09fd10b0a2348249f6e4f4604d86b84a14e39d3f71c57c8842aa92c017"
    },
    "transport_context.weight": {
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
    "transport_context.bias": {
      "shape": [
        36
      ],
      "dtype": "<f4",
      "sha256": "81c611f35bff79491538b2f7cf201c7597a661a5c549633541c62bdc8af1613f"
    },
    "transport_head.weight": {
      "shape": [
        3,
        36,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "1fe2373734955e60c172999142934b52e69ba7ab9039b3c18ea54082ba32afcd"
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
    },
    "tracer_readout.head.coefficients.0.weight": {
      "shape": [
        16,
        43,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "64f665a280f8f80d1f560ee207a14608177bbddcaa8b97564df58cfa3d00f99f"
    },
    "tracer_readout.head.coefficients.0.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "sha256": "f78816fcecdc9c18977a1f46a34b29087c7ef05cd5b83e94e8a41e56bc736ce3"
    },
    "tracer_readout.head.coefficients.2.weight": {
      "shape": [
        27,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "7636c9f8da9a82f9a9fcd804a25795a9c900e1c7e19f7442a7b113c8ccd64eb2"
    },
    "tracer_readout.head.coefficients.2.bias": {
      "shape": [
        27
      ],
      "dtype": "<f4",
      "sha256": "77133f431d5e12dd850002c0d3d4e0fecbe3a7a699d604dc8c5eae9976e1d260"
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
      "sha256": "fbc476cf6e9e726c9ff86b218fe908f0115bda91d2a3652a4f8d97434fd923ba"
    },
    "fluid.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "e3f2a1b12a54084896a893e4f8fd846edf33e57f257bf1d9b526ca69c89aefc1"
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
      "sha256": "4a47bc4d9cc90726c63cb72ad4aed949f3cd8d466c6255d554fe4267ced6b7d7"
    },
    "fluid.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "acc657b7ebc481db95d58b501397891cc02e438a965d196d72001853663cd58b"
    },
    "left.weight": {
      "shape": [
        12,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "99c37d39961c59e84450b46b6f0e6ab2db57a8c14f77a5e13cf2215225ce2f94"
    },
    "right.weight": {
      "shape": [
        12,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "3d0f942f90368cdc3e8ed82dfe84c40178e4cff0242bac6a5819b8b8c3d8d51e"
    },
    "local_update.0.weight": {
      "shape": [
        12,
        39,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "433bda6f9c21c2146187336caa2bc580a488819c1872afbad3b531ec5da61f64"
    },
    "local_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "23cd05f6113c2e1b0f73df89edda1a6ced68241084d7d4746e771ef094e8a4a9"
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
      "sha256": "57dbd14291f1843416a10033ac6dbf769e5cec68fa72c0c19a8bfbc45ebf98a5"
    },
    "local_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "723dc444475f02565c06d0976cf3ee0cdc58ef54e067176813e69e42b3bd498e"
    },
    "transport_context.weight": {
      "shape": [
        36,
        12,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "bff5708340198fde721db8a469c8eedf508aa04c42f9d8bfaf1f8524643db68b"
    },
    "transport_context.bias": {
      "shape": [
        36
      ],
      "dtype": "<f4",
      "sha256": "eb926732d09d6708957e8ceac45d6b50bed93d2d637e12ea3c82d1a47b24924c"
    },
    "transport_head.weight": {
      "shape": [
        3,
        36,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "cbbb890a76099205b554f2c40cfbf5ee9d83290ff131e03fbc86159fdacca4e0"
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
      "sha256": "030815dbc25ffa0ec791c84facfd162d2141c9dc07d18b33e6c28afa095b3217"
    },
    "tracer_readout.head.coefficients.0.weight": {
      "shape": [
        16,
        43,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "64222dc59c17576c458470c6bc85ec180cd8f6ba8e4425344ce1f4dad81a6681"
    },
    "tracer_readout.head.coefficients.0.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "sha256": "7c1e12d3de1f5d398421f1ef272e30905a439d3804ed11083c4ce368fa6dd740"
    },
    "tracer_readout.head.coefficients.2.weight": {
      "shape": [
        27,
        16,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "ea05c532338755dcdf67df988f038dc96015ad439dca05859ae870f1256a6c68"
    },
    "tracer_readout.head.coefficients.2.bias": {
      "shape": [
        27
      ],
      "dtype": "<f4",
      "sha256": "cf51c7d37c3b72916bbc68dded0be56eec9971dadbe257d3134a8994392d1465"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 1301565952
  },
  "outputs": {
    "probe.npy": {
      "bytes": 196648,
      "sha256": "91b2a6e0ff65de80cbb39edcf656513fe2fcacef9fb9266cec0b4a3584db9340"
    },
    "weights.npz": {
      "bytes": 414052,
      "sha256": "e47cc3ca55f2392d26ea1916e383583ce99f813d51246f64bdce0b1dc8a6746f"
    },
    "initial-weights.npz": {
      "bytes": 414052,
      "sha256": "a3d49b170d8dfc5c87d4ee423e6e52c004b8e22b67b2bc8549954ff9e8141708"
    },
    "history.json": {
      "bytes": 278957,
      "sha256": "5a5ead0bc184388245f19a393cbcf521a69d801223a775e024d0ef7a62897b3c"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 791.020926001016,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-temporal-fit-coupled-v1",
  "velocity_representation": "shared_neural_flux_and_momentum_coefficients",
  "supervision_mode": "increments",
  "architecture": "coupled",
  "feedback_protocol": 2,
  "identifier": "b2-coupled",
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
  "initialization": "Frozen neural pressure map and calibrated neural flux only; seeded shared width12 cell and zero local transport/viscosity heads",
  "inference": "One shared encoder; Kcheap local updates and global learned pressure/flux/momentum readouts; raw full-state outputs, no native pressure or transport solver or external projection",
  "sample_step_range": [
    0,
    120
  ],
  "loss": "Matched MSE pressure/equation/physical terms; velocity/tracer errors in fixed aggregate native-step units",
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
    "pressure/configuration.json": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1",
    "pressure/fine-weights.npz": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6",
    "pressure/geometry-gains.npz": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d",
    "pressure/receipt.json": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544",
    "request-initialization.json": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab",
    "request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
  },
  "decoder_representation": "Coupled learned provisional velocity, flux and momentum; entering-velocity tracer staging; same shared cell across K",
  "pretrained_nonlinear_request_weights_used": false,
  "full_fit_registration_sha256": "08ed048038db26f1838a6e74f82ae521c848f06b954a42aaf3000a17794d0510"
}

```
