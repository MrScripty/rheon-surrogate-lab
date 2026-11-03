# Full AI temporal b1-fixed-viscosity-s3 train

Retrospective source review · fit-e7801e768b065ddc

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

## Matched GPU benchmark

Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.

K = 1; each output advances 1/60 second. Full block duration is 1/60 seconds. The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.

| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |
| --- | --- | --- | --- | --- |
| 8 | 15x6x8 | 0.000128065 | 5.05547 | 0.25071 |
| 8 | 16x16x16 | 3.40279e-05 | 5.29968 | 0.27377 |
| 8 | 32x32x32 | 0.000384222 | 11.33485 | 0.58808 |
| 8 | 9x11x11 | 4.44132e-05 | 4.94758 | 0.25054 |
| 16 | 15x6x8 | 9.33514e-05 | 5.13472 | 0.24466 |
| 16 | 16x16x16 | 5.34572e-05 | 5.35750 | 0.26408 |
| 16 | 32x32x32 | 0.000523739 | 11.22978 | 0.58521 |
| 16 | 9x11x11 | 6.63626e-05 | 5.07712 | 0.25399 |

Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.

Retained measurement source: 72069eca9de4b6389615baa5c863b99f128efc57cea50b83b056de51a702ba48; subject: b1-fixed-viscosity-s3.

**Method.** A100,456-parameter recurrent full-state generator for K=1:6,600trainable coefficients and93,856frozen spatial-pressure/flux/point/viscosity coefficients. One width12fluid encoder is shared across a pass. Each output updates retained features, predicts four carrier displacements from a fused frozen-point/trainable-local projection with an alpha-squared curvature correction, and reconstructs velocity/tracer through fixed convex trilinear gathers. Frozen learned viscosity then acts on forced velocity, followed by learned flux, the inherited pressure map and tied momentum. Koutputs still require Kpressure/flux/momentum readouts and Ktransport reconstructions; there is no cheap endpoint-only pressure shortcut.

**Purpose.** Test the fixed learned243-coefficient viscosity calibration within a displacement-based full-state generator, at K=1and model seed2026102047. Keep the trainable architecture independent of K and balance the number of predicted physical training states across horizons.

**Experiment.** Original16consumed scenes/2064states,8/16cells per metre and four shapes including32cubed/rectangles, fixed1/60s outputs with K=1advancing1/60s. The fit runs20000CUDA Adam updates, first4000one-block then two own-history blocks, for36000encoder calls and36,000physical predictions. Rate0.001drops to0.0003at update16001. Private sampling seed2026101924is shared within this seed group; both model and sampling seeds differ between repetitions. Same-seed initial tensors match exactly across available K. A training-only native-qualified midpoint backtrace supplies detached displacement labels from the actual entering predicted state; target error is scaled by alpha squared. Labels are consumed after prediction and are absent from the inference module. All original receipts/recipes/inputs/outputs, nested assets, tensors and work counters verify.

**Observed outcome.** Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are 0.000523739, 0.000291008 and 3.4463e-06. Median time is 5.30128 ms per K1 block (5.30128 ms per physical output); the worst case median is 11.73795 ms/output. The retained F64 N1 composition has better worst pressure error and 20.52 times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.

**Lesson.** The controlled K1 diagnostic locates the fixed-viscosity series' shared 32-cubed pressure limit mainly in its CNN fallback. Other pressure-map and request errors remain. Freezing accurate local coefficients does not ensure a comparably accurate full generator; compare the complete pressure, velocities and tracer with the F64 composition before promoting a design.

**Reproduction.** Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. Inherited derived pressure maps are explicitly GPU-built precision variants. Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-pair-01/train/receipt.json",
    "sha256": "e7801e768b065ddcfceb69a8ca75c951e0a06749dba8aefee4c996aa1e39249b"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-pair-01/train/history.json",
    "sha256": "8f270908529237b7072c675ea82b3308db10078c9c5b80a4a0819450f77659e1"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/bundle_model_coupled.py",
      "sha256": "103a7fd05bfff77cca9db22390dcb86ed11eabf2438cdedce20297b9d42c4d01"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/bundle_model_decoder.py",
      "sha256": "84825ec60584f94a0eab4d8afe6fe773374c490a48d3a662df8e73b6d663fb45"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/bundle_model_displacement.py",
      "sha256": "75d352ef7a464da7d2a216e4f970cd0ab52244dc6ce75c5ab0c612968736c131"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/bundle_model_fixed_viscosity.py",
      "sha256": "5b64ab447d40c19774bace1f246d3dff301534fa6c6f7c714376b1e3bdca6b33"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/bundle_model_pressure_expert.py",
      "sha256": "9cc6527ccf6d273de4d37d8ccff2752159edf040789c74dc1ca85e4382588b75"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/displacement_readout.py",
      "sha256": "fd684b49468f1769fdd60d8ccb3e42d1983d6d171aba6ee6878ce3a84f822e1d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/coupled-asset-provenance.json",
      "sha256": "4cecfc1a7803f6921f4013ecfbc7fe9f85ba649360de05dd30d25e1a9aeb0407"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/gate-bounds.json",
      "sha256": "99ba099508587fa8cd722af3e6ece4ba46eeb9c0d445000599e6b5bc81ebd498"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/gate-bounds.source.json",
      "sha256": "7ea101b7d0bc4ce14ab4e6689ca3ffad7a9faf256263a463c44822952124b020"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/point-initialization.json",
      "sha256": "0be966772e00fb551c7551e596a82bf96aa2fb336a914e06cfe4b12fd725b0ea"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/point-weights.npz",
      "sha256": "88446b9820c746a458f3fc5da008e1de16822c63c5aaed130088962b9bfc9635"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/pressure/configuration.json",
      "sha256": "2c06da3852d27e0789c90d06f6842fb829258c6241fdf18ea99aa6c13728b5b1"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/pressure/fine-weights.npz",
      "sha256": "1b651a98bb8ef9ed8e30ffb761e47db36ad5943b3191970c61a1b3dc40bd11c6"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/pressure/geometry-gains.npz",
      "sha256": "6384d7f83526187243e4f8aedcda452baf2b387e720664affbceffce395abb2d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/pressure/receipt.json",
      "sha256": "d3a9a08d25bfce560d3685c5fd1cce0b5fc6959e41a7cbd3d946bd35b246c544"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/request-initialization.json",
      "sha256": "486cd9be0174c48bb454a3d0659871f567d4c531c8ecc8c380872ec3ff4c32ab"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/request-weights.npz",
      "sha256": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/viscosity-initialization.json",
      "sha256": "b530f9d4b9bd0f9b763ce26244b7bd7fefa97a4a16e7cca18cf96bd39aa22ffa"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/expert-assets/viscosity-weights.npz",
      "sha256": "f7c76bf1e24ce44d7d4b863140445c30f36a547526168f61917d7505b16953dd"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/factored_request_model.py",
      "sha256": "39577e2b6b275850f61f2e440e646ab435def01da1d16b5301fb5329e1fec6b5"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/frozen_temporal_pressure_core.py",
      "sha256": "5e83b77ec90ea3709dab8a84975aa068a60c1acf9e364af0a2b8f545e3f49666"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/full_learned_pressure_map.py",
      "sha256": "a71dd3279eae019267334b113ba3e069f46149842c44ac7ea045e0ad55f20388"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_displacement_objective.py",
      "sha256": "29f409f28c325bf35985440f391e75ce30b2c38df69aeacf930d71097e7d1022"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_features.py",
      "sha256": "88632d5e3da899e72005e1778c9a89cbcd0b8b812a754e98d40a95f9b2cb0d5a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_feedback.py",
      "sha256": "086fec14f9a88a7406adba33d83a8dc0e86d7d6481d00546369645fe48460801"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_objective.py",
      "sha256": "0081c3ba8f3a45465d561a7cbf0349a8b5b613682e03e09ae523028997a3f250"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_objective_increment.py",
      "sha256": "30e149224f0f335921fe225d1ba0b6d90e4d204c51d8eab80aac6202af61fc5e"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_objective_mse.py",
      "sha256": "0a0a7ec0faf8e5984802c89a6878e156535eef7b5b6f17a843150d98d1331136"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_train_coupled.py",
      "sha256": "3f8465a23e83a07ae64f22936dc2a57d6370224c083cc293a5341c25fb30a89c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_train_displacement.py",
      "sha256": "73752b4cb01669739d369938d542a59e4668f507dd9e145d71b6e213abc52dd7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/temporal_train_fixed_viscosity.py",
      "sha256": "5c8af4dca071a228402b8fea5460b3a6599bb5d290a76c81fa6edf5317471f17"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/tied_momentum_readout.py",
      "sha256": "589e4086ed8fca8b9b4df89efa6b2c8b5f59778bb2cbe96f6999f16473e8d53c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/training_displacement_teacher.py",
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
  "lesson": "Equal predicted-state budgets do not mean equal optimizer-update counts or encoder calls. K1/2/3/4use20,000/10,000/6,667/5,000updates. Repetitions change model initialization and sampling order together, so they do not isolate initialization sensitivity alone. The main encoder is shared, while costly pressure/transport work repeats for every output. Keep fixed readout interpolation, precision conversions, constructor matrix work and training-only numerical teachers explicit rather than treating every Koutput architecture as equivalent compute sharing.",
  "outcome": "Original training completed20000updates; last logged composite objective 0.0033332558591759633. Frozen pressure/flux/point/viscosity tensors remain byte-identical. Independent counts reproduce6,600trainable plus93,856frozen coefficients and36,000physical training predictions. No fresh comparable GPU field accuracy, full-block latency, drift or scientific promotion is established by this review.",
  "reproduction": "Retain exact captured config/typed recipe/model/objective/runtime and private seeds/order, the original16scene F32archive/normalization and step-scales.json, all expert assets including frozen point/viscosity calibrations, learned pressure gains/bounds and parent provenance. Preserve the K-specific update/curriculum/rate budget and actual-own-history displacement labels. The old constructor and final publication/restore probes execute CPU neural work; qualify a GPU-only variant before executing them. Derived learned pressure matrices are nonpersistent buffers. Clean retraining, new-Rheon regeneration, fresh common benchmarks and600-output drift remain unqualified."
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-e7801e768b065ddc",
    "kind": "training_attempt",
    "name": "Full AI temporal b1-fixed-viscosity-s3 train",
    "task": "full_state_generator",
    "schema": "rheon-temporal-training-result-v1",
    "method": {},
    "receipt_sha256": "e7801e768b065ddcfceb69a8ca75c951e0a06749dba8aefee4c996aa1e39249b",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "f2a41e2042624421bfbde13f02a1de0c",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI temporal b1-fixed-viscosity-s3 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 20000,
    "parameter_fits": null,
    "declared_parameter_count": 100456,
    "training_closure": {
      "config_sha256": "5e2ebb6db140699a202ec8f74717eac03dfa204545e34342a93834cafc840d3f",
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
        "bundle_model_fixed_viscosity.py": "5b64ab447d40c19774bace1f246d3dff301534fa6c6f7c714376b1e3bdca6b33",
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
        "expert-assets/viscosity-initialization.json": "b530f9d4b9bd0f9b763ce26244b7bd7fefa97a4a16e7cca18cf96bd39aa22ffa",
        "expert-assets/viscosity-weights.npz": "f7c76bf1e24ce44d7d4b863140445c30f36a547526168f61917d7505b16953dd",
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
        "temporal_train_fixed_viscosity.py": "5c8af4dca071a228402b8fea5460b3a6599bb5d290a76c81fa6edf5317471f17",
        "tied_momentum_readout.py": "589e4086ed8fca8b9b4df89efa6b2c8b5f59778bb2cbe96f6999f16473e8d53c",
        "training_displacement_teacher.py": "bf620c6b177d66ac9e7fc6195173732c791e09cc55240237d5f13f87b8dd12ad"
      },
      "horizons": 1,
      "normalization_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "c5bcc002385125180388a7680b53a2b920f9ad4278d1cebcf9ba669c8acfeff9",
        "bytes": 412266,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "37cd276d1b84c0cba7817e738e58a58198fb43892b6120c87e03969a2e860a52",
        "bytes": 412266,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/28",
      "temporal-executions/14"
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
      "receipt": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_generator",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-pair-01/train/receipt.json",
    "sha256": "e7801e768b065ddcfceb69a8ca75c951e0a06749dba8aefee4c996aa1e39249b",
    "bytes": 29094
  },
  "bound_training_configuration": {
    "config_sha256": "5e2ebb6db140699a202ec8f74717eac03dfa204545e34342a93834cafc840d3f",
    "graph_id": "c41c16fe08be46dcb5ce084bdf9ed3ad",
    "graph_document_sha256": "e32c9c7ffba4a1deab576d37451c09fcd40a21f8a673ad730884bea8ed14d185",
    "recipe_path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/train-recipe.json",
    "recipe_sha256": "4fa3779e9fa09a3f43713ed5fc1309ae2806966cfae22350f27db2c0e8fdb987",
    "config_path": "runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-project-01/config.json",
    "configuration": {
      "schema": "rheon-fixed-neural-viscosity-fit-v1",
      "velocity_representation": "shared_neural_flux_and_momentum_coefficients",
      "supervision_mode": "increments",
      "architecture": "displacement",
      "feedback_protocol": 2,
      "identifier": "b1-fixed-viscosity-s3",
      "horizons": 1,
      "width": 12,
      "updates": 20000,
      "single_block_updates": 4000,
      "learning_rate": 0.001,
      "schedule": [
        {
          "update": 16001,
          "learning_rate": 0.0003
        }
      ],
      "seed": 2026102047,
      "sampling_seed": 2026101924,
      "wall_seconds": 1800,
      "physical_context": {
        "dt_s": 0.016666666666666666,
        "rho_kg_m3": 1000.0,
        "mu_pa_s": 0.001,
        "boundaries": "closed regular grid"
      },
      "initialization": "Fresh shared cell, common seed; managed point/flux/pressure and managed243viscosity coefficients frozen",
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
        "request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd",
        "viscosity-initialization.json": "b530f9d4b9bd0f9b763ce26244b7bd7fefa97a4a16e7cca18cf96bd39aa22ffa",
        "viscosity-weights.npz": "f7c76bf1e24ce44d7d4b863140445c30f36a547526168f61917d7505b16953dd"
      },
      "decoder_representation": "Neural point displacement plus learned second-order curvature; shared velocity/tracer transport readout",
      "pretrained_nonlinear_request_weights_used": false,
      "displacement_auxiliary_weight": 1.0,
      "viscosity_coefficients_frozen_from_start": true,
      "replication_ordinal": 3,
      "replicates_primary": "b1-fixed-viscosity",
      "primary_config_sha256": "308699687d13d9e72da6ad71a7b513ac35dc2487b886b1b8f5849fc76c39ce6d",
      "replication_selection_sha256": "370b5f96cff86b451a7281dadd3aa07fb971c25a04db22148f7de78e224eea02"
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
  "optimizer_updates": 20000,
  "parameters": 100456,
  "trainable_parameters": 6600,
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
      "sha256": "cc4836fbce947f9956aeacf004ed185042f0fc3fc213b657945a9eef0d7a35d6"
    }
  },
  "parent_pressure_learned_parameters": 128938,
  "pressure_expert_calls_per_block": 1,
  "horizons": 1,
  "closure": {
    "config_sha256": "5e2ebb6db140699a202ec8f74717eac03dfa204545e34342a93834cafc840d3f",
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
      "bundle_model_fixed_viscosity.py": "5b64ab447d40c19774bace1f246d3dff301534fa6c6f7c714376b1e3bdca6b33",
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
      "expert-assets/viscosity-initialization.json": "b530f9d4b9bd0f9b763ce26244b7bd7fefa97a4a16e7cca18cf96bd39aa22ffa",
      "expert-assets/viscosity-weights.npz": "f7c76bf1e24ce44d7d4b863140445c30f36a547526168f61917d7505b16953dd",
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
      "temporal_train_fixed_viscosity.py": "5c8af4dca071a228402b8fea5460b3a6599bb5d290a76c81fa6edf5317471f17",
      "tied_momentum_readout.py": "589e4086ed8fca8b9b4df89efa6b2c8b5f59778bb2cbe96f6999f16473e8d53c",
      "training_displacement_teacher.py": "bf620c6b177d66ac9e7fc6195173732c791e09cc55240237d5f13f87b8dd12ad"
    },
    "horizons": 1,
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
      "sha256": "b7b677948fb838338e6e8ef0ddeb37d67d0a219cdaf36bf0fe0661d73f85296b"
    },
    "fluid.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "ee7bd37e8dc0217e7b3813394eaea41b5127312219bfb139ce1cd3742661d498"
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
      "sha256": "420b331b024a6c3a773d703909f7ebbafac8cfe64d811cb8efba97ab1c23c253"
    },
    "fluid.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "924195f67adf9c4db585299f083cf1ef67da7ae9cdac47dff0d7f113124a2bbc"
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
      "sha256": "8e19ce0d9fd9e33ebbc9794466db3c70e27bc21e5a755781c98a84a0f3aa3a7a"
    },
    "local_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "103f4ba34a2ad13ca8964d1e975b9353fa729917399598765f63cc3fd6a0d652"
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
      "sha256": "7b90c77ce1c8acc6585ecf6908c6ed5c776c0d1de428dd81fe14aeef0373a4e2"
    },
    "local_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "baa3e69eeb8840fda2c985fc6d075b48bfe6b6b7d2f53f2e9e63f614281c5aaf"
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
      "sha256": "5cfa1a0b2351054b441921e9cfb2aaf71dafa1b9efdd8f072378540723c3425b"
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
      "sha256": "cc4836fbce947f9956aeacf004ed185042f0fc3fc213b657945a9eef0d7a35d6"
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
      "sha256": "3e591a27c518abdd35bebcac55a976061eb35c8ad471d3105a4b3f9e87fb74f3"
    },
    "fluid.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "a952e8d5a975f54b45fb71bf7d2ed58254702d220fd9137b3490c3e6547569ba"
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
      "sha256": "0ccaa846b63c18753bbac47fa25362d5bc4da8a7d9a2caf449f2dc4b027e0507"
    },
    "fluid.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "3e73b731758d78b1540cefd2ac0d86279f55354ba666a97a51639848bdce1fcf"
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
      "sha256": "a2ab446579f736f1cc4a23117237a6298bae58008842904e850f72c8c3e5243b"
    },
    "local_update.0.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "abe47c7973a067d9ee33f9cd7973b4d1dabbea4a0ac224892d67466c0fa16ad2"
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
      "sha256": "a0b8d7bd65ceacfba8939b694ad1a7472c0dcbd8dcda3597c662dd12af0764a6"
    },
    "local_update.2.bias": {
      "shape": [
        12
      ],
      "dtype": "<f4",
      "sha256": "68000287c8dacb57119a95f7077d1c61fc0612edb924f153a0225f127307376b"
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
      "sha256": "980e30c43b31608ee917a99dadcdf28526cc31f91043d809bd8a9e52c6132975"
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
      "sha256": "dcf52c3a79bb834fc62118e3410cf627e2dc56374b4f16eadb4488e8d5ae49f0"
    },
    "displacement.context.bias": {
      "shape": [
        36
      ],
      "dtype": "<f4",
      "sha256": "5e39e9abf728912f7a269e1a59b76b52855e9e73e1d1029a34b4a879bbea6560"
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
      "sha256": "90a7e9f83d957eebea5a60806df0bb636f827cfc3bca7e81b9e3a9c0eeb935c6"
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
      "sha256": "cc4836fbce947f9956aeacf004ed185042f0fc3fc213b657945a9eef0d7a35d6"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 1285530624
  },
  "outputs": {
    "probe.npy": {
      "bytes": 98388,
      "sha256": "e1ed71cd2e9bd3f652f0552264605c219d836425ad1ca3b9a64b2237ddba92ed"
    },
    "weights.npz": {
      "bytes": 412266,
      "sha256": "c5bcc002385125180388a7680b53a2b920f9ad4278d1cebcf9ba669c8acfeff9"
    },
    "initial-weights.npz": {
      "bytes": 412266,
      "sha256": "37cd276d1b84c0cba7817e738e58a58198fb43892b6120c87e03969a2e860a52"
    },
    "history.json": {
      "bytes": 565028,
      "sha256": "8f270908529237b7072c675ea82b3308db10078c9c5b80a4a0819450f77659e1"
    },
    "normalization.json": {
      "bytes": 981,
      "sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 1161.4603154949727,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-fixed-neural-viscosity-fit-v1",
  "velocity_representation": "shared_neural_flux_and_momentum_coefficients",
  "supervision_mode": "increments",
  "architecture": "displacement",
  "feedback_protocol": 2,
  "identifier": "b1-fixed-viscosity-s3",
  "horizons": 1,
  "width": 12,
  "updates": 20000,
  "single_block_updates": 4000,
  "learning_rate": 0.001,
  "schedule": [
    {
      "update": 16001,
      "learning_rate": 0.0003
    }
  ],
  "seed": 2026102047,
  "sampling_seed": 2026101924,
  "wall_seconds": 1800,
  "physical_context": {
    "dt_s": 0.016666666666666666,
    "rho_kg_m3": 1000.0,
    "mu_pa_s": 0.001,
    "boundaries": "closed regular grid"
  },
  "initialization": "Fresh shared cell, common seed; managed point/flux/pressure and managed243viscosity coefficients frozen",
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
    "request-weights.npz": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd",
    "viscosity-initialization.json": "b530f9d4b9bd0f9b763ce26244b7bd7fefa97a4a16e7cca18cf96bd39aa22ffa",
    "viscosity-weights.npz": "f7c76bf1e24ce44d7d4b863140445c30f36a547526168f61917d7505b16953dd"
  },
  "decoder_representation": "Neural point displacement plus learned second-order curvature; shared velocity/tracer transport readout",
  "pretrained_nonlinear_request_weights_used": false,
  "displacement_auxiliary_weight": 1.0,
  "viscosity_coefficients_frozen_from_start": true,
  "replication_ordinal": 3,
  "replicates_primary": "b1-fixed-viscosity",
  "primary_config_sha256": "308699687d13d9e72da6ad71a7b513ac35dc2487b886b1b8f5849fc76c39ce6d",
  "replication_selection_sha256": "370b5f96cff86b451a7281dadd3aa07fb971c25a04db22148f7de78e224eea02"
}

```
