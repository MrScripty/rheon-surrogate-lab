# Full AI a-tail8-s3 train

Retrospective source review · fit-516f4d627c6425fd

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A pressure-only routed C107 successor with 128,938 total parameters. Its unchanged 94,956-parameter parent makes two fine CNN corrections with learned coarse blocks. For domains with every axis at most 16 cells, a new 33,982-parameter tail predicts 511 positive gains for non-DC cosine modes 0..7 from log(Nz/16), log(Ny/16), log(Nx/16), using 3→64 tanh→511 plus 511 logits. Float64 restriction/prolongation applies the tail to the remaining physical residual; an actual fine-equation norm check rejects a worsening proposal. The inference tail evaluates no numerical inverse. Above 16 cells on any axis, it returns the parent prediction unchanged. Two pressure corrections and three learned global blocks are internal solver work, not multiple simulation outputs; this component does not advance velocity or tracer.

**Purpose.** Reduce the small-grid pressure-error tail by fitting a final global correction while preserving the selected parent, its CNN and both routed coarse heads. This is a separate initialization/sampling-seed replicate of the tail calibration, not continuation of the completed earlier fit.

**Experiment.** The executed recipe runs head_train.py with calibration.json: 10,000 CUDA Adam updates, batch 256 with replacement, initialization seed 2026101983 and sampling seed 2026101984. The rate is 0.001, then 0.0003 at update 6001 and 0.0001 at 9001; gradients are clipped to norm one. Only five tail tensors train; all parent inventories remain byte-identical. The objective is mean valid-mode squared log-gain-fraction error plus 0.1 times each example’s worst valid-mode error. Offline inverse-mode labels cover all 1,728 rectangular geometries with axes 5..16: 1,526 train and 202 development geometries. The history records 2,560,000 sampled teacher examples and zero spatial CNN forwards during fitting. No physical trajectory fields train this tail. Inherited scene-feedback/batch-one context fields remain in the original config; they do not replace this separately selected calibration policy.

**Observed outcome.** The recorded final teacher-development objective is 0.0022176431957632303; it is a modal calibration loss. On the fresh shared 128 consumed pressure requests, this exact C130 checkpoint has worst mean-free relative pressure L2 8.5900931347869758e-05 and equation relative L2 0.00022552098518024733. 118 requests use teacher-training geometries and 10 use teacher-development geometries. Those partitions describe grid shape, not independent scene selection. The common pressure requests cover 8/16 cells per metre, small grids and 1/60-second steps. These results provide no new qualified latency, larger-domain, full-state, multistep or rollout-drift result.

**Lesson.** The three seeds yield similar small-grid pressure tails, but the lower teacher-development loss of the later replicates does not produce a lower worst pressure-field error on this common diagnostic. Evaluate modal fitting, pressure-field error and equation residual separately. The grid-count bypass makes this improvement conditional; it cannot be extrapolated to 32³ or larger domains. This calibration also supplies no new training coverage for timestep variation, forcing schedules or temporal extrapolation. Total parameter count, inherited context fields and historical CPU probes do not describe the active training scope or establish a complete fluid-generator frontier.

**Reproduction.** Retain the exact six typed inputs, all witnessed files, calibration.json and original train/restore recipes. The teacher manifest binds prepare_tail_teacher.py and all eight label-array inventories; the generator is an offline sealed-Neumann geometry teacher, separate from physical Rheon scene generation. Recreate a fresh admitted output rather than overwrite the guarded bank. The original recipe includes a historical CPU deployment probe on two grids; this source review runs no model and future reproduction must respect the current GPU-only inference policy. A changed recipe needs a separately identified execution. Clean managed retraining and complete-generator reproduction remain unverified; new Rheon physical data belongs to a separate reference epoch.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-pair-01/train/receipt.json",
    "sha256": "516f4d627c6425fdb6dbb67a69688eea0114c8b3bafc1f63f8af9c51958038e2"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-pair-01/train/history.json",
    "sha256": "884bc7ef7990c062ad2e1963e9b0fe8a8889fadc3f3773fe8880461309c004b5"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/calibration.json",
      "sha256": "8739b43ee9922c6eb08270b359044c05ced34648f03f4a643609c8af919e3555"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/coarse.py",
      "sha256": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/config.json",
      "sha256": "9195053067cb1548571f8958af4517d915632b4d8939d86e5fe6dee2562298e8"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/data.py",
      "sha256": "09231fe0119461205b74578dab9ce77cbcbea7c537900941432e97f52e51f16d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/distillation.json",
      "sha256": "e99ea86fd1abaee220071225e2730c0fbae5b22ec0f19510007b80fde424bf7d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/feedback_objective.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/fine-tuning.json",
      "sha256": "08f51f5235f0a7051ec640209ed07eb488f4b6957a411727cf5f32b1a1e5f2cb"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/fine_tuning.py",
      "sha256": "bc1d0ec5f7d496d0d79af767a7032e5108750a43f6525594c45c57e126c8780c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/frozen_gn.py",
      "sha256": "ce280f6ce86aeccc31521b13e4df039780fda8c112c63616457771377903e08d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/head_objective.py",
      "sha256": "24f4751520d7d194797bab223e0d254a5ebbbafd95dedb7f96324a052cbf0ed1"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/head_optimizer.py",
      "sha256": "e976b949d6d7695ae1335ac4e6d881d734945d1105cd5871c8aaf5decb35bdcf"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/head_train.py",
      "sha256": "0ac89f32e954100119b2547aee93e70472eddaf8a3f543a4023cd04b17483ebf"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/head_utils.py",
      "sha256": "066b58bf36d675a343e0663151ac2489addec3e1322189d2b5c5c75e351d0bf4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/hybrid-method.json",
      "sha256": "7286631ecc547c35be01b10413acc6e3f85649357bac8c308e25019cffccfbe7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/initial-context.json",
      "sha256": "d43b85d16bc9bdc82a5d542c694d5eb76784974b3232356364c6347a55ca1088"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/learned_coarse_model.py",
      "sha256": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/methods_view.py",
      "sha256": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/mixed_order.py",
      "sha256": "93d987b0f10101fb5b380e7310aa77eb6f5c7557198f6d43d90f102393e47b04"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/model.py",
      "sha256": "5b52bcf474739b556a89a789fea04223280e892d130e16ed64c6bb630a62b4c3"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/native_reconstruct.py",
      "sha256": "d2db7ae0a145e82560ede85518a542a2fc83f7e7c6de33697823457b96d74445"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/original_train.py",
      "sha256": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/parent_model.py",
      "sha256": "09348d81ff55ee247599ceb15dae34f7dc5eaaaf68263774fb743e8e2d355147"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/physical_residual.py",
      "sha256": "e0141080ca31cb8c1ce297b3efae1bcfbdd682fe2151c5be5947da78c592ab63"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/policy.py",
      "sha256": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/portable_policy.py",
      "sha256": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/reference_spectral.py",
      "sha256": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/routed_base_model.py",
      "sha256": "fe0a0b3bfbcc9479f14efe1b6c54981fb82ad8a7576b92470e0f1f9f63d24de9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/routed_descriptor.py",
      "sha256": "24217d1d3c593ab1cb2d1607cd47adcfaf12bb2bb4f186f326d2357b9d07c781"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/routing.json",
      "sha256": "78bab9e89fd29bf8e2e5c508b86d1165d98bc5c500ca85ee024d37a3502b94f0"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/student_slice.py",
      "sha256": "f144329caed584de6715fb45e2581c9148e559a183f66559349bfe17df58f580"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/tail_head_descriptor.py",
      "sha256": "4329d163a16b396b28eb9a7aee6fa5112ae633deb3bdf2d37194dca289c54c82"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/tail_physical.py",
      "sha256": "67ddfc7dde843863f1641481fb9a5aa36dfa8a47e2abc806c33e7cacde342696"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/teacher_model.py",
      "sha256": "63425e0a06f96ccd35476c94fabccb9bdfbd3691c706d98f2fa362ea00b3b4d9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/train.py",
      "sha256": "9e75fd34f56923f2fbc3702b762ea0d51b0f23760da0a477a626b66b0277aef7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-project-01/source-witness.json",
    "sha256": "e4d6579254efd6513e70055bc878c7c70d9fb481456f588d69e96800bbd9d3ae"
  },
  "training_inputs": [
    {
      "slot": "teacher",
      "path": "runs/full-ai-next-priorities-v1/tail-teacher-bank-01/teacher.npz",
      "sha256": "bdc5dc286758c2bb3a5c3b3ef775a53d680b3ffac9af8a81ed8f73ccbd28644f",
      "bytes": 5674172
    },
    {
      "slot": "teacher_manifest",
      "path": "runs/full-ai-next-priorities-v1/tail-teacher-bank-01/manifest.json",
      "sha256": "7a1d8578ab4760db4f07faa9fb2079ce6c4bc5a4ed47425cad18d990c247dff8",
      "bytes": 2247
    },
    {
      "slot": "initial_weights",
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-initial-01/weights.npz",
      "sha256": "14b5ad502e3e7bdc250b6583f3df9ef9517fa08125acd769b486260b7ad78190",
      "bytes": 525226
    },
    {
      "slot": "initial_inventory",
      "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-initial-01/initial-inventory.json",
      "sha256": "1e19f86244cfcde8c5ed964ba3ab5304fd458aa1167cf60c3adf04b3a9922eb2",
      "bytes": 14001
    },
    {
      "slot": "normalization",
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-s2-routed-assembly-01/normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "bytes": 6752
    },
    {
      "slot": "runtime_manifest",
      "path": "runs/dependency-migration-v1/runtime/current-manifest-v2.json",
      "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "bytes": 7989288
    }
  ],
  "data_generation_source": {
    "path": "experiments/full-ai-next-priorities-v1/prepare_tail_teacher.py",
    "sha256": "a94f6365e1696b8c3f98476da1d51f4482985b2426faee6c6cd65bb3093259fb",
    "manifest_slot": "teacher_manifest"
  }
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-516f4d627c6425fd",
    "kind": "training_attempt",
    "name": "Full AI a-tail8-s3 train",
    "task": "pressure_head_component",
    "schema": "rheon-neural-head-training-result-v1",
    "method": {
      "schema": "rheon-physical-residual-pressure-v1",
      "input_channels": 1,
      "encoder_widths": [
        4,
        8,
        12,
        16
      ],
      "bottleneck_width": 24,
      "decoder_widths": [
        16,
        12,
        8,
        4
      ],
      "kernel": [
        3,
        3,
        3
      ],
      "activation": "linear",
      "dtype": "float32",
      "pressure_dtype": "float64",
      "correction": {
        "activation": "linear",
        "steps": 2
      },
      "learned_coarse_revision": {
        "basis": "Orthonormal separable Neumann cosine modes 0..3, excluding DC",
        "trainable_parameters": 63,
        "coefficients": "Positive learned sigmoid gains; no analytical inverse eigenvalues",
        "bound": "max_axis_cells^2/2",
        "initialization": "Gain 1 on 16^3; all original neural tensors preserved",
        "gate": "Accept learned coarse proposal only when its actual fine-equation residual norm does not increase",
        "placement": "Learned coarse proposal before every fine neural pass",
        "limitation": "Gate controls coarse proposal only; final neural pass and trajectory still require independent evaluation"
      },
      "student_distillation": {
        "width_fraction": [
          1,
          2
        ],
        "teacher_weights_sha256": "42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4",
        "distillation_weight": 0.0,
        "target": "Teacher fine correction on each student-generated inner pressure residual",
        "teacher_used_at_inference": false,
        "normalization": "Shared fixed original C53 scales"
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 2097152
    },
    "receipt_sha256": "516f4d627c6425fdb6dbb67a69688eea0114c8b3bafc1f63f8af9c51958038e2",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/a-tail8-s3-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "7576d00e48c642729ff8a95b8fe300ab",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI a-tail8-s3 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 128938,
    "training_closure": {
      "source_witness_sha256": "e4d6579254efd6513e70055bc878c7c70d9fb481456f588d69e96800bbd9d3ae",
      "experiment_sha256": "9195053067cb1548571f8958af4517d915632b4d8939d86e5fe6dee2562298e8",
      "calibration_sha256": "8739b43ee9922c6eb08270b359044c05ced34648f03f4a643609c8af919e3555",
      "inputs": {
        "teacher": {
          "bytes": 5674172,
          "sha256": "bdc5dc286758c2bb3a5c3b3ef775a53d680b3ffac9af8a81ed8f73ccbd28644f"
        },
        "teacher_manifest": {
          "bytes": 2247,
          "sha256": "7a1d8578ab4760db4f07faa9fb2079ce6c4bc5a4ed47425cad18d990c247dff8"
        },
        "initial_weights": {
          "bytes": 525226,
          "sha256": "14b5ad502e3e7bdc250b6583f3df9ef9517fa08125acd769b486260b7ad78190"
        },
        "initial_inventory": {
          "bytes": 14001,
          "sha256": "1e19f86244cfcde8c5ed964ba3ab5304fd458aa1167cf60c3adf04b3a9922eb2"
        },
        "normalization": {
          "bytes": 6752,
          "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        }
      },
      "sources": {
        "analytical.py": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb",
        "band_feedback.py": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d",
        "baseline_model.py": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04",
        "coarse.py": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895",
        "custody-validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "data.py": "09231fe0119461205b74578dab9ce77cbcbea7c537900941432e97f52e51f16d",
        "direction_feedback.py": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f",
        "evidence.py": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210",
        "feedback_objective.py": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b",
        "fine_tuning.py": "bc1d0ec5f7d496d0d79af767a7032e5108750a43f6525594c45c57e126c8780c",
        "frozen_gn.py": "ce280f6ce86aeccc31521b13e4df039780fda8c112c63616457771377903e08d",
        "gradient_clipping.py": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd",
        "head_objective.py": "24f4751520d7d194797bab223e0d254a5ebbbafd95dedb7f96324a052cbf0ed1",
        "head_optimizer.py": "e976b949d6d7695ae1335ac4e6d881d734945d1105cd5871c8aaf5decb35bdcf",
        "head_train.py": "0ac89f32e954100119b2547aee93e70472eddaf8a3f543a4023cd04b17483ebf",
        "head_utils.py": "066b58bf36d675a343e0663151ac2489addec3e1322189d2b5c5c75e351d0bf4",
        "historical_data.py": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f",
        "historical_physical.py": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5",
        "learned_coarse_model.py": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0",
        "methods_view.py": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22",
        "mixed_order.py": "93d987b0f10101fb5b380e7310aa77eb6f5c7557198f6d43d90f102393e47b04",
        "model.py": "5b52bcf474739b556a89a789fea04223280e892d130e16ed64c6bb630a62b4c3",
        "native_reconstruct.py": "d2db7ae0a145e82560ede85518a542a2fc83f7e7c6de33697823457b96d74445",
        "original_data.py": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd",
        "original_model.py": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb",
        "original_train.py": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9",
        "parent_model.py": "09348d81ff55ee247599ceb15dae34f7dc5eaaaf68263774fb743e8e2d355147",
        "physical_residual.py": "e0141080ca31cb8c1ce297b3efae1bcfbdd682fe2151c5be5947da78c592ab63",
        "policy.py": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7",
        "portable_model.py": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935",
        "portable_policy.py": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460",
        "prepared.py": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76",
        "reference_spectral.py": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f",
        "retained_domain_model.py": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231",
        "routed_base_model.py": "fe0a0b3bfbcc9479f14efe1b6c54981fb82ad8a7576b92470e0f1f9f63d24de9",
        "routed_descriptor.py": "24217d1d3c593ab1cb2d1607cd47adcfaf12bb2bb4f186f326d2357b9d07c781",
        "scenes.py": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383",
        "smoke_train.py": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58",
        "student_slice.py": "f144329caed584de6715fb45e2581c9148e559a183f66559349bfe17df58f580",
        "support.py": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25",
        "tail_head_descriptor.py": "4329d163a16b396b28eb9a7aee6fa5112ae633deb3bdf2d37194dca289c54c82",
        "tail_physical.py": "67ddfc7dde843863f1641481fb9a5aa36dfa8a47e2abc806c33e7cacde342696",
        "teacher_model.py": "63425e0a06f96ccd35476c94fabccb9bdfbd3691c706d98f2fa362ea00b3b4d9",
        "test_inference.py": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6",
        "train.py": "9e75fd34f56923f2fbc3702b762ea0d51b0f23760da0a477a626b66b0277aef7",
        "training_diagnostic.py": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
      }
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "14b5ad502e3e7bdc250b6583f3df9ef9517fa08125acd769b486260b7ad78190",
        "bytes": 525226,
        "stage": "initial"
      },
      {
        "name": "weights.npz",
        "sha256": "67c39c78e7fceea7dc38facbf5f6ab5ae774a9915f5a88d59b538ae69c31e466",
        "bytes": 525226,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "14b5ad502e3e7bdc250b6583f3df9ef9517fa08125acd769b486260b7ad78190",
        "bytes": 525226,
        "stage": "initial"
      },
      {
        "name": "checkpoint-10000.npz",
        "sha256": "67c39c78e7fceea7dc38facbf5f6ab5ae774a9915f5a88d59b538ae69c31e466",
        "bytes": 525226,
        "stage": "retained_snapshot"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "cohorts/128"
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
      "receipt": "runs/full-ai-next-priorities-v1/a-tail8-s3-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_head_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/a-tail8-s3-pair-01/train/receipt.json",
    "sha256": "516f4d627c6425fdb6dbb67a69688eea0114c8b3bafc1f63f8af9c51958038e2",
    "bytes": 28209
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C130": {
        "name": "a-tail8-s3",
        "weights_sha256": "67c39c78e7fceea7dc38facbf5f6ab5ae774a9915f5a88d59b538ae69c31e466",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.00022552098518024733,
          "pressure_mean_free_relative_l2": 8.590093134786976e-05,
          "pressure_absolute_rms": 0.0013555508453719067,
          "pressure_absolute_max": 0.004913532678613575
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0060",
            "native_frame": 63,
            "counts_xyz": [
              5,
              6,
              15
            ],
            "cell_size_m": 0.125
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0054",
            "native_frame": 15,
            "counts_xyz": [
              15,
              12,
              5
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.00022552098518024733,
              "pressure_mean_free_relative_l2": 8.590093134786976e-05,
              "pressure_absolute_rms": 0.0013555508453719067,
              "pressure_absolute_max": 0.004913532678613575
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.00019706265199967104,
              "pressure_mean_free_relative_l2": 4.8968599552085654e-05,
              "pressure_absolute_rms": 0.0006805330178268666,
              "pressure_absolute_max": 0.0019779480352854215
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C130/results.json",
          "sha256": "c0765ad60d4a669e8cd3b49e6b5851f75797ab75a4da167b19525494bed3d8cc"
        },
        "CNN_operation_AST_preserved": null,
        "CPU_neural_forward": false
      }
    },
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
  "schema": "rheon-neural-head-training-result-v1",
  "status": "complete",
  "mode": "train",
  "optimizer_updates": 10000,
  "task": "pressure_tail_calibration",
  "pressure_stages": 2,
  "learned_global_blocks": 3,
  "parameters": 128938,
  "trainable_parameters": 33982,
  "architecture": {
    "schema": "rheon-physical-residual-pressure-v1",
    "input_channels": 1,
    "encoder_widths": [
      4,
      8,
      12,
      16
    ],
    "bottleneck_width": 24,
    "decoder_widths": [
      16,
      12,
      8,
      4
    ],
    "convolutions_per_block": 2,
    "kernel": [
      3,
      3,
      3
    ],
    "padding": "replicate",
    "activation": "linear",
    "pool": "per-axis average, kernel=min(2,count), ceil, exclude padding",
    "global_path": "add_spatial_mean_broadcast",
    "upsampling": "nearest_to_skip_shape",
    "head": "linear_1x1x1",
    "dtype": "float32",
    "pressure_dtype": "float64",
    "counts_supported": "positive per-axis integers within cell budget",
    "gauge": "subtract_first_correction_cell",
    "block_normalization": {
      "name": "Identity"
    },
    "max_cells": 2097152,
    "physical_cells": "one scalar h per sample; no cell-size network channels",
    "coordinates": "implicit grid indices; no coordinate feature maps",
    "initialization": "Verified selected routed C107 successor plus seeded tail head; all parent tensors frozen",
    "channels": [
      "dimensionless_laplacian_rhs"
    ],
    "conditioning": "physical-residual",
    "correction": {
      "activation": "linear",
      "steps": 2
    },
    "preprocessing": "q=(rho/h)*residual; q / train-only q RMS",
    "reconstruction": "float32 raw times selected train-only RMS, cell-zero gauge, add supplied base",
    "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
    "training_objective": "canonical-intermediate-contraction",
    "output_scale": "previous-pressure-rms",
    "first_correction_pressure_feedback_weight": 0.0,
    "learned_coarse_revision": {
      "basis": "Orthonormal separable Neumann cosine modes 0..3, excluding DC",
      "trainable_parameters": 63,
      "coefficients": "Positive learned sigmoid gains; no analytical inverse eigenvalues",
      "bound": "max_axis_cells^2/2",
      "initialization": "Gain 1 on 16^3; all original neural tensors preserved",
      "gate": "Accept learned coarse proposal only when its actual fine-equation residual norm does not increase",
      "placement": "Learned coarse proposal before every fine neural pass",
      "limitation": "Gate controls coarse proposal only; final neural pass and trajectory still require independent evaluation"
    },
    "shape_informed_coarse": {
      "features": "log(Nz/16), log(Ny/16), log(Nx/16)",
      "network": "Two learned3->16tanh->63heads; one selected by observable grid counts",
      "added_parameters": 1135,
      "head_learning_rate_multiplier": null,
      "fine_initialization": "Exact unchanged C107 fine CNN",
      "operator": "Existing sealed constant-density equation",
      "additional_head_parameters": 1198
    },
    "student_distillation": {
      "width_fraction": [
        1,
        2
      ],
      "teacher_weights_sha256": "42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4",
      "distillation_weight": 0.0,
      "target": "Teacher fine correction on each student-generated inner pressure residual",
      "teacher_used_at_inference": false,
      "normalization": "Shared fixed original C53 scales"
    },
    "parameter_count": 128938,
    "neural_head_routing": {
      "maximum_small_axis_cells": 16,
      "small_head_component": "a-head-calibration-s2",
      "large_head_component": "C107",
      "original_fine_parameters": 92560,
      "original_large_head_parameters": 1198,
      "calibrated_small_head_parameters": 1198,
      "parameters": 94956,
      "parameter_increase_fraction": 0.012777576313487915,
      "selection_input": "Grid counts only; no labels or measured errors",
      "candidate_numerical_pressure_solve": false,
      "registration_sha256": "5fb51cd5e73b808acd4e74e433e156f81d45a6400425ffe8d75d6d2378ee1aaa"
    },
    "parent_parameter_count": 94956,
    "tail_calibration": {
      "schema": "rheon-neural-tail-calibration-v1",
      "updates": 10000,
      "batch_size": 256,
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
      "wall_seconds": 1800,
      "seed": 2026101983,
      "sampling_seed": 2026101984,
      "objective": "Mean valid-mode squared log-gain-fraction error plus0.1per-example maximum valid-mode error",
      "identifier": "a-tail8-s3",
      "registration_sha256": "38cc6e478e9954ab85b795c1ac3be5447d577a60de3cb9670c1cbb9d6568eaab"
    },
    "final_learned_global_block": {
      "modes_per_axis": 8,
      "non_DC_gains": 511,
      "network": "3->64tanh->511 plus511learned logits",
      "trainable_parameters": 33982,
      "frozen_parent_parameters": 94956,
      "small_grid_fine_calls": 2,
      "small_grid_global_blocks": 3,
      "larger_grid_behavior": "Unchanged parent prediction; no tail above16cellsperaxis",
      "numerical_inverse_at_inference": false,
      "teacher": "Offline numerical modal labels, whole-geometry split"
    }
  },
  "closure": {
    "source_witness_sha256": "e4d6579254efd6513e70055bc878c7c70d9fb481456f588d69e96800bbd9d3ae",
    "experiment_sha256": "9195053067cb1548571f8958af4517d915632b4d8939d86e5fe6dee2562298e8",
    "calibration_sha256": "8739b43ee9922c6eb08270b359044c05ced34648f03f4a643609c8af919e3555",
    "inputs": {
      "teacher": {
        "bytes": 5674172,
        "sha256": "bdc5dc286758c2bb3a5c3b3ef775a53d680b3ffac9af8a81ed8f73ccbd28644f"
      },
      "teacher_manifest": {
        "bytes": 2247,
        "sha256": "7a1d8578ab4760db4f07faa9fb2079ce6c4bc5a4ed47425cad18d990c247dff8"
      },
      "initial_weights": {
        "bytes": 525226,
        "sha256": "14b5ad502e3e7bdc250b6583f3df9ef9517fa08125acd769b486260b7ad78190"
      },
      "initial_inventory": {
        "bytes": 14001,
        "sha256": "1e19f86244cfcde8c5ed964ba3ab5304fd458aa1167cf60c3adf04b3a9922eb2"
      },
      "normalization": {
        "bytes": 6752,
        "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      }
    },
    "sources": {
      "analytical.py": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb",
      "band_feedback.py": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d",
      "baseline_model.py": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04",
      "coarse.py": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895",
      "custody-validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "data.py": "09231fe0119461205b74578dab9ce77cbcbea7c537900941432e97f52e51f16d",
      "direction_feedback.py": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f",
      "evidence.py": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210",
      "feedback_objective.py": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b",
      "fine_tuning.py": "bc1d0ec5f7d496d0d79af767a7032e5108750a43f6525594c45c57e126c8780c",
      "frozen_gn.py": "ce280f6ce86aeccc31521b13e4df039780fda8c112c63616457771377903e08d",
      "gradient_clipping.py": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd",
      "head_objective.py": "24f4751520d7d194797bab223e0d254a5ebbbafd95dedb7f96324a052cbf0ed1",
      "head_optimizer.py": "e976b949d6d7695ae1335ac4e6d881d734945d1105cd5871c8aaf5decb35bdcf",
      "head_train.py": "0ac89f32e954100119b2547aee93e70472eddaf8a3f543a4023cd04b17483ebf",
      "head_utils.py": "066b58bf36d675a343e0663151ac2489addec3e1322189d2b5c5c75e351d0bf4",
      "historical_data.py": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f",
      "historical_physical.py": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5",
      "learned_coarse_model.py": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0",
      "methods_view.py": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22",
      "mixed_order.py": "93d987b0f10101fb5b380e7310aa77eb6f5c7557198f6d43d90f102393e47b04",
      "model.py": "5b52bcf474739b556a89a789fea04223280e892d130e16ed64c6bb630a62b4c3",
      "native_reconstruct.py": "d2db7ae0a145e82560ede85518a542a2fc83f7e7c6de33697823457b96d74445",
      "original_data.py": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd",
      "original_model.py": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb",
      "original_train.py": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9",
      "parent_model.py": "09348d81ff55ee247599ceb15dae34f7dc5eaaaf68263774fb743e8e2d355147",
      "physical_residual.py": "e0141080ca31cb8c1ce297b3efae1bcfbdd682fe2151c5be5947da78c592ab63",
      "policy.py": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7",
      "portable_model.py": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935",
      "portable_policy.py": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460",
      "prepared.py": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76",
      "reference_spectral.py": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f",
      "retained_domain_model.py": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231",
      "routed_base_model.py": "fe0a0b3bfbcc9479f14efe1b6c54981fb82ad8a7576b92470e0f1f9f63d24de9",
      "routed_descriptor.py": "24217d1d3c593ab1cb2d1607cd47adcfaf12bb2bb4f186f326d2357b9d07c781",
      "scenes.py": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383",
      "smoke_train.py": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58",
      "student_slice.py": "f144329caed584de6715fb45e2581c9148e559a183f66559349bfe17df58f580",
      "support.py": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25",
      "tail_head_descriptor.py": "4329d163a16b396b28eb9a7aee6fa5112ae633deb3bdf2d37194dca289c54c82",
      "tail_physical.py": "67ddfc7dde843863f1641481fb9a5aa36dfa8a47e2abc806c33e7cacde342696",
      "teacher_model.py": "63425e0a06f96ccd35476c94fabccb9bdfbd3691c706d98f2fa362ea00b3b4d9",
      "test_inference.py": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6",
      "train.py": "9e75fd34f56923f2fbc3702b762ea0d51b0f23760da0a477a626b66b0277aef7",
      "training_diagnostic.py": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  },
  "initial_inventory": {
    "coarse_logits": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "bytes": 252,
      "sha256": "4929ad3ab79008e14060c51600dc69e5f05cf82ec7f110c17c932ade8c5d5f64"
    },
    "small_coarse_logits": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "bytes": 252,
      "sha256": "e4f5066d77bd2de9ecd9dba3ab738c39745d26e9f610810e73f5285651784ebb"
    },
    "tail_logits": {
      "shape": [
        511
      ],
      "dtype": "<f4",
      "bytes": 2044,
      "sha256": "12bde97f82e70ce48e4957c6923a2860cebb876f633ae75418ccc32f8fac55b4"
    },
    "encoders.0.layers.0.weight": {
      "shape": [
        4,
        1,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 432,
      "sha256": "f37ee2e1dca35520b1efe258b0e9b494da3d1c16ae50aa7a2fe4f811147bfb8f"
    },
    "encoders.0.layers.3.weight": {
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 1728,
      "sha256": "02337c2d9fbdf1911b2a26172edb5102a611748ac130b22a3c449521f4c0b424"
    },
    "encoders.1.layers.0.weight": {
      "shape": [
        8,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 3456,
      "sha256": "b41b9311a838779c264cc28b24c78b2497ed657927e4537b08bdc1949ed255fc"
    },
    "encoders.1.layers.3.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 6912,
      "sha256": "060174e35a0610d5e1d0b0639310767f8b878a3d8e06e7759b286540899af539"
    },
    "encoders.2.layers.0.weight": {
      "shape": [
        12,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 10368,
      "sha256": "8b543e79d54ee0752c48ea07b4d4ff1f267174ba5e4e1e6d36994bfa10f27e98"
    },
    "encoders.2.layers.3.weight": {
      "shape": [
        12,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 15552,
      "sha256": "4a7c38eee7eb8eaa4e3890e9e8ce861ab6975c1e1416684b1fb1862d1782b76a"
    },
    "encoders.3.layers.0.weight": {
      "shape": [
        16,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 20736,
      "sha256": "7932627cfd088d4874ffaa18c62f6b80b84fc4be11c478232bb5cf4e6e3b9d08"
    },
    "encoders.3.layers.3.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 27648,
      "sha256": "67a1fe4c5403bc257879f8c1b785872619c234839817be792dce919f724ed5de"
    },
    "bottleneck.layers.0.weight": {
      "shape": [
        24,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 41472,
      "sha256": "2a9862fb35c4f0c618738f42c6af669ce8071a53c576a224ec723fedeb936922"
    },
    "bottleneck.layers.3.weight": {
      "shape": [
        24,
        24,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 62208,
      "sha256": "d8e092e6be01936d786e8b17a9e604e461ba4af6cb31cde38099893417c90008"
    },
    "decoders.0.layers.0.weight": {
      "shape": [
        16,
        40,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 69120,
      "sha256": "58d51c78e13a8b2c36b4c035b74dd8a87c2db92b75f8298b629fc5c148a2c02b"
    },
    "decoders.0.layers.3.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 27648,
      "sha256": "38c9644c2c242ad75ad9070d2ca6e2d2984d3f8e45248a3bd39cd931703e36c2"
    },
    "decoders.1.layers.0.weight": {
      "shape": [
        12,
        28,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 36288,
      "sha256": "0dd3fffaaf1eb31e81742ac35625c41eb0d61a7577ce0d7e76d5c7e8c538097e"
    },
    "decoders.1.layers.3.weight": {
      "shape": [
        12,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 15552,
      "sha256": "e662b00c70c4f8eb3fd6af5daefe62d8670faa4381f380ca0952ea272d22e428"
    },
    "decoders.2.layers.0.weight": {
      "shape": [
        8,
        20,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 17280,
      "sha256": "897216db24db4a4df34d9273d197508f3d0261dfdc5decfafb6982cea0a4337b"
    },
    "decoders.2.layers.3.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 6912,
      "sha256": "414758deeb72af51fc5c494b967bf9e77167def358b2884608ee0f906080ce38"
    },
    "decoders.3.layers.0.weight": {
      "shape": [
        4,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 5184,
      "sha256": "817316b8873759212cf3b458109374d35b52a3a77bb3e82aa9e29f83650a0b66"
    },
    "decoders.3.layers.3.weight": {
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 1728,
      "sha256": "64631b34d1c4b8aa25fde2aad93974bb888bb1ad7166a9b3b4a4cf8bd4cd6c49"
    },
    "head.weight": {
      "shape": [
        1,
        4,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "bytes": 16,
      "sha256": "62c58dcd44aa770b0d67da6cc12dd66057e882a29bc3678a7c1d1bbd6cfd3ee8"
    },
    "shape_head.0.weight": {
      "shape": [
        16,
        3
      ],
      "dtype": "<f4",
      "bytes": 192,
      "sha256": "64d0d38958cdaa168eabb7c172d665d3d3a6581fb3c8593c530c2de5c66d348a"
    },
    "shape_head.0.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "bytes": 64,
      "sha256": "0f12311b7a8f9ce0ff2342fc5eda41ddecad4a844a02a905e6df1ceff7a9ab62"
    },
    "shape_head.2.weight": {
      "shape": [
        63,
        16
      ],
      "dtype": "<f4",
      "bytes": 4032,
      "sha256": "4ac627a15dd54f89e88d1089dc5011e848933fcda6e236fbf571b28244c0bd4f"
    },
    "shape_head.2.bias": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "bytes": 252,
      "sha256": "85631fce43497001054ebc1928ca0b7ca3b11d2125464fab20f21dce52f08e24"
    },
    "small_shape_head.0.weight": {
      "shape": [
        16,
        3
      ],
      "dtype": "<f4",
      "bytes": 192,
      "sha256": "707d11a169d05af5aae4d433cd64214bd30cd75ad2a3e08e3934c729420e2204"
    },
    "small_shape_head.0.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "bytes": 64,
      "sha256": "3def506d22faf8bcf4bc62ba119407caa5b959655682de09565ff3055b70a0c4"
    },
    "small_shape_head.2.weight": {
      "shape": [
        63,
        16
      ],
      "dtype": "<f4",
      "bytes": 4032,
      "sha256": "896c5e25aa2822ed0309735b5a1bb1f5bf206aef5dcebae9cb76d58f8b93bbdd"
    },
    "small_shape_head.2.bias": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "bytes": 252,
      "sha256": "44813b6cc636038d62416d28baef7463afe5a39533f68336632a4c8748d302c1"
    },
    "tail_shape_head.0.weight": {
      "shape": [
        64,
        3
      ],
      "dtype": "<f4",
      "bytes": 768,
      "sha256": "2d29c9e7f6491413945a5991723733f3f9c9019f64ad45a7094f708f21bc9d32"
    },
    "tail_shape_head.0.bias": {
      "shape": [
        64
      ],
      "dtype": "<f4",
      "bytes": 256,
      "sha256": "fef00ff7bc570f940afa37bc1f0aea4edb8a0a91677328e987b064ce86ef171f"
    },
    "tail_shape_head.2.weight": {
      "shape": [
        511,
        64
      ],
      "dtype": "<f4",
      "bytes": 130816,
      "sha256": "41d8f36471292af08b586a2f5ff523983efcf454264e74a5f8044c37fa9d0322"
    },
    "tail_shape_head.2.bias": {
      "shape": [
        511
      ],
      "dtype": "<f4",
      "bytes": 2044,
      "sha256": "ff662105e12fcc74e80dbc270db8a47b29437318961d1fffa93e08e7e165c211"
    }
  },
  "final_inventory": {
    "coarse_logits": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "bytes": 252,
      "sha256": "4929ad3ab79008e14060c51600dc69e5f05cf82ec7f110c17c932ade8c5d5f64"
    },
    "small_coarse_logits": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "bytes": 252,
      "sha256": "e4f5066d77bd2de9ecd9dba3ab738c39745d26e9f610810e73f5285651784ebb"
    },
    "tail_logits": {
      "shape": [
        511
      ],
      "dtype": "<f4",
      "bytes": 2044,
      "sha256": "f38c6118c8c0a60c988ce2d116fc0e7b69aa6ccb55b3ab3d641ed612862eb6e5"
    },
    "encoders.0.layers.0.weight": {
      "shape": [
        4,
        1,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 432,
      "sha256": "f37ee2e1dca35520b1efe258b0e9b494da3d1c16ae50aa7a2fe4f811147bfb8f"
    },
    "encoders.0.layers.3.weight": {
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 1728,
      "sha256": "02337c2d9fbdf1911b2a26172edb5102a611748ac130b22a3c449521f4c0b424"
    },
    "encoders.1.layers.0.weight": {
      "shape": [
        8,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 3456,
      "sha256": "b41b9311a838779c264cc28b24c78b2497ed657927e4537b08bdc1949ed255fc"
    },
    "encoders.1.layers.3.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 6912,
      "sha256": "060174e35a0610d5e1d0b0639310767f8b878a3d8e06e7759b286540899af539"
    },
    "encoders.2.layers.0.weight": {
      "shape": [
        12,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 10368,
      "sha256": "8b543e79d54ee0752c48ea07b4d4ff1f267174ba5e4e1e6d36994bfa10f27e98"
    },
    "encoders.2.layers.3.weight": {
      "shape": [
        12,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 15552,
      "sha256": "4a7c38eee7eb8eaa4e3890e9e8ce861ab6975c1e1416684b1fb1862d1782b76a"
    },
    "encoders.3.layers.0.weight": {
      "shape": [
        16,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 20736,
      "sha256": "7932627cfd088d4874ffaa18c62f6b80b84fc4be11c478232bb5cf4e6e3b9d08"
    },
    "encoders.3.layers.3.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 27648,
      "sha256": "67a1fe4c5403bc257879f8c1b785872619c234839817be792dce919f724ed5de"
    },
    "bottleneck.layers.0.weight": {
      "shape": [
        24,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 41472,
      "sha256": "2a9862fb35c4f0c618738f42c6af669ce8071a53c576a224ec723fedeb936922"
    },
    "bottleneck.layers.3.weight": {
      "shape": [
        24,
        24,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 62208,
      "sha256": "d8e092e6be01936d786e8b17a9e604e461ba4af6cb31cde38099893417c90008"
    },
    "decoders.0.layers.0.weight": {
      "shape": [
        16,
        40,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 69120,
      "sha256": "58d51c78e13a8b2c36b4c035b74dd8a87c2db92b75f8298b629fc5c148a2c02b"
    },
    "decoders.0.layers.3.weight": {
      "shape": [
        16,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 27648,
      "sha256": "38c9644c2c242ad75ad9070d2ca6e2d2984d3f8e45248a3bd39cd931703e36c2"
    },
    "decoders.1.layers.0.weight": {
      "shape": [
        12,
        28,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 36288,
      "sha256": "0dd3fffaaf1eb31e81742ac35625c41eb0d61a7577ce0d7e76d5c7e8c538097e"
    },
    "decoders.1.layers.3.weight": {
      "shape": [
        12,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 15552,
      "sha256": "e662b00c70c4f8eb3fd6af5daefe62d8670faa4381f380ca0952ea272d22e428"
    },
    "decoders.2.layers.0.weight": {
      "shape": [
        8,
        20,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 17280,
      "sha256": "897216db24db4a4df34d9273d197508f3d0261dfdc5decfafb6982cea0a4337b"
    },
    "decoders.2.layers.3.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 6912,
      "sha256": "414758deeb72af51fc5c494b967bf9e77167def358b2884608ee0f906080ce38"
    },
    "decoders.3.layers.0.weight": {
      "shape": [
        4,
        12,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 5184,
      "sha256": "817316b8873759212cf3b458109374d35b52a3a77bb3e82aa9e29f83650a0b66"
    },
    "decoders.3.layers.3.weight": {
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "bytes": 1728,
      "sha256": "64631b34d1c4b8aa25fde2aad93974bb888bb1ad7166a9b3b4a4cf8bd4cd6c49"
    },
    "head.weight": {
      "shape": [
        1,
        4,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "bytes": 16,
      "sha256": "62c58dcd44aa770b0d67da6cc12dd66057e882a29bc3678a7c1d1bbd6cfd3ee8"
    },
    "shape_head.0.weight": {
      "shape": [
        16,
        3
      ],
      "dtype": "<f4",
      "bytes": 192,
      "sha256": "64d0d38958cdaa168eabb7c172d665d3d3a6581fb3c8593c530c2de5c66d348a"
    },
    "shape_head.0.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "bytes": 64,
      "sha256": "0f12311b7a8f9ce0ff2342fc5eda41ddecad4a844a02a905e6df1ceff7a9ab62"
    },
    "shape_head.2.weight": {
      "shape": [
        63,
        16
      ],
      "dtype": "<f4",
      "bytes": 4032,
      "sha256": "4ac627a15dd54f89e88d1089dc5011e848933fcda6e236fbf571b28244c0bd4f"
    },
    "shape_head.2.bias": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "bytes": 252,
      "sha256": "85631fce43497001054ebc1928ca0b7ca3b11d2125464fab20f21dce52f08e24"
    },
    "small_shape_head.0.weight": {
      "shape": [
        16,
        3
      ],
      "dtype": "<f4",
      "bytes": 192,
      "sha256": "707d11a169d05af5aae4d433cd64214bd30cd75ad2a3e08e3934c729420e2204"
    },
    "small_shape_head.0.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "bytes": 64,
      "sha256": "3def506d22faf8bcf4bc62ba119407caa5b959655682de09565ff3055b70a0c4"
    },
    "small_shape_head.2.weight": {
      "shape": [
        63,
        16
      ],
      "dtype": "<f4",
      "bytes": 4032,
      "sha256": "896c5e25aa2822ed0309735b5a1bb1f5bf206aef5dcebae9cb76d58f8b93bbdd"
    },
    "small_shape_head.2.bias": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "bytes": 252,
      "sha256": "44813b6cc636038d62416d28baef7463afe5a39533f68336632a4c8748d302c1"
    },
    "tail_shape_head.0.weight": {
      "shape": [
        64,
        3
      ],
      "dtype": "<f4",
      "bytes": 768,
      "sha256": "b7a01948d85c44a2f492864865e63bc9b0888e8db6ff0e9515b7b54efb0b354a"
    },
    "tail_shape_head.0.bias": {
      "shape": [
        64
      ],
      "dtype": "<f4",
      "bytes": 256,
      "sha256": "a9ac4a52f42cd5d28d0192792e3ba5e270f975c145c2366ae034527c7ce24f04"
    },
    "tail_shape_head.2.weight": {
      "shape": [
        511,
        64
      ],
      "dtype": "<f4",
      "bytes": 130816,
      "sha256": "6a2c3655c7ba952782eedeac2a9de4fcf0e1c1d4be5a66cb26d95949993cc0c2"
    },
    "tail_shape_head.2.bias": {
      "shape": [
        511
      ],
      "dtype": "<f4",
      "bytes": 2044,
      "sha256": "df59e2e721000bf9f8759957cc77eb26ba6b4f235e7d37e6fb88fcad312ed16d"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 75342336
  },
  "outputs": {
    "checkpoint-000.npz": {
      "bytes": 525226,
      "sha256": "14b5ad502e3e7bdc250b6583f3df9ef9517fa08125acd769b486260b7ad78190"
    },
    "probe.npy": {
      "bytes": 41608,
      "sha256": "0d5900facb1aaddac180667abdd576f7e311a5d325dd6d71090fb4a43f2b79d0"
    },
    "weights.npz": {
      "bytes": 525226,
      "sha256": "67c39c78e7fceea7dc38facbf5f6ab5ae774a9915f5a88d59b538ae69c31e466"
    },
    "initial-weights.npz": {
      "bytes": 525226,
      "sha256": "14b5ad502e3e7bdc250b6583f3df9ef9517fa08125acd769b486260b7ad78190"
    },
    "checkpoint-10000.npz": {
      "bytes": 525226,
      "sha256": "67c39c78e7fceea7dc38facbf5f6ab5ae774a9915f5a88d59b538ae69c31e466"
    },
    "history.json": {
      "bytes": 301585,
      "sha256": "884bc7ef7990c062ad2e1963e9b0fe8a8889fadc3f3773fe8880461309c004b5"
    },
    "normalization.json": {
      "bytes": 6752,
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "fine_tensors_bitwise_unchanged": true,
  "candidate_numerical_pressure_solve": false,
  "elapsed_s": 36.332333597005345,
  "scientific_acceptance_or_promotion": false
}

```
