# Full AI a-head-calibration train

Retrospective source review · fit-c80b7d18912c7d9d

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** C125 keeps the complete 93,758-parameter C107 pressure graph and starts from its exact checkpoint. It freezes 92,560 fine-network parameters and calibrates only the five existing coarse-head tensors (1,198 parameters). The head adds a 3→16 tanh→63 shape network to63 modal logits; sigmoid gives positive gains bounded by max-axis-cells²/2. Separable cosine modes0..3 excluding DC restrict/prolong the physical residual in float64. Before each of two float32 fine-network corrections, a coarse proposal is accepted only if its actual fine-equation residual norm does not increase. This predicts pressure for one request, not velocity, tracer, future endpoints or simultaneous physical steps.

**Purpose.** Test whether offline geometry-teacher calibration improves the C107 shape-dependent global response while preserving its fine tensors and inference graph. The three fits are siblings from the same complete initial inventory; the different torch seeds do not create different loaded initial weights. Their private sampling seeds differ. They are not a sequential continuation or three new architectures.

**Experiment.** Actual recipe entrypoint head_train.py uses calibration.json, not the inherited scene trainer/configuration. The offline teacher has6,728 unique Z,Y,X geometries: all1,728 triples with axes5..16 plus5,000 sampled larger triples with axes up to128. Whole-geometry hashing gives1,526 small+4,478 large training geometries and202 small+522 large development geometries. Labels are log of the analytical sealed-Neumann modal inverse gain divided by max-axis-cells²/2; no physical fields or Rheon trajectories train this calibration. 10,000 CUDA Adam updates each sample256 small+128 large geometries with replacement: 3,840,000 teacher examples and zero fine spatial-network forwards during fitting. Loss is mean squared log-gain error plus0.1 times mean per-example maximum squared mode error; learning rate.001→.0003 at6001→.0001 at9001, gradient clipping1, deterministic float32 fitting, TF32/AMP disabled. Fixed physical pressure normalization remains C107’s.

**Observed outcome.** The retained common128-pressure-request diagnostic gives worst mean-free pressure relative L2 0.0024197172641509169 and worst equation relative L2 0.00046812095601858427. Final teacher-development loss is 0.028558602556586266; this is a distinct metric. The common pressure requests cover small grids at8/16cells per metre and dt1/60; 118 have teacher-training geometries and10 teacher-development geometries. Geometry exposure is not a physical-scene held-out split. Exact historical zero-update CPU restore and bitwise probe evidence is retained; this review executes no new inference.

**Lesson.** All three improve this diagnostic over C107’s worst pressure0.00815329, with modest seed variation around0.0024; C126 is best of these three on both worst pressure and equation error. Training on large geometry labels does not prove accurate physical pressure solves on large domains. No speed improvement follows from a calibration that keeps the same inference graph, and the coarse acceptance gate does not bound the final fine correction or own-history drift. Preserve separate teacher loss, pressure-field error, equation residual, runtime and rollout claims.

**Reproduction.** Retain the original53-file witness, exact C107 initialization/normalization, six typed inputs, calibration and train/restore recipes. The teacher manifest binds the original prepare_head_teacher.py and all seven array inventories. Recreate a fresh admitted teacher output instead of overwriting the guarded bank. Original recipes include CPU deployment/restore probes; current reproduction must use separately identified GPU-only probe executions to respect the active inference policy. Clean retraining and complete-generator reproduction remain unqualified. New Rheon reference epochs must remain separate from these older pressure requests.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/a-head-calibration-pair-01/train/receipt.json",
    "sha256": "c80b7d18912c7d9d4b9fbcf7931b9cbe198b0fcf69afd79b3f8aa1e08b1e41d6"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/a-head-calibration-pair-01/train/history.json",
    "sha256": "e7943394dd175e0529c199d47219c63d2122a57716b9f0e30aa8990292b07ce2"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/calibration.json",
      "sha256": "0b9155f6a6689a0f40bf3af73cacde2a97fecc326e281ccc7549f778bffde4e9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/calibration_base_model.py",
      "sha256": "fe0a0b3bfbcc9479f14efe1b6c54981fb82ad8a7576b92470e0f1f9f63d24de9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/coarse.py",
      "sha256": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/config.json",
      "sha256": "5c309ba53e4bb92dda3a788c94b2aa8eaebdc8500cafb146d497895ad710df02"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/data.py",
      "sha256": "09231fe0119461205b74578dab9ce77cbcbea7c537900941432e97f52e51f16d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/distillation.json",
      "sha256": "e99ea86fd1abaee220071225e2730c0fbae5b22ec0f19510007b80fde424bf7d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/feedback_objective.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/fine-tuning.json",
      "sha256": "ca4b7bb9f8d78846b075dffb08bb26ce9f90e1c099379055af9f6cb680f7e2e7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/fine_tuning.py",
      "sha256": "bc1d0ec5f7d496d0d79af767a7032e5108750a43f6525594c45c57e126c8780c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/fit-sampling.json",
      "sha256": "d15473fcc282354208fe1077b03a0e803600b81c93b40574a5aa5a2ca73eaf3a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/frozen_gn.py",
      "sha256": "ce280f6ce86aeccc31521b13e4df039780fda8c112c63616457771377903e08d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/head_descriptor.py",
      "sha256": "3581afb32f0772df9d1a94791ee2983cb4c31bc7233bd661561f20840e2bf40f"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/head_objective.py",
      "sha256": "1d3e6cc26496e69b388b9bdb430543087c22f84be628a522ed6a25d391a54798"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/head_optimizer.py",
      "sha256": "2033de17a5467517426b6d5390b647aeca31b7078ab685bd69fadaf238074a67"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/head_train.py",
      "sha256": "3b7ec07366c6ed308c2cbc29f679f4fd7a803218e70e5f47b29225a2113b2524"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/head_utils.py",
      "sha256": "066b58bf36d675a343e0663151ac2489addec3e1322189d2b5c5c75e351d0bf4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/hybrid-method.json",
      "sha256": "7286631ecc547c35be01b10413acc6e3f85649357bac8c308e25019cffccfbe7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/learned_coarse_model.py",
      "sha256": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/methods_view.py",
      "sha256": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/mixed_order.py",
      "sha256": "9a439127ddae653786a7a0962e13b63d5698c1c38ffba720eea7acd2a9b1d6d2"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/model.py",
      "sha256": "4a224850603d94d138e9e5001246ad7275196ad847936636eec86f49943418f9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/native_reconstruct.py",
      "sha256": "d2db7ae0a145e82560ede85518a542a2fc83f7e7c6de33697823457b96d74445"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/original_train.py",
      "sha256": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/physical_residual.py",
      "sha256": "6e1392c26880c034d09c72437fefd4c1865d6540f40e245825a6ad6a4e7f4204"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/policy.py",
      "sha256": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/portable_policy.py",
      "sha256": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/reference_spectral.py",
      "sha256": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/student_slice.py",
      "sha256": "f144329caed584de6715fb45e2581c9148e559a183f66559349bfe17df58f580"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/teacher_model.py",
      "sha256": "63425e0a06f96ccd35476c94fabccb9bdfbd3691c706d98f2fa362ea00b3b4d9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/train.py",
      "sha256": "9e75fd34f56923f2fbc3702b762ea0d51b0f23760da0a477a626b66b0277aef7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/source-witness.json",
    "sha256": "b609dfc1edad25ef22a6c1b0448da6468e38d9d238129882149394fcce67094e"
  },
  "training_inputs": [
    {
      "slot": "teacher",
      "path": "runs/full-ai-next-priorities-v1/head-teacher-bank-01/teacher.npz",
      "sha256": "73c9be78b2a832c878e3045645f6a8e3ec4ef3c99ab9e23447c870f969a510e2",
      "bytes": 3499602
    },
    {
      "slot": "teacher_manifest",
      "path": "runs/full-ai-next-priorities-v1/head-teacher-bank-01/manifest.json",
      "sha256": "4480cd73a33a981d30399d3ab781c948873c2a9c22ebcb60ffd78d69e7f6cecb",
      "bytes": 2389
    },
    {
      "slot": "initial_weights",
      "path": "runs/pressure-methods-v4/v4-student50-control-study-01/restore/weights.npz",
      "sha256": "9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365",
      "bytes": 381750
    },
    {
      "slot": "initial_inventory",
      "path": "runs/full-ai-next-priorities-v1/a-head-calibration-project-01/initial-inventory.json",
      "sha256": "30849db6b8e931fd268289bb7f96c1f8aed46b5fef5bcb8e869632acffebdadb",
      "bytes": 7199
    },
    {
      "slot": "normalization",
      "path": "runs/pressure-methods-v4/v4-student50-control-study-01/restore/normalization.json",
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
    "path": "experiments/full-ai-next-priorities-v1/prepare_head_teacher.py",
    "sha256": "5104c7e242a87cc5ed0f88d7b40dc0ddc7632be59f9210619d0e7d341c0a0dda",
    "manifest_slot": "teacher_manifest"
  }
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-c80b7d18912c7d9d",
    "kind": "training_attempt",
    "name": "Full AI a-head-calibration train",
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
        "initialization": "Exact C107 learned head, followed by offline modal calibration",
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
    "receipt_sha256": "c80b7d18912c7d9d4b9fbcf7931b9cbe198b0fcf69afd79b3f8aa1e08b1e41d6",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/a-head-calibration-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "61c8d8b92a1845839e421f172089885e",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI a-head-calibration train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 93758,
    "training_closure": {
      "source_witness_sha256": "b609dfc1edad25ef22a6c1b0448da6468e38d9d238129882149394fcce67094e",
      "experiment_sha256": "5c309ba53e4bb92dda3a788c94b2aa8eaebdc8500cafb146d497895ad710df02",
      "calibration_sha256": "0b9155f6a6689a0f40bf3af73cacde2a97fecc326e281ccc7549f778bffde4e9",
      "inputs": {
        "teacher": {
          "bytes": 3499602,
          "sha256": "73c9be78b2a832c878e3045645f6a8e3ec4ef3c99ab9e23447c870f969a510e2"
        },
        "teacher_manifest": {
          "bytes": 2389,
          "sha256": "4480cd73a33a981d30399d3ab781c948873c2a9c22ebcb60ffd78d69e7f6cecb"
        },
        "initial_weights": {
          "bytes": 381750,
          "sha256": "9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365"
        },
        "initial_inventory": {
          "bytes": 7199,
          "sha256": "30849db6b8e931fd268289bb7f96c1f8aed46b5fef5bcb8e869632acffebdadb"
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
        "calibration_base_model.py": "fe0a0b3bfbcc9479f14efe1b6c54981fb82ad8a7576b92470e0f1f9f63d24de9",
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
        "head_descriptor.py": "3581afb32f0772df9d1a94791ee2983cb4c31bc7233bd661561f20840e2bf40f",
        "head_objective.py": "1d3e6cc26496e69b388b9bdb430543087c22f84be628a522ed6a25d391a54798",
        "head_optimizer.py": "2033de17a5467517426b6d5390b647aeca31b7078ab685bd69fadaf238074a67",
        "head_train.py": "3b7ec07366c6ed308c2cbc29f679f4fd7a803218e70e5f47b29225a2113b2524",
        "head_utils.py": "066b58bf36d675a343e0663151ac2489addec3e1322189d2b5c5c75e351d0bf4",
        "historical_data.py": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f",
        "historical_physical.py": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5",
        "learned_coarse_model.py": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0",
        "methods_view.py": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22",
        "mixed_order.py": "9a439127ddae653786a7a0962e13b63d5698c1c38ffba720eea7acd2a9b1d6d2",
        "model.py": "4a224850603d94d138e9e5001246ad7275196ad847936636eec86f49943418f9",
        "native_reconstruct.py": "d2db7ae0a145e82560ede85518a542a2fc83f7e7c6de33697823457b96d74445",
        "original_data.py": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd",
        "original_model.py": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb",
        "original_train.py": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9",
        "physical_residual.py": "6e1392c26880c034d09c72437fefd4c1865d6540f40e245825a6ad6a4e7f4204",
        "policy.py": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7",
        "portable_model.py": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935",
        "portable_policy.py": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460",
        "prepared.py": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76",
        "reference_spectral.py": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f",
        "retained_domain_model.py": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231",
        "scenes.py": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383",
        "smoke_train.py": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58",
        "student_slice.py": "f144329caed584de6715fb45e2581c9148e559a183f66559349bfe17df58f580",
        "support.py": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25",
        "teacher_model.py": "63425e0a06f96ccd35476c94fabccb9bdfbd3691c706d98f2fa362ea00b3b4d9",
        "test_inference.py": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6",
        "train.py": "9e75fd34f56923f2fbc3702b762ea0d51b0f23760da0a477a626b66b0277aef7",
        "training_diagnostic.py": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
      }
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365",
        "bytes": 381750,
        "stage": "initial"
      },
      {
        "name": "weights.npz",
        "sha256": "cc25fb7820c83b2b23b2259c4e306c247abdc01ec7e53c81ac3b7d86ae63e494",
        "bytes": 381750,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365",
        "bytes": 381750,
        "stage": "initial"
      },
      {
        "name": "checkpoint-10000.npz",
        "sha256": "cc25fb7820c83b2b23b2259c4e306c247abdc01ec7e53c81ac3b7d86ae63e494",
        "bytes": 381750,
        "stage": "retained_snapshot"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "cohorts/123"
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
      "receipt": "runs/full-ai-next-priorities-v1/a-head-calibration-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_head_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/a-head-calibration-pair-01/train/receipt.json",
    "sha256": "c80b7d18912c7d9d4b9fbcf7931b9cbe198b0fcf69afd79b3f8aa1e08b1e41d6",
    "bytes": 22603
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C125": {
        "name": "a-head-calibration",
        "weights_sha256": "cc25fb7820c83b2b23b2259c4e306c247abdc01ec7e53c81ac3b7d86ae63e494",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.00046812095601858427,
          "pressure_mean_free_relative_l2": 0.002419717264150917,
          "pressure_absolute_rms": 0.04860781579713077,
          "pressure_absolute_max": 0.11815401358592821
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0025",
            "native_frame": 15,
            "counts_xyz": [
              9,
              11,
              11
            ],
            "cell_size_m": 0.0625
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
              "common_equation_relative_l2": 0.0003601219314566797,
              "pressure_mean_free_relative_l2": 0.002419717264150917,
              "pressure_absolute_rms": 0.04860781579713077,
              "pressure_absolute_max": 0.11815401358592821
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.00046812095601858427,
              "pressure_mean_free_relative_l2": 0.0009465425862238681,
              "pressure_absolute_rms": 0.009457141889561718,
              "pressure_absolute_max": 0.027773286715076964
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C125/results.json",
          "sha256": "6b0c0dde0663374e2fe7040c31c1b35dfe6a5d5789e2c430e96bf73d046d5b11"
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
  "task": "pressure_head_calibration",
  "pressure_stages": 2,
  "parameters": 93758,
  "trainable_parameters": 1198,
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
    "initialization": "retain original first-convolution residual column and all surviving seeded tensors",
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
    "training_objective": "offline modal teacher log-gain calibration; fine network frozen",
    "output_scale": "previous-pressure-rms",
    "first_correction_pressure_feedback_weight": 0.0,
    "learned_coarse_revision": {
      "basis": "Orthonormal separable Neumann cosine modes 0..3, excluding DC",
      "trainable_parameters": 63,
      "coefficients": "Positive learned sigmoid gains; no analytical inverse eigenvalues",
      "bound": "max_axis_cells^2/2",
      "initialization": "Exact C107 learned head, followed by offline modal calibration",
      "gate": "Accept learned coarse proposal only when its actual fine-equation residual norm does not increase",
      "placement": "Learned coarse proposal before every fine neural pass",
      "limitation": "Gate controls coarse proposal only; final neural pass and trajectory still require independent evaluation"
    },
    "shape_informed_coarse": {
      "features": "log(Nz/16), log(Ny/16), log(Nx/16)",
      "network": "3 -> 16 tanh -> 63 additive coarse logits; final layer initially zero",
      "added_parameters": 1135,
      "head_learning_rate_multiplier": 1,
      "fine_initialization": "Exact C107 fine tensors frozen throughout calibration",
      "operator": "Existing sealed constant-density equation"
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
    "head_calibration": {
      "schema": "rheon-neural-head-calibration-v1",
      "updates": 10000,
      "batch_size": 384,
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
      "seed": 2026101953,
      "sampling_seed": 2026101954,
      "objective": "Mean squared log-gain-fraction error plus0.1 times the mean per-example maximum squared mode error",
      "identifier": "a-head-calibration",
      "registration_sha256": "21efa404f5e7908e030835092a13df6621bd3000af6c83b75d5d71ccf9795cab"
    }
  },
  "closure": {
    "source_witness_sha256": "b609dfc1edad25ef22a6c1b0448da6468e38d9d238129882149394fcce67094e",
    "experiment_sha256": "5c309ba53e4bb92dda3a788c94b2aa8eaebdc8500cafb146d497895ad710df02",
    "calibration_sha256": "0b9155f6a6689a0f40bf3af73cacde2a97fecc326e281ccc7549f778bffde4e9",
    "inputs": {
      "teacher": {
        "bytes": 3499602,
        "sha256": "73c9be78b2a832c878e3045645f6a8e3ec4ef3c99ab9e23447c870f969a510e2"
      },
      "teacher_manifest": {
        "bytes": 2389,
        "sha256": "4480cd73a33a981d30399d3ab781c948873c2a9c22ebcb60ffd78d69e7f6cecb"
      },
      "initial_weights": {
        "bytes": 381750,
        "sha256": "9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365"
      },
      "initial_inventory": {
        "bytes": 7199,
        "sha256": "30849db6b8e931fd268289bb7f96c1f8aed46b5fef5bcb8e869632acffebdadb"
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
      "calibration_base_model.py": "fe0a0b3bfbcc9479f14efe1b6c54981fb82ad8a7576b92470e0f1f9f63d24de9",
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
      "head_descriptor.py": "3581afb32f0772df9d1a94791ee2983cb4c31bc7233bd661561f20840e2bf40f",
      "head_objective.py": "1d3e6cc26496e69b388b9bdb430543087c22f84be628a522ed6a25d391a54798",
      "head_optimizer.py": "2033de17a5467517426b6d5390b647aeca31b7078ab685bd69fadaf238074a67",
      "head_train.py": "3b7ec07366c6ed308c2cbc29f679f4fd7a803218e70e5f47b29225a2113b2524",
      "head_utils.py": "066b58bf36d675a343e0663151ac2489addec3e1322189d2b5c5c75e351d0bf4",
      "historical_data.py": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f",
      "historical_physical.py": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5",
      "learned_coarse_model.py": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0",
      "methods_view.py": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22",
      "mixed_order.py": "9a439127ddae653786a7a0962e13b63d5698c1c38ffba720eea7acd2a9b1d6d2",
      "model.py": "4a224850603d94d138e9e5001246ad7275196ad847936636eec86f49943418f9",
      "native_reconstruct.py": "d2db7ae0a145e82560ede85518a542a2fc83f7e7c6de33697823457b96d74445",
      "original_data.py": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd",
      "original_model.py": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb",
      "original_train.py": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9",
      "physical_residual.py": "6e1392c26880c034d09c72437fefd4c1865d6540f40e245825a6ad6a4e7f4204",
      "policy.py": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7",
      "portable_model.py": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935",
      "portable_policy.py": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460",
      "prepared.py": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76",
      "reference_spectral.py": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f",
      "retained_domain_model.py": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231",
      "scenes.py": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383",
      "smoke_train.py": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58",
      "student_slice.py": "f144329caed584de6715fb45e2581c9148e559a183f66559349bfe17df58f580",
      "support.py": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25",
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
    }
  },
  "final_inventory": {
    "coarse_logits": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "bytes": 252,
      "sha256": "d0f6ab7d8b8bb14e12d662b7a99be6160eca69f86f1adeaf223052cb43fe815f"
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
      "sha256": "ce539653d2012b641ddabf964679732d3f734f2c70f467e0ca7825e955e2092a"
    },
    "shape_head.0.bias": {
      "shape": [
        16
      ],
      "dtype": "<f4",
      "bytes": 64,
      "sha256": "1afde0d56f5e42e5c61b56356fee4ec8ad2761f0238922d17f638833b23cf48c"
    },
    "shape_head.2.weight": {
      "shape": [
        63,
        16
      ],
      "dtype": "<f4",
      "bytes": 4032,
      "sha256": "d93ea8a343ee507c9e95b01e86f66b73d6ee821d2f47aa15a2359009c2455e7a"
    },
    "shape_head.2.bias": {
      "shape": [
        63
      ],
      "dtype": "<f4",
      "bytes": 252,
      "sha256": "4405f6e9eb7688d036ffb8273a25644dd2c066f80e1f3ee3359c13501a78d3ac"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 70027776
  },
  "outputs": {
    "checkpoint-000.npz": {
      "bytes": 381750,
      "sha256": "9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365"
    },
    "probe.npy": {
      "bytes": 41608,
      "sha256": "699c497d7fbbecd1226a7c8fdba9b6e281f988c63d9c7e05ad126822f7a9d8b3"
    },
    "weights.npz": {
      "bytes": 381750,
      "sha256": "cc25fb7820c83b2b23b2259c4e306c247abdc01ec7e53c81ac3b7d86ae63e494"
    },
    "initial-weights.npz": {
      "bytes": 381750,
      "sha256": "9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365"
    },
    "checkpoint-10000.npz": {
      "bytes": 381750,
      "sha256": "cc25fb7820c83b2b23b2259c4e306c247abdc01ec7e53c81ac3b7d86ae63e494"
    },
    "history.json": {
      "bytes": 294829,
      "sha256": "e7943394dd175e0529c199d47219c63d2122a57716b9f0e30aa8990292b07ce2"
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
  "elapsed_s": 24.051838881976437,
  "scientific_acceptance_or_promotion": false
}

```
