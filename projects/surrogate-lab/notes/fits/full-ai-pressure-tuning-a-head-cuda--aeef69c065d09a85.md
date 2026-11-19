# Full AI pressure tuning · a-head CUDA

Retrospective source review · fit-aeef69c065d09a85

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** C120 is a pressure-only residual predictor with93,758 trained parameters, encoder widths[4, 8, 12, 16], bottleneck24, reversed decoder and2 internal pressure correction pass(es). The fine path uses bias-free linear convolutions, replicated boundaries, average pooling, nearest upsampling, skip concatenation and a global pooled addition. Input is normalized physical residual q=(rho/h)r. Pressure and equation arithmetic use float64 around the float32 CNN. A shape-conditioned positive learned cosine-mode proposal precedes each fine pass and is accepted only if its actual fine-equation residual norm does not increase. The gate does not bound the final neural correction. The model does not advance velocity/tracer or generate simultaneous future physical states.

**Purpose.** Test a lower coarse-head learning rate: use the fine-network rate rather than10/3 times that rate, paired with C119’s initial weights and sample order. This trains the full network; it is not the later head-only offline geometry calibration.

**Experiment.** Initialize fromthe same exact C107 checkpoint: 9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365. Use1,200 CUDA Adam updates, batch1 and800 small and400 larger draws; larger draws rotate through32/64/128 cell tiers. The34 captured train-only parents provide68 selected frames at8/16/32 cells per metre, dt1/60s and fluid density1000kg/m³; the domains include rectangular shapes and cubes through128³. Hard-case sampling uses frozen training-only rank weights up to4 and is disabled. The actual head rate multiplier is1, while inherited architecture metadata says 1000; preserve this discrepancy. Fine rate3e-5 falls to1e-5 at update901; seed2026101907, private sampling seed2026101908 and fixed C53 normalization. The canonical physical loss adds logarithmic final equation feedback and total0.05 earlier-stage feedback where applicable. Inherited auxiliary banks are disabled and both teacher-forward counts are zero. All arms record2,400 actual optimizer-loss CNN forwards; forward-cell counts differ by sampling. These are parent/frame counts, not proof of independent scenes. Trainer held-out evaluations are zero; the evidence wrapper’s historical post-fit consumed pressure probe does not create an independent validation split.

**Observed outcome.** On the same128 original consumed pressure requests, worst mean-free pressure relative L2 is0.0077217365590031431, and worst equation relative L2 is0.0011061111530836956. C119–C122 range about 0.00752–0.00797 pressure error, C123 reaches0.01583 and C1240.04810. The control has the lowest worst pressure of those four paired arms; the combined arm has the lowest worst equation error among them. The larger redistributed and one-stage graphs do worse on these tails. The common requests cover only small grids at8/16cells per metre and dt1/60s. No larger-grid physical accuracy, complete-generator drift or freshly qualified speed follows from this review.

**Lesson.** Reducing a head learning rate, resampling hard cases, changing channel allocation or increasing global modes does not guarantee better worst pressure error. Pressure-field and equation tails can rank the same four controlled arms differently. Equal CNN-forward counts do not imply equal FLOPs, memory traffic or measured latency when architectures and sampled grids differ. The one-stage accuracy loss is a real same-request result; its saved pressure pass is distinct from multistep physical-time prediction. Keep the old solver epoch and preserve negative outcomes rather than promote an attractive architecture label.

**Reproduction.** The original46-file witness,49 recipe source bindings, recorded twelve fitting-input declarations, exact rehashed initialization, train population, original train/restore recipes and runtime contract are bound below or in this review report. Dataset/input archives are not freshly decoded by this source-only review; clean reproduction must verify and consume their exact original bytes under the pinned old physics contract. Captured recipes include historical CPU deployment probes; follow current GPU-only inference policy via a separately identified execution rather than silently changing an original Run. Opening/importing a project must remain inert. Clean retraining, independent generalization and end-to-end reproduction remain unverified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/a-head-study-01/train/receipt.json",
    "sha256": "aeef69c065d09a85773f249d9611e2d22ff65bd5a5ea0dd2afa1740291ec24a8"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/a-head-study-01/train/history.json",
    "sha256": "499098408b5464419e84caa538aa112515dc4c16b3826cdff6e3fe7c31847169"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/coarse.py",
      "sha256": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/config.json",
      "sha256": "8ecefd850ccd694f43ca06d330936836da39713b4dc241f3362a882302a519e9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/data.py",
      "sha256": "09231fe0119461205b74578dab9ce77cbcbea7c537900941432e97f52e51f16d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/distillation.json",
      "sha256": "e99ea86fd1abaee220071225e2730c0fbae5b22ec0f19510007b80fde424bf7d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/feedback_objective.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/fine-tuning.json",
      "sha256": "0a636be36ae8b359f23a327104bd7dc7abae12f55161800f531ddbb00d004dcf"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/fine_tuning.py",
      "sha256": "bc1d0ec5f7d496d0d79af767a7032e5108750a43f6525594c45c57e126c8780c"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/fit-sampling.json",
      "sha256": "d737f149daf3b054c5a8d53d16f8cdee6f4f3be2a292ffb970e809839b9efcec"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/frozen_gn.py",
      "sha256": "ce280f6ce86aeccc31521b13e4df039780fda8c112c63616457771377903e08d"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/head_optimizer.py",
      "sha256": "2033de17a5467517426b6d5390b647aeca31b7078ab685bd69fadaf238074a67"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/hybrid-method.json",
      "sha256": "7286631ecc547c35be01b10413acc6e3f85649357bac8c308e25019cffccfbe7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/learned_coarse_model.py",
      "sha256": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/methods_view.py",
      "sha256": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/mixed_order.py",
      "sha256": "9a439127ddae653786a7a0962e13b63d5698c1c38ffba720eea7acd2a9b1d6d2"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/model.py",
      "sha256": "fe0a0b3bfbcc9479f14efe1b6c54981fb82ad8a7576b92470e0f1f9f63d24de9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/native_reconstruct.py",
      "sha256": "d2db7ae0a145e82560ede85518a542a2fc83f7e7c6de33697823457b96d74445"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/original_train.py",
      "sha256": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/physical_residual.py",
      "sha256": "6e1392c26880c034d09c72437fefd4c1865d6540f40e245825a6ad6a4e7f4204"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/policy.py",
      "sha256": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/portable_policy.py",
      "sha256": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/reference_spectral.py",
      "sha256": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/student_slice.py",
      "sha256": "f144329caed584de6715fb45e2581c9148e559a183f66559349bfe17df58f580"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/teacher_model.py",
      "sha256": "63425e0a06f96ccd35476c94fabccb9bdfbd3691c706d98f2fa362ea00b3b4d9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/train.py",
      "sha256": "9e75fd34f56923f2fbc3702b762ea0d51b0f23760da0a477a626b66b0277aef7"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/a-head-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/full-ai-next-priorities-v1/a-head-project-01/source-witness.json",
    "sha256": "a10ce10d83493817857284da5349d19fcd20c6c318030c4fe2108d528891b1d7"
  },
  "training_population": {
    "path": "runs/full-ai-next-priorities-v1/a-head-study-01/train/splits.json",
    "sha256": "8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"
  },
  "training_inputs": [
    {
      "slot": "initial_weights",
      "path": "runs/pressure-methods-v4/v4-student50-control-study-01/restore/weights.npz",
      "sha256": "9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365",
      "bytes": 381750
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-aeef69c065d09a85",
    "kind": "training_attempt",
    "name": "Full AI pressure tuning · a-head CUDA",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
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
        "bound": "max_axis_cells^2/2",
        "coefficients": "Positive learned sigmoid gains; no analytical inverse eigenvalues",
        "gate": "Accept learned coarse proposal only when its actual fine-equation residual norm does not increase",
        "initialization": "Gain 1 on 16^3; all original neural tensors preserved",
        "limitation": "Gate controls coarse proposal only; final neural pass and trajectory still require independent evaluation",
        "placement": "Learned coarse proposal before every fine neural pass",
        "trainable_parameters": 63
      },
      "student_distillation": {
        "distillation_weight": 0.0,
        "normalization": "Shared fixed original C53 scales",
        "target": "Teacher fine correction on each student-generated inner pressure residual",
        "teacher_used_at_inference": false,
        "teacher_weights_sha256": "42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4",
        "width_fraction": [
          1,
          2
        ]
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 2097152
    },
    "receipt_sha256": "aeef69c065d09a85773f249d9611e2d22ff65bd5a5ea0dd2afa1740291ec24a8",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/a-head-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "562d2edcffa345bab9e269996eab982c",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI pressure tuning · a-head CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1200,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "8ecefd850ccd694f43ca06d330936836da39713b4dc241f3362a882302a519e9",
      "inputs": {
        "dataset": {
          "bytes": 22711895,
          "sha256": "20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5"
        },
        "dataset_extra": {
          "bytes": 25135473,
          "sha256": "2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368"
        },
        "initial_weights": {
          "bytes": 381750,
          "sha256": "9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365"
        },
        "splits": {
          "bytes": 57361,
          "sha256": "8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"
        },
        "teacher_weights": {
          "bytes": 1491574,
          "sha256": "42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4"
        },
        "view_02": {
          "bytes": 22277708,
          "sha256": "85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb"
        },
        "view_03": {
          "bytes": 23650806,
          "sha256": "98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8"
        },
        "view_04": {
          "bytes": 22239100,
          "sha256": "1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc"
        },
        "view_05": {
          "bytes": 24113006,
          "sha256": "6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda"
        },
        "view_06": {
          "bytes": 19897861,
          "sha256": "8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080"
        },
        "view_07": {
          "bytes": 22257574,
          "sha256": "4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9"
        },
        "view_08": {
          "bytes": 18319876,
          "sha256": "2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "9ea38c6e70ca1ae5822f9a47cca631683d064cd369c933e2ba66e39ccd67e3be",
      "source_witness_sha256": "a10ce10d83493817857284da5349d19fcd20c6c318030c4fe2108d528891b1d7"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365",
        "bytes": 381750,
        "stage": "initial"
      },
      {
        "name": "checkpoint-1200.npz",
        "sha256": "728cd489571444a23b368285ee7c9521fd486b91bbc5e1f35ffd6f2c7718aaf0",
        "bytes": 381750,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-600.npz",
        "sha256": "4c2cf023da52fad2c56e490473946542e6717e46169b2130511445397b9e7d37",
        "bytes": 381750,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "728cd489571444a23b368285ee7c9521fd486b91bbc5e1f35ffd6f2c7718aaf0",
        "bytes": 381750,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 4156129280,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/118"
    ],
    "purpose": {
      "status": "source_review_required",
      "evidence": "Captured architecture, training configuration and linked named entries."
    },
    "outcome": {
      "status": "retained_training_evidence_only",
      "fresh_common_benchmark": "pending",
      "scientific_promotion": false
    },
    "lesson": {
      "status": "review_required_no_automatic_causal_claim"
    },
    "reproduction": {
      "receipt": "runs/full-ai-next-priorities-v1/a-head-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/a-head-study-01/train/receipt.json",
    "sha256": "aeef69c065d09a85773f249d9611e2d22ff65bd5a5ea0dd2afa1740291ec24a8",
    "bytes": 14547
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C120": {
        "name": "a-head",
        "weights_sha256": "728cd489571444a23b368285ee7c9521fd486b91bbc5e1f35ffd6f2c7718aaf0",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.0011061111530836956,
          "pressure_mean_free_relative_l2": 0.007721736559003143,
          "pressure_absolute_rms": 0.0686730622583786,
          "pressure_absolute_max": 0.15035101304458742
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0025",
            "native_frame": 63,
            "counts_xyz": [
              9,
              11,
              11
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0060",
            "native_frame": 63,
            "counts_xyz": [
              5,
              6,
              15
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.0004872866178289869,
              "pressure_mean_free_relative_l2": 0.007721736559003143,
              "pressure_absolute_rms": 0.0686730622583786,
              "pressure_absolute_max": 0.15035101304458742
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.0011061111530836956,
              "pressure_mean_free_relative_l2": 0.0019456506845797117,
              "pressure_absolute_rms": 0.027015524534175547,
              "pressure_absolute_max": 0.07927993931851926
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C120/results.json",
          "sha256": "a9519d7a44a213a7af29b5ed9bb86bc7555d68d13b40fa95a69fabf4d2655816"
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
{"architecture":{"activation":"linear","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":24,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[16,12,8,4],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[4,8,12,16],"fine_tuning":{"arm":"a-head","early_weight":0.05,"initial_weights_sha256":"9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"learned_coarse_revision":{"basis":"Orthonormal separable Neumann cosine modes 0..3, excluding DC","bound":"max_axis_cells^2/2","coefficients":"Positive learned sigmoid gains; no analytical inverse eigenvalues","gate":"Accept learned coarse proposal only when its actual fine-equation residual norm does not increase","initialization":"Gain 1 on 16^3; all original neural tensors preserved","limitation":"Gate controls coarse proposal only; final neural pass and trajectory still require independent evaluation","placement":"Learned coarse proposal before every fine neural pass","trainable_parameters":63},"max_cells":2097152,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Balanced spatial tiers; private RNG; fixed update budget.","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","shape_informed_coarse":{"added_parameters":1135,"features":"log(Nz/16), log(Ny/16), log(Nx/16)","fine_initialization":"Exact completed v2-headshape CPU restore; matched continuation control","head_learning_rate_multiplier":1000,"network":"3 -> 16 tanh -> 63 additive coarse logits; final layer initially zero","operator":"Existing sealed constant-density equation"},"student_distillation":{"distillation_weight":0.0,"normalization":"Shared fixed original C53 scales","target":"Teacher fine correction on each student-generated inner pressure residual","teacher_used_at_inference":false,"teacher_weights_sha256":"42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4","width_fraction":[1,2]},"supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[12,6,11],"frame":1,"native_frame":63,"parent":3,"split":"train","trajectory_id":"baseline--water-train-0003"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":0,"native_frame":15,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":0,"native_frame":15,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":15,"parent":18,"split":"train","trajectory_id":"large-train-64-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[31,23,17],"frame":1,"native_frame":63,"parent":21,"split":"train","trajectory_id":"v2-train-00-cpm16"},{"cell_size_m":0.125,"counts_xyz":[48,13,9],"frame":0,"native_frame":15,"parent":24,"split":"train","trajectory_id":"v2-train-02-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[64,64,64],"frame":1,"native_frame":15,"parent":27,"split":"train","trajectory_id":"v3-train-00-n64-cpm16"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":0,"parent":30,"split":"train","trajectory_id":"v3-train-01-n64-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[128,128,128],"frame":1,"native_frame":15,"parent":33,"split":"train","trajectory_id":"v3-train-01-n128-cpm32"}],"closure":{"experiment_sha256":"8ecefd850ccd694f43ca06d330936836da39713b4dc241f3362a882302a519e9","inputs":{"dataset":{"bytes":22711895,"sha256":"20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5"},"dataset_extra":{"bytes":25135473,"sha256":"2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368"},"initial_weights":{"bytes":381750,"sha256":"9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365"},"splits":{"bytes":57361,"sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"teacher_weights":{"bytes":1491574,"sha256":"42b2504fd922e9123823427857b4de36b153dbc343e157ad6fac36e3760612f4"},"view_02":{"bytes":22277708,"sha256":"85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb"},"view_03":{"bytes":23650806,"sha256":"98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8"},"view_04":{"bytes":22239100,"sha256":"1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc"},"view_05":{"bytes":24113006,"sha256":"6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda"},"view_06":{"bytes":19897861,"sha256":"8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080"},"view_07":{"bytes":22257574,"sha256":"4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9"},"view_08":{"bytes":18319876,"sha256":"2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"9ea38c6e70ca1ae5822f9a47cca631683d064cd369c933e2ba66e39ccd67e3be","source_witness_sha256":"a10ce10d83493817857284da5349d19fcd20c6c318030c4fe2108d528891b1d7"},"diagnostic_population":"train_only","elapsed_s":86.97493628802476,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.485331226902531e-06,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":1759897088,"peak_reserved_bytes":2002780160,"tf32":false},"fine_tuning_statistics":{"distillation_loss_calls":0,"distillation_teacher_forward_cells":0,"distillation_teacher_forwards":0,"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"frozen_full_teacher_unchanged":true,"heldout_evaluations":0,"mode":"train","optimizer_updates":1200,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":381750,"role":"checkpoint","sha256":"9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365"},"checkpoint-1200.npz":{"bytes":381750,"role":"checkpoint","sha256":"728cd489571444a23b368285ee7c9521fd486b91bbc5e1f35ffd6f2c7718aaf0"},"checkpoint-600.npz":{"bytes":381750,"role":"checkpoint","sha256":"4c2cf023da52fad2c56e490473946542e6717e46169b2130511445397b9e7d37"},"history.json":{"bytes":17242,"role":"training_diagnostics","sha256":"499098408b5464419e84caa538aa112515dc4c16b3826cdff6e3fe7c31847169"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":1238286,"role":"predictions","sha256":"fe03a1224c904e24327e0f938fc469465a9ad4a2ea3cdf68894f0c5507708ffb"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":57361,"role":"training_splits","sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"summary.json":{"bytes":684,"role":"result_summary","sha256":"8f49251181a2381803c18708dacc5e42d49ee7bf109ef3d50850a7ac32bbe2c6"},"weights.npz":{"bytes":381750,"role":"checkpoint","sha256":"728cd489571444a23b368285ee7c9521fd486b91bbc5e1f35ffd6f2c7718aaf0"}},"per_case_inference_s":[0.010097438003867865,0.004510200000368059,0.0038559490349143744,0.0045796490157954395,0.0049303569830954075,0.005146900948602706,0.6740070999949239,0.022175395977683365,0.01048519799951464,0.6051313169882633,0.6156920900102705,3.4183775520068593],"prediction_export_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"f1dd0aacd4cbf364be938b002c739838e904e248b5ec15e0edfd6bef1d6ff635","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"c567c2b1ea6542395eadaf51f6e963c7e0682e361d8fee69ee108f0e6837cc49","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"b1ec2c9c9bef2e0ef36689367a6227003b87391c52de41a11f37ee8998b48da8","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"2e8b1cd99071876ce1588012d86bf77ce882985e9e2cd7ed89508ce2d296b291","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"2b03b95c474c9fe96e882e02131480ffa8581ecab623c303888227362be8713f","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"6fb5b538259532bb9be290f2e1fd1d49a688f35475d48b7f3e2779819e898d5d","shape":[7,15,14]},"case_006":{"bytes":262144,"dtype":"<f8","sha256":"d747c8fc00aea7850d3c75b367949d389ee3e7ba88994b9c3d040286009084a7","shape":[32,32,32]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"ef2ea9eecaf1fff739fd371bbce57dbbed85e430f935faecddd309e8266ba92b","shape":[17,23,31]},"case_008":{"bytes":22464,"dtype":"<f8","sha256":"f1116e5987eba71321cde83f2dc47688b0370bb1b9dcaf3dfd31dd45af0adcec","shape":[9,13,24]},"case_009":{"bytes":262144,"dtype":"<f8","sha256":"4cbdc438329aa8ac1a2189c71193d056469bd4fe825c7ea295735939858de7d2","shape":[32,32,32]},"case_010":{"bytes":262144,"dtype":"<f8","sha256":"4f8c8b84075c51e025c554e766163e4cc33815e74d5329a510453de43883ac80","shape":[32,32,32]},"case_011":{"bytes":262144,"dtype":"<f8","sha256":"85477f72877f3d7a365fc682620413dc4314deaf70c7d7ac8353769a91af98be","shape":[32,32,32]}},"prediction_export_policy":"NPZ samples at most32cells/axis; full prediction_inventory hashes every full-resolution value and is checked exactly on fresh CPU restore","prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"f1dd0aacd4cbf364be938b002c739838e904e248b5ec15e0edfd6bef1d6ff635","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"c567c2b1ea6542395eadaf51f6e963c7e0682e361d8fee69ee108f0e6837cc49","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"b1ec2c9c9bef2e0ef36689367a6227003b87391c52de41a11f37ee8998b48da8","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"2e8b1cd99071876ce1588012d86bf77ce882985e9e2cd7ed89508ce2d296b291","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"2b03b95c474c9fe96e882e02131480ffa8581ecab623c303888227362be8713f","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"6fb5b538259532bb9be290f2e1fd1d49a688f35475d48b7f3e2779819e898d5d","shape":[7,15,14]},"case_006":{"bytes":2097152,"dtype":"<f8","sha256":"dba5ad8e1bce2825c6a57a39981d7466115504040c52f2acfb5abd37747f2e85","shape":[64,64,64]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"ef2ea9eecaf1fff739fd371bbce57dbbed85e430f935faecddd309e8266ba92b","shape":[17,23,31]},"case_008":{"bytes":44928,"dtype":"<f8","sha256":"57c093ac59b90f7201a2918360cc9d368402ce25a614f9e06868f0c6c9532c27","shape":[9,13,48]},"case_009":{"bytes":2097152,"dtype":"<f8","sha256":"0352ea19e4a94c701bb33ec041b2b40347a3da54795d439332f540f28365cbed","shape":[64,64,64]},"case_010":{"bytes":2097152,"dtype":"<f8","sha256":"634fc52a1e6ce0bf608e4c04f97ab644792b4e9a1deb55ba5c7c8e031526fa53","shape":[64,64,64]},"case_011":{"bytes":16777216,"dtype":"<f8","sha256":"4b67e050fb7b2cd9ed306664cbdf92ffabc7927ece4411ebeac4bc93d80dd576","shape":[128,128,128]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":4156129280,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
