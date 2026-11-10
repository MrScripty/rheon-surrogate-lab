# Pressure methods v5 · v5-multilevel CUDA

Retrospective source review · fit-c78b892406a60d33

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** Three bias-free replicated 3³ convolutions, channels 1→4→4→1, with 648 learned F32 coefficients and a fixed residual skip q/8. All layers are linear; F32 execution order remains relevant even when a combined mathematical stencil exists. The input is the normalized physical residual q=(rho/h)r; the F64 reconstructed local pressure correction is cell-zero gauged and added to supplied base pressure. The captured predict_pressure function performs this local readout only; it does not run a V-cycle, GCR, advection or a full simulation advance.

**Purpose.** Train the same tiny local smoother after one frozen numerical multilevel cycle, targeting the remaining correction while retaining original pressure/RHS labels.

**Experiment.** 2,400 CUDA Adam updates, batch one, clip norm one; fine rate 0.001 with captured late drops. There are 34 parent groups / 68 saved examples, all train-only. The sampler preserves 1,600 small draws and interleaves 800 larger-tier draws; exact parent/frame exposures and voxel work verify. Grids extend through 128³ at 8/16/32 cells per metre, with fixed dt=1/60 s and fluid density 1,000 kg/m³. Before fitting, the source materializes one frozen CPU numerical V-cycle for each of the 68 saved frames, replacing base/residual while retaining original reference pressure and RHS labels. This preprocessing is outside the learned graph. The local objective weights pressure 0.25, pressure-gradient 0.25, equation 0.4 and spatial maximum-equation 0.1, plus log1p(relative equation squared/1e-12). Captured learning work is 2,400 root neural forwards / 612,197,560 input voxels; inherited auxiliary banks and the old teacher are inactive.

**Observed outcome.** Retained final logged training loss 23.139797445385376. The existing common 128-request result has worst mean-free pressure relative L2 0.23135373320406569, equation relative L2 0.10438271669853232 and GPU median of case-median pressure calls 0.72940375000000002 ms. Scope: Local smoother only on the common raw requests. Numerical hierarchy/GCR composition is excluded and requires a separate complete-solver benchmark. Timings include host-owned input/output, validation/packing, transfers and synchronization for the captured readout, with setup/scoring separate. These are retained consumed small-grid cases, not independent large-grid or trajectory results. No new model execution occurs here.

**Lesson.** The deployed local readout is unchanged while training bases/residuals are conditioned by a V-cycle. Its common raw-request benchmark omits that hierarchy and evaluates a different residual context from fitting. Do not treat this as a complete hybrid-system regression.

**Reproduction.** Retain the exact typed native recipe, all witnessed source/config/runtime, fixed normalization, complete original input packet, source population and initial/final tensors. Preserve balanced private RNG streams, physical parent/frame counts and final-update selection. Retain native augmentation receipts or manufactured-pattern generator as applicable; post-coarse and differentiable hierarchy training require their exact numerical arithmetic and gauge/adjoint. The historical wrapper/restore runs CPU neural inference and needs a qualified GPU-only route before replay. Clean retraining, new-Rheon regeneration, complete-composition benchmarking and drift remain unqualified; no historical Trial ancestry is invented.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v5/v5-multilevel-study-01/train/receipt.json",
    "sha256": "c78b892406a60d3372da4764e5043d4548b1f8bd492482fad98abd032ba7877a"
  },
  "history": {
    "path": "runs/pressure-methods-v5/v5-multilevel-study-01/train/history.json",
    "sha256": "6736a964ade953ffb59a9c77e47be889c897efa3e0c689848f840ca7e2fa541d"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/coarse.py",
      "sha256": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/config.json",
      "sha256": "55e9ef702d01b855ff633af220aa389795fc7a9c160ba683c41115f5b1b87533"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/data.py",
      "sha256": "ecb49d27bb91054a088a7ea2718e59a00708ef878bcd295305c6a5ed3ea982b7"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/fine-tuning.json",
      "sha256": "d514bd2a662b1f93e4cb3fd49adbf303932180df16090a6be101de1e3ce4c22f"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/fine_tuning.py",
      "sha256": "35645824cfb6190d933b73146b99146097c7b684ed57f8d0be1e9b740d48f948"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/frozen_gn.py",
      "sha256": "ce280f6ce86aeccc31521b13e4df039780fda8c112c63616457771377903e08d"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/head_optimizer.py",
      "sha256": "e280386a3df5ca474903a4e6c42ec69177348d76b59226897e41c39af0000818"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/hybrid-method.json",
      "sha256": "7286631ecc547c35be01b10413acc6e3f85649357bac8c308e25019cffccfbe7"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/learned_coarse_model.py",
      "sha256": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/methods_view.py",
      "sha256": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/mixed_order.py",
      "sha256": "93d987b0f10101fb5b380e7310aa77eb6f5c7557198f6d43d90f102393e47b04"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/model.py",
      "sha256": "eed615467eee2ba923587c782ecf350329551e51b294a1b28a05df8c82926cb0"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/native_reconstruct.py",
      "sha256": "bf9f9e5712a22ae01a1dcdeffbdd89a9ba177b50f8aa36deb9692e7f15c61335"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/original_train.py",
      "sha256": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/physical_residual.py",
      "sha256": "6e1392c26880c034d09c72437fefd4c1865d6540f40e245825a6ad6a4e7f4204"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/policy.py",
      "sha256": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/portable_policy.py",
      "sha256": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/reference_spectral.py",
      "sha256": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/train.py",
      "sha256": "b5b11637cbe2a3a4742504781689f35581731290b78101a86d7378ec2a1a1f4d"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    },
    {
      "path": "runs/pressure-methods-v5/v5-multilevel-project-02/training_multilevel.py",
      "sha256": "eb8dfc7f963455112eb430701a941c3a9921b9ca4331e02ee1786fa2e3ad5dba"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v5/v5-multilevel-project-02/source-witness.json",
    "sha256": "b3bdcc9172bf43d83399b2ce55e0a4a228690cb2f8858bd63e4b7c3952db5a5e"
  },
  "training_population": {
    "path": "runs/pressure-methods-v5/v5-multilevel-study-01/train/splits.json",
    "sha256": "8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"
  },
  "training_inputs": [
    {
      "bytes": 22711895,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-00.npz",
      "sha256": "20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5",
      "slot": "dataset"
    },
    {
      "bytes": 25135473,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-01.npz",
      "sha256": "2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368",
      "slot": "dataset_extra"
    },
    {
      "bytes": 3400,
      "path": "runs/pressure-methods-v5/tiny-initialization-01/weights.npz",
      "sha256": "ce457a138e56100e481ce0216d608792b6c466cf5eb2a34b2ac454ff0cf3b419",
      "slot": "initial_weights"
    },
    {
      "bytes": 57361,
      "path": "runs/pressure-methods-v3/derived-training-view-01/manifest.json",
      "sha256": "8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169",
      "slot": "splits"
    },
    {
      "bytes": 22277708,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-02.npz",
      "sha256": "85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb",
      "slot": "view_02"
    },
    {
      "bytes": 23650806,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-03.npz",
      "sha256": "98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8",
      "slot": "view_03"
    },
    {
      "bytes": 22239100,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-04.npz",
      "sha256": "1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc",
      "slot": "view_04"
    },
    {
      "bytes": 24113006,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-05.npz",
      "sha256": "6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda",
      "slot": "view_05"
    },
    {
      "bytes": 19897861,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-06.npz",
      "sha256": "8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080",
      "slot": "view_06"
    },
    {
      "bytes": 22257574,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-07.npz",
      "sha256": "4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9",
      "slot": "view_07"
    },
    {
      "bytes": 18319876,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-08.npz",
      "sha256": "2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30",
      "slot": "view_08"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-c78b892406a60d33",
    "kind": "training_attempt",
    "name": "Pressure methods v5 · v5-multilevel CUDA",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-multilevel-local-smoother-v1",
      "input_channels": 1,
      "activation": "linear",
      "restrictions": "Uniform-density cubic sealed Neumann operator, up to 128 cells per axis for training"
    },
    "receipt_sha256": "c78b892406a60d3372da4764e5043d4548b1f8bd492482fad98abd032ba7877a",
    "receipt_paths": [
      "runs/pressure-methods-v5/v5-multilevel-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "a57aa6c7caa24f5ab4499aa31ad6d01b",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods v5 · v5-multilevel CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2400,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "55e9ef702d01b855ff633af220aa389795fc7a9c160ba683c41115f5b1b87533",
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
          "bytes": 3400,
          "sha256": "ce457a138e56100e481ce0216d608792b6c466cf5eb2a34b2ac454ff0cf3b419"
        },
        "splits": {
          "bytes": 57361,
          "sha256": "8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"
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
      "sample_order_sha256": "157178a36ed5bebdd0971c3643a6720aa377962175614cdde7115ddef5cfbf56",
      "source_witness_sha256": "b3bdcc9172bf43d83399b2ce55e0a4a228690cb2f8858bd63e4b7c3952db5a5e"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "ce457a138e56100e481ce0216d608792b6c466cf5eb2a34b2ac454ff0cf3b419",
        "bytes": 3400,
        "stage": "initial"
      },
      {
        "name": "checkpoint-2400.npz",
        "sha256": "860d1d359ca5491f0f794068db239cadde3c7f55cbca7a8c35c8b4112fbaec7f",
        "bytes": 3400,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "860d1d359ca5491f0f794068db239cadde3c7f55cbca7a8c35c8b4112fbaec7f",
        "bytes": 3400,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 3171110912,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/111"
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
      "receipt": "runs/pressure-methods-v5/v5-multilevel-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v5/v5-multilevel-study-01/train/receipt.json",
    "sha256": "c78b892406a60d3372da4764e5043d4548b1f8bd492482fad98abd032ba7877a",
    "bytes": 9844
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C113": {
        "name": "v5-multilevel",
        "weights_sha256": "860d1d359ca5491f0f794068db239cadde3c7f55cbca7a8c35c8b4112fbaec7f",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.10438271669853232,
          "pressure_mean_free_relative_l2": 0.2313537332040657,
          "pressure_absolute_rms": 15.420200374858755,
          "pressure_absolute_max": 43.82292168435832
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
            "trajectory_id": "baseline--water-validation-0031",
            "native_frame": 15,
            "counts_xyz": [
              14,
              6,
              16
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.07985713648371512,
              "pressure_mean_free_relative_l2": 0.15697496159500904,
              "pressure_absolute_rms": 15.420200374858755,
              "pressure_absolute_max": 43.82292168435832
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.10438271669853232,
              "pressure_mean_free_relative_l2": 0.2313537332040657,
              "pressure_absolute_rms": 4.587366165972367,
              "pressure_absolute_max": 13.203846711673323
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-02/C113/results.json",
          "sha256": "c28b61033034f68a68a1a1c600f48d95c260a1117edc845393124d0273a86fd6"
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
{"architecture":{"activation":"linear","bias":false,"boundary":"Replicated one-cell halo at each convolution","channels":["dimensionless_laplacian_rhs"],"coarse_training_cycles":1,"deployment":"Local correction called after an independently validated numerical multilevel cycle","fixed_skip":"q / 8","input_channels":1,"kernels":[3,3,3],"parameters":648,"preprocessing":"q=(rho/h)*residual, divided by fixed original train-only q RMS","reconstruction":"Raw correction times the same q RMS; cell-zero gauge; add supplied base","restrictions":"Uniform-density cubic sealed Neumann operator, up to 128 cells per axis for training","schema":"rheon-multilevel-local-smoother-v1","widths":[1,4,4,1]},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[12,6,11],"frame":1,"native_frame":63,"parent":3,"split":"train","trajectory_id":"baseline--water-train-0003"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":0,"native_frame":15,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":0,"native_frame":15,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":15,"parent":18,"split":"train","trajectory_id":"large-train-64-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[31,23,17],"frame":1,"native_frame":63,"parent":21,"split":"train","trajectory_id":"v2-train-00-cpm16"},{"cell_size_m":0.125,"counts_xyz":[48,13,9],"frame":0,"native_frame":15,"parent":24,"split":"train","trajectory_id":"v2-train-02-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[64,64,64],"frame":1,"native_frame":15,"parent":27,"split":"train","trajectory_id":"v3-train-00-n64-cpm16"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":0,"parent":30,"split":"train","trajectory_id":"v3-train-01-n64-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[128,128,128],"frame":1,"native_frame":15,"parent":33,"split":"train","trajectory_id":"v3-train-01-n128-cpm32"}],"closure":{"experiment_sha256":"55e9ef702d01b855ff633af220aa389795fc7a9c160ba683c41115f5b1b87533","inputs":{"dataset":{"bytes":22711895,"sha256":"20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5"},"dataset_extra":{"bytes":25135473,"sha256":"2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368"},"initial_weights":{"bytes":3400,"sha256":"ce457a138e56100e481ce0216d608792b6c466cf5eb2a34b2ac454ff0cf3b419"},"splits":{"bytes":57361,"sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"view_02":{"bytes":22277708,"sha256":"85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb"},"view_03":{"bytes":23650806,"sha256":"98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8"},"view_04":{"bytes":22239100,"sha256":"1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc"},"view_05":{"bytes":24113006,"sha256":"6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda"},"view_06":{"bytes":19897861,"sha256":"8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080"},"view_07":{"bytes":22257574,"sha256":"4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9"},"view_08":{"bytes":18319876,"sha256":"2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"157178a36ed5bebdd0971c3643a6720aa377962175614cdde7115ddef5cfbf56","source_witness_sha256":"b3bdcc9172bf43d83399b2ce55e0a4a228690cb2f8858bd63e4b7c3952db5a5e"},"diagnostic_population":"train_only","elapsed_s":44.85572264000075,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":4.6172505818731224e-09,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":1052439552,"peak_reserved_bytes":1103101952,"tf32":false},"fine_tuning_statistics":{"loss_calls":2400,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":2400,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":3400,"role":"checkpoint","sha256":"ce457a138e56100e481ce0216d608792b6c466cf5eb2a34b2ac454ff0cf3b419"},"checkpoint-2400.npz":{"bytes":3400,"role":"checkpoint","sha256":"860d1d359ca5491f0f794068db239cadde3c7f55cbca7a8c35c8b4112fbaec7f"},"history.json":{"bytes":9406,"role":"training_diagnostics","sha256":"6736a964ade953ffb59a9c77e47be889c897efa3e0c689848f840ca7e2fa541d"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":1238286,"role":"predictions","sha256":"1b919ec40d3f0d7a3c4bbd776bb4095094943ca0d9ce9c01fd21180ecb499bb7"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":57361,"role":"training_splits","sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"summary.json":{"bytes":677,"role":"result_summary","sha256":"fe3bd7c00d0f28c5fc355a6d4b3192a1f814c1e1eeb4400046f2a94a82807057"},"weights.npz":{"bytes":3400,"role":"checkpoint","sha256":"860d1d359ca5491f0f794068db239cadde3c7f55cbca7a8c35c8b4112fbaec7f"}},"per_case_inference_s":[0.0012409510090947151,0.0003800659906119108,0.0002652220136951655,0.0003491929965093732,0.0004778050060849637,0.0004276669933460653,0.17800625300151296,0.003055329987546429,0.0010769210057333112,0.16942305301199667,0.16540418501244858,0.4877935209951829],"prediction_export_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"ee62e5ff09a9acfa72fe35e86eb100034e64b6f56292baf77fee90244fb21d48","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"9028851c0bbd3f69278c8a4d89b8f024b4375a2495935bc13e52eda23f928772","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"5dc4ad9a3ada3d04030222886bbe91b80bf0b73e09c3f7370351912784f5c18b","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"b3fdbef1ba842d5ea603d0781cc57b6a5b683ed954d6bf052bc35fa6c3795678","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"103a8ccf42a082481d6e9def57ecdbe81d5d23df22a7211ba246c04fb6a37979","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"f6a989d8c0617f9439dcfa9d2dbc44afec93a98a0818db063859d49cb5c195c6","shape":[7,15,14]},"case_006":{"bytes":262144,"dtype":"<f8","sha256":"0f7810e7917d0699ced40c501801b23f30946f2c94db618f0f43f54181aa6ac9","shape":[32,32,32]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"478fb61e8c57ba0729392f1ed81dfb3282a0595f4f6d1811408eb673d435af48","shape":[17,23,31]},"case_008":{"bytes":22464,"dtype":"<f8","sha256":"ad833b70269ce58112ee4929ff078ee8722ada37c19127aaea57b3e01dcb4812","shape":[9,13,24]},"case_009":{"bytes":262144,"dtype":"<f8","sha256":"44fbf4d5a6138af304bbf6ec582a50d2415db80978aadf75879844ccb2996228","shape":[32,32,32]},"case_010":{"bytes":262144,"dtype":"<f8","sha256":"7ce375cd4b358254a3371c9cc08667ebc3e391168513987a636455aa648db400","shape":[32,32,32]},"case_011":{"bytes":262144,"dtype":"<f8","sha256":"7be373936d2177d593b9eabbe1edb81c8046d37903d97d178ad71c35ce322943","shape":[32,32,32]}},"prediction_export_policy":"NPZ samples at most32cells/axis; full prediction_inventory hashes every full-resolution value and is checked exactly on fresh CPU restore","prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"ee62e5ff09a9acfa72fe35e86eb100034e64b6f56292baf77fee90244fb21d48","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"9028851c0bbd3f69278c8a4d89b8f024b4375a2495935bc13e52eda23f928772","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"5dc4ad9a3ada3d04030222886bbe91b80bf0b73e09c3f7370351912784f5c18b","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"b3fdbef1ba842d5ea603d0781cc57b6a5b683ed954d6bf052bc35fa6c3795678","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"103a8ccf42a082481d6e9def57ecdbe81d5d23df22a7211ba246c04fb6a37979","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"f6a989d8c0617f9439dcfa9d2dbc44afec93a98a0818db063859d49cb5c195c6","shape":[7,15,14]},"case_006":{"bytes":2097152,"dtype":"<f8","sha256":"7f0ca9853265cfaba741baf0f94d53b1fa6cf9bc963ae6ef5346751ab66d4cd3","shape":[64,64,64]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"478fb61e8c57ba0729392f1ed81dfb3282a0595f4f6d1811408eb673d435af48","shape":[17,23,31]},"case_008":{"bytes":44928,"dtype":"<f8","sha256":"bd537660367623cebf3e5ee58ea2225d5e417c0d1478cc9f0f092e5f2a6ac4a8","shape":[9,13,48]},"case_009":{"bytes":2097152,"dtype":"<f8","sha256":"2a009a48c9ced80f79d11cf660185ddd58bb509ef5b2a5fe64471fabfc32eb41","shape":[64,64,64]},"case_010":{"bytes":2097152,"dtype":"<f8","sha256":"b30f784c2bd44925ee4e6fcaa569e66544f93520b885750fcf1c807e46d197da","shape":[64,64,64]},"case_011":{"bytes":16777216,"dtype":"<f8","sha256":"d5a82a9248b7b22e685f405924a3ec16c6dac94acffe0456478d587344f42926","shape":[128,128,128]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":3171110912,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
