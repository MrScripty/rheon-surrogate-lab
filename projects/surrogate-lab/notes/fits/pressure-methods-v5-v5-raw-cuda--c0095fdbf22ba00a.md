# Pressure methods v5 · v5-raw CUDA

Retrospective source review · fit-c0095fdbf22ba00a

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** Three bias-free replicated 3³ convolutions, channels 1→4→4→1, with 648 learned F32 coefficients and a fixed residual skip q/8. All layers are linear; F32 execution order remains relevant even when a combined mathematical stencil exists. The input is the normalized physical residual q=(rho/h)r; the F64 reconstructed local pressure correction is cell-zero gauged and added to supplied base pressure. The captured predict_pressure function performs this local readout only; it does not run a V-cycle, GCR, advection or a full simulation advance.

**Purpose.** Establish a tiny local pressure smoother trained directly on raw recorded requests, as a control for post-coarse residual conditioning.

**Experiment.** 2,400 CUDA Adam updates, batch one, clip norm one; fine rate 0.001 with captured late drops. There are 34 parent groups / 68 saved examples, all train-only. The sampler preserves 1,600 small draws and interleaves 800 larger-tier draws; exact parent/frame exposures and voxel work verify. Grids extend through 128³ at 8/16/32 cells per metre, with fixed dt=1/60 s and fluid density 1,000 kg/m³. Raw original base/residual fields enter fitting. The local objective weights pressure 0.25, pressure-gradient 0.25, equation 0.4 and spatial maximum-equation 0.1, plus log1p(relative equation squared/1e-12). Captured learning work is 2,400 root neural forwards / 612,197,560 input voxels; inherited auxiliary banks and the old teacher are inactive.

**Observed outcome.** Retained final logged training loss 24.154032279770384. The existing common 128-request result has worst mean-free pressure relative L2 0.22061744925986818, equation relative L2 0.092286374818681405 and GPU median of case-median pressure calls 0.69514125000000004 ms. Scope: Local smoother only on the common raw requests. Numerical hierarchy/GCR composition is excluded and requires a separate complete-solver benchmark. Timings include host-owned input/output, validation/packing, transfers and synchronization for the captured readout, with setup/scoring separate. These are retained consumed small-grid cases, not independent large-grid or trajectory results. No new model execution occurs here.

**Lesson.** A local correction has limited global pressure reach. The common raw-request error is a valid local-candidate diagnostic, but neither its low latency nor its pressure error measures a complete numerical-hierarchy solve.

**Reproduction.** Retain the exact typed native recipe, all witnessed source/config/runtime, fixed normalization, complete original input packet, source population and initial/final tensors. Preserve balanced private RNG streams, physical parent/frame counts and final-update selection. Retain native augmentation receipts or manufactured-pattern generator as applicable; post-coarse and differentiable hierarchy training require their exact numerical arithmetic and gauge/adjoint. The historical wrapper/restore runs CPU neural inference and needs a qualified GPU-only route before replay. Clean retraining, new-Rheon regeneration, complete-composition benchmarking and drift remain unqualified; no historical Trial ancestry is invented.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v5/v5-raw-study-01/train/receipt.json",
    "sha256": "c0095fdbf22ba00a9a647532c772ecdd421b28eda6044af500252c2c0f539084"
  },
  "history": {
    "path": "runs/pressure-methods-v5/v5-raw-study-01/train/history.json",
    "sha256": "bfabbc61e77fd82c098476c4fc0c306adde7bc060c8bd7865962093c45de5292"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/coarse.py",
      "sha256": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/config.json",
      "sha256": "c6b834df8ede6df9d4c6e6c0a02af9b579eee2dc27d44d322f03fafd5f50d470"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/data.py",
      "sha256": "ecb49d27bb91054a088a7ea2718e59a00708ef878bcd295305c6a5ed3ea982b7"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/fine-tuning.json",
      "sha256": "7a1fa742236d0ec9b20fef9524d574c44e02f150c31e585122cd3d9ec7cae97e"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/fine_tuning.py",
      "sha256": "35645824cfb6190d933b73146b99146097c7b684ed57f8d0be1e9b740d48f948"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/frozen_gn.py",
      "sha256": "ce280f6ce86aeccc31521b13e4df039780fda8c112c63616457771377903e08d"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/head_optimizer.py",
      "sha256": "e280386a3df5ca474903a4e6c42ec69177348d76b59226897e41c39af0000818"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/hybrid-method.json",
      "sha256": "7286631ecc547c35be01b10413acc6e3f85649357bac8c308e25019cffccfbe7"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/learned_coarse_model.py",
      "sha256": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/methods_view.py",
      "sha256": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/mixed_order.py",
      "sha256": "93d987b0f10101fb5b380e7310aa77eb6f5c7557198f6d43d90f102393e47b04"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/model.py",
      "sha256": "eed615467eee2ba923587c782ecf350329551e51b294a1b28a05df8c82926cb0"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/native_reconstruct.py",
      "sha256": "bf9f9e5712a22ae01a1dcdeffbdd89a9ba177b50f8aa36deb9692e7f15c61335"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/original_train.py",
      "sha256": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/physical_residual.py",
      "sha256": "6e1392c26880c034d09c72437fefd4c1865d6540f40e245825a6ad6a4e7f4204"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/policy.py",
      "sha256": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/portable_policy.py",
      "sha256": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/reference_spectral.py",
      "sha256": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/train.py",
      "sha256": "b5b11637cbe2a3a4742504781689f35581731290b78101a86d7378ec2a1a1f4d"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    },
    {
      "path": "runs/pressure-methods-v5/v5-raw-project-02/training_multilevel.py",
      "sha256": "eb8dfc7f963455112eb430701a941c3a9921b9ca4331e02ee1786fa2e3ad5dba"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v5/v5-raw-project-02/source-witness.json",
    "sha256": "a8242fa05cb0f29c9fcf7da1c103cddd1ce1f2cdbd0f3b086ae8d9e3d1fa58d5"
  },
  "training_population": {
    "path": "runs/pressure-methods-v5/v5-raw-study-01/train/splits.json",
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
    "item_id": "fit-c0095fdbf22ba00a",
    "kind": "training_attempt",
    "name": "Pressure methods v5 · v5-raw CUDA",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-multilevel-local-smoother-v1",
      "input_channels": 1,
      "activation": "linear",
      "restrictions": "Uniform-density cubic sealed Neumann operator, up to 128 cells per axis for training"
    },
    "receipt_sha256": "c0095fdbf22ba00a9a647532c772ecdd421b28eda6044af500252c2c0f539084",
    "receipt_paths": [
      "runs/pressure-methods-v5/v5-raw-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "995a33dfb5ba499e9259d8f9b0d17a87",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods v5 · v5-raw CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2400,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "c6b834df8ede6df9d4c6e6c0a02af9b579eee2dc27d44d322f03fafd5f50d470",
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
      "source_witness_sha256": "a8242fa05cb0f29c9fcf7da1c103cddd1ce1f2cdbd0f3b086ae8d9e3d1fa58d5"
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
        "sha256": "45a394686a497c39d7f6e76e1d2e4e6efe54128d92353c0c139062ed05505d0a",
        "bytes": 3400,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "45a394686a497c39d7f6e76e1d2e4e6efe54128d92353c0c139062ed05505d0a",
        "bytes": 3400,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 3123339264,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/110"
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
      "receipt": "runs/pressure-methods-v5/v5-raw-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v5/v5-raw-study-01/train/receipt.json",
    "sha256": "c0095fdbf22ba00a9a647532c772ecdd421b28eda6044af500252c2c0f539084",
    "bytes": 9846
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C112": {
        "name": "v5-raw",
        "weights_sha256": "45a394686a497c39d7f6e76e1d2e4e6efe54128d92353c0c139062ed05505d0a",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.0922863748186814,
          "pressure_mean_free_relative_l2": 0.22061744925986818,
          "pressure_absolute_rms": 14.519719966807433,
          "pressure_absolute_max": 39.00748500190723
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
              "common_equation_relative_l2": 0.07038072713289877,
              "pressure_mean_free_relative_l2": 0.14797468765036287,
              "pressure_absolute_rms": 14.519719966807433,
              "pressure_absolute_max": 39.00748500190723
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.0922863748186814,
              "pressure_mean_free_relative_l2": 0.22061744925986818,
              "pressure_absolute_rms": 4.240475239873603,
              "pressure_absolute_max": 12.475278498220211
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-02/C112/results.json",
          "sha256": "48471954d4c9d1cff76e1dffa38ea50003e9da747ea7964db0ae282b788bbc50"
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
{"architecture":{"activation":"linear","bias":false,"boundary":"Replicated one-cell halo at each convolution","channels":["dimensionless_laplacian_rhs"],"coarse_training_cycles":0,"deployment":"Local correction called after an independently validated numerical multilevel cycle","fixed_skip":"q / 8","input_channels":1,"kernels":[3,3,3],"parameters":648,"preprocessing":"q=(rho/h)*residual, divided by fixed original train-only q RMS","reconstruction":"Raw correction times the same q RMS; cell-zero gauge; add supplied base","restrictions":"Uniform-density cubic sealed Neumann operator, up to 128 cells per axis for training","schema":"rheon-multilevel-local-smoother-v1","widths":[1,4,4,1]},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[12,6,11],"frame":1,"native_frame":63,"parent":3,"split":"train","trajectory_id":"baseline--water-train-0003"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":0,"native_frame":15,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":0,"native_frame":15,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":15,"parent":18,"split":"train","trajectory_id":"large-train-64-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[31,23,17],"frame":1,"native_frame":63,"parent":21,"split":"train","trajectory_id":"v2-train-00-cpm16"},{"cell_size_m":0.125,"counts_xyz":[48,13,9],"frame":0,"native_frame":15,"parent":24,"split":"train","trajectory_id":"v2-train-02-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[64,64,64],"frame":1,"native_frame":15,"parent":27,"split":"train","trajectory_id":"v3-train-00-n64-cpm16"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":0,"parent":30,"split":"train","trajectory_id":"v3-train-01-n64-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[128,128,128],"frame":1,"native_frame":15,"parent":33,"split":"train","trajectory_id":"v3-train-01-n128-cpm32"}],"closure":{"experiment_sha256":"c6b834df8ede6df9d4c6e6c0a02af9b579eee2dc27d44d322f03fafd5f50d470","inputs":{"dataset":{"bytes":22711895,"sha256":"20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5"},"dataset_extra":{"bytes":25135473,"sha256":"2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368"},"initial_weights":{"bytes":3400,"sha256":"ce457a138e56100e481ce0216d608792b6c466cf5eb2a34b2ac454ff0cf3b419"},"splits":{"bytes":57361,"sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"view_02":{"bytes":22277708,"sha256":"85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb"},"view_03":{"bytes":23650806,"sha256":"98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8"},"view_04":{"bytes":22239100,"sha256":"1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc"},"view_05":{"bytes":24113006,"sha256":"6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda"},"view_06":{"bytes":19897861,"sha256":"8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080"},"view_07":{"bytes":22257574,"sha256":"4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9"},"view_08":{"bytes":18319876,"sha256":"2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"157178a36ed5bebdd0971c3643a6720aa377962175614cdde7115ddef5cfbf56","source_witness_sha256":"a8242fa05cb0f29c9fcf7da1c103cddd1ce1f2cdbd0f3b086ae8d9e3d1fa58d5"},"diagnostic_population":"train_only","elapsed_s":43.211547924001934,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.477519901982305e-07,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":1052439552,"peak_reserved_bytes":1103101952,"tf32":false},"fine_tuning_statistics":{"loss_calls":2400,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":2400,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":3400,"role":"checkpoint","sha256":"ce457a138e56100e481ce0216d608792b6c466cf5eb2a34b2ac454ff0cf3b419"},"checkpoint-2400.npz":{"bytes":3400,"role":"checkpoint","sha256":"45a394686a497c39d7f6e76e1d2e4e6efe54128d92353c0c139062ed05505d0a"},"history.json":{"bytes":9373,"role":"training_diagnostics","sha256":"bfabbc61e77fd82c098476c4fc0c306adde7bc060c8bd7865962093c45de5292"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":1238286,"role":"predictions","sha256":"179abacd687573c9ba06e43a5b449dd6114295798b82fcf372f4ab02a8babbb8"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":57361,"role":"training_splits","sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"summary.json":{"bytes":678,"role":"result_summary","sha256":"fb54dbb3a46748d094aa6503decfefa5e2fa82003001c2fda7650af620f05577"},"weights.npz":{"bytes":3400,"role":"checkpoint","sha256":"45a394686a497c39d7f6e76e1d2e4e6efe54128d92353c0c139062ed05505d0a"}},"per_case_inference_s":[0.0011732909770216793,0.00034603799576871097,0.0002480760158505291,0.00032102499972097576,0.00035114301135763526,0.0003874940157402307,0.16912731699994765,0.00289160298416391,0.0009880670113489032,0.16184705001069233,0.1566471460100729,0.46538452900131233],"prediction_export_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"74720a8a0749c3eafb06e70f517946a65fdf55d80f22d0df3a767b7066ec1406","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"d5ef68eb781c6f75afd638bd571e4f86eb7bacf54089f6e83e0f5a75647d1a47","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"025679013c224ccafbab13db8f59a00b533bd4da15b663caa266690b17a29758","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"3b3c942bd8ecf28b793411273ead3c1570d502bbd03efeb2c6741b95a8db1c01","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"268abb518a06c0f5ecf10602705a8f1592a83ec39faa41dc0ef3f0ceb0f21ddb","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"325a9f95446c1953c4ed4cd06f271ea6487222508e71dd1fd1bc8354f7ac3c2c","shape":[7,15,14]},"case_006":{"bytes":262144,"dtype":"<f8","sha256":"e73b84a9eb10d0db632db0500c85d8fa4fcb808cd17618a620b0f0fbd3c877de","shape":[32,32,32]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"45bedee5ddad3f64a71cf8a9ced8c357010fe49e2c548e7b06513735fba2c0e8","shape":[17,23,31]},"case_008":{"bytes":22464,"dtype":"<f8","sha256":"f7bc1330b8388d9435514e591b659ee5f109316f20c814f8125a9755d366183a","shape":[9,13,24]},"case_009":{"bytes":262144,"dtype":"<f8","sha256":"bcce25c660fc52827421cba915af2ca4dbaff5d4e46b10040092633293ec1f47","shape":[32,32,32]},"case_010":{"bytes":262144,"dtype":"<f8","sha256":"efc171b689ea383cb56540ca6ee05fb02cf36a28ab3806434302d1aba801bd74","shape":[32,32,32]},"case_011":{"bytes":262144,"dtype":"<f8","sha256":"e4a6ed23976106d55ff60cc1fe7bcf566889ed9189b342b8be8a2e1baff324d3","shape":[32,32,32]}},"prediction_export_policy":"NPZ samples at most32cells/axis; full prediction_inventory hashes every full-resolution value and is checked exactly on fresh CPU restore","prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"74720a8a0749c3eafb06e70f517946a65fdf55d80f22d0df3a767b7066ec1406","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"d5ef68eb781c6f75afd638bd571e4f86eb7bacf54089f6e83e0f5a75647d1a47","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"025679013c224ccafbab13db8f59a00b533bd4da15b663caa266690b17a29758","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"3b3c942bd8ecf28b793411273ead3c1570d502bbd03efeb2c6741b95a8db1c01","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"268abb518a06c0f5ecf10602705a8f1592a83ec39faa41dc0ef3f0ceb0f21ddb","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"325a9f95446c1953c4ed4cd06f271ea6487222508e71dd1fd1bc8354f7ac3c2c","shape":[7,15,14]},"case_006":{"bytes":2097152,"dtype":"<f8","sha256":"9b4947b30c711078721334ad159f2d4f3441b5984d6ac9256e1bd295cbb8ec1d","shape":[64,64,64]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"45bedee5ddad3f64a71cf8a9ced8c357010fe49e2c548e7b06513735fba2c0e8","shape":[17,23,31]},"case_008":{"bytes":44928,"dtype":"<f8","sha256":"9c00a5fd527e23c17b077e9143f2a5af58cda044c86a59b7b5e500789f45bf2e","shape":[9,13,48]},"case_009":{"bytes":2097152,"dtype":"<f8","sha256":"536e75cefafa321f099cd9d165ae8e157a2939f568c65d708e29baec5665dc1c","shape":[64,64,64]},"case_010":{"bytes":2097152,"dtype":"<f8","sha256":"9ae276b8811e7534c0623372df098205ff5b5776790eb6b673157392bb7502c4","shape":[64,64,64]},"case_011":{"bytes":16777216,"dtype":"<f8","sha256":"3f83067a4aa27ebd952a7c61dd587e1c1849457f3266f4692d3d0bb8cf6bc62f","shape":[128,128,128]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":3123339264,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
