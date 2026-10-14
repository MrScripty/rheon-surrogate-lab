"""Pretraining contracts and arithmetic only: no optimizer construction/steps."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import model
import qualify
import train

_saved_path = list(sys.path)
try:
    original_tests = model.load_original("test_model", "bf9351c5bfb86f4803c160d262740abaa09e252bce10eb5d70f6be2a07c41b7e")
finally:
    # The frozen fixture adds its original directory to sys.path. Keep that
    # behavior local to fixture loading, never alter sibling test resolution.
    sys.path[:] = _saved_path


class Qualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("qualification cannot construct optimizer")):
            cls.metrics, cls.proof = qualify.measure()

    def test_exact_initialization_and_normalization_inventory(self):
        self.assertEqual(len(self.proof["mapping"]), 38)
        self.assertEqual(self.proof["parameter_count"], 372377)
        self.assertEqual(self.proof["raw_weight_bytes"], 1489508)
        torch.manual_seed(20261014)
        network = model.PressureCorrectionNet()
        modules = [m for m in network.modules() if isinstance(m, nn.GroupNorm)]
        self.assertEqual(len(modules), 18)
        self.assertTrue(all(m.num_groups == 4 and m.eps == 1e-5 and not m.affine for m in modules))
        self.assertEqual(network.head.__class__, nn.Conv3d)
        self.assertFalse(any(isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.Dropout)) for m in network.modules()))
        for block in [*network.encoders, network.bottleneck, *network.decoders]:
            self.assertEqual([type(m) for m in block.layers], [nn.Conv3d, nn.GroupNorm, nn.SiLU] * 2)

    def test_closed_actual_qualification_metrics_pass(self):
        source = {"executable_sources": train.executable_sources(), "native_qualification": {
            "base_commit": train.NATIVE_BASE, "patch_sha256": "a" * 64,
            "prepared_tree_sha256": "b" * 64, "operator_binary_sha256": "c" * 64}}
        report = {"format": "rheon-global-pressure-groupnorm-qualification-v1", "status": "passed",
                  "executable_sources": source["executable_sources"],
                  "runtime": {"torch": "2.8.0+cpu", "numpy": "2.3.5", "threads": 1, "interop_threads": 1},
                  "parameter_count": 372377, "raw_weight_bytes": 1489508,
                  "tests": {"suite_sha256": "d" * 64, "passed_count": 1},
                  "evidence": {"gates": train.QUALIFICATION_GATES, "metrics": self.metrics,
                               "initialization_mapping_sha256": train.sha256(train.canonical(self.proof)),
                               "no_optimizer_constructed": True, "native_operator": {**source["native_qualification"], "oracle_action_sha256": "e" * 64}, "output_budget_sha256": "f" * 64}}
        train.validate_qualification(report, source)
        for key, replacement in (("metrics", {"placeholder": 0}), ("metrics", {**self.metrics, "global_pool_gradient_l1": 0}),
                                 ("metrics", {**self.metrics, "observed_peak_rss_bytes": True}),
                                 ("metrics", {**self.metrics, "model_parameter_fd_abs_error": 1}),
                                 ("no_optimizer_constructed", 1)):
            bad = copy.deepcopy(report)
            bad["evidence"][key] = replacement
            with self.assertRaises(train.TrainingContractError):
                train.validate_qualification(bad, source)

    def test_actual_gauge_gn_parameter_and_global_derivatives(self):
        self.assertLess(self.metrics["gauge_finite_difference_max_abs_error"], 1e-6)
        self.assertLess(self.metrics["groupnorm_finite_difference_max_abs_error"], 1e-6)
        self.assertGreater(self.metrics["model_parameter_gradient_abs"], 1e-6)
        self.assertLess(self.metrics["model_parameter_fd_abs_error"], max(1e-4, .05 * self.metrics["model_parameter_gradient_abs"]))
        self.assertGreater(self.metrics["distant_input_derivative_abs"], 0)
        self.assertGreater(self.metrics["global_pool_gradient_l1"], 0)
        self.assertGreater(self.metrics["amplitude_derivative_abs"], 0)

    def test_batch_train_eval_shapes_edges_and_zero_bypass(self):
        self.assertEqual(self.metrics["batch_neighbor_max_abs_error"], 0)
        self.assertEqual(self.metrics["train_eval_max_abs_error"], 0)
        self.assertEqual(self.metrics["zero_residual_max_abs_error"], 0)
        self.assertLessEqual(self.metrics["group_variance_max"], 1.0001)
        self.assertLessEqual(self.metrics["group_mean_max_abs"], 2e-6)
        self.assertEqual(self.metrics["native_operator_max_abs_error"], 0)


class Boundary(unittest.TestCase):
    def setUp(self):
        model.configure_cpu()
        self.guard = patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("no optimizer in pretraining tests"))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_dangling_captured_model_never_falls_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "original_model.py").symlink_to(directory / "absent")
            with patch.object(model, "__file__", str(directory / "model.py")):
                with self.assertRaisesRegex(RuntimeError, "invalid pinned original"):
                    model.load_original("model", model.ORIGINAL_MODEL_SHA256)

    def test_dangling_captured_source_witness_never_falls_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "original_data.py").symlink_to(directory / "absent")
            with patch.object(train, "__file__", str(directory / "train.py")):
                with self.assertRaises(train.TrainingContractError) as caught:
                    train.source_bytes("original_data.py")
                self.assertEqual(caught.exception.code, "unsafe_path")

    def test_bad_captured_bytes_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "original_model.py").write_text("raise AssertionError('must not execute')")
            with patch.object(model, "__file__", str(directory / "model.py")):
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    model.load_original("model", model.ORIGINAL_MODEL_SHA256)

    def test_numeric_new_checkpoint_roundtrip_and_original_inventory_rejected(self):
        torch.manual_seed(20261014)
        network = model.PressureCorrectionNet().eval()
        arrays, _, normalization = qualify.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = train.publish_checkpoint(network, directory, 0)
            copy_path = directory / "weights.npz"
            raw = (directory / "checkpoint-000.npz").read_bytes()
            train.copy_new(copy_path, raw)
            other = model.PressureCorrectionNet().eval()
            train.load_weights(other, raw, checkpoint["inventory"])
            with torch.no_grad():
                batch = train.tensors(arrays, [0])
                original = train.prediction(network, batch, normalization)
                restored = train.prediction(other, batch, normalization)
            self.assertEqual(original.numpy().tobytes(), restored.numpy().tobytes())
            self.assertEqual(copy_path.read_bytes(), raw)
            legacy_inventory = train.state_inventory(model.original.PressureCorrectionNet())
            with self.assertRaises(train.TrainingContractError) as caught:
                train.load_weights(other, raw, legacy_inventory)
            self.assertEqual(caught.exception.code, "weights_inventory")

    def test_old_training_identity_rejected_before_state_use(self):
        # Complete key envelope is supplied to isolate identity discriminant.
        names = {"schema", "status", "mode", "optimizer_updates", "trained_optimizer_updates", "architecture", "parameter_count",
                 "weights_sha256", "weights_inventory", "predictions_sha256", "predictions_inventory", "witness_cases", "per_case_inference_s",
                 "closure", "sample_order_sha256", "training_history", "validation", "elapsed_s", "runtime_observations", "outputs",
                 "originating_training_run", "checkpoints", "diagnostics_sha256", "initialization_mapping_sha256"}
        old = {name: None for name in names}
        old.update(schema="rheon-global-pressure-training-receipt-v1", status="complete", mode="train")
        with self.assertRaises(train.TrainingContractError) as caught:
            train.validate_restore(old, {}, None, None, None, None, {})
        self.assertEqual(caught.exception.code, "restore_identity")

    def test_restore_predictions_reject_equal_bytes_with_wrong_header_or_keys(self):
        arrays = {"pressure": np.arange(8, dtype="<f8").reshape(1, 2, 2, 2), "row_indices": np.array([0], dtype="<i8")}
        train.validate_prediction_parity(arrays, {name: values.copy() for name, values in arrays.items()})
        for retained in ({**arrays, "pressure": arrays["pressure"].reshape(1, 8)},
                         {**arrays, "pressure": arrays["pressure"].view("<i8")},
                         {**arrays, "extra": np.array([1], dtype="<i4")}):
            self.assertEqual(retained["pressure"].tobytes(), arrays["pressure"].tobytes())
            with self.assertRaises(train.TrainingContractError) as caught:
                train.validate_prediction_parity(arrays, retained)
            self.assertEqual(caught.exception.code, "restore_predictions")

    def test_train_only_rms_and_sampling_unchanged(self):
        arrays, rows, norm = qualify.fixture()
        self.assertEqual(train.canonical(norm), train.canonical(model.original.fit_normalization(arrays, rows)))
        self.assertEqual(train.sample_order(rows).tobytes(), train.base.sample_order(rows).tobytes())
        self.assertEqual(train.sample_order(rows).shape, (200, 2))
        self.assertEqual(set(train.sample_order(rows).flatten()), {0})

    def test_read_only_observation_preserves_numeric_state_and_grad_buffers(self):
        torch.manual_seed(20261014)
        network = model.PressureCorrectionNet()
        arrays, _, norm = qualify.fixture()
        before = train.canonical(train.state_inventory(network))
        observation = train.observe_batch(network, arrays, [0], norm)
        train.validate_observation(observation, network)
        self.assertGreater(observation["global_gradient"]["l2"], 0)
        self.assertGreater(observation["raw_output_gradient"]["l2"], 0)
        self.assertEqual(before, train.canonical(train.state_inventory(network)))
        self.assertTrue(all(parameter.grad is None for parameter in network.parameters()))
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in network.modules()))

    def test_typed_observation_and_history_reject_metadata_aliases(self):
        torch.manual_seed(20261014)
        network = model.PressureCorrectionNet()
        arrays, _, norm = qualify.fixture()
        observation = train.observe_batch(network, arrays, [0], norm)
        bad = copy.deepcopy(observation)
        bad["raw_output_gradient"]["nonzero_count"] = True
        with self.assertRaises(train.TrainingContractError) as caught:
            train.validate_observation(bad, network)
        self.assertEqual(caught.exception.code, "diagnostic_types")
        bad = copy.deepcopy(observation)
        bad["activations"]["head:output"]["zero_fraction"] = False
        with self.assertRaises(train.TrainingContractError):
            train.validate_observation(bad, network)
        history = [{"update": update, "train_loss": 1., "gradient_norm_before_clip": .5} for update in train.CHECKPOINT_UPDATES[1:]]
        validation = [{"update": 0, "loss": 1.}, {"update": 200, "loss": .5}]
        train.validate_history(history, validation)
        validation[0]["update"] = False
        with self.assertRaises(train.TrainingContractError) as caught:
            train.validate_history(history, validation)
        self.assertEqual(caught.exception.code, "restore_validation")

    def test_all_outputs_and_inherited_slots_are_explicit(self):
        self.assertEqual(len(train.OUTPUTS), 18)
        self.assertEqual(len(train.TRAINED_INPUTS), 17)
        self.assertEqual(train.CHECKPOINT_UPDATES, tuple(range(0, 201, 20)))
        self.assertEqual({name for name, _ in train.TRAINED_INPUTS.values()}, set(train.OUTPUTS) - {"summary.json"})
        self.assertEqual(train.strict_json(Path(train.__file__).with_name("config.json").read_bytes()), train.FIXED_CONFIG)


class RestoreCustody(original_tests.RestoreCustodyContract):
    # Reuse exact accepted cold-owner fixture on the adapted17slot inventory.
    # Inherited tests never construct an optimizer or scientific Run.
    def test_checkpoint_and_diagnostic_share_exact_original_run(self):
        self.assertEqual(len(self.raw), 17)
        evidence = copy.deepcopy(self.evidence)
        evidence["trained_checkpoint_200"]["source"]["artifact_name"] = "weights.npz"
        with self.assertRaises(train.TrainingContractError) as caught:
            train.validate_trained_custody(evidence, self.raw, self.evidence_path)
        self.assertEqual(caught.exception.code, "trained_origin")
        evidence = copy.deepcopy(self.evidence)
        evidence["trained_diagnostics"]["source"]["catalog_id"] = "f" * 32
        with self.assertRaises(train.TrainingContractError) as caught:
            train.validate_trained_custody(evidence, self.raw, self.evidence_path)
        self.assertEqual(caught.exception.code, "trained_origin")


if __name__ == "__main__":
    unittest.main()
