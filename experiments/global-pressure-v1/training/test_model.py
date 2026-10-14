#!/usr/bin/env python3
"""Focused model/numerics contracts. No scientific 200-update run is executed."""
import copy
import base64
import os
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model
import train


def dense_matrix(counts, spacing, rho):
    """Independent hand assembly: explicit endpoint incidence, not padded Torch."""
    nx, ny, nz = counts
    matrix = np.zeros((nx * ny * nz,) * 2, dtype=np.float64)
    def index(x, y, z): return x + nx * (y + ny * z)
    weights = [spacing[1] * spacing[2] / (rho * spacing[0]), spacing[0] * spacing[2] / (rho * spacing[1]), spacing[0] * spacing[1] / (rho * spacing[2])]
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                low = index(x, y, z)
                for axis, coordinate in enumerate((x, y, z)):
                    if coordinate + 1 < counts[axis]:
                        high_coords = [x, y, z]; high_coords[axis] += 1
                        high = index(*high_coords); weight = weights[axis]
                        matrix[low, low] += weight; matrix[high, high] += weight
                        matrix[low, high] -= weight; matrix[high, low] -= weight
    return matrix


def normalization_fixture():
    scales = {}
    for name in ("residual", "pressure", "rhs", "hx", "hy", "hz", "rho", "dt"):
        scales[name] = {"value": 2.0 if name == "pressure" else 3.0 if name == "rhs" else 1.0, "units": "fixture", "method": "fixture", "source": name}
    return {"schema": "rheon-global-pressure-normalization-v1", "channels": model.CHANNELS, "scales": scales, "training_provenance": {}}


def arithmetic_fixture():
    # Disposable two-parent/two-frame data, never passed to the production reader.
    count = 4
    values = np.arange(16**3, dtype=np.float64).reshape(16, 16, 16) / (16**3)
    base = np.stack([values * (0.1 + 0.01 * row) for row in range(count)])
    reference = base + np.stack([values[::-1].copy() * 0.01 for _ in range(count)])
    reference -= reference[:, :1, :1, :1]
    arrays = {"base": base, "reference": reference, "rhs": np.stack([(values - 0.5) * 0.01 for _ in range(count)]), "native_input_residual": np.stack([(values - 0.5) * 0.02 for _ in range(count)]), "spacing": np.asarray([[.0625, .0625, .0625], [.075, .0625, .05], [.0625, .0625, .0625], [.075, .0625, .05]], dtype="<f8"), "rho": np.full(count, 1.25, dtype="<f8"), "dt": np.full(count, 1 / 60, dtype="<f8")}
    rows = [{"trajectory_id": "arithmetic-a" if row < 2 else "arithmetic-b", "frame": row % 2} for row in range(count)]
    return arrays, rows


class FiniteVolumeAndGauge(unittest.TestCase):
    @classmethod
    def setUpClass(cls): model.configure_cpu()

    def test_literal_all_axis_dyadic_full_gauge_row(self):
        pressure = torch.arange(8, dtype=torch.float64).reshape(1, 1, 2, 2, 2)
        spacing = torch.tensor([[.5, .25, 2.]], dtype=torch.float64)
        rho = torch.tensor([2.], dtype=torch.float64)
        literal = [-4.625, -3.625, 3.375, 4.375, -4.375, -3.375, 3.625, 4.625]
        self.assertEqual(model.full_operator(pressure, spacing, rho).flatten().tolist(), literal)
        self.assertEqual(model.full_operator(torch.ones_like(pressure), spacing, rho).count_nonzero(), 0)

    def test_actual_native_qualified_anisotropic_oracle_literal(self):
        p = torch.tensor([0., 1., -2., .5, 3., -1.], dtype=torch.float64).reshape(1, 1, 1, 2, 3)
        got = model.full_operator(p, torch.tensor([[.25, .4, .7]], dtype=torch.float64), torch.tensor([1.25], dtype=torch.float64)).flatten().numpy()
        np.testing.assert_allclose(got, [-1.071, 2.8839999999999995, -3.038, -2.065, 6.524, -3.2339999999999995], rtol=0, atol=2e-15)
        np.testing.assert_allclose(got, dense_matrix([3, 2, 1], [.25, .4, .7], 1.25) @ p.flatten().numpy(), rtol=0, atol=2e-15)

    def test_operator_gradient_matches_independent_dense_transpose(self):
        counts = [3, 2, 2]; spacing = [.25, .5, 1.]
        pressure = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(1, 1, 2, 2, 3).requires_grad_()
        vector = torch.linspace(.3, 1.5, 12, dtype=torch.float64).reshape_as(pressure)
        action = model.full_operator(pressure, torch.tensor([spacing], dtype=torch.float64), torch.tensor([2.], dtype=torch.float64))
        gradient, = torch.autograd.grad((action * vector).sum(), pressure)
        expected = dense_matrix(counts, spacing, 2.).T @ vector.flatten().numpy()
        np.testing.assert_allclose(gradient.flatten().numpy(), expected, rtol=0, atol=5e-16)

    def test_loss_gauge_gradient_dense_and_finite_difference(self):
        shape = (1, 1, 2, 2, 2); h = [.5, .25, 2.]; rho = 2.; scale = 2.
        matrix = dense_matrix([2, 2, 2], h, rho)
        base = np.asarray([0., .1, -.2, .3, -.1, .2, .4, -.3])
        reference = np.asarray([0., .2, -.1, .4, -.3, .1, .5, -.2])
        rhs = np.asarray([-.1, .2, -.3, .4, -.2, .1, -.4, .3])
        delta = np.asarray([.03, -.02, .01, .02, -.01, .04, -.03, .02])
        raw = torch.tensor(delta.reshape(shape), dtype=torch.float64, requires_grad=True)
        predicted = torch.tensor(base.reshape(shape)) + model.gauge_correction(raw, scale)
        loss = model.pressure_loss(predicted, torch.tensor(reference.reshape(shape)), torch.tensor(rhs.reshape(shape)), torch.tensor([h], dtype=torch.float64), torch.tensor([rho], dtype=torch.float64), normalization_fixture()).mean()
        gradient, = torch.autograd.grad(loss, raw)
        p = base + scale * (delta - delta[0]); denp = max(float(reference @ reference), (1e-6 * math.sqrt(8) * scale)**2); denb = max(float(rhs @ rhs), (1e-6 * math.sqrt(8) * 3.)**2)
        g = (p - reference) / denp + matrix.T @ (matrix @ p - rhs) / denb
        expected = scale * g; expected[0] = -scale * math.fsum(g[1:])
        np.testing.assert_allclose(gradient.flatten().numpy(), expected, rtol=1e-14, atol=1e-14)
        self.assertAlmostEqual(float(gradient.sum()), 0., places=13)
        def independent_loss(values):
            candidate = base + scale * (values - values[0]); residual = matrix @ candidate - rhs
            return .5 * float((candidate-reference) @ (candidate-reference)) / denp + .5 * float(residual @ residual) / denb
        for index in range(8):
            plus = delta.copy(); minus = delta.copy(); plus[index] += 1e-6; minus[index] -= 1e-6
            estimate = (independent_loss(plus) - independent_loss(minus)) / 2e-6
            self.assertAlmostEqual(estimate, expected[index], delta=2e-9)

    def test_eps_b_uses_original_rhs_scale_not_input_residual_scale(self):
        norm = normalization_fixture(); other = copy.deepcopy(norm); other["scales"]["residual"]["value"] = 1000.
        p = torch.arange(8, dtype=torch.float64).reshape(1, 1, 2, 2, 2)
        args = (p, torch.zeros_like(p), torch.zeros_like(p), torch.tensor([[.5, .25, 2.]], dtype=torch.float64), torch.tensor([2.], dtype=torch.float64))
        self.assertEqual(model.pressure_loss(*args, norm).item(), model.pressure_loss(*args, other).item())


class ArchitectureAndInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls): model.configure_cpu()

    def test_exact_parameter_count_inventory_and_16_32_shapes(self):
        torch.manual_seed(20261014); network = model.PressureCorrectionNet()
        self.assertEqual(model.parameter_count(network), 372377)
        self.assertEqual(sum(spec["bytes"] for spec in train.state_inventory(network).values()), 1489508)
        self.assertFalse(any(isinstance(layer, (torch.nn.BatchNorm3d, torch.nn.Dropout)) for layer in network.modules()))
        self.assertTrue(any(parameter.count_nonzero() for parameter in (network.head.weight,)))
        with torch.no_grad():
            for count in (16, 32): self.assertEqual(tuple(network(torch.zeros(1, 10, count, count, count)).shape), (1, 1, count, count, count))

    def test_input_channels_coordinates_and_labels_cannot_enter(self):
        arrays, rows = arithmetic_fixture(); norm = model.fit_normalization(arrays, rows)
        batch = train.tensors(arrays, [0, 1], labels=False)
        self.assertNotIn("reference", batch); self.assertNotIn("rhs", batch)
        inputs = model.build_inputs(batch["native_input_residual"], batch["base"], batch["spacing"], batch["rho"], batch["dt"], norm)
        self.assertEqual(inputs.shape, (2, 10, 16, 16, 16))
        self.assertEqual(inputs[0, 2, 0, 0, 0].item(), .5 / 16)
        self.assertEqual(inputs[0, 2, 0, 0, -1].item(), 15.5 / 16)
        self.assertEqual(inputs[0, 3, 0, -1, 0].item(), 15.5 / 16)
        self.assertEqual(inputs[0, 4, -1, 0, 0].item(), 15.5 / 16)
        self.assertEqual(norm["scales"]["rho"]["value"], 1.)
        self.assertEqual(norm["scales"]["dt"]["method"], "constant_unit_scale")

    def test_zero_residual_bypasses_inference_per_row(self):
        arrays, rows = arithmetic_fixture(); norm = model.fit_normalization(arrays, rows)
        batch = train.tensors(arrays, [0, 1], labels=False)
        class Counted(torch.nn.Module):
            def __init__(self): super().__init__(); self.rows = []
            def forward(self, inputs): self.rows.append(inputs.shape[0]); return torch.ones_like(inputs[:, :1])
        network = Counted(); batch["native_input_residual"].zero_()
        estimated = train.prediction(network, batch, norm)
        self.assertEqual(network.rows, []); self.assertTrue(torch.equal(estimated, batch["base"]))
        batch["native_input_residual"][1, 0, 0, 0, 1] = 1.
        estimated = train.prediction(network, batch, norm)
        self.assertEqual(network.rows, [1]); self.assertTrue(torch.equal(estimated[0], batch["base"][0])); self.assertTrue((estimated[:, :, 0, 0, 0] == 0).all())

    def test_local_input_has_distant_dependence_and_global_pool_gradient(self):
        torch.manual_seed(20261014); network = model.PressureCorrectionNet()
        x = torch.randn(1, 10, 32, 32, 32, requires_grad=True)
        captured = {}
        hook = network.global_pool.register_forward_hook(lambda module, inputs, output: captured.update(pool=output))
        try:
            output = network(x)
            grad_x, grad_pool = torch.autograd.grad(output[0, 0, -1, -1, -1], (x, captured["pool"]), retain_graph=True)
            gauge_pool, = torch.autograd.grad(output[0, 0, -1, -1, -1] - output[0, 0, 0, 0, 0], (captured["pool"],))
        finally: hook.remove()
        self.assertNotEqual(grad_x[0, 0, 0, 0, 0].item(), 0.)
        self.assertGreater(grad_pool.abs().sum().item(), 0.)
        self.assertGreater(gauge_pool.abs().sum().item(), 0.)

    def test_model_head_weight_gradient_finite_difference(self):
        arrays, rows = arithmetic_fixture(); norm = model.fit_normalization(arrays, rows)
        torch.manual_seed(20261014); network = model.PressureCorrectionNet(); batch = train.tensors(arrays, [0])
        loss = train.losses(network, batch, norm).mean(); loss.backward()
        expected = network.head.weight.grad[0, 0, 0, 0, 0].item()
        original = network.head.weight[0, 0, 0, 0, 0].item(); epsilon = 1e-3
        with torch.no_grad():
            network.head.weight[0, 0, 0, 0, 0] = original + epsilon; plus = train.losses(network, batch, norm).mean().item()
            network.head.weight[0, 0, 0, 0, 0] = original - epsilon; minus = train.losses(network, batch, norm).mean().item()
            network.head.weight[0, 0, 0, 0, 0] = original
        self.assertAlmostEqual((plus-minus)/(2*epsilon), expected, delta=max(1e-5, abs(expected)*.02))


class NormalizationAndFixedUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls): model.configure_cpu()

    def test_validation_perturbation_cannot_change_norm_sampling_or_one_update(self):
        arrays, rows = arithmetic_fixture(); validation = {name: values.copy() for name, values in arrays.items()}
        altered = {name: values.copy() for name, values in validation.items()}
        for name in ("reference", "rhs", "native_input_residual"): altered[name] *= 7
        altered["base"] *= 3
        altered["spacing"] *= 1.5
        norm_a = model.fit_normalization(arrays, rows); norm_b = model.fit_normalization(arrays, rows)
        self.assertEqual(train.canonical(norm_a), train.canonical(norm_b))
        order_a = train.sample_order(rows); order_b = train.sample_order(rows)
        self.assertEqual(order_a.shape, (200, 2)); self.assertEqual(order_a.tobytes(), order_b.tobytes())
        states = []
        observed = []
        for observed_validation in (validation, altered):
            torch.manual_seed(20261014); network = model.PressureCorrectionNet()
            observed.append(train.evaluate(network, observed_validation, rows, norm_a))
            optimizer = torch.optim.Adam(network.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
            train.one_update(network, optimizer, arrays, order_a[0], norm_a)
            states.append({name: tensor.detach().numpy().tobytes() for name, tensor in network.state_dict().items()})
        self.assertEqual(states[0], states[1])
        self.assertNotEqual(observed[0], observed[1])

    def test_zero_training_scale_is_rejected(self):
        arrays, rows = arithmetic_fixture(); arrays["base"].fill(0.)
        with self.assertRaises(model.ModelContractError): model.fit_normalization(arrays, rows)

    def test_scales_match_training_only_numeric_authority(self):
        arrays, rows = arithmetic_fixture(); normalization = model.fit_normalization(arrays, rows)
        self.assertEqual(normalization["training_provenance"]["frame_counts"], {"arithmetic-a": 2, "arithmetic-b": 2})
        self.assertEqual(normalization["scales"]["pressure"]["value"], float(np.sqrt(np.mean(arrays["base"] ** 2))))
        self.assertNotEqual(normalization["scales"]["pressure"]["value"], float(np.sqrt(np.mean(arrays["reference"] ** 2))))
        self.assertNotEqual(normalization["scales"]["rhs"]["value"], normalization["scales"]["residual"]["value"])


class RestoreCustodyContract(unittest.TestCase):
    def setUp(self):
        # Synthetic retained metadata exercises the actual accepted cold validator;
        # it is not evidence that a historical Run occurred or was drained.
        source = Path(os.environ.get("TULDOK_CUSTODY_SOURCE", "/tmp/global-pressure-tuldok-reference/recipe_process.py"))
        if not source.exists(): self.skipTest("requires exact accepted owner source")
        self.evidence_path = source.with_name("input-evidence.json")
        helper = {"pid": 1, "start_ticks": "1", "boot_id": "00000000-0000-0000-0000-000000000000"}
        identity = {"attempt_id": "a" * 32, "request_sha256": "b" * 64, "spec_sha256": "c" * 64, "helper": helper}
        receipt = {**identity, "schema_version": 1, "go_received": True, "spawn_closed": True, "drained": True, "echild": True, "outcome": "completed", "reason": "completed", "leader_returncode": 0, "streams": {name: {"bytes": 0, "sha256": train.sha256(b"")} for name in ("stdout", "stderr")}}
        raw_receipt = train.canonical(receipt)
        self.raw = {slot: ("synthetic-" + slot).encode() for slot in train.TRAINED_INPUTS}
        self.evidence = {}
        for slot, (artifact, role) in train.TRAINED_INPUTS.items():
            source = {"schema_version": 1, "catalog_id": "d" * 32, "run_id": "a" * 32, "attempt_id": "a" * 32, "request_sha256": "b" * 64, "material_sha256": "e" * 64, "artifact_name": artifact, "artifact_role": role, "artifact_bytes": len(self.raw[slot]), "measured_sha256": train.sha256(self.raw[slot]), "custody_receipt_sha256": train.sha256(raw_receipt), "custody_identity": identity, "custody_receipt_base64": base64.b64encode(raw_receipt).decode()}
            self.evidence[slot] = {"kind": "run_artifact", "source": source}
        self.assertEqual(train.validate_trained_custody(self.evidence, self.raw, self.evidence_path)["run_id"], "a" * 32)

    def test_project_file_is_not_relabelled_as_trained_run_artifact(self):
        self.evidence["trained_weights"]["kind"] = "project_file"
        with self.assertRaises(train.TrainingContractError) as error: train.validate_trained_custody(self.evidence, self.raw, self.evidence_path)
        self.assertEqual(error.exception.code, "trained_origin")

    def test_typed_owner_rejects_boolean_schema_and_payload_drift(self):
        for change in ({"schema_version": True}, {"measured_sha256": "f" * 64}):
            evidence = copy.deepcopy(self.evidence); evidence["trained_weights"]["source"].update(change)
            with self.assertRaises(train.TrainingContractError) as error: train.validate_trained_custody(evidence, self.raw, self.evidence_path)
            self.assertEqual(error.exception.code, "trained_custody")

    def test_exact_artifact_role_and_common_catalog_origin_required(self):
        evidence = copy.deepcopy(self.evidence); evidence["trained_weights"]["source"]["artifact_role"] = "preprocessing"
        with self.assertRaises(train.TrainingContractError) as error: train.validate_trained_custody(evidence, self.raw, self.evidence_path)
        self.assertEqual(error.exception.code, "trained_origin")
        evidence = copy.deepcopy(self.evidence); evidence["trained_splits"]["source"]["catalog_id"] = "f" * 32
        with self.assertRaises(train.TrainingContractError) as error: train.validate_trained_custody(evidence, self.raw, self.evidence_path)
        self.assertEqual(error.exception.code, "trained_origin")

    def test_noncompleted_custody_is_rejected(self):
        evidence = copy.deepcopy(self.evidence); source = evidence["trained_weights"]["source"]
        receipt = train.strict_json(base64.b64decode(source["custody_receipt_base64"]))
        receipt.update(outcome="unknown", drained=False, echild=False)
        raw = train.canonical(receipt); source.update(custody_receipt_base64=base64.b64encode(raw).decode(), custody_receipt_sha256=train.sha256(raw))
        with self.assertRaises(train.TrainingContractError) as error: train.validate_trained_custody(evidence, self.raw, self.evidence_path)
        self.assertEqual(error.exception.code, "trained_custody")


class NumericCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls): model.configure_cpu()

    def test_numeric_restore_exact_bytes_and_bitwise_predictions(self):
        torch.manual_seed(20261014); original = model.PressureCorrectionNet()
        inputs = torch.randn(1, 10, 16, 16, 16)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.npz"; copied = Path(temp) / "copy.npz"
            inventory = train.save_weights(original, path)
            self.assertLess(path.stat().st_size, 8 * 1024 * 1024)
            restored = model.PressureCorrectionNet(); train.load_weights(restored, path, inventory)
            with torch.no_grad(): self.assertEqual(original(inputs).numpy().tobytes(), restored(inputs).numpy().tobytes())
            train.copy_new(copied, path.read_bytes()); self.assertEqual(path.read_bytes(), copied.read_bytes())
            with self.assertRaises(FileExistsError): train.save_weights(original, path)

    def test_shape_inventory_and_rehashed_nonfinite_fail_at_intended_boundary(self):
        torch.manual_seed(20261014); network = model.PressureCorrectionNet()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.npz"; inventory = train.save_weights(network, path)
            wrong = copy.deepcopy(inventory); first = next(iter(wrong)); wrong[first]["shape"][0] += 1
            with self.assertRaises(train.TrainingContractError) as error: train.load_weights(network, path, wrong)
            self.assertEqual(error.exception.code, "weights_inventory")
            arrays = {"a": np.asarray([float("nan")], dtype="<f4")}
            corrupt = Path(temp) / "nan.npz"
            with corrupt.open("wb") as stream: np.savez(stream, **arrays)
            spec = {"a": {"dtype": "<f4", "shape": [1], "bytes": 4, "sha256": train.sha256(arrays["a"].tobytes())}}
            with self.assertRaises(train.TrainingContractError) as error: train.read_npz(corrupt, spec, 1024)
            self.assertEqual(error.exception.code, "npz_nonfinite")

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw, code in [(b'{"updates":200,"updates":0}', "duplicate_json_key"), (b'{"updates":NaN}', "json_nonfinite"), (b'{"loss":1e999}', "json_nonfinite")]:
            with self.assertRaises(train.TrainingContractError) as error: train.strict_json(raw)
            self.assertEqual(error.exception.code, code)


if __name__ == "__main__": unittest.main()
