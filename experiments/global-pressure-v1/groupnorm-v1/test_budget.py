"""Output-envelope boundary tests and disposable numeric serialization fixtures."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("groupnorm_output_budget", Path(__file__).with_name("budget.py"))
budget = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = budget
SPEC.loader.exec_module(budget)


def declarations():
    result = {
        "weights.npz": ("checkpoint", 8 * budget.MIB),
        "predictions.npz": ("predictions", 8 * budget.MIB),
        "splits.json": ("training_splits", 256 * 1024),
        "normalization.json": ("preprocessing", 256 * 1024),
        "receipt.json": ("training_result", 512 * 1024),
        "summary.json": ("result_summary", 64 * 1024),
        "diagnostics.json": ("training_diagnostics", 2 * budget.MIB),
    }
    result.update({f"checkpoint-{u:03d}.npz": ("checkpoint", 2 * budget.MIB)
                   for u in budget.CHECKPOINT_UPDATES})
    return result


class OutputBudgetTests(unittest.TestCase):
    def test_complete_eighteen_outputs_and_eleven_checkpoint_slots(self):
        outputs = declarations()
        self.assertEqual(len(budget.admit_inventory(outputs)), 18)
        self.assertEqual(budget.CHECKPOINT_UPDATES, tuple(range(0, 201, 20)))
        self.assertGreater(sum(cap for _, cap in outputs.values()), budget.AGGREGATE_LIMIT)
        measured = {name: 100 for name in outputs}
        projected = {name: cap if name in budget.VARIABLE_OUTPUTS else 100
                     for name, (_, cap) in outputs.items()}
        self.assertEqual(budget.check_sizes(outputs, measured, projected), sum(projected.values()))

    def test_missing_initial_intermediate_final_and_unknown_output_fail(self):
        for missing in ("checkpoint-000.npz", "checkpoint-100.npz", "checkpoint-200.npz",
                        "diagnostics.json", "receipt.json"):
            outputs = declarations()
            outputs.pop(missing)
            with self.assertRaisesRegex(budget.BudgetError, "inventory"):
                budget.admit_inventory(outputs)
        outputs = declarations()
        outputs["undeclared-debug.log"] = ("diagnostic", 1024)
        with self.assertRaisesRegex(budget.BudgetError, "inventory"):
            budget.admit_inventory(outputs)

    def test_bool_float_nonpositive_caps_and_sizes_fail_closed(self):
        for invalid in (True, 2.0, 0, -1):
            outputs = declarations()
            outputs["weights.npz"] = ("checkpoint", invalid)
            with self.assertRaisesRegex(budget.BudgetError, "inventory_type"):
                budget.admit_inventory(outputs)
            outputs = declarations()
            measured = {name: 100 for name in outputs}
            projected = dict(measured)
            measured["weights.npz"] = invalid
            with self.assertRaisesRegex(budget.BudgetError, "artifact_bound"):
                budget.check_sizes(outputs, measured, projected)

    def test_individual_cap_fails_even_when_total_fits(self):
        outputs = declarations()
        values = {name: 1 for name in outputs}
        values["checkpoint-020.npz"] = 2 * budget.MIB + 1
        self.assertLess(sum(values.values()), budget.AGGREGATE_LIMIT)
        with self.assertRaisesRegex(budget.BudgetError, "artifact_bound: checkpoint-020"):
            budget.check_sizes(outputs, values, values)

    def test_aggregate_boundary_exact_pass_and_one_byte_over_fail(self):
        outputs = declarations()
        values = {name: 1 for name in outputs}
        for update in budget.CHECKPOINT_UPDATES:
            values[f"checkpoint-{update:03d}.npz"] = 2 * budget.MIB
        values["weights.npz"] = 8 * budget.MIB
        values["predictions.npz"] += budget.AGGREGATE_LIMIT - sum(values.values())
        self.assertEqual(budget.check_sizes(outputs, values, values), budget.AGGREGATE_LIMIT)
        values["predictions.npz"] += 1
        self.assertTrue(all(values[name] <= maximum for name, (_, maximum) in outputs.items()))
        with self.assertRaisesRegex(budget.BudgetError, "aggregate_bound"):
            budget.check_sizes(outputs, values, values)

    def test_optimistic_projection_cannot_be_smaller_than_measured_or_omit_file(self):
        outputs = declarations()
        measured = {name: 100 for name in outputs}
        projected = dict(measured)
        projected["diagnostics.json"] = 99
        with self.assertRaisesRegex(budget.BudgetError, "artifact_bound"):
            budget.check_sizes(outputs, measured, projected)
        projected = dict(measured)
        projected.pop("checkpoint-200.npz")
        with self.assertRaisesRegex(budget.BudgetError, "inventory"):
            budget.check_sizes(outputs, measured, projected)

    def test_checkpoint_cap_cannot_silently_expand_or_change_role(self):
        for specification in (("checkpoint", 8 * budget.MIB), ("trained_state", 2 * budget.MIB)):
            outputs = declarations()
            outputs["checkpoint-100.npz"] = specification
            with self.assertRaisesRegex(budget.BudgetError, "checkpoint_cap"):
                budget.admit_inventory(outputs)

    def test_regular_file_reader_rejects_symlink_directory_and_oversize(self):
        with tempfile.TemporaryDirectory(prefix="gn-budget-file-test-") as temporary:
            root = Path(temporary)
            target = root / "numeric"
            target.write_bytes(b"numeric")
            linked = root / "link"
            linked.symlink_to(target)
            for path in (linked, root):
                with self.assertRaisesRegex(budget.BudgetError, "unsafe_path"):
                    budget.regular_bytes(path)
            with self.assertRaisesRegex(budget.BudgetError, "file_size"):
                budget.regular_bytes(target, 3)

    def test_numeric_text_envelope_reserves_optional_floats_and_preserves_fixed_strings(self):
        value = {"loss": 0.25, "cosine": None, "nonzero_count": 20,
                 "checkpoint_bytes": 1500186, "rows": [1, 2], "label": "initialized",
                 "qualification_only": True}
        enlarged = budget.numeric_text_envelope(value)
        self.assertGreaterEqual(len(budget.canonical(enlarged)), len(budget.canonical(value)))
        self.assertEqual(len(budget.canonical(enlarged["loss"])), 24)
        self.assertEqual(enlarged["checkpoint_bytes"], 1500186)
        self.assertEqual(enlarged["label"], "initialized")
        self.assertIs(enlarged["qualification_only"], True)

    def test_normalization_uses_actual_trainer_json_bytes_including_newline(self):
        trainer = budget.load_trainer()
        value = {"scale": 1.0}
        self.assertEqual(trainer.canonical(value), trainer.base.canonical(value))
        self.assertEqual(trainer.canonical(value), budget.canonical(value) + b"\n")

    def test_actual_new_initialized_state_serialization_is_value_independent_without_optimizer(self):
        import numpy as np
        import torch

        trainer = budget.load_trainer()
        self.assertEqual(trainer.OUTPUTS, declarations())
        trainer.model.configure_cpu()
        torch.manual_seed(20261014)
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer forbidden")):
            network = trainer.model.PressureCorrectionNet()
            proof = trainer.model.initialization_proof(network)
            self.assertEqual(proof["parameter_count"], 372377)
            self.assertEqual(proof["raw_weight_bytes"], 1489508)
            with tempfile.TemporaryDirectory(prefix="gn-budget-numeric-test-") as temporary:
                root = Path(temporary)
                before = trainer.save_weights(network, root / "initialized.npz")
                original_size = (root / "initialized.npz").stat().st_size
                # Disposable arithmetic byte fixture, not another scientific
                # candidate or an optimizer update: change one stored value.
                with torch.no_grad():
                    next(network.parameters()).flatten()[0].add_(0.125)
                after = trainer.save_weights(network, root / "arithmetic-byte-fixture.npz")
                self.assertNotEqual(before, after)
                self.assertEqual(original_size, (root / "arithmetic-byte-fixture.npz").stat().st_size)
                self.assertLessEqual(original_size, 2 * budget.MIB)
                restored = trainer.read_npz(root / "arithmetic-byte-fixture.npz", after, 2 * budget.MIB)
                self.assertEqual(set(restored), set(network.state_dict()))
                self.assertTrue(all(v.dtype == np.dtype("<f4") for v in restored.values()))
                self.assertTrue(all(parameter.grad is None for parameter in network.parameters()))


if __name__ == "__main__":
    unittest.main()
