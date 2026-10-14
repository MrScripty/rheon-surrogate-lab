"""Lab project contract regressions; generic consumer checks are opt-in."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tuldok"))
import project
import worker


class ConfigContracts(unittest.TestCase):
    def test_receipt_publication_refuses_a_racing_file_or_dangling_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "receipt.json"
            project.write(path, {"status": "original"})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                project.write(path, {"status": "replacement"})
            self.assertEqual(path.read_bytes(), original)
            path.unlink()
            path.symlink_to(root / "missing-target")
            with self.assertRaises(FileExistsError):
                project.write(path, {"status": "replacement"})
            self.assertTrue(path.is_symlink())
            self.assertFalse((root / "missing-target").exists())

    def test_existing_or_dangling_process_receipt_is_preserved_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "new-run-process.json"
            receipt.write_text("retained work")
            with self.assertRaisesRegex(ValueError, "receipt destination already exists"):
                project.local_run(root / "unused-project", "train", root / "new-run", sys.executable, root / "unused-policy")
            self.assertEqual(receipt.read_text(), "retained work")
            self.assertFalse((root / "new-run").exists())
            receipt.unlink()
            receipt.symlink_to(root / "missing-retained-target")
            with self.assertRaisesRegex(ValueError, "receipt destination already exists"):
                project.local_run(root / "unused-project", "train", root / "new-run", sys.executable, root / "unused-policy")
            self.assertTrue(receipt.is_symlink())
            self.assertFalse((root / "missing-retained-target").exists())

    def test_declared_output_cap_and_inventory_fail_before_local_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "payload.json").write_bytes(b"123")
            recipe = {"outputs": [{"path": "payload.json", "max_bytes": 2}],
                      "resources": {"max_output_bytes": 100}}
            with self.assertRaisesRegex(ValueError, "declared cap"):
                project.validate_outputs(root, recipe)
            recipe["outputs"][0]["max_bytes"] = 4
            project.validate_outputs(root, recipe)
            (root / "undeclared.json").write_text("extra")
            with self.assertRaisesRegex(ValueError, "declared recipe"):
                project.validate_outputs(root, recipe)

    def test_only_empty_cargo_cache_marker_is_exempt_from_source_clean_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            self.assertTrue(project.cargo_source_clean(root))
            marker = root / ".cargo-ok"
            marker.write_bytes(b"")
            self.assertTrue(project.cargo_source_clean(root))
            marker.write_bytes(b"changed")
            self.assertFalse(project.cargo_source_clean(root))
            marker.write_bytes(b"")
            (root / "unexpected.rs").write_text("changed source")
            self.assertFalse(project.cargo_source_clean(root))

    def test_original_source_pins_and_fixed_split_contract(self):
        config = worker.read_config(ROOT / "tuldok/configs/pressure-v1.json")
        self.assertEqual(len(config["training_seeds"]) * config["training_frames"], 96)
        self.assertTrue(set(config["training_seeds"]).isdisjoint(config["test_seeds"]))
        self.assertTrue(set(config["validation_seeds"]).isdisjoint(config["test_seeds"]))
        for name, digest in worker.ALGORITHMS.items():
            self.assertEqual(hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest(), digest)

    def test_changed_protocol_bool_and_duplicate_fields_are_refused(self):
        value = json.loads((ROOT / "tuldok/configs/pressure-v1.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for field, changed in (("training_seeds", [900]), ("threads", True),
                                   ("repetitions", 1), ("device", "cuda")):
                path.write_text(json.dumps({**value, field: changed}))
                with self.assertRaisesRegex(ValueError, "Unsupported pressure configuration"):
                    worker.read_config(path)
            path.write_text('{"n":32,"n":32}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                worker.read_config(path)

    def test_native_pin_is_a_git_revision_and_root_baseline_is_preserved(self):
        manifest = (ROOT / "probes/tuldok-pressure/Cargo.toml").read_text()
        self.assertIn('rev = "' + project.RHEON_REVISION + '"', manifest)
        self.assertNotIn('path = "../../../Rheon"', manifest)
        self.assertIn('34235d33c886aa05b5bdf95d920b863d1fb2ad6f', (ROOT / "Cargo.toml").read_text())
        self.assertIn('path = "../../../Rheon"', (ROOT / "probes/trajectory-parallel/Cargo.toml").read_text())

    def test_invalid_scientific_value_is_not_a_success(self):
        for value in (float("nan"), float("inf"), -1, True, 1e-5):
            with self.assertRaisesRegex(ValueError, "numerical acceptance"):
                worker.finite(value, 1e-5, strict=True)
        worker.finite(1e-8, 1e-5, strict=True)


@unittest.skipUnless(os.environ.get("TULDOK_CONTRACT_ROOT"), "requires explicit read-only Tuldok source root")
class GenericConsumerContracts(unittest.TestCase):
    def test_both_recipes_are_accepted_by_the_actual_generic_reader(self):
        sources = {"worker.py": b"captured fixture", "native.py": b"fixture", project.NATIVE: b"native fixture"}
        with tempfile.TemporaryDirectory() as temporary:
            for command in ("train", "compare"):
                path = Path(temporary) / (command + ".json")
                path.write_text(json.dumps(project.definition(command, sources, "0" * 64)))
                # This tests representation only. Zero digest is an explicit
                # fixture; no capture, approval, runtime qualification or Run.
                code = "import sys; from pathlib import Path; import recipe_contract; d=recipe_contract.decode_definition(Path(sys.argv[1]).read_bytes()); print(d['id'])"
                result = subprocess.run([sys.executable, "-c", code, str(path)],
                                        env={**os.environ, "PYTHONPATH": os.environ["TULDOK_CONTRACT_ROOT"]},
                                        capture_output=True, text=True, check=True)
                self.assertIn("lab-pressure-" + command, result.stdout)

    def test_project_reference_schema_is_accepted(self):
        value = {"schema_version": 1, "id": "1" * 32, "name": "Lab project",
                 "data": [], "recipes": [{"id": "train", "label": "Train", "path": "recipes/train.json", "sha256": "2" * 64}],
                 "evidence": [], "requirements": [{"id": "cpu", "label": "Qualified CPU snapshot required"}]}
        code = "import json,sys; import projects; projects.manifest(json.loads(sys.argv[1])); print('accepted')"
        result = subprocess.run([sys.executable, "-c", code, json.dumps(value)],
                                env={**os.environ, "PYTHONPATH": os.environ["TULDOK_CONTRACT_ROOT"]},
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "accepted")


if __name__ == "__main__":
    unittest.main()
